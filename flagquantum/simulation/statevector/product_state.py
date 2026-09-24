"""Exact CPU statevector execution over dynamically merged product components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ..gate_matrix import gate_matrix as _gate_matrix
from .controlled_phase import (
    _apply_controlled_phase_graph_cpu,
    _controlled_phase_graph_factors_cpu,
)
from .operations import (
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_single_qubit_fixed,
    _diagonal_region,
    _environment_flag,
    _fused_gate_matrix,
)
from .program import (
    _StatevectorControlledPhaseDecompositionStep,
    _StatevectorControlledPhaseGraphStep,
    _StatevectorCXSequenceStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorProgramStep,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


_PRODUCT_STATE_MIN_WIRES = 16
_PRODUCT_STATE_MAX_ESTIMATED_WORK_RATIO = 0.30
# Static 18- and 20-wire Clifford plans measured 1.96x and 2.97x faster at
# estimated ratios 0.350 and 0.317. Keep the evidence-bound extension narrow;
# every other program retains the established conservative ceiling.
_PRODUCT_STATE_MAX_STATIC_CLIFFORD_WORK_RATIO = 0.36
# Weighted phase-graph plans measured faster than dense graph execution at
# estimated ratios below 0.45 on 18 and 22 wires. This extension is selected
# only when the compiled plan contains that exact static graph step.
_PRODUCT_STATE_MAX_CONTROLLED_PHASE_GRAPH_WORK_RATIO = 0.45
_STATIC_CLIFFORD_GATES = frozenset({"h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"})
_FIXED_SINGLE_QUBIT_CLIFFORD_GATES = frozenset({"h", "s", "sdg", "y", "z"})


def _cpu_product_state_fixed_clifford_enabled() -> bool:
    """Whether product components use fixed CPU Clifford kernels."""

    return _environment_flag("FQ_CPU_PRODUCT_STATE_FIXED_CLIFFORD", default=True)


def _apply_fixed_clifford_gate(
    state: torch.Tensor,
    name: str,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    axis = int(wire) + 1
    tensor = state.reshape((state.shape[0],) + (2,) * n_wires)
    zero, one = tensor.unbind(dim=axis)
    if name == "h":
        scale = 2**-0.5
        values = ((zero + one) * scale, (zero - one) * scale)
    elif name == "s":
        values = (zero, 1j * one)
    elif name == "sdg":
        values = (zero, -1j * one)
    elif name == "z":
        values = (zero, -one)
    else:
        return _apply_single_qubit_fixed(state, name, wire, n_wires)
    return torch.stack(values, dim=axis).reshape(state.shape)


@dataclass(frozen=True)
class _ProductComponent:
    wires: tuple[int, ...]
    state: torch.Tensor


def _step_wire_groups(step: _StatevectorProgramStep) -> tuple[tuple[int, ...], ...]:
    if isinstance(step, _StatevectorGateStep):
        return (tuple(step.instruction.wires),)
    if isinstance(step, _StatevectorFusedGateStep):
        return (step.wires,)
    if isinstance(step, _StatevectorControlledPhaseDecompositionStep):
        return ((step.control, step.target),)
    if isinstance(step, _StatevectorControlledPhaseGraphStep):
        return (
            tuple(
                sorted(
                    {
                        wire
                        for control, target, _ in step.edges
                        for wire in (control, target)
                    }
                )
            ),
        )
    if isinstance(step, _StatevectorCXSequenceStep):
        return tuple(zip(step.controls, step.targets, strict=True))
    return ()


def _swap_wires(step: _StatevectorProgramStep) -> tuple[int, int] | None:
    if not isinstance(step, _StatevectorGateStep):
        return None
    if canonical_opcode(step.instruction.name) != "swap":
        return None
    left, right = step.instruction.wires
    return left, right


def _is_static_clifford_step(step: _StatevectorProgramStep) -> bool:
    """Whether a compiled step is an exact parameter-free Clifford operation."""

    instructions: tuple[Instruction, ...]
    if isinstance(step, _StatevectorCXSequenceStep):
        return True
    if isinstance(step, _StatevectorGateStep):
        instructions = (step.instruction,)
    elif isinstance(step, _StatevectorFusedGateStep):
        instructions = step.instructions
    else:
        return False
    return all(
        instruction.matrix is None
        and not instruction.params
        and canonical_opcode(instruction.name) in _STATIC_CLIFFORD_GATES
        for instruction in instructions
    )


def product_state_execution_is_beneficial(
    program: Sequence[_StatevectorProgramStep],
    n_wires: int,
    *,
    enable_swap_remapping: bool = True,
) -> bool:
    """Select plans whose component work is far below dense execution work."""

    if n_wires < _PRODUCT_STATE_MIN_WIRES or not program:
        return False
    component_by_wire = list(range(n_wires))
    component_members = {wire: {wire} for wire in range(n_wires)}

    def merge(wires: Sequence[int]) -> int:
        component_ids = tuple(dict.fromkeys(component_by_wire[wire] for wire in wires))
        merged_id = component_ids[0]
        for component_id in component_ids[1:]:
            moved_wires = component_members.pop(component_id)
            component_members[merged_id].update(moved_wires)
            for wire in moved_wires:
                component_by_wire[wire] = merged_id
        return len(component_members[merged_id])

    def remap_swap(left: int, right: int) -> tuple[int, ...]:
        left_id = component_by_wire[left]
        right_id = component_by_wire[right]
        if left_id == right_id:
            return (len(component_members[left_id]),)
        component_members[left_id].remove(left)
        component_members[left_id].add(right)
        component_members[right_id].remove(right)
        component_members[right_id].add(left)
        component_by_wire[left], component_by_wire[right] = right_id, left_id
        return (
            len(component_members[left_id]),
            len(component_members[right_id]),
        )

    estimated_work = 2**n_wires
    operation_count = 0
    static_clifford = True
    contains_controlled_phase_graph = False
    for step in program:
        static_clifford = static_clifford and _is_static_clifford_step(step)
        contains_controlled_phase_graph = contains_controlled_phase_graph or isinstance(
            step, _StatevectorControlledPhaseGraphStep
        )
        swap_wires = _swap_wires(step)
        if enable_swap_remapping and swap_wires is not None:
            left, right = swap_wires
            estimated_work += sum(2**size for size in remap_swap(left, right))
            operation_count += 1
            continue
        groups = _step_wire_groups(step)
        if not groups:
            return False
        for wires in groups:
            estimated_work += 2 ** merge(wires)
            operation_count += 1
    dense_work = max(len(program), operation_count) * 2**n_wires
    if contains_controlled_phase_graph:
        maximum_ratio = _PRODUCT_STATE_MAX_CONTROLLED_PHASE_GRAPH_WORK_RATIO
    elif static_clifford:
        maximum_ratio = _PRODUCT_STATE_MAX_STATIC_CLIFFORD_WORK_RATIO
    else:
        maximum_ratio = _PRODUCT_STATE_MAX_ESTIMATED_WORK_RATIO
    return bool(estimated_work <= maximum_ratio * dense_work)


def _merge_components(
    components: dict[int, _ProductComponent], wires: Sequence[int]
) -> _ProductComponent:
    selected: list[_ProductComponent] = []
    seen: set[int] = set()
    for wire in wires:
        component = components[wire]
        identity = id(component)
        if identity not in seen:
            selected.append(component)
            seen.add(identity)
    if len(selected) == 1:
        return selected[0]
    selected.sort(key=lambda component: component.wires)
    concatenated_wires = tuple(wire for item in selected for wire in item.wires)
    merged = selected[0].state
    for item in selected[1:]:
        merged = (merged.unsqueeze(-1) * item.state.unsqueeze(1)).reshape(1, -1)
    ordered_wires = tuple(sorted(concatenated_wires))
    if concatenated_wires != ordered_wires:
        axes = {wire: index + 1 for index, wire in enumerate(concatenated_wires)}
        merged = (
            merged.reshape((1,) + (2,) * len(concatenated_wires))
            .permute((0,) + tuple(axes[wire] for wire in ordered_wires))
            .reshape(1, -1)
            .contiguous()
        )
    component = _ProductComponent(ordered_wires, merged)
    for wire in ordered_wires:
        components[wire] = component
    return component


def _relabel_component(
    component: _ProductComponent, substitutions: dict[int, int]
) -> _ProductComponent:
    relabeled_wires = tuple(substitutions.get(wire, wire) for wire in component.wires)
    ordered_wires = tuple(sorted(relabeled_wires))
    state = component.state
    if relabeled_wires != ordered_wires:
        axes = {wire: index + 1 for index, wire in enumerate(relabeled_wires)}
        state = (
            state.reshape((1,) + (2,) * len(relabeled_wires))
            .permute((0,) + tuple(axes[wire] for wire in ordered_wires))
            .reshape(1, -1)
            .contiguous()
        )
    return _ProductComponent(ordered_wires, state)


def _remap_swap_components(
    components: dict[int, _ProductComponent], left: int, right: int
) -> None:
    left_component = components[left]
    right_component = components[right]
    substitutions = {left: right, right: left}
    updated_components: tuple[_ProductComponent, ...]
    if left_component is right_component:
        updated_components = (_relabel_component(left_component, substitutions),)
    else:
        updated_components = (
            _relabel_component(left_component, substitutions),
            _relabel_component(right_component, substitutions),
        )
    for component in updated_components:
        for wire in component.wires:
            components[wire] = component


def _controlled_phase_matrix(
    step: _StatevectorControlledPhaseDecompositionStep,
    state: torch.Tensor,
) -> torch.Tensor:
    real_dtype = torch.float32 if state.dtype == torch.complex64 else torch.float64
    angle = torch.tensor(step.half_angle / 2, device=state.device, dtype=real_dtype)
    phases = torch.stack((-angle, -angle, -angle, 3 * angle))
    diagonal = torch.polar(torch.ones_like(phases), phases).to(dtype=state.dtype)
    return torch.diag(diagonal)


def _apply_step(
    component: _ProductComponent,
    step: (
        _StatevectorGateStep
        | _StatevectorFusedGateStep
        | _StatevectorControlledPhaseDecompositionStep
        | _StatevectorControlledPhaseGraphStep
    ),
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    constant_cache: dict[tuple[int, str, torch.dtype, int], torch.Tensor] | None,
    enable_fixed_clifford: bool,
) -> torch.Tensor:
    global_wires: tuple[int, ...]
    if isinstance(step, _StatevectorControlledPhaseGraphStep):
        global_wires = tuple(
            sorted(
                {
                    wire
                    for control, target, _ in step.edges
                    for wire in (control, target)
                }
            )
        )
        key = (
            id(step),
            str(component.state.device),
            component.state.dtype,
            len(global_wires),
        )
        factors = None if constant_cache is None else constant_cache.get(key)
        if factors is None:
            factors, built_wires = _controlled_phase_graph_factors_cpu(
                step.edges,
                device=component.state.device,
                dtype=component.state.dtype,
            )
            if built_wires != global_wires:
                raise RuntimeError(
                    "compiled controlled-phase graph wire order changed unexpectedly"
                )
            if constant_cache is not None:
                constant_cache[key] = factors
        local_wires = tuple(component.wires.index(wire) for wire in global_wires)
        return _apply_controlled_phase_graph_cpu(
            component.state,
            factors,
            local_wires,
            len(component.wires),
        )
    if isinstance(step, _StatevectorControlledPhaseDecompositionStep):
        global_wires = (step.control, step.target)
        matrix = _controlled_phase_matrix(step, component.state)
        diagonal = True
    elif isinstance(step, _StatevectorFusedGateStep):
        global_wires = step.wires
        matrix = _fused_gate_matrix(
            step,
            bsz=1,
            device=component.state.device,
            dtype=component.state.dtype,
            parameter_bindings=parameter_bindings,
        )
        diagonal = step.diagonal
    else:
        instruction = step.instruction
        global_wires = tuple(instruction.wires)
        name = canonical_opcode(instruction.name)
        local_wires = tuple(component.wires.index(wire) for wire in global_wires)
        if name in {"x", "cx", "swap"}:
            return _apply_fixed_permutation(
                component.state, name, local_wires, len(component.wires)
            )
        if enable_fixed_clifford and name in _FIXED_SINGLE_QUBIT_CLIFFORD_GATES:
            return _apply_fixed_clifford_gate(
                component.state, name, local_wires[0], len(component.wires)
            )
        matrix = _gate_matrix(
            instruction,
            bsz=1,
            device=component.state.device,
            dtype=component.state.dtype,
            parameter_bindings=parameter_bindings,
        )
        diagonal = _diagonal_region((instruction,))
    local_wires = tuple(component.wires.index(wire) for wire in global_wires)
    apply = _apply_diagonal_matrix if diagonal else _apply_matrix
    return apply(
        component.state,
        matrix,
        local_wires,
        len(component.wires),
    )


def execute_product_state_program(
    program: Sequence[_StatevectorProgramStep],
    *,
    n_wires: int,
    device: torch.device,
    dtype: torch.dtype,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    enable_swap_remapping: bool = True,
    enable_fixed_clifford: bool = True,
    constant_cache: dict[tuple[int, str, torch.dtype, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute an exact program while merging components only when required."""

    zero = torch.zeros((1, 2), device=device, dtype=dtype)
    zero[:, 0] = 1
    components = {
        wire: _ProductComponent((wire,), zero.clone()) for wire in range(n_wires)
    }
    for step in program:
        swap_wires = _swap_wires(step)
        if enable_swap_remapping and swap_wires is not None:
            left, right = swap_wires
            _remap_swap_components(components, left, right)
            continue
        if isinstance(step, _StatevectorCXSequenceStep):
            for control, target in zip(step.controls, step.targets, strict=True):
                component = _merge_components(components, (control, target))
                local_wires = (
                    component.wires.index(control),
                    component.wires.index(target),
                )
                state = _apply_fixed_permutation(
                    component.state, "cx", local_wires, len(component.wires)
                )
                updated = _ProductComponent(component.wires, state)
                for wire in updated.wires:
                    components[wire] = updated
            continue
        groups = _step_wire_groups(step)
        if len(groups) != 1 or not isinstance(
            step,
            (
                _StatevectorGateStep,
                _StatevectorFusedGateStep,
                _StatevectorControlledPhaseDecompositionStep,
                _StatevectorControlledPhaseGraphStep,
            ),
        ):
            raise TypeError(f"unsupported product-state program step: {type(step)!r}")
        component = _merge_components(components, groups[0])
        updated = _ProductComponent(
            component.wires,
            _apply_step(
                component,
                step,
                parameter_bindings,
                constant_cache,
                enable_fixed_clifford,
            ),
        )
        for wire in updated.wires:
            components[wire] = updated
    return _merge_components(components, tuple(range(n_wires))).state
