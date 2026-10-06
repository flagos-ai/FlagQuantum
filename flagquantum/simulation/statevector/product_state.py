"""Exact CPU statevector execution over dynamically merged product components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ..gate_matrix import gate_matrix as _gate_matrix
from ..native_cpu.rotation import fused_static_clifford_layer_
from .clifford_matching import apply_native_clifford_matching
from .controlled_phase import (
    _apply_controlled_phase_graph_cpu,
    _controlled_phase_graph_factors_cpu,
)
from .cz_graph import _apply_cz_graph_cpu, _cz_graph_signs_cpu
from .operations import (
    _apply_cx_sequence_gather,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_single_qubit_fixed,
    _cx_sequence_permutation_index,
    _diagonal_region,
    _environment_flag,
    _fused_gate_matrix,
)
from .program import (
    _StatevectorCliffordMatchingStep,
    _StatevectorControlledPhaseDecompositionStep,
    _StatevectorControlledPhaseGraphStep,
    _StatevectorCXSequenceStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorProgramStep,
)
from .wire_permutation import _apply_wire_permutation_gather

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...circuit import _StatevectorExecutionStatistics


_PRODUCT_STATE_MIN_WIRES = 16
_PRODUCT_STATE_MAX_ESTIMATED_WORK_RATIO = 0.30
# Static 18- and 20-qubit Clifford plans measured 1.96x and 2.97x faster at
# estimated ratios 0.350 and 0.317. Keep the evidence-bound extension narrow;
# every other program retains the established conservative ceiling.
_PRODUCT_STATE_MAX_STATIC_CLIFFORD_WORK_RATIO = 0.36
# Weighted phase-graph plans measured faster than dense graph execution at
# estimated ratios below 0.45 on 18 and 22 qubits. This extension is selected
# only when the compiled plan contains that exact static graph step.
_PRODUCT_STATE_MAX_CONTROLLED_PHASE_GRAPH_WORK_RATIO = 0.45
_STATIC_CLIFFORD_GATES = frozenset({"h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"})
_FIXED_SINGLE_QUBIT_CLIFFORD_GATES = frozenset({"h", "s", "sdg", "y", "z"})
_PRODUCT_STATE_CX_GATHER_MINIMUM_LENGTH = 5
_PRODUCT_STATE_NATIVE_STATIC_CLIFFORD_MIN_WIRES = 16
_PRODUCT_STATE_NATIVE_MIXED_CLIFFORD_MIN_WIRES = 16
_NATIVE_CLIFFORD_CODES = {"h": 1, "s": 2, "x": 4}


def _cpu_product_state_execution_enabled() -> bool:
    """Whether eligible CPU circuits may keep product components separate."""

    return _environment_flag("FQ_CPU_PRODUCT_STATE_EXECUTION", default=True)


def _cpu_product_state_swap_remapping_enabled() -> bool:
    """Whether product components treat SWAP as qubit remapping."""

    return _environment_flag("FQ_CPU_PRODUCT_STATE_SWAP_REMAPPING", default=True)


def _cpu_product_state_fixed_clifford_enabled() -> bool:
    """Whether product components use fixed CPU Clifford kernels."""

    return _environment_flag("FQ_CPU_PRODUCT_STATE_FIXED_CLIFFORD", default=True)


def _cpu_product_state_clifford_matching_enabled() -> bool:
    """Whether product-state Clifford matchings may share permutation passes."""

    return _environment_flag("FQ_CPU_PRODUCT_STATE_CLIFFORD_MATCHING", default=True)


def _cpu_product_state_native_static_clifford_enabled() -> bool:
    """Whether wide product components may use the native Clifford kernel."""

    return _environment_flag(
        "FQ_CPU_PRODUCT_STATE_NATIVE_STATIC_CLIFFORD", default=True
    )


def _cpu_product_state_deferred_swap_enabled() -> bool:
    """Whether product components defer SWAP tensor materialization."""

    return _environment_flag("FQ_CPU_PRODUCT_STATE_DEFER_SWAP", default=True)


def _apply_fixed_clifford_gate(
    state: torch.Tensor,
    name: str,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    axis = int(qubit) + 1
    tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
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
        return _apply_single_qubit_fixed(state, name, qubit, n_qubits)
    return torch.stack(values, dim=axis).reshape(state.shape)


def _fixed_clifford_layer_gate(
    step: _StatevectorProgramStep,
) -> tuple[str, int] | None:
    if not isinstance(step, _StatevectorGateStep):
        return None
    instruction = step.instruction
    name = canonical_opcode(instruction.name)
    if (
        instruction.matrix is not None
        or instruction.params
        or len(instruction.wires) != 1
        or name not in {"h", "s", "x"}
    ):
        return None
    return name, int(instruction.wires[0])


def _apply_fixed_clifford_layer(
    component: _ProductComponent,
    gates: Sequence[tuple[str, int]],
) -> torch.Tensor:
    """Apply one disjoint H/S/X layer with bounded passes over a component."""

    n_qubits = len(component.qubits)
    local_gates = tuple((name, component.qubits.index(qubit)) for name, qubit in gates)
    if len(local_gates) == 1:
        name, qubit = local_gates[0]
        if name == "x":
            return _apply_fixed_permutation(component.state, name, (qubit,), n_qubits)
        return _apply_fixed_clifford_gate(component.state, name, qubit, n_qubits)

    if (
        n_qubits >= _PRODUCT_STATE_NATIVE_STATIC_CLIFFORD_MIN_WIRES
        and not component.state.requires_grad
        and _cpu_product_state_native_static_clifford_enabled()
    ):
        output = component.state.clone()
        gate_codes = torch.tensor(
            tuple(_NATIVE_CLIFFORD_CODES[name] for name, _ in local_gates),
            dtype=torch.int8,
            device=output.device,
        )
        wires = torch.tensor(
            tuple(wire for _, wire in local_gates),
            dtype=torch.int64,
            device=output.device,
        )
        if fused_static_clifford_layer_(
            output,
            gate_codes,
            wires,
            n_qubits=n_qubits,
        ):
            return output

    state = component.state
    for name, qubit in local_gates:
        if name == "h":
            state = _apply_fixed_clifford_gate(state, name, qubit, n_qubits)

    s_qubits = tuple(qubit for name, qubit in local_gates if name == "s")
    if s_qubits:
        factor_shape = [1] * (n_qubits + 1)
        phases = torch.ones((), device=state.device, dtype=state.dtype)
        local_phase = torch.tensor((1, 1j), device=state.device, dtype=state.dtype)
        for qubit in s_qubits:
            factor_shape[qubit + 1] = 2
            phases = phases.unsqueeze(-1) * local_phase
        tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
        state = (tensor * phases.reshape(factor_shape)).reshape(state.shape)

    x_qubits = tuple(qubit for name, qubit in local_gates if name == "x")
    if x_qubits:
        tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
        state = torch.flip(tensor, dims=tuple(qubit + 1 for qubit in x_qubits)).reshape(
            state.shape
        )
    return state


@dataclass(frozen=True)
class _ProductComponent:
    qubits: tuple[int, ...]
    state: torch.Tensor


def _apply_product_clifford_matching(
    component: _ProductComponent,
    step: _StatevectorCliffordMatchingStep,
    execution_statistics: _StatevectorExecutionStatistics | None,
) -> _ProductComponent:
    """Apply one matching after its edges have joined product components."""

    wire_set = set(component.qubits)
    controls = tuple(
        component.qubits.index(control)
        for control, target in zip(step.controls, step.targets, strict=True)
        if control in wire_set and target in wire_set
    )
    targets = tuple(
        component.qubits.index(target)
        for control, target in zip(step.controls, step.targets, strict=True)
        if control in wire_set and target in wire_set
    )
    cz_edges = tuple(
        (component.qubits.index(left), component.qubits.index(right))
        for left, right in step.cz_edges
        if left in wire_set and right in wire_set
    )
    local_step = _StatevectorCliffordMatchingStep(
        controls=controls,
        targets=targets,
        cz_edges=cz_edges,
    )
    matching_state: torch.Tensor | None = None
    if (
        len(component.qubits) >= _PRODUCT_STATE_NATIVE_MIXED_CLIFFORD_MIN_WIRES
        and controls
    ):
        matching_state, _ = apply_native_clifford_matching(
            local_step,
            component.state,
            n_qubits=len(component.qubits),
            scratch=None,
            reuse_output=False,
            owns_state=False,
            mapping_builder=_cx_sequence_permutation_index,
        )
    if matching_state is not None:
        if execution_statistics is not None:
            execution_statistics["native_cpu_clifford_matching_regions"] = (
                execution_statistics.get("native_cpu_clifford_matching_regions", 0) + 1
            )
        return _ProductComponent(component.qubits, matching_state)

    matching_state = component.state
    if controls:
        if len(controls) >= _PRODUCT_STATE_CX_GATHER_MINIMUM_LENGTH:
            matching_state = _apply_cx_sequence_gather(
                matching_state,
                controls,
                targets,
                len(component.qubits),
            )
        else:
            for control, target in zip(controls, targets, strict=True):
                matching_state = _apply_fixed_permutation(
                    matching_state,
                    "cx",
                    (control, target),
                    len(component.qubits),
                )
    if cz_edges:
        signs, qubits = _cz_graph_signs_cpu(cz_edges, device=matching_state.device)
        matching_state = _apply_cz_graph_cpu(
            matching_state,
            signs,
            qubits,
            len(component.qubits),
        )
    return _ProductComponent(component.qubits, matching_state)


def _step_qubit_groups(step: _StatevectorProgramStep) -> tuple[tuple[int, ...], ...]:
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
                        qubit
                        for control, target, _ in step.edges
                        for qubit in (control, target)
                    }
                )
            ),
        )
    if isinstance(step, _StatevectorCXSequenceStep):
        return tuple(zip(step.controls, step.targets, strict=True))
    if isinstance(step, _StatevectorCliffordMatchingStep):
        return (*step.cz_edges, *tuple(zip(step.controls, step.targets, strict=True)))
    return ()


def _swap_qubits(step: _StatevectorProgramStep) -> tuple[int, int] | None:
    if not isinstance(step, _StatevectorGateStep):
        return None
    if canonical_opcode(step.instruction.name) != "swap":
        return None
    left, right = step.instruction.wires
    return left, right


def _is_static_clifford_step(step: _StatevectorProgramStep) -> bool:
    """Whether a compiled step is an exact parameter-free Clifford operation."""

    instructions: tuple[Instruction, ...]
    if isinstance(step, (_StatevectorCXSequenceStep, _StatevectorCliffordMatchingStep)):
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
    n_qubits: int,
    *,
    enable_swap_remapping: bool = True,
) -> bool:
    """Select plans whose component work is far below dense execution work."""

    if n_qubits < _PRODUCT_STATE_MIN_WIRES or not program:
        return False
    component_by_qubit = list(range(n_qubits))
    component_members = {qubit: {qubit} for qubit in range(n_qubits)}

    def merge(qubits: Sequence[int]) -> int:
        component_ids = tuple(
            dict.fromkeys(component_by_qubit[qubit] for qubit in qubits)
        )
        merged_id = component_ids[0]
        for component_id in component_ids[1:]:
            moved_qubits = component_members.pop(component_id)
            component_members[merged_id].update(moved_qubits)
            for qubit in moved_qubits:
                component_by_qubit[qubit] = merged_id
        return len(component_members[merged_id])

    def remap_swap(left: int, right: int) -> tuple[int, ...]:
        left_id = component_by_qubit[left]
        right_id = component_by_qubit[right]
        if left_id == right_id:
            return (len(component_members[left_id]),)
        component_members[left_id].remove(left)
        component_members[left_id].add(right)
        component_members[right_id].remove(right)
        component_members[right_id].add(left)
        component_by_qubit[left], component_by_qubit[right] = right_id, left_id
        return (
            len(component_members[left_id]),
            len(component_members[right_id]),
        )

    estimated_work = 2**n_qubits
    operation_count = 0
    static_clifford = True
    contains_controlled_phase_graph = False
    for step in program:
        static_clifford = static_clifford and _is_static_clifford_step(step)
        contains_controlled_phase_graph = contains_controlled_phase_graph or isinstance(
            step, _StatevectorControlledPhaseGraphStep
        )
        swap_qubits = _swap_qubits(step)
        if enable_swap_remapping and swap_qubits is not None:
            left, right = swap_qubits
            estimated_work += sum(2**size for size in remap_swap(left, right))
            operation_count += 1
            continue
        groups = _step_qubit_groups(step)
        if not groups:
            return False
        for qubits in groups:
            estimated_work += 2 ** merge(qubits)
            operation_count += 1
    dense_work = max(len(program), operation_count) * 2**n_qubits
    if contains_controlled_phase_graph:
        maximum_ratio = _PRODUCT_STATE_MAX_CONTROLLED_PHASE_GRAPH_WORK_RATIO
    elif static_clifford:
        maximum_ratio = _PRODUCT_STATE_MAX_STATIC_CLIFFORD_WORK_RATIO
    else:
        maximum_ratio = _PRODUCT_STATE_MAX_ESTIMATED_WORK_RATIO
    return bool(estimated_work <= maximum_ratio * dense_work)


def _merge_components(
    components: dict[int, _ProductComponent], qubits: Sequence[int]
) -> _ProductComponent:
    selected: list[_ProductComponent] = []
    seen: set[int] = set()
    for qubit in qubits:
        component = components[qubit]
        identity = id(component)
        if identity not in seen:
            selected.append(component)
            seen.add(identity)
    if len(selected) == 1:
        return selected[0]
    selected.sort(key=lambda component: component.qubits)
    concatenated_qubits = tuple(qubit for item in selected for qubit in item.qubits)
    merged = selected[0].state
    for item in selected[1:]:
        merged = (merged.unsqueeze(-1) * item.state.unsqueeze(1)).reshape(1, -1)
    ordered_qubits = tuple(sorted(concatenated_qubits))
    if concatenated_qubits != ordered_qubits:
        axes = {qubit: index + 1 for index, qubit in enumerate(concatenated_qubits)}
        merged = (
            merged.reshape((1,) + (2,) * len(concatenated_qubits))
            .permute((0,) + tuple(axes[qubit] for qubit in ordered_qubits))
            .reshape(1, -1)
            .contiguous()
        )
    component = _ProductComponent(ordered_qubits, merged)
    for qubit in ordered_qubits:
        components[qubit] = component
    return component


def _relabel_component(
    component: _ProductComponent,
    substitutions: dict[int, int],
    *,
    defer_materialization: bool,
) -> _ProductComponent:
    relabeled_qubits = tuple(
        substitutions.get(qubit, qubit) for qubit in component.qubits
    )
    if defer_materialization:
        return _ProductComponent(relabeled_qubits, component.state)
    return _canonicalize_component(
        _ProductComponent(relabeled_qubits, component.state), use_gather=False
    )


def _canonicalize_component(
    component: _ProductComponent, *, use_gather: bool
) -> _ProductComponent:
    """Return a component whose tensor axes follow ascending logical qubits."""

    relabeled_qubits = component.qubits
    ordered_qubits = tuple(sorted(relabeled_qubits))
    if relabeled_qubits != ordered_qubits:
        if use_gather:
            state = _apply_wire_permutation_gather(component.state, relabeled_qubits)
        else:
            axes = {qubit: index + 1 for index, qubit in enumerate(relabeled_qubits)}
            state = (
                component.state.reshape(
                    (component.state.shape[0],) + (2,) * len(relabeled_qubits)
                )
                .permute((0,) + tuple(axes[qubit] for qubit in ordered_qubits))
                .reshape(component.state.shape)
                .contiguous()
            )
        return _ProductComponent(ordered_qubits, state)
    return component


def _remap_swap_components(
    components: dict[int, _ProductComponent],
    left: int,
    right: int,
    *,
    defer_materialization: bool,
) -> None:
    left_component = components[left]
    right_component = components[right]
    substitutions = {left: right, right: left}
    updated_components: tuple[_ProductComponent, ...]
    if left_component is right_component:
        updated_components = (
            _relabel_component(
                left_component,
                substitutions,
                defer_materialization=defer_materialization,
            ),
        )
    else:
        updated_components = (
            _relabel_component(
                left_component,
                substitutions,
                defer_materialization=defer_materialization,
            ),
            _relabel_component(
                right_component,
                substitutions,
                defer_materialization=defer_materialization,
            ),
        )
    for component in updated_components:
        for qubit in component.qubits:
            components[qubit] = component


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
    global_qubits: tuple[int, ...]
    if isinstance(step, _StatevectorControlledPhaseGraphStep):
        global_qubits = tuple(
            sorted(
                {
                    qubit
                    for control, target, _ in step.edges
                    for qubit in (control, target)
                }
            )
        )
        key = (
            id(step),
            str(component.state.device),
            component.state.dtype,
            len(global_qubits),
        )
        factors = None if constant_cache is None else constant_cache.get(key)
        if factors is None:
            factors, built_qubits = _controlled_phase_graph_factors_cpu(
                step.edges,
                device=component.state.device,
                dtype=component.state.dtype,
            )
            if built_qubits != global_qubits:
                raise RuntimeError(
                    "compiled controlled-phase graph qubit order changed unexpectedly"
                )
            if constant_cache is not None:
                constant_cache[key] = factors
        local_qubits = tuple(component.qubits.index(qubit) for qubit in global_qubits)
        return _apply_controlled_phase_graph_cpu(
            component.state,
            factors,
            local_qubits,
            len(component.qubits),
        )
    if isinstance(step, _StatevectorControlledPhaseDecompositionStep):
        global_qubits = (step.control, step.target)
        matrix = _controlled_phase_matrix(step, component.state)
        diagonal = True
    elif isinstance(step, _StatevectorFusedGateStep):
        global_qubits = step.wires
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
        global_qubits = tuple(instruction.wires)
        name = canonical_opcode(instruction.name)
        local_qubits = tuple(component.qubits.index(qubit) for qubit in global_qubits)
        if name in {"x", "cx", "swap"}:
            return _apply_fixed_permutation(
                component.state, name, local_qubits, len(component.qubits)
            )
        if enable_fixed_clifford and name in _FIXED_SINGLE_QUBIT_CLIFFORD_GATES:
            return _apply_fixed_clifford_gate(
                component.state, name, local_qubits[0], len(component.qubits)
            )
        matrix = _gate_matrix(
            instruction,
            bsz=1,
            device=component.state.device,
            dtype=component.state.dtype,
            parameter_bindings=parameter_bindings,
        )
        diagonal = _diagonal_region((instruction,))
    local_qubits = tuple(component.qubits.index(qubit) for qubit in global_qubits)
    apply = _apply_diagonal_matrix if diagonal else _apply_matrix
    return apply(
        component.state,
        matrix,
        local_qubits,
        len(component.qubits),
    )


def execute_product_state_program(
    program: Sequence[_StatevectorProgramStep],
    *,
    n_qubits: int,
    device: torch.device,
    dtype: torch.dtype,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    enable_swap_remapping: bool = True,
    enable_deferred_swap: bool = True,
    enable_fixed_clifford: bool = True,
    enable_clifford_matching: bool = True,
    constant_cache: dict[tuple[int, str, torch.dtype, int], torch.Tensor] | None = None,
    execution_statistics: _StatevectorExecutionStatistics | None = None,
) -> torch.Tensor:
    """Execute an exact program while merging components only when required."""

    zero = torch.zeros((1, 2), device=device, dtype=dtype)
    zero[:, 0] = 1
    components = {
        qubit: _ProductComponent((qubit,), zero.clone()) for qubit in range(n_qubits)
    }
    index = 0
    while index < len(program):
        step = program[index]
        if enable_clifford_matching and enable_fixed_clifford:
            first_gate = _fixed_clifford_layer_gate(step)
            if first_gate is not None:
                gates = [first_gate]
                occupied_qubits = {first_gate[1]}
                cursor = index + 1
                while cursor < len(program):
                    candidate = _fixed_clifford_layer_gate(program[cursor])
                    if candidate is None or candidate[1] in occupied_qubits:
                        break
                    gates.append(candidate)
                    occupied_qubits.add(candidate[1])
                    cursor += 1
                if len(gates) >= 2:
                    gates_by_component: dict[
                        int, tuple[_ProductComponent, list[tuple[str, int]]]
                    ] = {}
                    for name, qubit in gates:
                        component = components[qubit]
                        matching_entry = gates_by_component.get(id(component))
                        if matching_entry is None:
                            matching_entry = (component, [])
                            gates_by_component[id(component)] = matching_entry
                        matching_entry[1].append((name, qubit))
                    if not any(
                        len(component_gates) >= 2
                        for _, component_gates in gates_by_component.values()
                    ):
                        gates_by_component.clear()
                    for component, component_gates in gates_by_component.values():
                        updated = _ProductComponent(
                            component.qubits,
                            _apply_fixed_clifford_layer(component, component_gates),
                        )
                        for qubit in updated.qubits:
                            components[qubit] = updated
                    if gates_by_component:
                        index = cursor
                        continue
        swap_qubits = _swap_qubits(step)
        if enable_swap_remapping and swap_qubits is not None:
            left, right = swap_qubits
            _remap_swap_components(
                components,
                left,
                right,
                defer_materialization=enable_deferred_swap,
            )
            index += 1
            continue
        if isinstance(step, _StatevectorCXSequenceStep):
            if not enable_clifford_matching:
                for control, target in zip(step.controls, step.targets, strict=True):
                    component = _merge_components(components, (control, target))
                    local_qubits = (
                        component.qubits.index(control),
                        component.qubits.index(target),
                    )
                    state = _apply_fixed_permutation(
                        component.state, "cx", local_qubits, len(component.qubits)
                    )
                    updated = _ProductComponent(component.qubits, state)
                    for qubit in updated.qubits:
                        components[qubit] = updated
                index += 1
                continue

            for control, target in zip(step.controls, step.targets, strict=True):
                _merge_components(components, (control, target))

            edges_by_component: dict[
                int, tuple[_ProductComponent, list[tuple[int, int]]]
            ] = {}
            for control, target in zip(step.controls, step.targets, strict=True):
                component = components[control]
                edge_entry = edges_by_component.get(id(component))
                if edge_entry is None:
                    edge_entry = (component, [])
                    edges_by_component[id(component)] = edge_entry
                edge_entry[1].append((control, target))

            for component, edges in edges_by_component.values():
                controls = tuple(
                    component.qubits.index(control) for control, _ in edges
                )
                targets = tuple(component.qubits.index(target) for _, target in edges)
                if len(edges) >= _PRODUCT_STATE_CX_GATHER_MINIMUM_LENGTH:
                    state = _apply_cx_sequence_gather(
                        component.state,
                        controls,
                        targets,
                        len(component.qubits),
                    )
                else:
                    state = component.state
                    for control, target in zip(controls, targets, strict=True):
                        state = _apply_fixed_permutation(
                            state,
                            "cx",
                            (control, target),
                            len(component.qubits),
                        )
                updated = _ProductComponent(component.qubits, state)
                for qubit in updated.qubits:
                    components[qubit] = updated
            index += 1
            continue
        if isinstance(step, _StatevectorCliffordMatchingStep):
            matching_edges = (
                *step.cz_edges,
                *tuple(zip(step.controls, step.targets, strict=True)),
            )
            for left, right in matching_edges:
                _merge_components(components, (left, right))

            selected: dict[int, _ProductComponent] = {}
            for left, _ in matching_edges:
                component = components[left]
                selected[id(component)] = component
            for component in selected.values():
                updated = _apply_product_clifford_matching(
                    component, step, execution_statistics
                )
                for qubit in updated.qubits:
                    components[qubit] = updated
            index += 1
            continue
        groups = _step_qubit_groups(step)
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
            component.qubits,
            _apply_step(
                component,
                step,
                parameter_bindings,
                constant_cache,
                enable_fixed_clifford,
            ),
        )
        for qubit in updated.qubits:
            components[qubit] = updated
        index += 1
    final_component = _merge_components(components, tuple(range(n_qubits)))
    return _canonicalize_component(
        final_component, use_gather=enable_deferred_swap
    ).state
