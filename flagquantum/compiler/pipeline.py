"""Native FlagQuantum optimization, scheduling, and lowering pipeline."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from typing import Any, Iterable

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig, get_runtime_config
from ..errors import CompilationError
from .routing import (
    CouplingMap,
    record_post_routing_optimization,
    route_to_topology,
    select_routing_strategy,
)

_SELF_INVERSE = {"x", "y", "z", "h", "cx", "cy", "cz", "swap", "ccx", "cswap"}
_ROTATION_PARAM = {
    "rx": "theta",
    "ry": "theta",
    "rz": "theta",
    "phase": "theta",
    "u1": "theta",
    "rxx": "theta",
    "ryy": "theta",
    "rzz": "theta",
    "crx": "theta",
    "cry": "theta",
    "crz": "theta",
    "cphase": "theta",
}


def _as_ir(circuit_or_ir: Any) -> CircuitIR:
    return ensure_circuit_ir(circuit_or_ir)


def _is_zero(value: Any, atol: float = 1e-12) -> bool:
    try:
        if bool(getattr(value, "requires_grad", False)):
            return False
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and value.numel() != 1:
            return False
        if hasattr(value, "item"):
            value = value.item()
        return isclose(float(value), 0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _add_values(left: Any, right: Any) -> Any:
    return left + right


def _replace_param(instruction: Instruction, key: str, value: Any) -> Instruction:
    params = dict(instruction.params)
    params[key] = value
    return Instruction(
        name=instruction.name,
        wires=instruction.wires,
        params=params,
        matrix=instruction.matrix,
        metadata=instruction.metadata,
    )


def remove_identity_gates(ir: CircuitIR) -> CircuitIR:
    """Remove explicit identity gates and zero-angle rotations."""

    instructions = []
    for instruction in ir:
        if instruction.name in {"i", "id"}:
            continue
        param_name = _ROTATION_PARAM.get(instruction.name)
        if param_name is not None and _is_zero(instruction.params.get(param_name)):
            continue
        instructions.append(instruction)
    return replace(ir, instructions=tuple(instructions))


def _last_touching_instruction(
    instructions: list[Instruction],
    wires: tuple[int, ...],
) -> int | None:
    target_wires = set(wires)
    for index in range(len(instructions) - 1, -1, -1):
        if not target_wires.isdisjoint(instructions[index].wires):
            return index
    return None


def merge_self_inverse(ir: CircuitIR) -> CircuitIR:
    """Remove identical self-inverse gates adjacent on their wires."""

    out: list[Instruction] = []
    for instruction in ir:
        previous_index = _last_touching_instruction(out, instruction.wires)
        previous = out[previous_index] if previous_index is not None else None
        if (
            instruction.name in _SELF_INVERSE
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and not instruction.params
            and not previous.params
        ):
            assert previous_index is not None
            out.pop(previous_index)
        else:
            out.append(instruction)
    return replace(ir, instructions=tuple(out))


def merge_adjacent_rotations(ir: CircuitIR) -> CircuitIR:
    """Merge rotations adjacent on their wires."""

    out: list[Instruction] = []
    for instruction in ir:
        param_name = _ROTATION_PARAM.get(instruction.name)
        previous_index = _last_touching_instruction(out, instruction.wires)
        previous = out[previous_index] if previous_index is not None else None
        if (
            param_name is not None
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and previous.matrix is None
            and instruction.matrix is None
            and param_name in previous.params
            and param_name in instruction.params
        ):
            assert previous_index is not None
            merged_value = _add_values(
                previous.params[param_name], instruction.params[param_name]
            )
            if _is_zero(merged_value):
                out.pop(previous_index)
            else:
                out[previous_index] = _replace_param(
                    previous,
                    param_name,
                    merged_value,
                )
            continue
        out.append(instruction)
    return replace(ir, instructions=tuple(out))


def schedule_layers(ir: CircuitIR) -> list[list[Instruction]]:
    """ASAP-schedule instructions without reversing per-wire dependencies."""

    layers: list[list[Instruction]] = []
    last_layer_by_wire: dict[int, int] = {}
    for instruction in ir:
        layer_index = 1 + max(
            (last_layer_by_wire.get(wire, -1) for wire in instruction.wires),
            default=-1,
        )
        if layer_index == len(layers):
            layers.append([instruction])
        else:
            layers[layer_index].append(instruction)
        for wire in instruction.wires:
            last_layer_by_wire[wire] = layer_index
    return layers


def simple_compile(circuit_or_ir: Any) -> CircuitIR:
    ir = _as_ir(circuit_or_ir)
    max_rounds = len(ir) + 1
    for _ in range(max_rounds):
        previous_count = len(ir)
        ir = remove_identity_gates(ir)
        ir = merge_self_inverse(ir)
        ir = merge_adjacent_rotations(ir)
        ir = remove_identity_gates(ir)
        if len(ir) == previous_count:
            return ir
    raise CompilationError("compiler optimization passes did not reach a fixed point")


def compile_for_backend(
    circuit_or_ir: Any,
    *,
    coupling_map: CouplingMap | Iterable[tuple[int, int]] | None = None,
    routing_strategy: str = "restore_after_each_gate",
    optimize: bool = True,
    config: RuntimeConfig | None = None,
) -> CircuitIR:
    """Compile an IR for a backend topology and local optimizer stack."""

    ir = _as_ir(circuit_or_ir)
    selected_config = config or get_runtime_config()
    metadata = dict(ir.metadata)
    metadata["runtime_config"] = selected_config.to_manifest()
    ir = replace(ir, metadata=metadata)
    if optimize:
        ir = simple_compile(ir)
    if coupling_map is not None:
        coupling = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(ir.n_wires, coupling_map)
        )
        strategy_selection = None
        selected_routing_strategy = routing_strategy
        if routing_strategy == "auto":
            strategy_selection = select_routing_strategy(ir, coupling)
            selected_routing_strategy = strategy_selection.selected_strategy
        ir = route_to_topology(
            ir,
            coupling,
            strategy=selected_routing_strategy,
        )
        if optimize:
            ir = simple_compile(ir)
        ir = record_post_routing_optimization(ir)
        if strategy_selection is not None:
            metadata = dict(ir.metadata)
            metadata["routing_strategy_selection"] = strategy_selection.summary()
            ir = replace(ir, metadata=metadata)
    return ir


__all__ = [
    "CouplingMap",
    "compile_for_backend",
    "simple_compile",
    "remove_identity_gates",
    "merge_self_inverse",
    "merge_adjacent_rotations",
    "route_to_topology",
    "schedule_layers",
]
