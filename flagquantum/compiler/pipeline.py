"""Native FlagQuantum optimization, scheduling, and lowering pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from math import isclose
from typing import Any

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig, get_runtime_config
from ..errors import CompilationError
from .routing import (
    CouplingMap,
    record_post_routing_optimization,
    route_to_topology,
    select_routing_strategy,
)
from .zero_state_reset import remove_zero_state_resets

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


class _WireLocalProgram:
    """Instructions in program order, indexed by the latest writer of each wire.

    Both merge passes need the most recent instruction that touches a wire set,
    and they need to remove or rewrite it. Scanning the output list for that
    instruction costs the length of the untouched prefix, which is quadratic on a
    circuit whose gates share no wire -- the shape a routing pass produces on a
    wide device. Indexing by wire instead makes a pass linear in the gates times
    their wires.

    ``position`` is the insertion sequence number. Sequence numbers only
    increase, so the largest live one touching a wire is also the latest one in
    program order, and positions do not shift when an earlier instruction is
    removed.
    """

    def __init__(self) -> None:
        self._instructions: dict[int, Instruction] = {}
        self._last_by_wire: dict[int, list[int]] = {}
        self._position = 0

    def last_touching(self, wires: tuple[int, ...]) -> int | None:
        """Return the position of the latest instruction touching any of ``wires``."""

        latest: int | None = None
        for wire in wires:
            stack = self._last_by_wire.get(wire)
            if stack and (latest is None or stack[-1] > latest):
                latest = stack[-1]
        return latest

    def append(self, instruction: Instruction) -> None:
        self._position += 1
        self._instructions[self._position] = instruction
        for wire in instruction.wires:
            self._last_by_wire.setdefault(wire, []).append(self._position)

    def pop(self, position: int) -> None:
        """Remove the instruction at ``position``, a wire's latest writer.

        Both passes only ever remove the instruction they just looked up, so the
        removed one is by construction the latest writer of every wire it touches
        and the per-wire stack needs no repair beyond dropping its top entry.
        """

        instruction = self._instructions.pop(position)
        for wire in instruction.wires:
            stack = self._last_by_wire[wire]
            assert stack[-1] == position
            stack.pop()

    def rewrite(self, position: int, instruction: Instruction) -> None:
        """Replace the instruction at ``position``, keeping the wires it touches."""

        assert instruction.wires == self._instructions[position].wires
        self._instructions[position] = instruction

    def __getitem__(self, position: int) -> Instruction:
        return self._instructions[position]

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions.values())


def merge_self_inverse(ir: CircuitIR) -> CircuitIR:
    """Remove identical self-inverse gates adjacent on their wires."""

    program = _WireLocalProgram()
    for instruction in ir:
        position = program.last_touching(instruction.wires)
        previous = None if position is None else program[position]
        if (
            instruction.name in _SELF_INVERSE
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and not instruction.params
            and not previous.params
        ):
            assert position is not None
            program.pop(position)
        else:
            program.append(instruction)
    return replace(ir, instructions=tuple(program))


def merge_adjacent_rotations(ir: CircuitIR) -> CircuitIR:
    """Merge rotations adjacent on their wires."""

    program = _WireLocalProgram()
    for instruction in ir:
        param_name = _ROTATION_PARAM.get(instruction.name)
        position = program.last_touching(instruction.wires)
        previous = None if position is None else program[position]
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
            assert position is not None
            merged_value = _add_values(
                previous.params[param_name], instruction.params[param_name]
            )
            if _is_zero(merged_value):
                program.pop(position)
            else:
                program.rewrite(
                    position,
                    _replace_param(previous, param_name, merged_value),
                )
        else:
            program.append(instruction)
    return replace(ir, instructions=tuple(program))


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


def _optimize_to_fixed_point(circuit_or_ir: Any) -> CircuitIR:
    # Imported here rather than at module scope: `one_qubit_optimization` reads
    # the Euler tables and `_is_zero` from this layer, so a module-level import
    # in this direction would be circular.
    from .one_qubit_optimization import collapse_one_qubit_runs

    ir = _as_ir(circuit_or_ir)
    max_rounds = len(ir) + 1
    for _ in range(max_rounds):
        previous_count = len(ir)
        # First, because a reset this pass can remove is removable whatever the passes
        # below do, and removing it hands them a shorter program. The loop, not this
        # ordering, is what earns the reach on a wire the passes below only empty out
        # later: `x(0) x(0) reset(0)` needs a second round.
        ir = remove_zero_state_resets(ir)
        ir = remove_identity_gates(ir)
        ir = merge_self_inverse(ir)
        ir = merge_adjacent_rotations(ir)
        ir = remove_identity_gates(ir)
        ir = collapse_one_qubit_runs(ir)
        if len(ir) == previous_count:
            return ir
    raise CompilationError("compiler optimization passes did not reach a fixed point")


def optimize(circuit_or_ir: Any) -> CircuitIR:
    """Apply target-independent circuit optimizations to a fixed point."""

    return _optimize_to_fixed_point(circuit_or_ir)


def compile(
    circuit_or_ir: Any,
    *,
    coupling_map: CouplingMap | Iterable[tuple[int, int]] | None = None,
    routing_strategy: str = "restore_after_each_gate",
    optimize: bool = True,
    config: RuntimeConfig | None = None,
) -> CircuitIR:
    """Compile a circuit with optional topology routing and optimization."""

    ir = _as_ir(circuit_or_ir)
    selected_config = config or get_runtime_config()
    metadata = dict(ir.metadata)
    metadata["runtime_config"] = selected_config.to_manifest()
    ir = replace(ir, metadata=metadata)
    if optimize:
        ir = _optimize_to_fixed_point(ir)
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
            ir = _optimize_to_fixed_point(ir)
        ir = record_post_routing_optimization(ir)
        if strategy_selection is not None:
            metadata = dict(ir.metadata)
            metadata["routing_strategy_selection"] = strategy_selection.summary()
            ir = replace(ir, metadata=metadata)
    return ir


__all__ = [
    "CouplingMap",
    "compile",
    "optimize",
    "route_to_topology",
    "schedule_layers",
]
