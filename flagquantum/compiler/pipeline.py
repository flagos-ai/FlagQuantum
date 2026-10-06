"""Native FlagQuantum optimization, scheduling, and lowering pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from math import isclose
from typing import Any

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig, get_runtime_config
from ..errors import CompilationError
from .optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    optimization_level_stages,
)
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
    """Remove every gate that is exactly the identity operator.

    A single-qubit instruction is decided by `is_identity_one_qubit`, which reads
    the angle triple the operator schema declares rather than a table of opcode
    names kept here: that deletes the explicit `i`/`id` set this pass used to
    carry and admits `u3(0, phi, lam)` when `phi + lam` vanishes, a zero-angle
    `rx`, `ry`, `rz`, `phase` or `u1`, and every full turn that returns to `+I`.
    See that function for why a trainable angle refuses, why only exact multiples
    are considered, and why the fold is modulo `4*pi` rather than `2*pi` -- the
    shorter fold would remove `rz(2*pi)`, which is `-I`, and the IR has nowhere
    to record the sign. The import is local because `one_qubit_synthesis` reads
    `_is_zero` from this module.

    A multi-qubit instruction keeps the older rule, which reads the angle
    parameter `_ROTATION_PARAM` names for its opcode, because
    `canonical_euler_angles` describes the single-qubit group only. That leaves
    the `crz(0)`, `rxx(0)`, `cphase(0)` and `rzz(0)` removals this pass already
    performed untouched, and it declines a two-qubit half turn for the same reason
    the single-qubit branch does: `rzz(2*pi)` is `-I`.
    """

    from .one_qubit_synthesis import is_identity_one_qubit

    instructions = []
    for instruction in ir:
        if is_identity_one_qubit(instruction):
            continue
        param_name = _ROTATION_PARAM.get(instruction.name)
        if param_name is not None and _is_zero(instruction.params.get(param_name)):
            continue
        instructions.append(instruction)
    return replace(ir, instructions=tuple(instructions))


class _WireLocalProgram:
    """Instructions in program order, indexed by the latest writer of each qubit.

    Both merge passes need the most recent instruction that touches a qubit set,
    and they need to remove or rewrite it. Scanning the output list for that
    instruction costs the length of the untouched prefix, which is quadratic on a
    circuit whose gates share no qubit -- the shape a routing pass produces on a
    wide device. Indexing by qubit instead makes a pass linear in the gates times
    their qubits.

    ``position`` is the insertion sequence number. Sequence numbers only
    increase, so the largest live one touching a qubit is also the latest one in
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
        """Remove the instruction at ``position``, a qubit's latest writer.

        Both passes only ever remove the instruction they just looked up, so the
        removed one is by construction the latest writer of every qubit it touches
        and the per-qubit stack needs no repair beyond dropping its top entry.
        """

        instruction = self._instructions.pop(position)
        for wire in instruction.wires:
            stack = self._last_by_wire[wire]
            assert stack[-1] == position
            stack.pop()

    def rewrite(self, position: int, instruction: Instruction) -> None:
        """Replace the instruction at ``position``, keeping the qubits it touches."""

        assert instruction.wires == self._instructions[position].wires
        self._instructions[position] = instruction

    def __getitem__(self, position: int) -> Instruction:
        return self._instructions[position]

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions.values())


def merge_self_inverse(ir: CircuitIR) -> CircuitIR:
    """Remove identical self-inverse gates adjacent on their qubits."""

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
    """Merge rotations adjacent on their qubits."""

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
    """ASAP-schedule instructions without reversing per-qubit dependencies."""

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


def _round_passes() -> dict[str, Callable[[CircuitIR], CircuitIR]]:
    """Resolve the declared pass names to the implementations that own them.

    The mapping is keyed on the names a level's declaration lists, so a name a
    level lists and this mapping does not define is a `KeyError` at the first call
    rather than a silently skipped pass.
    """

    from .commutation_cancellation import (
        cancel_commuting_self_inverse,
        merge_commuting_rotations,
    )
    from .diagonal_before_measure import remove_diagonal_gates_before_measure
    from .inverse_cancellation import merge_inverse_pairs
    from .one_qubit_optimization import collapse_one_qubit_runs
    from .two_qubit_optimization import collapse_two_qubit_blocks

    return {
        "remove_zero_state_resets": remove_zero_state_resets,
        "remove_diagonal_gates_before_measure": remove_diagonal_gates_before_measure,
        "remove_identity_gates": remove_identity_gates,
        "merge_self_inverse": merge_self_inverse,
        "merge_inverse_pairs": merge_inverse_pairs,
        "merge_adjacent_rotations": merge_adjacent_rotations,
        "merge_commuting_rotations": merge_commuting_rotations,
        "cancel_commuting_self_inverse": cancel_commuting_self_inverse,
        "collapse_one_qubit_runs": collapse_one_qubit_runs,
        "collapse_two_qubit_blocks": collapse_two_qubit_blocks,
    }


def _with_recorded_level(ir: CircuitIR, level: int) -> CircuitIR:
    """Record the level the caller asked for, so the request is readable.

    A caller who asked for level 0 and a caller who passed `optimize=False` to
    `compile` get different programs only in this field: level 0 is a request
    that was honored, and no field at all means no request was made.
    """

    metadata = dict(ir.metadata)
    metadata["optimization"] = {"level": level}
    return replace(ir, metadata=metadata)


def _optimize_to_fixed_point(
    circuit_or_ir: Any,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> CircuitIR:
    """Run one declared level's pass sequence to a fixed point.

    The sequence, and the reason each pass sits where it does, are declared in
    `optimization_levels.OPTIMIZATION_LEVEL_STAGES`; this function is the loop
    that runs the declaration rather than a second copy of it. A level whose
    sequence is empty is answered with the program unchanged, which is what
    level 0 means.
    """

    stages = optimization_level_stages(optimization_level)
    ir = _as_ir(circuit_or_ir)
    if not stages:
        return _with_recorded_level(ir, optimization_level)
    # `_round_passes` imports the passes here rather than at module scope: they read
    # this layer, so a module-level import in this direction would be circular.
    # `commutation_cancellation` reads `_SELF_INVERSE`, `_ROTATION_PARAM` and the
    # parameter arithmetic below, `inverse_cancellation` reads
    # `_WireLocalProgram`, `one_qubit_optimization` reads the Euler tables and
    # `_is_zero`, `two_qubit_optimization` reads `one_qubit_synthesis` for the same
    # reason, and `diagonal_before_measure` reads one_qubit_synthesis.
    passes = _round_passes()
    max_rounds = len(ir) + 1
    for _ in range(max_rounds):
        previous_count = len(ir)
        for name in stages:
            ir = passes[name](ir)
        if len(ir) == previous_count:
            return _with_recorded_level(ir, optimization_level)
    raise CompilationError("compiler optimization passes did not reach a fixed point")


def optimize(
    circuit_or_ir: Any,
    *,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> CircuitIR:
    """Apply target-independent circuit optimizations to a fixed point.

    `optimization_level` selects one of the declared pass sequences in
    `flagquantum.compiler.optimization_levels`: 1 collapses adjacent gates and
    removes work that cannot be observed, 2 additionally commutes gates past one
    another, and 0 is the request to leave the program as submitted. The level
    this call used is recorded under `metadata["optimization"]`. A reserved or
    unknown level raises `CompilationError` rather than falling back, so a
    caller cannot be handed a weaker program than it asked for.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.compiler import optimize
        >>> program = fq.Circuit(1).h(0).h(0).to_ir()
        >>> len(optimize(program, optimization_level=1).instructions)
        0
    """

    return _optimize_to_fixed_point(circuit_or_ir, optimization_level)


def compile(
    circuit_or_ir: Any,
    *,
    coupling_map: CouplingMap | Iterable[tuple[int, int]] | None = None,
    routing_strategy: str = "restore_after_each_gate",
    optimize: bool = True,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    config: RuntimeConfig | None = None,
) -> CircuitIR:
    """Compile a circuit with optional topology routing and optimization.

    `optimization_level` is forwarded to `optimize` at both points this function
    optimizes -- before routing and again after it, because routing inserts SWAPs
    and the second pass is what cancels them. It is validated here whether or not
    `optimize` is set: a reserved level is a capability this release does not
    have, and a caller that also asked for no optimization is the one case where
    ignoring the level would hide that instead of reporting it.
    """

    optimization_level_stages(optimization_level)
    ir = _as_ir(circuit_or_ir)
    selected_config = config or get_runtime_config()
    metadata = dict(ir.metadata)
    metadata["runtime_config"] = selected_config.to_manifest()
    ir = replace(ir, metadata=metadata)
    if optimize:
        ir = _optimize_to_fixed_point(ir, optimization_level)
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
            ir = _optimize_to_fixed_point(ir, optimization_level)
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
