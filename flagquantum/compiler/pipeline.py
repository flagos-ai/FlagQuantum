"""Native FlagQuantum optimization, scheduling, and lowering pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from math import isclose
from types import MappingProxyType
from typing import Any, cast

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig, get_runtime_config
from .commutation import CommutationAnalysis, analyze_commutation
from .directed_topology import DirectedCouplingMap
from .pass_manager import (
    OPTIMIZATION_PIPELINE,
    AnalysisSpec,
    PassFunction,
    PassManager,
    PassRegistry,
    PassSpec,
)
from .routing import (
    CouplingMap,
    record_post_routing_optimization,
    route_to_topology,
    select_routing_strategy,
)
from .zero_state_reset import remove_zero_state_resets

# The opcodes the pipeline treats as their own inverse. `i` is deliberately absent
# even though the operator schema declares it self-inverse: `remove_identity_gates`
# deletes it one pass earlier, so including it here would add a candidate that can
# never reach these passes. The difference is reported, not closed, by
# `benchmarks/compiler_inverse_cancellation.py`.
_SELF_INVERSE = frozenset(
    {"x", "y", "z", "h", "cx", "cy", "cz", "swap", "ccx", "cswap"}
)

# The opcodes whose single angle parameter is `theta`, which is the parameter their
# adjoint negates. Measured to be exactly the schema's `negate_parameters` opcodes.
_ROTATION_PARAM: Mapping[str, str] = MappingProxyType(
    {
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
)


# The two opcode tables above and the commutation partition below are the facts the
# optimization passes read, and they are registered as analyses rather than reached
# as module globals. The tables follow from the operator schema instead of from the
# program, so they are declared program-independent: the manager computes each once
# and carries it for the rest of the run. Nothing about the values changes here; what
# changes is that a pass now says which facts it reads, and a caller can hand the
# same pass a different registry rather than editing this module.
def _self_inverse_opcodes(_program: CircuitIR) -> frozenset[str]:
    """Every opcode whose declared adjoint is itself, as the pipeline uses the term."""

    return _SELF_INVERSE


def _rotation_parameters(_program: CircuitIR) -> Mapping[str, str]:
    """Every opcode whose single angle parameter its adjoint negates, to that name."""

    return _ROTATION_PARAM


def _commutation_blocks(program: CircuitIR) -> CommutationAnalysis:
    """The commuting blocks of ``program``, one ordered partition per qubit."""

    return analyze_commutation(program)


_BUILTIN_ANALYSES: Mapping[str, AnalysisSpec] = MappingProxyType(
    {
        "self_inverse_opcodes": AnalysisSpec(
            _self_inverse_opcodes, program_independent=True
        ),
        "rotation_parameters": AnalysisSpec(
            _rotation_parameters, program_independent=True
        ),
        # A function of the program, so it is recomputed from the program a round
        # actually produced and no pass may claim to preserve it.
        "commutation_blocks": AnalysisSpec(_commutation_blocks),
    }
)
"""The analyses the built-in passes declare, by name."""


def _rotation_table(analyses: Mapping[str, object] | None) -> Mapping[str, str]:
    """The rotation-parameter table a pass was handed, or this module's own."""

    if analyses is None:
        return _ROTATION_PARAM
    return cast(Mapping[str, str], analyses["rotation_parameters"])


def _self_inverse_table(analyses: Mapping[str, object] | None) -> frozenset[str]:
    """The self-inverse opcode set a pass was handed, or this module's own."""

    if analyses is None:
        return _SELF_INVERSE
    return cast(frozenset[str], analyses["self_inverse_opcodes"])


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


def remove_identity_gates(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Remove explicit identity gates and zero-angle rotations."""

    rotations = _rotation_table(analyses)
    instructions = []
    for instruction in ir:
        if instruction.name in {"i", "id"}:
            continue
        param_name = rotations.get(instruction.name)
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


def merge_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Remove identical self-inverse gates adjacent on their wires."""

    self_inverse = _self_inverse_table(analyses)
    program = _WireLocalProgram()
    for instruction in ir:
        position = program.last_touching(instruction.wires)
        previous = None if position is None else program[position]
        if (
            instruction.name in self_inverse
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


def merge_adjacent_rotations(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Merge rotations adjacent on their wires."""

    rotations = _rotation_table(analyses)
    program = _WireLocalProgram()
    for instruction in ir:
        param_name = rotations.get(instruction.name)
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
    return PassManager(default_pass_registry()).to_fixed_point(
        _as_ir(circuit_or_ir), OPTIMIZATION_PIPELINE
    )


# Four of the eight built-in passes live in their own module, and each of those
# modules reads this one: `commutation_cancellation` reads `_SELF_INVERSE`,
# `inverse_cancellation` reads `_WireLocalProgram`, `one_qubit_optimization` reads
# the Euler tables and `_is_zero`, and `diagonal_before_measure` reads
# one_qubit_synthesis. An eager import in this direction would therefore be a
# cycle, so each is bound through a wrapper that imports it at call time. The four
# remaining passes are defined here and need no wrapper.
#
# Each wrapper that needs one forwards the analyses it was handed rather than
# reading this module's tables itself, so a pass in another module reads the same
# declared facts as the passes defined here.
def _remove_diagonal_gates_before_measure(ir: CircuitIR) -> CircuitIR:
    from .diagonal_before_measure import remove_diagonal_gates_before_measure

    return remove_diagonal_gates_before_measure(ir)


def _merge_inverse_pairs(ir: CircuitIR) -> CircuitIR:
    from .inverse_cancellation import merge_inverse_pairs

    return merge_inverse_pairs(ir)


def _cancel_commuting_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    from .commutation_cancellation import cancel_commuting_self_inverse

    return cancel_commuting_self_inverse(ir, analyses)


def _collapse_one_qubit_runs(ir: CircuitIR) -> CircuitIR:
    from .one_qubit_optimization import collapse_one_qubit_runs

    return collapse_one_qubit_runs(ir)


# The built-in passes, bound to the functions that implement them, in the order and
# with the calls the inline fixed-point loop used.
BUILTIN_PASSES: Mapping[str, PassFunction] = MappingProxyType(
    {
        "remove_zero_state_resets": remove_zero_state_resets,
        "remove_diagonal_gates_before_measure": (_remove_diagonal_gates_before_measure),
        "remove_identity_gates": remove_identity_gates,
        "merge_self_inverse": merge_self_inverse,
        "merge_inverse_pairs": _merge_inverse_pairs,
        "merge_adjacent_rotations": merge_adjacent_rotations,
        "cancel_commuting_self_inverse": _cancel_commuting_self_inverse,
        "collapse_one_qubit_runs": _collapse_one_qubit_runs,
    }
)
"""Maps every name in `OPTIMIZATION_PIPELINE` to the function that implements it.

Every binding is callable with one program argument, which is what
`PassRegistry.resolve` returns and what a caller that does not care about analyses
calls. A binding that declares analyses through `_BUILTIN_PASS_SPECS` also accepts
the resolved facts as a second argument, and the manager passes them.
"""


# What each built-in pass declares about the pipeline: the analyses it reads and the
# ones it leaves valid. The two opcode tables are program-independent and every
# reader asserts it preserves its own, which is what lets the manager compute them
# once for a whole fixed-point run instead of once per round. `commutation_blocks`
# is a function of the program, so it is declared read and never preserved.
_BUILTIN_PASS_SPECS: Mapping[str, PassSpec] = MappingProxyType(
    {
        "remove_zero_state_resets": PassSpec(remove_zero_state_resets),
        "remove_diagonal_gates_before_measure": PassSpec(
            _remove_diagonal_gates_before_measure
        ),
        "remove_identity_gates": PassSpec(
            remove_identity_gates,
            uses=("rotation_parameters",),
            preserves=("rotation_parameters",),
        ),
        "merge_self_inverse": PassSpec(
            merge_self_inverse,
            uses=("self_inverse_opcodes",),
            preserves=("self_inverse_opcodes",),
        ),
        "merge_inverse_pairs": PassSpec(_merge_inverse_pairs),
        "merge_adjacent_rotations": PassSpec(
            merge_adjacent_rotations,
            uses=("rotation_parameters",),
            preserves=("rotation_parameters",),
        ),
        "cancel_commuting_self_inverse": PassSpec(
            _cancel_commuting_self_inverse,
            uses=("self_inverse_opcodes", "commutation_blocks"),
            preserves=("self_inverse_opcodes",),
        ),
        "collapse_one_qubit_runs": PassSpec(_collapse_one_qubit_runs),
    }
)
"""What each built-in pass declares, by name. Every function identity matches
`BUILTIN_PASSES`; a registry refuses a spec whose function is a different object."""


def default_pass_registry() -> PassRegistry:
    """The registry that resolves exactly the built-in optimization passes."""

    return PassRegistry(BUILTIN_PASSES, _BUILTIN_PASS_SPECS, _BUILTIN_ANALYSES)


def optimize(circuit_or_ir: Any) -> CircuitIR:
    """Apply target-independent circuit optimizations to a fixed point."""

    return _optimize_to_fixed_point(circuit_or_ir)


def compile(
    circuit_or_ir: Any,
    *,
    coupling_map: (
        CouplingMap | DirectedCouplingMap | Iterable[tuple[int, int]] | None
    ) = None,
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
        # A device the router understands is handed over as it stands: an ordered
        # device carries a rule an edge sequence cannot express, and rebuilding it
        # as a ``CouplingMap`` here would settle the routing before the router saw
        # it.
        coupling = (
            coupling_map
            if isinstance(coupling_map, (CouplingMap, DirectedCouplingMap))
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
    "BUILTIN_PASSES",
    "CouplingMap",
    "compile",
    "default_pass_registry",
    "optimize",
    "route_to_topology",
    "schedule_layers",
]
