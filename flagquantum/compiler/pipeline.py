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
from .optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    optimization_level_stages,
)
from .pass_manager import OPTIMIZATION_PIPELINE as OPTIMIZATION_PIPELINE
from .pass_manager import (
    AnalysisSpec,
    PassBinding,
    PassFunction,
    PassManager,
    PassRegistry,
    PassSpec,
)
from .routing import (
    CouplingMap,
    LinkErrorRates,
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
# The analyses the built-in passes declare, by name.


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


def _is_zero(value: Any, atol: float = 0.0) -> bool:
    """Use exact scalar zero unless a numerical synthesis caller supplies a tolerance."""
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

    An instruction carrying its own matrix is never removed. The angle says what
    the opcode's matrix would be, not what the caller supplied, so a
    matrix-bearing `rz(0)` is not the identity and `is_identity_one_qubit`
    already declines it on that ground. This branch declines for the same reason
    rather than for a different one, so the two branches cannot disagree about
    the same instruction.
    """

    from .one_qubit_synthesis import is_identity_one_qubit

    rotations = _rotation_table(analyses)
    instructions = []
    for instruction in ir:
        if is_identity_one_qubit(instruction):
            continue
        if instruction.matrix is None:
            param_name = rotations.get(instruction.name)
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


def merge_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Remove identical self-inverse gates adjacent on their qubits."""

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
    """Merge rotations adjacent on their qubits."""

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


# Six of the ten built-in passes live in their own module, and each of those
# modules reads this one: `commutation_cancellation` reads `_SELF_INVERSE`,
# `inverse_cancellation` reads the program index above, `one_qubit_optimization`
# reads the Euler tables and `_is_zero`, `two_qubit_optimization` reads
# `one_qubit_synthesis`, and `diagonal_before_measure` reads one_qubit_synthesis.
# An eager import in this direction would therefore be a cycle, so each is bound
# through a wrapper that imports it at call time. The four remaining passes are
# defined here and need no wrapper.
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


def _merge_commuting_rotations(ir: CircuitIR) -> CircuitIR:
    from .commutation_cancellation import merge_commuting_rotations

    return merge_commuting_rotations(ir)


def _collapse_one_qubit_runs(ir: CircuitIR) -> CircuitIR:
    from .one_qubit_optimization import collapse_one_qubit_runs

    return collapse_one_qubit_runs(ir)


def _collapse_two_qubit_blocks(ir: CircuitIR) -> CircuitIR:
    from .two_qubit_optimization import collapse_two_qubit_blocks

    return collapse_two_qubit_blocks(ir)


# The built-in passes, bound to the functions that implement them.
BUILTIN_PASSES: Mapping[str, PassFunction] = MappingProxyType(
    {
        "remove_zero_state_resets": remove_zero_state_resets,
        "remove_diagonal_gates_before_measure": (_remove_diagonal_gates_before_measure),
        "remove_identity_gates": remove_identity_gates,
        "merge_self_inverse": merge_self_inverse,
        "merge_inverse_pairs": _merge_inverse_pairs,
        "merge_adjacent_rotations": merge_adjacent_rotations,
        "merge_commuting_rotations": _merge_commuting_rotations,
        "cancel_commuting_self_inverse": _cancel_commuting_self_inverse,
        "collapse_one_qubit_runs": _collapse_one_qubit_runs,
        "collapse_two_qubit_blocks": _collapse_two_qubit_blocks,
    }
)
# Maps every name in `OPTIMIZATION_PIPELINE` to the function that implements it.
# Every binding is callable with one program argument, which is what
# `PassRegistry.resolve` returns and what a caller that does not care about analyses
# calls. A binding that declares analyses through `_BUILTIN_PASS_SPECS` also accepts
# the resolved facts as a second argument, and the manager passes them.


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
        "merge_commuting_rotations": PassSpec(_merge_commuting_rotations),
        "cancel_commuting_self_inverse": PassSpec(
            _cancel_commuting_self_inverse,
            uses=("self_inverse_opcodes", "commutation_blocks"),
            preserves=("self_inverse_opcodes",),
        ),
        "collapse_one_qubit_runs": PassSpec(_collapse_one_qubit_runs),
        "collapse_two_qubit_blocks": PassSpec(_collapse_two_qubit_blocks),
    }
)
# What each built-in pass declares, by name. Every function identity matches
# `BUILTIN_PASSES`; a registry refuses a spec whose function is a different object.


def default_pass_registry() -> PassRegistry:
    """The registry that resolves exactly the built-in optimization passes."""

    return PassRegistry(BUILTIN_PASSES, _BUILTIN_PASS_SPECS, _BUILTIN_ANALYSES)


def _with_recorded_level(
    ir: CircuitIR, level: int, admitted: tuple[str, ...] = ()
) -> CircuitIR:
    """Record the level the caller asked for, so the request is readable.

    A caller who asked for level 0 and a caller who passed `optimize=False` to
    `compile` get different programs only in this field: level 0 is a request
    that was honored, and no field at all means no request was made.

    A program built with admitted passes records their names beside the level,
    because a program a caller's own pass changed is not the program the level
    describes and the difference has to be readable from the result rather than
    only from the call. The key appears only when there is something to say, so a
    program that ran nothing but the built-in stages carries no new field.
    """

    metadata = dict(ir.metadata)
    metadata["optimization"] = {"level": level}
    if admitted:
        metadata["admitted_passes"] = list(admitted)
    return replace(ir, metadata=metadata)


def _registry_admitting(
    passes: Mapping[str, PassBinding] | None,
) -> PassRegistry:
    """The built-in registry, extended with the caller's own names.

    Registration completes before the first pass runs, so a reserved built-in name,
    a name already taken, an empty name, and a name bound to something that is not
    callable are all refused while the program is untouched. The names are added in
    the order the mapping declares them, because a pass runs on the program the
    previous one produced and the order is therefore part of the request.

    `with_pass` accepts no declaration, so a pass that reads or preserves an
    analysis is registered on a `PassRegistry` directly -- `default_pass_registry`
    is exported beside the manager for exactly that route.
    """

    registry = default_pass_registry()
    for name, function in (passes or {}).items():
        registry = registry.with_pass(name, function)
    return registry


def _optimize_to_fixed_point(
    circuit_or_ir: Any,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    passes: Mapping[str, PassBinding] | None = None,
) -> CircuitIR:
    """Run one declared level's pass sequence to a fixed point.

    The sequence, and the reason each pass sits where it does, are declared in
    `optimization_levels.OPTIMIZATION_LEVEL_STAGES`. This function resolves the
    declaration and hands it to `PassManager.to_fixed_point`, which owns the loop
    and the termination condition; neither the sequence nor the loop has a second
    copy here. A level whose sequence is empty is answered with the program
    unchanged, which is what level 0 means -- unless the caller admitted passes, in
    which case they run and the empty level is the whole of the built-in work.

    An admitted pass joins the sequence after the declared stages rather than
    replacing one of them, and it joins it inside the loop, so it runs whenever the
    loop runs a round. A pass that is not idempotent therefore keeps the loop going
    until `to_fixed_point` refuses to continue, which is the same verdict the
    built-in passes are held to.
    """

    stages = optimization_level_stages(optimization_level)
    ir = _as_ir(circuit_or_ir)
    admitted = tuple((passes or {}).keys())
    sequence = (*stages, *admitted)
    if not sequence:
        return _with_recorded_level(ir, optimization_level)
    optimized = PassManager(_registry_admitting(passes)).to_fixed_point(ir, sequence)
    return _with_recorded_level(optimized, optimization_level, admitted)


def optimize(
    circuit_or_ir: Any,
    *,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    passes: Mapping[str, PassBinding] | None = None,
) -> CircuitIR:
    """Apply target-independent circuit optimizations to a fixed point.

    `optimization_level` selects one of the declared pass sequences in
    `flagquantum.compiler.optimization_levels`: 1 collapses adjacent gates and
    removes work that cannot be observed, 2 additionally commutes gates past one
    another, and 0 is the request to leave the program as submitted. The level
    this call used is recorded under `metadata["optimization"]`. A reserved or
    unknown level raises `CompilationError` rather than falling back, so a
    caller cannot be handed a weaker program than it asked for.

    `passes` admits transformations this repository does not ship. Each name is a
    pass name the caller owns and each value is the transformation it resolves to,
    and the passes run after the level's declared stages, in the order the mapping
    declares them, inside the same fixed-point loop. The names they ran under are
    recorded under `metadata["admitted_passes"]`. A name a built-in pass already
    owns is refused with `PassRegistryError` before any pass runs, because a
    caller-supplied pass must never redefine what the level means. A pass that
    needs an analysis, or that vouches for one, is registered on a `PassRegistry`
    instead and run through `PassManager`: a mapping carries names and functions,
    not declarations.

    Examples:
        >>> import math
        >>> import flagquantum as fq
        >>> from dataclasses import replace
        >>> from flagquantum.compiler import optimize
        >>> from flagquantum.core.ir import Instruction
        >>> def lower_x_to_rx(program):
        ...     # One qubit wide, so the replacement names that qubit directly.
        ...     return replace(program, instructions=tuple(
        ...         Instruction("rx", (0,), {"theta": math.pi})
        ...         if held.name == "x" else held
        ...         for held in program.instructions
        ...     ))
        >>> program = fq.Circuit(1).x(0).to_ir()
        >>> out = optimize(program, optimization_level=0, passes={"acme.rx": lower_x_to_rx})
        >>> [held.name for held in out.instructions], out.metadata["admitted_passes"]
        (['rx'], ['acme.rx'])
    """

    return _optimize_to_fixed_point(circuit_or_ir, optimization_level, passes)


def compile(
    circuit_or_ir: Any,
    *,
    coupling_map: (
        CouplingMap
        | DirectedCouplingMap
        | LinkErrorRates
        | Iterable[tuple[int, int]]
        | None
    ) = None,
    routing_strategy: str = "restore_after_each_gate",
    optimize: bool = True,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    config: RuntimeConfig | None = None,
    passes: Mapping[str, PassBinding] | None = None,
) -> CircuitIR:
    """Compile a circuit with optional topology routing and optimization.

    `coupling_map` states the device. A `CouplingMap` says which qubits a device
    couples and can therefore be routed by hop count only; a `LinkErrorRates` says
    that and what each coupling costs to use, so the route choice, the price the
    estimate reports, and the routing metadata all read the calibration.

    `optimization_level` is forwarded to `optimize` at both points this function
    optimizes -- before routing and again after it, because routing inserts SWAPs
    and the second pass is what cancels them. It is validated here whether or not
    `optimize` is set: a reserved level is a capability this release does not
    have, and a caller that also asked for no optimization is the one case where
    ignoring the level would hide that instead of reporting it.

    `passes` is forwarded the same way and therefore runs at both points too: it is
    part of the optimization rather than an epilogue, so an admitted pass sees the
    program a level's stages would have handed it, and after routing it sees the
    SWAPs the router inserted. Admitting passes while `optimize` is off is refused
    rather than ignored, because the caller asked for a transformation and the same
    call asked for nothing to run.
    """

    optimization_level_stages(optimization_level)
    if passes and not optimize:
        raise ValueError(
            "compile() was given passes while optimization is switched off; the "
            "admitted passes would not run, so the request is refused rather than "
            "answered with the program unchanged"
        )
    ir = _as_ir(circuit_or_ir)
    selected_config = config or get_runtime_config()
    metadata = dict(ir.metadata)
    metadata["runtime_config"] = selected_config.to_manifest()
    ir = replace(ir, metadata=metadata)
    if optimize:
        ir = _optimize_to_fixed_point(ir, optimization_level, passes)
    if coupling_map is not None:
        # A device the router understands is handed over as it stands: an ordered
        # device carries a rule an edge sequence cannot express, and rebuilding it
        # as a ``CouplingMap`` here would settle the routing before the router saw
        # it.
        coupling = (
            coupling_map
            if isinstance(
                coupling_map, (CouplingMap, DirectedCouplingMap, LinkErrorRates)
            )
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
            ir = _optimize_to_fixed_point(ir, optimization_level, passes)
        ir = record_post_routing_optimization(ir)
        if strategy_selection is not None:
            metadata = dict(ir.metadata)
            metadata["routing_strategy_selection"] = strategy_selection.summary()
            ir = replace(ir, metadata=metadata)
    return ir


__all__ = [
    "BUILTIN_PASSES",
    "CouplingMap",
    "LinkErrorRates",
    "compile",
    "default_pass_registry",
    "optimize",
    "route_to_topology",
    "schedule_layers",
]
