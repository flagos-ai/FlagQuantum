"""Native FlagQuantum optimization, scheduling, and lowering pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig, get_runtime_config
from .lowering.schedule_legalization import schedule_layers as schedule_layers
from .optimization.canonicalization import _ROTATION_PARAM as _ROTATION_PARAM
from .optimization.canonicalization import _SELF_INVERSE as _SELF_INVERSE
from .optimization.canonicalization import _add_values as _add_values
from .optimization.canonicalization import _is_zero as _is_zero
from .optimization.canonicalization import _replace_param as _replace_param
from .optimization.canonicalization import _rotation_table as _rotation_table
from .optimization.canonicalization import _self_inverse_table as _self_inverse_table
from .optimization.canonicalization import _WireLocalProgram as _WireLocalProgram
from .optimization.canonicalization import (
    merge_adjacent_rotations as merge_adjacent_rotations,
)
from .optimization.canonicalization import merge_self_inverse as merge_self_inverse
from .optimization.canonicalization import (
    remove_identity_gates as remove_identity_gates,
)
from .optimization.commutation import CommutationAnalysis, analyze_commutation
from .optimization.optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    optimization_level_stages,
)
from .optimization.zero_state_reset import remove_zero_state_resets
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
from .topology.directed_topology import DirectedCouplingMap


# The canonical opcode tables and the commutation partition are the facts the
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


def _as_ir(circuit_or_ir: Any) -> CircuitIR:
    return ensure_circuit_ir(circuit_or_ir)


# Bind the pass library at call time. The registry forwards each declared
# analysis to the pass that reads it.


def _remove_diagonal_gates_before_measure(ir: CircuitIR) -> CircuitIR:
    from .optimization.diagonal_before_measure import (
        remove_diagonal_gates_before_measure,
    )

    return remove_diagonal_gates_before_measure(ir)


def _merge_inverse_pairs(ir: CircuitIR) -> CircuitIR:
    from .optimization.inverse_cancellation import merge_inverse_pairs

    return merge_inverse_pairs(ir)


def _cancel_commuting_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    from .optimization.commutation_cancellation import cancel_commuting_self_inverse

    return cancel_commuting_self_inverse(ir, analyses)


def _merge_commuting_rotations(ir: CircuitIR) -> CircuitIR:
    from .optimization.commutation_cancellation import merge_commuting_rotations

    return merge_commuting_rotations(ir)


def _collapse_one_qubit_runs(ir: CircuitIR) -> CircuitIR:
    from .optimization.one_qubit_optimization import collapse_one_qubit_runs

    return collapse_one_qubit_runs(ir)


def _collapse_two_qubit_blocks(ir: CircuitIR) -> CircuitIR:
    from .optimization.two_qubit_optimization import collapse_two_qubit_blocks

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
    `flagquantum.compiler.optimization.optimization_levels`: 1 collapses adjacent gates and
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
    from .lowering.deferred_operations import lower_deferred_operations
    from .select_lowering import lower_select_program

    ir = lower_select_program(_as_ir(circuit_or_ir), "dense")
    ir = replace(ir, instructions=lower_deferred_operations(ir.instructions))
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
