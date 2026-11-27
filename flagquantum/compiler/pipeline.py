"""Public compiler entry points composing the owned transformation stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..core.ir import CircuitIR
from ..core.ir import ensure_circuit_ir as ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig
from ..core.runtime_config import get_runtime_config as get_runtime_config
from .lowering.pipeline import compile_circuit as _compile_circuit
from .lowering.schedule_legalization import schedule_layers as schedule_layers
from .optimization import pipeline as _optimization_pipeline
from .optimization.canonicalization import (
    merge_adjacent_rotations as merge_adjacent_rotations,
)
from .optimization.canonicalization import merge_self_inverse as merge_self_inverse
from .optimization.canonicalization import (
    remove_identity_gates as remove_identity_gates,
)
from .optimization.commutation import CommutationAnalysis as CommutationAnalysis
from .optimization.commutation import analyze_commutation as analyze_commutation
from .optimization.optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
)
from .optimization.optimization_levels import (
    optimization_level_stages as optimization_level_stages,
)
from .optimization.pass_registry import BUILTIN_PASSES as BUILTIN_PASSES
from .optimization.pass_registry import default_pass_registry as _default_pass_registry
from .optimization.zero_state_reset import (
    remove_zero_state_resets as remove_zero_state_resets,
)
from .pass_manager import OPTIMIZATION_PIPELINE as OPTIMIZATION_PIPELINE
from .pass_manager import (
    AnalysisSpec as AnalysisSpec,
)
from .pass_manager import (
    PassBinding,
    PassRegistry,
)
from .pass_manager import (
    PassFunction as PassFunction,
)
from .pass_manager import (
    PassManager as PassManager,
)
from .pass_manager import (
    PassSpec as PassSpec,
)
from .routing import (
    CouplingMap,
    LinkErrorRates,
    route_to_topology,
)
from .routing import (
    record_post_routing_optimization as record_post_routing_optimization,
)
from .routing import (
    select_routing_strategy as select_routing_strategy,
)
from .topology.directed_topology import DirectedCouplingMap


def default_pass_registry() -> PassRegistry:
    """The registry that resolves exactly the built-in optimization passes."""

    return _default_pass_registry()


def _optimize_to_fixed_point(
    circuit_or_ir: Any,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    passes: Mapping[str, PassBinding] | None = None,
) -> CircuitIR:
    # Bind the public registry factory for optimize and post-routing optimization.
    return _optimization_pipeline._optimize_to_fixed_point(
        circuit_or_ir,
        optimization_level,
        passes,
        registry_factory=default_pass_registry,
    )


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
    routing_strategy: str = "auto",
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

    Automatic routing selects among existing deployable strategies, honoring
    calibration restrictions and restoring logical output placement. Request
    ``routing_strategy="restore_after_each_gate"`` to retain the previous policy.

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

    return _compile_circuit(
        circuit_or_ir,
        coupling_map=coupling_map,
        routing_strategy=routing_strategy,
        optimize=optimize,
        optimization_level=optimization_level,
        config=config,
        passes=passes,
        optimize_program=_optimize_to_fixed_point,
        route_program=route_to_topology,
    )


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
