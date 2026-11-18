"""Compose circuit lowering, optimization, and topology routing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from ...core.ir import CircuitIR, ensure_circuit_ir
from ...core.runtime_config import RuntimeConfig, get_runtime_config
from ..optimization.optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    optimization_level_stages,
)
from ..optimization.pass_manager import PassBinding
from ..topology.directed_topology import DirectedCouplingMap
from ..topology.routing import (
    CouplingMap,
    LinkErrorRates,
    _route_automatically,
    record_post_routing_optimization,
)


def compile_circuit(
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
    optimize_program: Callable[[Any, int, Mapping[str, PassBinding] | None], CircuitIR],
    route_program: Callable[..., CircuitIR],
) -> CircuitIR:
    """Lower a circuit and compose the supplied optimization and routing stages."""

    optimization_level_stages(optimization_level)
    if passes and not optimize:
        raise ValueError(
            "compile() was given passes while optimization is switched off; the "
            "admitted passes would not run, so the request is refused rather than "
            "answered with the program unchanged"
        )
    from .deferred_operations import lower_deferred_operations
    from .select_lowering import lower_select_program

    ir = lower_select_program(ensure_circuit_ir(circuit_or_ir), "dense")
    ir = replace(ir, instructions=lower_deferred_operations(ir.instructions))
    selected_config = config or get_runtime_config()
    metadata = dict(ir.metadata)
    metadata["runtime_config"] = selected_config.to_manifest()
    ir = replace(ir, metadata=metadata)
    if optimize:
        ir = optimize_program(ir, optimization_level, passes)
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
        if routing_strategy == "auto":
            ir, strategy_selection = _route_automatically(
                ir, coupling, route_program=route_program
            )
        else:
            ir = route_program(ir, coupling, strategy=routing_strategy)
        if optimize:
            ir = optimize_program(ir, optimization_level, passes)
        ir = record_post_routing_optimization(ir)
        if strategy_selection is not None:
            metadata = dict(ir.metadata)
            metadata["routing_strategy_selection"] = strategy_selection.summary()
            ir = replace(ir, metadata=metadata)
    return ir
