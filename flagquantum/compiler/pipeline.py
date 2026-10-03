"""Native FlagQuantum optimization, scheduling, and lowering pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.runtime_config import RuntimeConfig, get_runtime_config
from ..errors import CompilationError
from .passes import (
    IDENTITY_REMOVAL,
    ROTATION_MERGE,
    SELF_INVERSE_CANCELLATION,
    CompilerPass,
    PassManager,
    RegisteredPass,
)
from .routing import (
    CouplingMap,
    record_post_routing_optimization,
    route_to_topology,
    select_routing_strategy,
)
from .zero_state_reset import remove_zero_state_resets

# The target-independent optimization set, in the order the manager runs it.
# `identity_removal` runs first so that removing an identity can expose a pair
# for `self_inverse_cancellation`, and `rotation_merge` can leave a rotation at
# exactly zero only by removing it, so no trailing identity sweep is needed.
DEFAULT_OPTIMIZATION_PASSES: tuple[RegisteredPass, ...] = (
    IDENTITY_REMOVAL,
    SELF_INVERSE_CANCELLATION,
    ROTATION_MERGE,
)


def _as_ir(circuit_or_ir: Any) -> CircuitIR:
    return ensure_circuit_ir(circuit_or_ir)


def _optimization_manager(passes: Iterable[CompilerPass] | None) -> PassManager:
    """Return the manager for ``passes``, defaulting to the registered set."""

    if passes is None:
        return PassManager(passes=DEFAULT_OPTIMIZATION_PASSES)
    manager = PassManager()
    for item in passes:
        manager = manager.with_pass(item)
    return manager


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


def _optimize_to_fixed_point(
    circuit_or_ir: Any,
    *,
    passes: Iterable[CompilerPass] | None = None,
) -> CircuitIR:
    """Run the optimization set to a fixed point, with the two unmanaged passes.

    ``remove_zero_state_resets`` and ``collapse_one_qubit_runs`` are not
    declared passes yet: the first reads the register's initial state rather
    than an opcode table, and the second decides per wire set rather than per
    instruction. They are therefore composed around the manager instead of
    being registered with it, and each is called from exactly one place below,
    so the fixed-point loop the tests measure is the loop that runs.
    """

    from .one_qubit_optimization import collapse_one_qubit_runs

    manager = _optimization_manager(passes)
    ir = _as_ir(circuit_or_ir)
    max_rounds = len(ir) + 1
    for _ in range(max_rounds):
        previous_count = len(ir)
        # First, because a reset this pass can remove is removable whatever the
        # passes below do, and removing it hands them a shorter program. The
        # loop, not this ordering, is what earns the reach on a wire the passes
        # below only empty out later: `x(0) x(0) reset(0)` needs a second round.
        ir = remove_zero_state_resets(ir)
        ir = manager.run(ir)
        ir = collapse_one_qubit_runs(ir)
        if len(ir) == previous_count:
            return ir
    raise CompilationError("compiler optimization passes did not reach a fixed point")


def optimize(
    circuit_or_ir: Any,
    *,
    passes: Iterable[CompilerPass] | None = None,
) -> CircuitIR:
    """Apply target-independent circuit optimizations to a fixed point.

    ``passes`` replaces the registered optimization set for this call, so a
    caller-supplied or extension-supplied pipeline compiles through the same
    entry point as the default one.
    """

    return _optimize_to_fixed_point(circuit_or_ir, passes=passes)


def compile(
    circuit_or_ir: Any,
    *,
    coupling_map: CouplingMap | Iterable[tuple[int, int]] | None = None,
    routing_strategy: str = "restore_after_each_gate",
    optimize: bool = True,
    passes: Iterable[CompilerPass] | None = None,
    config: RuntimeConfig | None = None,
) -> CircuitIR:
    """Compile a circuit with optional topology routing and optimization."""

    ir = _as_ir(circuit_or_ir)
    selected_config = config or get_runtime_config()
    metadata = dict(ir.metadata)
    metadata["runtime_config"] = selected_config.to_manifest()
    ir = replace(ir, metadata=metadata)
    if optimize:
        ir = _optimize_to_fixed_point(ir, passes=passes)
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
            ir = _optimize_to_fixed_point(ir, passes=passes)
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
