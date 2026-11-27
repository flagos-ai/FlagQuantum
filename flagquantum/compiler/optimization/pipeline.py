"""Declared optimization levels executed by the pass manager."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from ...core.ir import CircuitIR, ensure_circuit_ir
from .commutation import _reuse_commutation_analysis
from .optimization_levels import DEFAULT_OPTIMIZATION_LEVEL, optimization_level_stages
from .pass_manager import PassBinding, PassManager, PassRegistry
from .pass_registry import default_pass_registry


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
    *,
    registry_factory: Callable[[], PassRegistry] = default_pass_registry,
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

    registry = registry_factory()
    for name, function in (passes or {}).items():
        registry = registry.with_pass(name, function)
    return registry


def _optimize_to_fixed_point(
    circuit_or_ir: Any,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    passes: Mapping[str, PassBinding] | None = None,
    *,
    registry_factory: Callable[[], PassRegistry] = default_pass_registry,
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
    ir = ensure_circuit_ir(circuit_or_ir)
    admitted = tuple((passes or {}).keys())
    sequence = (*stages, *admitted)
    if not sequence:
        return _with_recorded_level(ir, optimization_level)
    with _reuse_commutation_analysis():
        optimized = PassManager(
            _registry_admitting(passes, registry_factory=registry_factory)
        ).to_fixed_point(ir, sequence)
    return _with_recorded_level(optimized, optimization_level, admitted)
