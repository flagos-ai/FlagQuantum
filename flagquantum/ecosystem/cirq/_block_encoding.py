"""Convert concrete zero-signal external blocks to owned four-way snapshots."""

from typing import Any, cast

from ...algorithms.primitives.circuit_block_encoding import CircuitBlockEncoding
from ...compiler._component_trace.resource_model import ComponentCost
from ...core.ir import CircuitIR
from ._component_snapshot import _snapshot


def import_block_encoding(
    source: object,
    *,
    models: (
        tuple[
            ComponentCost | None,
            ComponentCost | None,
            ComponentCost | None,
            ComponentCost | None,
        ]
        | None
    ) = None,
    include_controlled: bool = True,
) -> CircuitBlockEncoding:
    """Capture actual forward/adjoint and positive-controlled source operations.

    Concrete system/ancilla/resource ports and an identity signal preparation
    are admitted. Shaped ports are flattened in source row-major order and must
    retain identical input/output shapes. The source decides how each control and
    adjoint decomposes; no measurement-containing stream is inverted afterward.
    Logical outputs are restored explicitly and binding swaps counted separately.
    Missing independent models remain unavailable; emission never implies a cost.
    Set include_controlled=False to capture only the ordinary directional pair.
    An unavailable source-controlled decomposition is not synthesized or ignored.
    """
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity

    if type(include_controlled) is not bool:
        raise ValueError("Controlled snapshot request must be Boolean")
    foreign = cast(Any, source)
    widths = (foreign.system_bitsize, foreign.ancilla_bitsize, foreign.resource_bitsize)
    if any(type(n) is not int or n < 0 for n in widths) or not widths[0]:
        raise ValueError("External block binding requires concrete nonnegative widths")
    if foreign.signal_state.prepare != PrepareIdentity.from_bitsizes([widths[1]]):
        raise ValueError("External block binding requires the identity signal state")
    ports = tuple(
        (name, width)
        for name, width in zip(("system", "ancilla", "resource"), widths, strict=True)
        if width
    )
    forward, fs = _snapshot(foreign, ports)
    backward, bs = _snapshot(foreign.adjoint(), ports)
    controlled_forward: CircuitIR | None = None
    controlled_backward: CircuitIR | None = None
    cs = ds = 0
    if include_controlled:
        controlled = foreign.controlled()
        controlled_forward, cs = _snapshot(controlled, (("ctrl", 1), *ports))
        controlled_backward, ds = _snapshot(controlled.adjoint(), (("ctrl", 1), *ports))
    return CircuitBlockEncoding(
        *widths,
        foreign.alpha,
        foreign.epsilon,
        (forward, backward, controlled_forward, controlled_backward),
        models,
        (fs, bs, cs, ds),
    )
