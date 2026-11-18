"""Convert concrete through-port oracles into native named emitters."""

from typing import Any, cast

from ...algorithms.primitives.circuit_register_emitter import CircuitRegisterEmitter
from ._component_snapshot import _snapshot


def import_register_emitter(
    source: object, *, local_ports: tuple[str, ...] = ()
) -> CircuitRegisterEmitter:
    """Capture both original directions for existing PREPARE and SELECT consumers.

    Logical junk remains a logical port; private allocations belong to the
    snapshot's clean workspace. Explicit local_ports name RIGHT outputs with
    caller-owned clean storage; these remain live until the captured adjoint
    consumes them. Without that request, one-sided ports are rejected. The
    source's own adjoint is captured independently, including its feedback.
    A scalar with no logical ports owns one clean physical carrier, as required
    by the existing CircuitIR. This carrier does not add a source logical port.
    """
    foreign = cast(Any, source)
    registers = tuple(foreign.signature)
    if any(reg.side.value != 3 and reg.name not in local_ports for reg in registers):
        raise ValueError("Named emitter capture requires through ports")
    if any(reg.name in local_ports and reg.side.value != 2 for reg in registers):
        raise ValueError("Forward local ports must be RIGHT outputs")
    # _snapshot validates the original widths and shapes before accepting them.
    ports = tuple((reg.name, reg.total_bits()) for reg in registers)
    forward, fs = _snapshot(
        foreign,
        ports,
        local_ports=local_ports,
        allow_empty_ports=True,
        scalar_carrier=not ports,
    )
    backward, bs = _snapshot(
        foreign.adjoint(),
        ports,
        local_ports=local_ports,
        allow_empty_ports=True,
        scalar_carrier=not ports,
    )
    return CircuitRegisterEmitter(ports, (forward, backward), (fs, bs))
