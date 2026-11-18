"""Convert concrete through-port oracles into native named emitters."""

from typing import Any, cast

from ...algorithms.primitives.circuit_register_emitter import CircuitRegisterEmitter
from ._component_snapshot import _snapshot


def import_register_emitter(source: object) -> CircuitRegisterEmitter:
    """Capture both original directions for existing PREPARE and SELECT consumers.

    Logical junk remains a logical port; private allocations belong to the
    snapshot's clean workspace. One-sided ports require an explicit lifetime
    binding and are rejected here rather than reclassified as through junk.
    The source's own adjoint is captured independently, including its feedback.
    """
    foreign = cast(Any, source)
    registers = tuple(foreign.signature)
    if any(reg.side.value != 3 for reg in registers):
        raise ValueError("Named emitter capture requires through ports")
    # _snapshot validates the original widths and shapes before accepting them.
    ports = tuple((reg.name, reg.total_bits()) for reg in registers)
    forward, fs = _snapshot(foreign, ports)
    backward, bs = _snapshot(foreign.adjoint(), ports)
    return CircuitRegisterEmitter(ports, (forward, backward), (fs, bs))
