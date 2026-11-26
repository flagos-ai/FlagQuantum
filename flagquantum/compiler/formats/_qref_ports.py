# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Port projection adapted from Qualtran v0.7.0 qref_interop/_bloq_to_qref.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/qref_interop/_bloq_to_qref.py

"""Directional QREF port names for native register declarations."""

from collections.abc import Iterable
from typing import Literal, TypedDict

from ...core.ir.program.registers import Register, RegisterSide


class QrefPort(TypedDict):
    name: str
    direction: Literal["input", "output"]
    size: int | str


_DIRECTIONS: tuple[tuple[str, Literal["input", "output"], RegisterSide], ...] = (
    ("in", "input", RegisterSide.LEFT),
    ("out", "output", RegisterSide.RIGHT),
)


def _register_indices(register: Register) -> Iterable[tuple[int, ...]]:
    try:
        shape = register.shape
    except ValueError:
        # The source represents an unresolved array shape by one unsuffixed port.
        return ((),)
    if len(shape) > 1:
        raise NotImplementedError(
            f"QREF register {register.name!r} has more than one shape dimension"
        )
    return register.all_idxs()


def _port_suffix(index: tuple[int, ...]) -> str:
    if len(index) > 1:
        raise NotImplementedError("QREF port indices are at most one-dimensional")
    return f"_{index[0]}" if index else ""


def qref_ports(registers: Iterable[Register]) -> list[QrefPort]:
    """Split through declarations and expand concrete rank-one register arrays."""
    ports: list[QrefPort] = []
    for register in registers:
        start = len(ports)
        size = register.bitsize
        wire_size = size if type(size) is int else str(size)
        for prefix, direction, side in _DIRECTIONS:
            if not register.side & side:
                continue
            for index in _register_indices(register):
                ports.append(
                    {
                        "name": f"{prefix}_{register.name}{_port_suffix(index)}",
                        "size": wire_size,
                        "direction": direction,
                    }
                )
        ports[start:] = sorted(ports[start:], key=lambda port: port["name"])
    return ports
