"""Shared Walsh/Gray expansion for checked uniformly controlled rotations."""

from __future__ import annotations

from collections.abc import Callable

from .._hybrid.model import Value


def emit_rotation_table(
    axis: str,
    selector: tuple[tuple[Value, int], ...],
    destination: tuple[Value, int],
    angles: tuple[float, ...],
    emit: Callable[[str, tuple[Value, ...], dict[str, object]], None],
) -> None:
    """Emit the existing cyclic Gray sequence; callers validate static operands."""
    target, wire = destination

    def gate(name: str, **attributes: object) -> None:
        emit("qreg.gate", (target,), {"gate": name, "index": wire, **attributes})

    if not selector:
        gate(f"r{axis.lower()}", angle=angles[0])
        return
    size = len(angles)
    coefficients = [angle / size for angle in angles]
    stride = 1
    while stride < size:
        for start in range(0, size, 2 * stride):
            for left in range(start, start + stride):
                right = left + stride
                a, b = coefficients[left], coefficients[right]
                coefficients[left], coefficients[right] = a + b, a - b
        stride *= 2
    if axis == "X":
        gate("h")
    rotation = "ry" if axis == "Y" else "rz"
    for step in range(size):
        gray = step ^ (step >> 1)
        gate(rotation, angle=coefficients[gray])
        following = (step + 1) % size
        edge = gray ^ following ^ (following >> 1)
        control, index = selector[edge.bit_length() - 1]
        emit(
            "qreg.pair_gate",
            (control, target),
            {"gate": "cx", "indices": (index, wire)},
        )
    if axis == "X":
        gate("h")
