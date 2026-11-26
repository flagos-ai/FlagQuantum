"""Shared SDK inputs for source execution and independent resource comparisons."""

from typing import Any


def pauli_phasor_case(cirq: Any, kind: str) -> Any:
    word, coefficient, negative, positive = {
        "phasor_z": ("Z", 1, 0.3, 0.2),
        "phasor_xy": ("XY", 1, 0.3, 0.2),
        "phasor_yz": ("YZ", 1, 0.3, 0.2),
        "phasor_xyz": ("XYZ", 1, 0.3, 0.2),
        "phasor_negative": ("XY", -1, 0.3, 0.2),
        "phasor_zero": ("XY", 1, 0.0, 0.0),
        "phasor_empty": ("", 1, 0.3, 0.2),
    }[kind]
    return cirq.PauliStringPhasorGate(
        cirq.DensePauliString(word, coefficient=coefficient),
        exponent_neg=negative,
        exponent_pos=positive,
    )
