"""Shared SDK inputs for source execution and independent resource comparisons."""

from typing import Any

REMAINING_CONTROL_FAMILIES = (
    "hpow",
    "cczpow",
    "xxpow",
    "yypow",
    "phasedx",
    "phasedxz",
    "phasediswap",
    "givens",
    "h",
    "identity2",
)


def remaining_control_case(cirq: Any, kind: str) -> Any:
    return {
        "hpow": cirq.HPowGate(exponent=0.375),
        "cczpow": cirq.CCZPowGate(exponent=0.375),
        "xxpow": cirq.XXPowGate(exponent=0.375),
        "yypow": cirq.YYPowGate(exponent=0.375),
        "phasedx": cirq.PhasedXPowGate(phase_exponent=0.2, exponent=0.375),
        "phasedxz": cirq.PhasedXZGate(
            x_exponent=0.375, z_exponent=0.13, axis_phase_exponent=0.2
        ),
        "phasediswap": cirq.PhasedISwapPowGate(phase_exponent=0.2, exponent=0.375),
        "givens": cirq.givens(0.375),
        "h": cirq.H,
        "identity2": cirq.IdentityGate(2),
    }[kind]


def pauli_phasor_case(cirq: Any, kind: str) -> Any:
    word, coefficient, negative, positive = {
        "phasor_z": ("Z", 1, 0.3, 0.2),
        "phasor_xy": ("XY", 1, 0.3, 0.2),
        "phasor_yz": ("YZ", 1, 0.3, 0.2),
        "phasor_xyz": ("XYZ", 1, 0.3, 0.2),
        "phasor_negative": ("XY", -1, 0.3, 0.2),
        "phasor_zero": ("XY", 1, 0.0, 0.0),
        "phasor_empty": ("", 1, 0.3, 0.2),
        "phasor_ix": ("IX", 1, 0.3, 0.2),
        "phasor_identity": ("II", 1, 0.3, 0.2),
    }[kind]
    return cirq.PauliStringPhasorGate(
        cirq.DensePauliString(word, coefficient=coefficient),
        exponent_neg=negative,
        exponent_pos=positive,
    )
