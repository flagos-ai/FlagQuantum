"""Exact zero-entry specialization of the retained tensor-hypercontraction walk."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    inplace_prepare_amplitudes,
    phase,
    qft,
    qfunc,
)

CLOCK_AMPLITUDES = (
    0.06302556465232,
    0.123904869291238,
    0.180564742213955,
    0.231075699404113,
    0.273717650795818,
    0.307038475877809,
    0.329903473915844,
    0.34153400482528,
    0.34153400482528,
    0.329903473915844,
    0.307038475877809,
    0.273717650795818,
    0.231075699404113,
    0.180564742213955,
    0.123904869291239,
    0.06302556465232,
)
ELL_FIELDS = (
    ("mu", 5),
    ("nu", 5),
    ("m1", 1),
    ("succ", 1),
    ("r_br", 1),
    ("theta", 1),
    ("s", 9),
    ("theta_alt", 1),
    ("mu_alt", 5),
    ("nu_alt", 5),
    ("keep", 18),
    ("sigma", 18),
    ("aux", 1),
)


@qfunc
def main(
    weights: list[float],
    clock_phase: float,
    spin_phase: float,
    t: Output[QArray[QBit]],
    ell: Output[QArray[QBit]],
    psi: Output[QArray[QBit]],
) -> None:
    allocate(4, t)
    allocate(71, ell)
    allocate(8, psi)
    inplace_prepare_amplitudes(weights, 0, t)
    for i in range(4):
        phase(t[i], clock_phase * (i + 1))
    H(psi[6])
    H(psi[7])
    phase(psi[6], spin_phase)
    # On the initialized ell=0 subspace, SELECT and each reflection are identity.
    # The derivation and its limits are documented in WORKFLOW.md.
    with adjoint():
        qft(t)


def build_program(
    weights: tuple[float, ...] = CLOCK_AMPLITUDES,
    clock_phase: float = 0.0,
    spin_phase: float = 0.0,
) -> synth.Program:
    if len(weights) != 16 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in weights
    ):
        raise ValueError("provide sixteen finite real clock amplitudes")
    if not math.isclose(math.fsum(v * v for v in weights), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("clock amplitudes must be normalized")
    if any(
        type(v) not in (int, float) or not math.isfinite(v)
        for v in (clock_phase, spin_phase)
    ):
        raise ValueError("phases must be finite real numbers")
    return synth.synthesize(
        main,
        classical_inputs={
            "weights": list(weights),
            "clock_phase": clock_phase,
            "spin_phase": spin_phase,
        },
        max_qubits=83,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("t", "psi"))))
    print(program.report["resources"])
