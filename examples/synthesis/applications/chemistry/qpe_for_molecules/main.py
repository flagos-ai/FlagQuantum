"""Native LiH QPE preserving power-specific second-order product formulas."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    control,
    hadamard_transform,
    inplace_prepare_amplitudes,
    qft,
    qfunc,
    suzuki_trotter,
)

from .physics import G0, G1, G2, G3, G4, G5, G6, G7, STEPS, ZERO_STATE


@qfunc
def power(time: float, n: int, state: QArray[QBit]) -> None:
    suzuki_trotter(G0, time / (2 * n), 1, 1, state)
    for i in range(n):
        suzuki_trotter(G1, time / (2 * n), 1, 1, state)
        suzuki_trotter(G2, time / (2 * n), 1, 1, state)
        suzuki_trotter(G3, time / (2 * n), 1, 1, state)
        suzuki_trotter(G4, time / (2 * n), 1, 1, state)
        suzuki_trotter(G5, time / (2 * n), 1, 1, state)
        suzuki_trotter(G6, time / (2 * n), 1, 1, state)
        suzuki_trotter(G7, time / n, 1, 1, state)
        suzuki_trotter(G6, time / (2 * n), 1, 1, state)
        suzuki_trotter(G5, time / (2 * n), 1, 1, state)
        suzuki_trotter(G4, time / (2 * n), 1, 1, state)
        suzuki_trotter(G3, time / (2 * n), 1, 1, state)
        suzuki_trotter(G2, time / (2 * n), 1, 1, state)
        suzuki_trotter(G1, time / (2 * n), 1, 1, state)
        if i < n - 1:
            suzuki_trotter(G0, time / n, 1, 1, state)
        else:
            suzuki_trotter(G0, time / (2 * n), 1, 1, state)


@qfunc
def selected_power(time: float, n: int, bit: QBit, state: QArray[QBit]) -> None:
    control(bit, lambda: power(time, n, state))


@qfunc
def main(
    times: list[float],
    steps: list[int],
    amplitudes: list[float],
    state: Output[QArray[QBit]],
    phase: Output[QArray[QBit]],
) -> None:
    allocate(6, state)
    allocate(7, phase)
    inplace_prepare_amplitudes(amplitudes, 0, state)
    hadamard_transform(phase)
    for i in range(7):
        selected_power(times[i], steps[i], phase[i], state)
    with adjoint():
        qft(phase)


def classical_inputs(
    scale: float = 1.0, amplitudes: tuple[float, ...] = ZERO_STATE
) -> dict[str, list[float] | list[int]]:
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or not 0 < scale <= 2
    ):
        raise ValueError("time scale must be a finite real in (0,2]")
    if len(amplitudes) != 64 or any(
        type(a) not in (int, float) or not math.isfinite(a) for a in amplitudes
    ):
        raise ValueError("provide 64 finite real input amplitudes")
    if not math.isclose(
        math.fsum(a * a for a in amplitudes), 1, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("input amplitudes must be normalized")
    return {
        "times": [-2 * math.pi * scale * (2**i) for i in range(7)],
        "steps": list(STEPS),
        "amplitudes": list(amplitudes),
    }


def build_program(
    scale: float = 1.0, amplitudes: tuple[float, ...] = ZERO_STATE
) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs=classical_inputs(scale, amplitudes),
        max_qubits=13,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    probabilities = program.probabilities(result, registers=("phase",))
    print(
        {
            (word if word < 64 else word - 128) / 128: probability
            for (word,), probability in probabilities.items()
        }
    )
    print(program.report["resources"])
