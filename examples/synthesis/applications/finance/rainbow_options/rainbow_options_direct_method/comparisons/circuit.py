"""Independent rational payoff branches and full joint-state amplification."""

import math
from fractions import Fraction

import torch

import flagquantum as fq

from ..physics import (
    EXPONENT_RATE,
    FRACTION_BITS,
    LOG_SHIFT,
    LOW_SCALE,
    PAYOFF_WIDTH,
    PRICE_SCALE,
    PROBABILITIES,
    STRIKE,
)


def prepared_state(rate: float = EXPONENT_RATE) -> torch.Tensor:
    """Evaluate branch amplitudes directly, without the synthesis expression graph."""
    state = torch.zeros((2, 4, 4, 32), dtype=torch.complex128)
    step = rate / (1 << FRACTION_BITS)
    maximum = ((1 << PAYOFF_WIDTH) - 1) / (1 << FRACTION_BITS)
    low_probability = LOW_SCALE**2 / max(
        PRICE_SCALE * math.exp(rate * maximum + LOG_SHIFT), STRIKE
    )
    normalization = math.fsum(PROBABILITIES)
    for x in range(4):
        for y in range(4):
            score = max(Fraction(x), Fraction(3 * (x + y) - 5, 4))
            raw = int(score * (1 << FRACTION_BITS))
            weight = math.sqrt(PROBABILITIES[x] * PROBABILITIES[y]) / normalization
            if score <= Fraction(3, 2):
                state[0, x, y, 0] = weight * math.sqrt(1 - low_probability)
                state[1, x, y, 0] = weight * math.sqrt(low_probability)
                continue
            for auxiliary in range(32):
                amplitude = weight
                for bit in range(PAYOFF_WIDTH):
                    selected = auxiliary >> bit & 1
                    if raw >> bit & 1:
                        if selected:
                            amplitude = 0.0
                            break
                    else:
                        zero_probability = math.exp(-step * (1 << bit))
                        amplitude *= math.sqrt(
                            1 - zero_probability if selected else zero_probability
                        )
                state[int(auxiliary == 0), x, y, auxiliary] = amplitude
    return state.flatten()


def reference_circuit(k: int = 1, rate: float = EXPONENT_RATE) -> fq.Circuit:
    """Build an independent dense Circuit for the ten physical model qubits.

    A Householder matrix prepares the directly evaluated joint state. Reflection
    about that state is independent of the native arithmetic/rotation lowering.
    Keep the oracle and every reflection as Circuit operations with exact phase.
    """
    state = prepared_state(rate)
    identity = torch.eye(state.numel(), dtype=torch.complex128)
    difference = identity[:, 0] - state
    preparation = identity - 2 * torch.outer(
        difference, difference.conj()
    ) / torch.vdot(difference, difference)
    reflection = 2 * torch.outer(state, state.conj()) - identity
    circuit = fq.Circuit(10, dtype=torch.complex128).any(*range(10), matrix=preparation)
    for _ in range(k):
        circuit.z(0).any(*range(10), matrix=reflection)
    return circuit


def run_example() -> None:
    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        actual = fq.run(program).statevector().reshape(1024, -1)
        expected = fq.run(reference_circuit(k)).statevector().flatten()
        torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
        assert float(actual[:, 1:].abs().square().sum()) < 1e-20
        print(k, program.report["resources"])
    print("Complete rainbow direct states and clean scratch agree.")


if __name__ == "__main__":
    run_example()
