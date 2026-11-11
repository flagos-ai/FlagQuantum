"""Independent rational branches and exact reflection on the complete model."""

import math
from fractions import Fraction

import torch

import flagquantum as fq

from ..physics import EXPONENT_RATE, LOG_SHIFT, PRICE_SCALE, PROBABILITIES, STRIKE


def prepared_state(rate: float = EXPONENT_RATE) -> torch.Tensor:
    """Evaluate the joint branch law without using native loading angles."""
    step = rate / 4
    scale = PRICE_SCALE * math.exp(LOG_SHIFT)
    low_probability = (STRIKE - scale * math.exp(-step)) / (
        scale * math.expm1(32 * step) * math.exp(-step)
    )
    weights = [math.exp(step * word) for word in range(32)]
    normalization = math.fsum(weights)
    input_normalization = math.fsum(PROBABILITIES)
    state = torch.zeros((2, 4, 4, 32), dtype=torch.complex128)
    for x in range(4):
        for y in range(4):
            score = max(Fraction(x), Fraction(3 * (x + y) - 5, 4))
            amplitude = (
                math.sqrt(PROBABILITIES[x] * PROBABILITIES[y]) / input_normalization
            )
            if score <= Fraction(3, 2):
                state[0, x, y, 0] = amplitude * math.sqrt(1 - low_probability)
                state[1, x, y, 0] = amplitude * math.sqrt(low_probability)
            else:
                for reference in range(32):
                    indicator = int(Fraction(reference, 4) <= score)
                    state[indicator, x, y, reference] = amplitude * math.sqrt(
                        weights[reference] / normalization
                    )
    return state.flatten()


def reference_circuit(k: int = 1, rate: float = EXPONENT_RATE) -> fq.Circuit:
    """Prepare the independently evaluated vector and retain the Grover sign."""
    state = prepared_state(rate)
    identity = torch.eye(1024, dtype=torch.complex128)
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
    print("Complete Rainbow integration states and clean scratch agree.")


if __name__ == "__main__":
    run_example()
