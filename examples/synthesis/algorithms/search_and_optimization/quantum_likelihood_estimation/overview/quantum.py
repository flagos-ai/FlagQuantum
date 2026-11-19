"""Use native qfunc for each experiment and batched Circuit for proposal scoring."""

import math
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

import flagquantum as fq
from flagquantum import synth
from flagquantum.observables import X, Z
from flagquantum.synth import PHASE, Output, QBit, R, U, allocate, qfunc, suzuki_trotter

HYPOTHESES = (("x", 1), ("x", 2), ("z", 1), ("z", 2))
OPERATORS = (X(0), 2 * X(0), Z(0), 2 * Z(0))
DTYPE = torch.complex128
RealArray = NDArray[np.float64]


def build_experiment(hypothesis: int, controls: Sequence[float]) -> synth.Program:
    """Bind time and four control angles for an actual one-shot experiment."""
    if not 0 <= hypothesis < len(OPERATORS) or len(controls) != 5:
        raise ValueError("Choose one hypothesis and exactly five finite controls")
    if not all(math.isfinite(value) for value in controls):
        raise ValueError("All controls must be finite")
    operator = OPERATORS[hypothesis]

    @qfunc
    def experiment(values: list[float], q: Output[QBit]) -> None:
        allocate(q)
        R(2 * values[1], math.pi / 2, q)
        PHASE(values[2], q)
        suzuki_trotter(operator, values[0], 1, 1, q)
        U(values[3], values[4], math.pi - values[4], 0, q)

    return synth.synthesize(
        experiment,
        classical_inputs={"values": tuple(controls)},
        max_qubits=1,
        max_workspace=0,
    )


def batch_state(hypothesis: int, controls: RealArray) -> torch.Tensor:
    """Execute every proposal with independent per-item gate angles."""
    time, alpha, beta, theta, phi = controls.T
    axis, coefficient = HYPOTHESES[hypothesis]
    circuit = fq.Circuit(1, bsz=len(controls), dtype=DTYPE)
    circuit.ry(0, theta=tuple(map(float, 2 * alpha)))
    circuit.phase(0, theta=tuple(map(float, beta)))
    circuit.gate("r" + axis, 0, theta=tuple(map(float, 2 * coefficient * time)))
    circuit.u3(
        0,
        theta=tuple(map(float, theta)),
        phi=tuple(map(float, phi)),
        lbd=tuple(map(float, math.pi - phi)),
    )
    output = fq.run(circuit).statevector()
    assert isinstance(output, torch.Tensor)
    return output


def likelihoods(controls: RealArray) -> RealArray:
    """Read both outcome probabilities for every hypothesis and proposal."""
    return np.asarray(
        [
            torch.square(torch.abs(batch_state(i, controls))).numpy()
            for i in range(len(HYPOTHESES))
        ],
        dtype=np.float64,
    )
