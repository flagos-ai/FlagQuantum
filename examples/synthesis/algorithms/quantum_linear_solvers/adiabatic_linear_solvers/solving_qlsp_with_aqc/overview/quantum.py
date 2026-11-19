"""Prepare the linear-system input and lower the declared Pauli evolution."""

from flagquantum import synth
from flagquantum.observables import I, Observable, X, Y, Z
from flagquantum.synth import Output, QArray, QBit, prepare_state, qfunc, suzuki_trotter

from .model import LinearSystem, PauliCoefficient, decompose


def _operator(terms: tuple[PauliCoefficient, ...]) -> Observable:
    paulis = (I, X, Y, Z)
    operator = 0.0 * I(0)
    for term in terms:
        word = I(0)
        for bit, axis in enumerate(reversed(term.axes)):
            if axis:
                word = word @ paulis[axis](bit)
        operator = operator + term.coefficient * word
    return operator


def build_program(
    model: LinearSystem, duration: int, steps: int, *, max_qubits: int = 3
) -> synth.Program:
    """Capture up to 100 source time slices, each with two ordered products."""
    if type(duration) is not int or not 1 <= duration <= 10:
        raise ValueError("The duration must be an integer from 1 to 10")
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("The step count must be an integer from 1 to 100")
    h0, h1 = _operator(decompose(model.h0)), _operator(decompose(model.h1))
    delta_time = duration / steps
    probabilities = tuple(float(value**2) for value in model.initial)
    weights: list[float] = []
    for step in range(steps):
        ratio = step * delta_time / duration
        weights.extend((delta_time * (1.0 - ratio), delta_time * ratio))

    @qfunc
    def evolve(
        probabilities: list[float], weights: list[float], state: Output[QArray[QBit]]
    ) -> None:
        prepare_state(probabilities, 0.0, state)
        for step in range(steps):
            suzuki_trotter(h0, weights[2 * step], 1, 1, state)
            suzuki_trotter(h1, weights[2 * step + 1], 1, 10, state)

    return synth.synthesize(
        evolve,
        classical_inputs={"probabilities": probabilities, "weights": weights},
        max_qubits=max_qubits,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
