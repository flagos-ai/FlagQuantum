"""Native source cost layer and static selection from the adaptive mixer pool."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    R,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)

from ..main import cost_layer
from .model import Mixer

XX = fq.X(0) @ fq.X(1)
YY = fq.Y(0) @ fq.Y(1)


@qfunc
def apply_mixer(
    kind: int, first: int, second: int, beta: float, v: QArray[QBit]
) -> None:
    if kind == 0:
        for bit in range(8):
            RX(2 * beta, v[bit])
    if kind == 1:
        for bit in range(8):
            R(2 * beta, math.pi / 2, v[bit])
    if kind == 2:
        RX(2 * beta, v[first])
    if kind == 3:
        R(2 * beta, math.pi / 2, v[first])
    if kind == 4:
        suzuki_trotter(XX, beta, 1, 1, [v[first], v[second]])
    if kind == 5:
        suzuki_trotter(YY, beta, 1, 1, [v[first], v[second]])


@qfunc
def adaptive(
    kinds: list[int],
    first: list[int],
    second: list[int],
    gammas: list[float],
    betas: list[float],
    extra_gamma: float,
    v: Output[QArray[QBit]],
) -> None:
    allocate(8, v)
    hadamard_transform(v)
    for layer in range(len(kinds)):
        cost_layer(gammas[layer], v)
        apply_mixer(kinds[layer], first[layer], second[layer], betas[layer], v)
    if extra_gamma != 0:
        cost_layer(extra_gamma, v)


def build_program(
    mixers: tuple[Mixer, ...], parameters: tuple[float, ...], extra_gamma: float = 0.0
) -> synth.Program:
    layers = len(mixers)
    if not 1 <= layers <= 24 or len(parameters) != 2 * layers:
        raise ValueError(
            "ADAPT-QAOA requires two angles per layer and at most 24 layers"
        )
    return synth.synthesize(
        adaptive,
        classical_inputs={
            "kinds": tuple(m.kind for m in mixers),
            "first": tuple(m.supports[0][0] for m in mixers),
            "second": tuple(m.supports[0][-1] for m in mixers),
            "gammas": parameters[:layers],
            "betas": parameters[layers:],
            "extra_gamma": extra_gamma,
        },
        max_qubits=8,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=1024**2,
    )
