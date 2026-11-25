"""Compose source X/Y/Z measurement bases with the frozen native ansatz."""

import math
from typing import Literal

import flagquantum as fq
from examples.synthesis.applications.optimization.robust_posture_optimization.main import (
    first_rotation,
    second_rotation,
)
from examples.synthesis.applications.optimization.robust_posture_optimization.main import (
    main as prepare,
)
from flagquantum import synth
from flagquantum.synth import H, Output, QArray, QBit, S, allocate, qfunc

Basis = Literal["X", "Y", "Z"]


@qfunc
def rotate(theta: list[float], q: QArray[QBit]) -> None:
    first_rotation(theta, q[0])
    second_rotation(theta, q[1])
    first_rotation(theta, q[1])


@qfunc
def measure_x(theta: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    rotate(theta, q)
    H(q[0])
    H(q[1])


@qfunc
def measure_y(theta: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    rotate(theta, q)
    S(q[0])
    H(q[0])
    S(q[1])
    H(q[1])


def build_program(parameters: tuple[float, ...], basis: Basis) -> synth.Program:
    if len(parameters) != 6 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("posture requires six finite real angles")
    if basis not in ("X", "Y", "Z"):
        raise ValueError("measurement basis must be X, Y or Z")
    entry = {"X": measure_x, "Y": measure_y, "Z": prepare}[basis]
    return synth.synthesize(
        entry,
        classical_inputs={"theta": parameters},
        max_qubits=2,
        max_workspace=0,
        max_operations=128,
        memory_limit_bytes=1024**2,
    )


def distribution(parameters: tuple[float, ...], basis: Basis) -> tuple[float, ...]:
    program = build_program(parameters, basis)
    values = program.probabilities(fq.run(program), registers=("q",), threshold=0)
    return tuple(values.get((word,), 0.0) for word in range(4))
