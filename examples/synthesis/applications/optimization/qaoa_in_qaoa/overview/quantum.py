"""Bounded native community QAOA specializes fresh signed graph coefficients."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def community_program(
    node_count: int,
    first: list[int],
    second: list[int],
    weights: list[float],
    params: list[float],
    target: Output[QArray[QBit]],
) -> None:
    allocate(node_count, target)
    hadamard_transform(target)
    for layer in range(2):
        for edge in range(len(weights)):
            phase(
                weights[edge]
                * (
                    target[first[edge]]
                    + target[second[edge]]
                    - 2 * target[first[edge]] * target[second[edge]]
                ),
                params[2 * layer],
            )
        for bit in range(node_count):
            RX(params[2 * layer + 1], target[bit])


def build_program(
    node_count: int,
    edges: tuple[tuple[int, int, float], ...],
    parameters: tuple[float, ...],
) -> synth.Program:
    if type(node_count) is not int or not 1 <= node_count <= 6:
        raise ValueError("communities require one to six nodes")
    if len(parameters) != 4 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in parameters
    ):
        raise ValueError("two layers require four finite real angles")
    if not edges or len(edges) > 15:
        raise ValueError("a quantum community requires one to fifteen edges")
    pairs = set()
    for first, second, weight in edges:
        if (
            type(first) is not int
            or type(second) is not int
            or not 0 <= first < second < node_count
        ):
            raise ValueError(
                "each edge requires distinct ordered node indices in range"
            )
        if (
            (first, second) in pairs
            or type(weight) not in (int, float)
            or not math.isfinite(weight)
            or weight == 0
        ):
            raise ValueError("edges must be unique with finite nonzero signed weights")
        pairs.add((first, second))
    return synth.synthesize(
        community_program,
        classical_inputs={
            "node_count": node_count,
            "first": tuple(e[0] for e in edges),
            "second": tuple(e[1] for e in edges),
            "weights": tuple(e[2] for e in edges),
            "params": parameters,
        },
        max_qubits=node_count,
        max_workspace=0,
        max_operations=2048,
        memory_limit_bytes=1024**2,
    )


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    if (
        tuple(program.registers) != ("target",)
        or not 1 <= len(program.registers["target"]) <= 6
    ):
        raise ValueError("community readout requires one to six named target bits")
    values = program.probabilities(fq.run(program), registers=("target",), threshold=0)
    return tuple(
        values.get((word,), 0.0)
        for word in range(1 << len(program.registers["target"]))
    )
