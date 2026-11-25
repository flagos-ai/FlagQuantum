"""Independent X followed by RY(-theta), including the absolute sign."""

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.molecule_eigensolver.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states

BINDINGS = (DEFAULT_PARAMS, (-0.81,))


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    return fq.Circuit(1, dtype=torch.complex128).x(0).ry(0, theta=-params[0])


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("state",))


if __name__ == "__main__":
    run_example()
