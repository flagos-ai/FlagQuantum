"""Independent Householder preparation, preserving signed amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    amplitudes = torch.tensor(
        [
            0.245034472820708,
            -0.170081350544172,
            0.167697366233728,
            -0.023905223276522,
            0.343936056437118,
            -0.296692966947323,
            -0.283916724581065,
            0.330582853784989,
            -0.323760293689634,
            0.257046476143475,
            0.153163141682669,
            -0.075162649766642,
            -0.314744872342388,
            -0.249922153985921,
            0.352662065451226,
            -0.015039183255469,
        ],
        dtype=torch.complex128,
    )
    amplitudes = amplitudes / amplitudes.norm()
    identity = torch.eye(16, dtype=torch.complex128)
    difference = identity[:, 0] - amplitudes
    matrix = (
        identity
        - 2 * torch.outer(difference, difference.conj()) / difference.norm().square()
    )
    return fq.Circuit(4, dtype=torch.complex128).any(0, 1, 2, 3, matrix=matrix)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("out",))


if __name__ == "__main__":
    run_example()
