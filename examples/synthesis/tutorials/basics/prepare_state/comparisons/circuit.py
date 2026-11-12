"""Build a dense reflection reference independently of synthesis lowering."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.prepare_state.main import main
from flagquantum import synth


def probabilities() -> torch.Tensor:
    return torch.tensor(
        [
            0,
            0.002,
            0.004,
            0.006,
            0.0081,
            0.0101,
            0.0121,
            0.0141,
            0.0161,
            0.0181,
            0.0202,
            0.0222,
            0.0242,
            0.0262,
            0.0282,
            0.0302,
            0.0323,
            0.0343,
            0.0363,
            0.0383,
            0.0403,
            0.0423,
            0.0444,
            0.0464,
            0.0484,
            0.0504,
            0.0524,
            0.0544,
            0.0565,
            0.0585,
            0.0605,
            0.0625,
        ],
        dtype=torch.float64,
    )


def reference_circuit() -> fq.Circuit:
    identity = torch.eye(32, dtype=torch.complex128)
    delta = identity[:, 0] - probabilities().sqrt()
    matrix = identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    return fq.Circuit(5, dtype=torch.complex128).any(*range(5), matrix=matrix)


def run_example() -> None:
    compare_states(synth.synthesize(main), reference_circuit(), ("io",))


if __name__ == "__main__":
    run_example()
