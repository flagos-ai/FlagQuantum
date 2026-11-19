"""Execute the original block-projection experiment with full native states."""

from dataclasses import dataclass
from typing import cast

import torch

import flagquantum as fq
from flagquantum.compiler.resource_estimation import (
    ResourceEstimate,
    estimate_resources,
)

from . import physics, quantum


@dataclass(frozen=True)
class Result:
    expected: tuple[complex, ...]
    projected: tuple[complex, ...]
    l2_error: float
    work_leak: float
    block_resources: ResourceEstimate
    qsvt_resources: ResourceEstimate


def run(seed: int = 17) -> Result:
    initial = physics.initial_state(seed)
    encoding = quantum.block_encoding()
    circuit = quantum.input_circuit(initial).compose(encoding)
    state = cast(torch.Tensor, fq.run(circuit).statevector()).reshape(1024, 128)
    projected = state[0] * quantum.NORMALIZATION
    expected = physics.matrix() @ initial.to(torch.complex128)
    error = float((projected - expected).norm())
    work_leak = float(state.reshape(4, 256, 128)[1:].abs().square().sum())
    if error >= 1e-10 or work_leak >= 1e-24:
        raise ValueError(
            "Full original-grid projection or clean work validation failed"
        )
    return Result(
        tuple(complex(value) for value in expected),
        tuple(complex(value) for value in projected),
        error,
        work_leak,
        estimate_resources(encoding.to_ir()),
        estimate_resources(quantum.qsvt_step().to_ir()),
    )
