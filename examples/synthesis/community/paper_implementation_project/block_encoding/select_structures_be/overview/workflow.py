"""Evaluate the actual, unnormalized flags-zero vectors."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq

from .model import DTYPE, Structure, make_structures
from .quantum import build_encoding


@dataclass(frozen=True)
class Projection:
    """The realized source-sized block and its independent signed reference."""

    structure: Structure
    circuit: fq.Circuit
    actual: torch.Tensor
    expected: torch.Tensor
    scale: int
    error: float


def project(structure: Structure, data: tuple[float, ...]) -> Projection:
    """Compare the entire complex projection against the classical matrix product."""
    circuit = build_encoding(structure, data)
    vector = fq.run(circuit).statevector().flatten()[: len(data)]
    scale = len(structure.addresses)
    expected = structure.matrix @ torch.tensor(data, dtype=DTYPE) / scale
    return Projection(
        structure,
        circuit,
        vector,
        expected,
        scale,
        float((vector - expected).abs().max()),
    )


def run_workflow() -> tuple[Projection, ...]:
    """Prepare every original uniform input and evaluate the four blocks."""
    results = []
    for model in make_structures():
        size = model.matrix.shape[0]
        results.append(project(model, (1 / math.sqrt(size),) * size))
    return tuple(results)
