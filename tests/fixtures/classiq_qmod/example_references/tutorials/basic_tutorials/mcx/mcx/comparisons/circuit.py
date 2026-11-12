"""Test-only exact product-MPS check of the byte-identical fifty-control source."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.mcx.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.ecosystem._qmod.source import load_qmod


def run_example() -> None:
    imported = load_qmod(Path(__file__).with_name("source.qmod"))
    selected = select_register_program(
        imported.program, imported.functions, include_boolean=True
    ).selected
    assert selected and selected.target
    options = fq.ExecutionOptions(
        mode="mps",
        precision="complex128",
        allow_approximate=False,
        allow_backend_fallback=False,
        memory_limit_bytes=2048,
    )
    actual = fq.run(selected.target.submitted, options=options).native()
    expected = fq.run(reference_circuit().to_ir(), options=options).native()
    for left, right in zip(
        getattr(actual, "tensors"), getattr(expected, "tensors"), strict=True
    ):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)


if __name__ == "__main__":
    run_example()
