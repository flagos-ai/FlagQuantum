"""Verify native arithmetic-oracle search against its original and a dense Circuit."""

import hashlib
from pathlib import Path

import pytest

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.main import (
    build_program,
)
from flagquantum import synth

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/tutorials/workshops/grover_workshop/grover_workshop.qmod"
)


def test_native_arithmetic_grover_matches_original_and_dense_circuit() -> None:
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "d91986e3901f5eda691500013b20f111bb1fb6c679b32d44a603173dea6d5ad8"
    )
    native = build_program()
    original = synth.synthesize(SOURCE, max_workspace=0)
    reference = reference_circuit()
    compare_states(native, reference, ("a", "b", "c"))
    compare_states(original, reference, ("a", "b", "c"))
    assert native.report["workspace_qubits"] == 0


if __name__ == "__main__":
    test_native_arithmetic_grover_matches_original_and_dense_circuit()
