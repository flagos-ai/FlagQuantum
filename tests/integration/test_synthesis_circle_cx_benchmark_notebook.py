"""Check zero-work phase, endpoint selection, and the pinned comparison data."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.discrete_quantum_walk_circle.overview import (
    archive,
    benchmark,
    compact,
)

pytestmark = pytest.mark.integration


def test_gray_code_parity_preserves_every_complex_amplitude() -> None:
    trial = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        trial.h(wire).rz(wire, 0.14 + 0.19 * wire)
    reference = trial.copy()
    compact.controlled_x(trial, (0, 3, 1), 2)
    permutation = list(range(16))
    permutation[13], permutation[15] = permutation[15], permutation[13]
    reference.any(*range(4), matrix=torch.eye(16, dtype=torch.complex128)[permutation])
    torch.testing.assert_close(
        fq.run(trial).statevector(), fq.run(reference).statevector(), rtol=0, atol=1e-12
    )


def test_complete_endpoint_states_and_actual_budgets() -> None:
    assert tuple(range(5, 12)) == benchmark.SIZES
    for width in (5, 11):
        result = benchmark.search(width)
        assert result.candidate_count == width
        assert result.width_first.resources.n_qubits == width + 1
        assert result.cx_first.resources.n_qubits <= 2 * width
        assert 0 < result.cx_first.cx < result.width_first.cx
        assert benchmark.check(result.width_first, width) < 1e-10
        assert benchmark.check(result.cx_first, width) < 1e-10
    with pytest.raises(ValueError, match="5 to 11"):
        benchmark.search(4)


def test_archival_data_and_notebook_stages() -> None:
    width, cx = archive.read()
    assert len(width) == len(cx) == 13
    assert (width[0].position_bits, width[-1].position_bits) == (5, 60)
    assert (width[-1].cx, cx[-1].cx) == (542090, 17148)
    assert len(archive.QISKIT) == 6
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview"
    )
    source = json.loads((directory / "provenance.json").read_text())["original_assets"][
        0
    ]
    assert (
        hashlib.sha256(archive.FILE.read_bytes()).hexdigest() == source["sha256_local"]
    )
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 15
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
