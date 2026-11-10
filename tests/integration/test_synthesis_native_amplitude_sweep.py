"""Reproduce the full preparation Notebook locally without a compression claim."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.bound_sweep import (
    plot_sweep,
    run_sweep,
    source_amplitudes,
)
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]


def test_complete_256_amplitude_ten_bound_workflow_against_independent_circuit(
    tmp_path: Path,
) -> None:
    vector = torch.tensor(source_amplitudes(), dtype=torch.complex128)
    difference = -vector.clone()
    difference[0] += 1
    matrix = torch.eye(256, dtype=torch.complex128) - (
        2
        * torch.outer(difference, difference.conj())
        / torch.vdot(difference, difference)
    )
    reference = fq.Circuit(8, dtype=torch.complex128).any(*range(8), matrix=matrix)
    points = run_sweep()
    assert len(points) == 10
    np.testing.assert_allclose(
        [p.bound for p in points], np.linspace(0, 0.3, 10), atol=0, rtol=0
    )
    for point in points:
        compare_states(point.program, reference, ("out",))
        assert point.amplitude_l2_error < 1e-12
        assert point.amplitude_l2_error <= point.bound + 1e-12
        assert point.probability_l2_error < 1e-12
        resources = point.program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert resources.n_qubits == 8
        assert point.program.report["workspace_qubits"] == 0
        assert point.depth == resources.depth
        assert point.cx_count == resources.operation_counts.get("cx", 0)
    # Exact preparation must honestly report its lack of bound-dependent savings.
    assert len({(p.depth, p.cx_count) for p in points}) == 1
    figure = tmp_path / "preparation_bounds.png"
    plot_sweep(points, figure)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_source_generator_keeps_eight_qubits_seed_and_signed_mean_centered_data() -> (
    None
):
    amplitudes = np.asarray(source_amplitudes())
    assert amplitudes.shape == (256,)
    x = np.linspace(-1, 1, 256)
    trend = np.polynomial.polynomial.polyval(x, [0.6, -0.4, 0.8, -0.3])
    expected = trend + 0.05 * np.random.RandomState(1).normal(size=256)
    expected -= expected.mean()
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(amplitudes, expected, atol=1e-15, rtol=0)
    assert float(amplitudes.min()) < 0 < float(amplitudes.max())
    assert abs(float(amplitudes.sum())) < 1e-12


def test_workflow_provenance_covers_every_source_code_cell_without_changing_qmod_vector() -> (
    None
):
    from examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.main import (
        AMPLITUDES,
        build_program,
    )

    assert len(AMPLITUDES) == 64
    assert len(build_program().registers["out"]) == 6
    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    group = next(
        g
        for g in inventory["groups"]
        if g["id"]
        == "tutorials/technology_demonstrations/approximated_state_preparation/approximated_state_preparation"
    )
    assert group["workflow_status"] == "validated"
    assert [
        c["index"] for c in group["workflow_validation"]["reviewed_code_cells"]
    ] == [2, 3, 4, 5, 6]
