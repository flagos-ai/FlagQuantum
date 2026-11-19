"""Check the native quantum kernels independently of the optional SVC runtime."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.qsvm.overview import quantum

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("feature", tuple(quantum.FeatureMap))
def test_complete_feature_maps_and_overlap(feature: quantum.FeatureMap) -> None:
    first, second = (0.23, -0.41), (-0.17, 0.37)
    forward = quantum.independent_circuit(feature, first)
    backward = quantum.independent_circuit(feature, second)
    ref_first = fq.run(forward).statevector().flatten()
    ref_second = fq.run(backward).statevector().flatten()
    for point, expected in ((first, ref_first), (second, ref_second)):
        actual = quantum.state(feature, point)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    reference = fq.run(forward.compose(backward.adjoint())).statevector()
    program = quantum.build(feature, first, second)
    actual_result = fq.run(program)
    torch.testing.assert_close(
        actual_result.statevector(), reference, atol=1e-12, rtol=0
    )
    probability = dict(program.probabilities(actual_result, registers=("q",)))[(0,)]
    assert probability == pytest.approx(
        float(torch.vdot(ref_second, ref_first).abs() ** 2), abs=1e-12
    )
    assert program.report["workspace_qubits"] == 0


def test_data_and_finite_kernel_counts() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.algorithms.machine_learning.qsvm.overview import (
        datasets,
        workflow,
    )

    split = datasets.synthetic()
    rng = np.random.RandomState(0)
    for group in (split.train, split.test, split.predict):
        first = np.stack(
            (rng.normal(1.23016026, 0.5, 20), rng.normal(1.72327701, 0.5, 20)), axis=-1
        )
        second = np.stack(
            (rng.normal(3.20331931, 0.5, 20), rng.normal(5.32365722, 0.5, 20)), axis=-1
        )
        expected = np.clip(np.vstack((first, second)), 0, 2 * np.pi) / (3 * np.pi) - 1
        np.testing.assert_allclose(group.points, expected, atol=1e-15, rtol=0)
        assert tuple(group.labels) == (0,) * 20 + (1,) * 20

    vectors = workflow.feature_states(quantum.FeatureMap.BLOCH, split.train)
    exact = workflow.exact_kernel(vectors, vectors)
    assert exact.shape == (40, 40)
    np.testing.assert_allclose(np.diag(exact), 1, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(exact).min() >= -1e-12
    observed, size = workflow.sample_kernel(vectors, None, np.random.default_rng(107))
    assert size == 820
    np.testing.assert_array_equal(observed, observed.T)
    np.testing.assert_array_equal(
        observed * workflow.SHOTS, np.round(observed * workflow.SHOTS)
    )
    assert not np.array_equal(observed, exact)
    assert np.max(np.abs(observed - exact)) < 0.05
    cross, count = workflow.sample_kernel(
        vectors[:10], vectors[:3], np.random.default_rng(107)
    )
    assert cross.shape == (10, 3) and count == 30


def test_workflow_source_shape_and_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="finite"):
        quantum.build(quantum.FeatureMap.PAULI, (0.2, float("nan")))
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/algorithms/machine_learning/qsvm/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 25
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
