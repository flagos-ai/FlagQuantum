"""Portable full-state and input-contract checks for the credit-card workflow."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.credit_card_fraud.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.finance.credit_card_fraud.overview import quantum

pytestmark = pytest.mark.integration


def test_full_native_overlap_and_exact_inner_product() -> None:
    for first, second in (
        ((0.17, -0.43, 0.29), (-0.31, 0.57, 0.11)),
        ((-0.71, 0.23, 0.44), (0.32, -0.19, -0.51)),
    ):
        program = quantum.overlap(first, second)
        actual = fq.run(program).statevector()
        reference = fq.run(reference_circuit(first, second)).statevector()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
        assert program.report["workspace_qubits"] == 0
        estimate = float(
            quantum.state(first).vdot(quantum.state(second)).abs().square()
        )
        assert estimate == pytest.approx(
            float(actual.flatten()[0].abs().square()), abs=1e-12
        )


def test_exact_original_partition_order_and_source_pin(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.applications.finance.credit_card_fraud.overview import model

    nominal = np.repeat(np.arange(200, dtype=np.float64)[:, None], 29, axis=1)
    fraud = np.repeat(np.arange(1000, 1045, dtype=np.float64)[:, None], 29, axis=1)
    split = model.partition((nominal, fraud))
    assert tuple(
        group.points.shape for group in (split.train, split.test, split.predict)
    ) == ((125, 29), (60, 29), (60, 29))
    for group, nominal_interval, fraud_interval in (
        (split.train, (0, 100), (1000, 1025)),
        (split.test, (100, 150), (1025, 1035)),
        (split.predict, (150, 200), (1035, 1045)),
    ):
        assert tuple(group.points[group.labels == 0, 0]) == tuple(
            range(*nominal_interval)
        )
        assert tuple(group.points[group.labels == 1, 0]) == tuple(
            range(*fraud_interval)
        )
    with pytest.raises(ValueError, match="200 nominal"):
        model.partition((nominal[:-1], fraud))
    wrong = tmp_path / "creditcard.csv"
    wrong.write_text("invalid data\n")
    with pytest.raises(ValueError, match="pinned 948-row"):
        model.prepare(wrong)


def test_unexecuted_notebook_and_nonfinite_refusal() -> None:
    with pytest.raises(ValueError, match="Three finite"):
        quantum.state((float("nan"), 1.0, 2.0))
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/finance/credit_card_fraud/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 26
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
