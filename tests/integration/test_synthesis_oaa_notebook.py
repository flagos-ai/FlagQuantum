"""Accept both OAA experiments and the actual good-block probability check."""

import json
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.physics import (
    AMPLITUDES,
)
from examples.synthesis.compare_states import compare_states
from tests.integration.synthesis_notebook_support import execute_code_cells

try:
    import numpy as np

    from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.overview import (
        main as oaa,
    )
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip(
        "This complete Notebook workflow requires NumPy.", allow_module_level=True
    )

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = (
    ROOT
    / "examples/synthesis/algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/overview"
)


@pytest.fixture(scope="module")
def workflow() -> tuple[oaa.Stage, oaa.Stage]:
    return oaa.run_workflow()


def test_source_random_input_and_normalization() -> None:
    amplitudes = oaa.source_amplitudes()
    assert amplitudes == pytest.approx(AMPLITUDES, abs=1e-15, rel=0)
    assert sum(a * a for a in amplitudes) == pytest.approx(1, abs=1e-15)
    assert all(a > 0 for a in amplitudes)
    assert oaa.source_amplitudes() == amplitudes


@pytest.mark.parametrize("repetitions", [0, 1])
def test_both_experiments_absolute_state_projection_order_and_clean_work(
    workflow: tuple[oaa.Stage, oaa.Stage], repetitions: int
) -> None:
    amplitudes = oaa.source_amplitudes()
    stage = workflow[repetitions]
    compare_states(
        stage.program, reference_circuit(repetitions, amplitudes), ("data", "block")
    )
    assert stage.program.report["workspace_qubits"] == 0
    assert tuple(len(w) for w in stage.program.registers.values()) == (2, 2)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / 2**0.5
    expected = torch.kron(hadamard, hadamard) @ torch.tensor(
        amplitudes, dtype=torch.complex128
    )
    if repetitions == 0:
        expected = expected / 2
    actual = fq.run(stage.program).statevector().reshape(4, 4)[:, 0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert dict(stage.good_rows()) == pytest.approx(
        {index: float(value.abs().square()) for index, value in enumerate(expected)},
        abs=1e-12,
        rel=0,
    )


def test_success_check_distinguishes_good_block_from_total_normalization(
    workflow: tuple[oaa.Stage, oaa.Stage],
) -> None:
    before, after = workflow
    for stage in workflow:
        assert sum(p for _, p in stage.probabilities) == pytest.approx(1, abs=1e-12)
        assert stage.good_probability == sum(p for _, p in stage.good_rows())
    assert before.good_probability == pytest.approx(0.25, abs=1e-12)
    assert after.good_probability == pytest.approx(1, abs=1e-12)
    assert before.good_probability < 0.5


@pytest.mark.parametrize("values", [(0.3, -0.4, 0.5, -0.7), (-0.2, 0.6, 0.4, 0.5)])
@pytest.mark.parametrize("repetitions", [0, 1])
def test_good_block_readout_with_nonzero_signed_data(
    values: tuple[float, ...], repetitions: int
) -> None:
    norm = sum(value * value for value in values) ** 0.5
    amplitudes = tuple(value / norm for value in values)
    stage = oaa.run_stage(repetitions, amplitudes)
    compare_states(
        stage.program, reference_circuit(repetitions, amplitudes), ("data", "block")
    )
    assert stage.good_probability == pytest.approx(
        0.25 if repetitions == 0 else 1, abs=1e-12
    )


def test_actual_notebook_both_experiments_and_source_provenance() -> None:
    path = DIRECTORY / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        assert "from classiq" not in text and "source_format" not in text
        assert "from workflow import" not in text
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, cells = execute_code_cells(path)
    assert cells == 5
    amplitudes = cast(tuple[float, ...], namespace["amplitudes"])
    before = cast(oaa.Stage, namespace["before"])
    after = cast(oaa.Stage, namespace["after"])
    for repetitions, stage in enumerate((before, after)):
        compare_states(
            stage.program,
            reference_circuit(repetitions, amplitudes),
            ("data", "block"),
        )
    assert before.good_probability == pytest.approx(0.25, abs=1e-12)
    assert after.good_probability == pytest.approx(1, abs=1e-12)
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [c["index"] for c in proof["reviewed_code_cells"]] == [
        16,
        18,
        19,
        22,
        24,
        25,
    ]
    assert np.isclose(sum(a * a for a in amplitudes), 1, rtol=0, atol=1e-12)
