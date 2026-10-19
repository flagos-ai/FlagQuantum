"""Check the cloud lessons' numerical predictions without submitting jobs."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit
WORKSHOP = Path(__file__).resolve().parents[2] / "workshops/flagos2026"


@pytest.mark.parametrize("edition", ("notebooks", "notebooks_cn"))
@pytest.mark.parametrize(
    "filename, expected",
    (
        ("05_jiuding_jobs.ipynb", [0, 0, 0, 0, 1, 0, 0, 0]),
        ("06_quafu_hardware.ipynb", [0.75, 0, 0, 0, 0.25, 0, 0, 0]),
    ),
)
def test_cloud_circuit_matches_its_asymmetric_prediction(
    edition: str, filename: str, expected: list[float]
) -> None:
    notebook = json.loads((WORKSHOP / edition / filename).read_text())
    source = "".join(notebook["cells"][4]["source"])
    tree = ast.parse(source.replace("%matplotlib inline", ""))
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "circuit"
            for target in node.targets
        )
    ]
    assert len(builders) == 1
    scope = {"fq": fq, "math": math}
    exec(compile(ast.Module(body=builders, type_ignores=[]), filename, "exec"), scope)
    probabilities = fq.run(scope["circuit"]).to_statevector().reshape(-1).abs().square()
    torch.testing.assert_close(probabilities, probabilities.new_tensor(expected))


@pytest.mark.parametrize("edition", ("notebooks", "notebooks_cn"))
def test_optional_batch_predicts_distinct_ordered_bit_strings(edition: str) -> None:
    notebook = json.loads((WORKSHOP / edition / "06_quafu_hardware.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "batch_circuits =" in "".join(cell["source"])
    )
    assignments = [
        node for node in ast.parse(source).body if isinstance(node, ast.Assign)
    ]
    scope = {"fq": fq}
    exec(
        compile(
            ast.Module(body=assignments, type_ignores=[]), "batch-circuits", "exec"
        ),
        scope,
    )
    assert scope["batch_expected"] == ["100", "010", "001"]
    for circuit, expected in zip(
        scope["batch_circuits"], scope["batch_expected"], strict=True
    ):
        assert fq.run(circuit, shots=32, outputs=fq.counts()).counts == [{expected: 32}]


def test_offline_notebook_validator_recognizes_batch_submission_switch() -> None:
    tree = ast.parse((WORKSHOP / "scripts/verify_notebooks.py").read_text())
    switches = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REMOTE_SWITCHES"
            for target in node.targets
        )
    )
    assert "submit_batch" in switches
