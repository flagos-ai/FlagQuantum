"""Native QML programs preserve the source quantum state, distinct from training."""

import json
from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.comparisons.circuit import (
    reference_circuit as qnn_reference,
)
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.main import (
    build_program as build_qnn,
)
from examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.comparisons.circuit import (
    reference_circuit as qgan_reference,
)
from examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.main import (
    build_program as build_qgan,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["qnn", "qgan"])
@pytest.mark.parametrize("scale", [1.0, -0.63, 8.7])
def test_native_qml_matches_original_and_independent_full_complex_state(
    name: Literal["qnn", "qgan"], scale: float
) -> None:
    build, reference, weights_count, width, original_path = (
        (build_qnn, qnn_reference, 3, 2, "hybrid_qnn/hybrid_qnn_for_subset_majority")
        if name == "qnn"
        else (build_qgan, qgan_reference, 8, 4, "qgan/qgan_bars_and_strips")
    )
    data = [0.2 * scale, -0.3 * scale, 0.4 * scale, -0.5 * scale]
    weights = [0.1 * scale * (index + 1) for index in range(weights_count)]
    native = build(data, weights)
    source = synth.synthesize(
        ROOT
        / f"tests/fixtures/classiq_qmod/corpus/original/algorithms/QML/{original_path}.qmod",
        classical_inputs={"input_vec": data, "weight": weights},
    )
    for program in (native, source):
        compare_states(program, reference(data, weights), ("result",))
        assert len(program.registers["result"]) == width
        assert program.report["workspace_qubits"] == 0
    assert native.report["source_format"] == "python"


@pytest.mark.parametrize("name", ["qnn", "qgan"])
@pytest.mark.parametrize("data_size", [0, 3, 5])
def test_fixed_source_feature_shapes_are_checked(
    name: Literal["qnn", "qgan"], data_size: int
) -> None:
    build, count = (build_qnn, 3) if name == "qnn" else (build_qgan, 8)
    with pytest.raises(ValueError, match="four features"):
        build([0.2] * data_size, [0.1] * count)


@pytest.mark.parametrize("name", ["qnn", "qgan"])
@pytest.mark.parametrize("offset", [-1, 1])
def test_fixed_source_weight_shapes_are_checked(
    name: Literal["qnn", "qgan"], offset: int
) -> None:
    build, count = (build_qnn, 3) if name == "qnn" else (build_qgan, 8)
    with pytest.raises(ValueError, match="weights"):
        build([0.2] * 4, [0.1] * (count + offset))


@pytest.mark.parametrize("name", ["qnn", "qgan"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_qml_helpers_preserve_finite_plain_real_parameter_checks(
    name: Literal["qnn", "qgan"], value: float
) -> None:
    build, count = (build_qnn, 3) if name == "qnn" else (build_qgan, 8)
    with pytest.raises(ValueError, match="finite reals"):
        build([value] + [0.1] * 3, [0.1] * count)
    with pytest.raises(ValueError, match="finite reals"):
        build([0.1] * 4, [value] + [0.1] * (count - 1))


def test_quantum_programs_do_not_certify_complete_training_notebooks() -> None:
    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    for group in inventory["groups"]:
        if group["id"] in {
            "algorithms/QML/hybrid_qnn/hybrid_qnn_for_subset_majority",
            "algorithms/QML/qgan/qgan_bars_and_strips",
        }:
            assert group["native_program"]
            validation = group.get("workflow_validation")
            if validation is None:
                assert group["workflow_status"] == "pending"
                assert "/overview/" not in group["native_program"]["path"]
                continue
            assert group["workflow_status"] == "validated"
            entry = ROOT / group["native_program"]["path"]
            assert entry.parent.name == "overview" and entry.is_file()
            provenance = json.loads((entry.parent / "provenance.json").read_text())
            assert validation == provenance
            source = next(row for row in group["sources"] if row["kind"] == "notebook")
            assert provenance["upstream_path"] == source["path"]
            assert provenance["git_blob_sha1"] == source["git_blob_sha1"]
            assert provenance["reviewed_code_cells"]
            notebook = json.loads((entry.parent / "walkthrough.ipynb").read_text())
            assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
            assert (
                ROOT / provenance["native_notebook"]
                == entry.parent / "walkthrough.ipynb"
            )
            assert (ROOT / provenance["acceptance_test"]).is_file()
