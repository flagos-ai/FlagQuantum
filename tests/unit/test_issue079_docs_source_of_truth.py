"""Contracts for product documentation sources of truth."""

import importlib.util
from pathlib import Path

import pytest

import flagquantum as fq

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "docs_source_of_truth", ROOT / "tools/docs_source_of_truth.py"
)
assert SPEC and SPEC.loader
DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS)


def test_generated_documentation_and_claims_are_current():
    assert DOCS.validate() == []


def test_stable_api_includes_module_contract():
    api = DOCS.load(ROOT / "docs/public_api_v1.json")
    assert {"Module", "QuantumModule", "ExecutionResult", "RuntimePolicy"} <= set(
        api["stable_exports"]
    )
    assert issubclass(fq.Module, __import__("torch").nn.Module)
    assert fq.Module is fq.QuantumModule
    for name in api["stable_exports"]:
        assert hasattr(fq, name)


def test_json_loader_rejects_duplicate_authority_keys(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"states": {}, "states": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key 'states'"):
        DOCS.load(duplicate)


def test_every_stable_export_has_an_existing_contract_mapping():
    api = DOCS.load(ROOT / "docs/public_api_v1.json")
    assert set(api["verification"]) == set(api["stable_exports"])
    for target in api["verification"].values():
        assert DOCS.test_node_exists(target)


def test_contract_mapping_rejects_file_only_and_missing_nodes():
    assert not DOCS.test_node_exists("tests/test_mps.py")
    assert not DOCS.test_node_exists("tests/test_mps.py::test_does_not_exist")
    assert DOCS.test_node_exists(
        "tests/test_mps.py::test_mps_bell_state_matches_statevector"
    )


def test_large_benchmark_authority_uses_its_dedicated_audit() -> None:
    config = DOCS.load(ROOT / "docs/source_of_truth.json")
    assert config["externally_validated_authorities"]["benchmark_evidence"] == (
        "python benchmarks/audit_results.py --input benchmarks/results"
    )


def test_stable_document_example_executes():
    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result = fq.run_native(circuit)
    assert result is not None
