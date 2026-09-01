from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.errors as fqe

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "errors-module-boundary-v1-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_errors_module_contract_is_frozen_without_whole_api_freeze() -> None:
    candidate = _load()

    assert candidate["status"] == "frozen"
    assert candidate["implementation_authorized"] is True
    assert candidate["root_manifest_change"] is False
    assert candidate["rules"]["candidate_is_frozen_contract"] is True
    assert candidate["rules"]["whole_api_freeze_implied"] is False


def test_error_namespace_matches_candidate_exactly() -> None:
    extension = _load()["stable_extension"]

    assert extension["namespace"] == "flagquantum.errors"
    assert set(fqe.__all__) == set(extension["additions"])
    assert set(extension["new_namespace_only"]) == set(extension["additions"])
    assert set(extension["additions"]).isdisjoint(dir(fq))


def test_module_signature_matches_candidate() -> None:
    signature = _load()["public_signatures"]["Module"]

    assert str(inspect.signature(fq.Module, eval_str=False)) == signature
    assert "deployment_binding" not in inspect.signature(fq.Module).parameters


def test_candidate_records_module_and_error_boundaries() -> None:
    candidate = _load()

    assert candidate["module_boundary"]["deployment_binding_parameter"] is False
    assert (
        candidate["module_boundary"]["deployment_binding_owner"]
        == "application_or_deployment_model"
    )
    assert candidate["error_contract"]["wrong_python_type_remains"] == "TypeError"
    assert (
        candidate["error_contract"]["backend_native_errors_cross_stable_boundary"]
        is False
    )
