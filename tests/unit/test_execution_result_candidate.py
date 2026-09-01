from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "execution-result-v1-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_result_contract_is_frozen_without_freezing_the_whole_api() -> None:
    candidate = _load()

    assert candidate["status"] == "frozen"
    assert candidate["implementation_authorized"] is True
    assert candidate["root_manifest_change"] is False
    assert candidate["rules"]["candidate_is_frozen_contract"] is True
    assert candidate["rules"]["whole_api_freeze_implied"] is False


def test_measurement_candidate_forbids_implicit_composition() -> None:
    rules = _load()["measurement_rules"]

    assert rules["program_measurements_are_canonical"] is True
    assert rules["explicit_and_program_measurements_conflict"] == "error"
    assert rules["implicit_replace"] is False
    assert rules["implicit_append"] is False


def test_result_candidate_accessors_exist_without_native_delegation() -> None:
    contract = _load()["result_contract"]

    assert all(
        callable(getattr(fq.ExecutionResult, name)) for name in contract["accessors"]
    )
    assert "__getattr__" not in fq.ExecutionResult.__dict__
    assert fq.ExecutionResult().summary()["schema"] == contract["summary_schema"]
    assert fq.ExecutionResult().summary()["version"] == contract["summary_version"]
