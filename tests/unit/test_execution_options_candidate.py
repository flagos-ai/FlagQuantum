from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from flagquantum.runtime.options import ExecutionOptions

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "execution-options-v1-candidate.json"
STABLE_CORE = ROOT / "contracts" / "public-api-v1-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_execution_options_candidate_is_authorized_but_not_frozen() -> None:
    candidate = _load(CANDIDATE)

    assert candidate["status"] == "approved_for_implementation"
    assert candidate["root_addition"] == "ExecutionOptions"
    assert candidate["implementation_authorized"] is True
    assert candidate["rules"]["candidate_is_frozen_contract"] is False


def test_execution_options_field_contract_is_exact_and_overlay_safe() -> None:
    candidate = _load(CANDIDATE)
    fields = candidate["contract"]["fields"]
    names = [field["name"] for field in fields]

    assert names == [
        "mode",
        "backend",
        "device",
        "target",
        "batch_size",
        "precision",
        "shots",
        "seed",
        "memory_limit_bytes",
        "require_gradients",
        "allow_approximate",
        "allow_backend_fallback",
    ]
    assert len(names) == len(set(names))
    assert all(field["default"] is None for field in fields)
    assert candidate["contract"]["frozen"] is True
    assert candidate["contract"]["slots"] is True


def test_execution_options_implementation_matches_candidate_fields() -> None:
    candidate = _load(CANDIDATE)
    contracted = candidate["contract"]["fields"]

    assert [field.name for field in fields(ExecutionOptions)] == [
        field["name"] for field in contracted
    ]
    assert ExecutionOptions().to_dict() == {
        "schema": candidate["contract"]["serialization_schema"],
        "version": candidate["contract"]["serialization_version"],
        **{field["name"]: field["default"] for field in contracted},
    }


def test_execution_mode_is_representation_not_distribution_topology() -> None:
    candidate = _load(CANDIDATE)
    modes = set(candidate["contract"]["allowed_values"]["mode"])

    assert modes == {"auto", "statevector", "mps", "tensor_network", "density_matrix"}
    assert not any("distributed" in mode for mode in modes)
    assert "tn" not in modes


def test_execution_options_defaults_fail_closed() -> None:
    defaults = _load(CANDIDATE)["framework_defaults"]

    assert defaults["allow_approximate"] is False
    assert defaults["allow_backend_fallback"] is False
    assert defaults["require_gradients"] is False


def test_execution_options_excludes_backend_and_orchestration_escape_hatches() -> None:
    candidate = _load(CANDIDATE)
    field_names = {field["name"] for field in candidate["contract"]["fields"]}
    excluded = set(candidate["excluded_fields"])

    assert {"world_size", "rank", "max_bond", "cutoff", "extras"} <= excluded
    assert field_names.isdisjoint(excluded)
    assert candidate["rules"]["unknown_fields_fail"] is True


def test_execution_options_is_registered_but_not_in_current_manifest() -> None:
    stable_core = _load(STABLE_CORE)
    current = _load(ROOT / "docs" / "public_api_v1.json")

    assert "ExecutionOptions" in stable_core["stable_core"]["planned_additions"]
    assert "ExecutionOptions" not in current["stable_exports"]
