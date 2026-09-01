from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "experimental-namespace-v1-candidate.json"


def test_experimental_root_contains_only_domain_namespaces() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert list(fq.experimental.__all__) == contract["top_level_namespaces"]
    assert set(dir(fq.experimental)) >= set(contract["top_level_namespaces"])
    assert contract["rules"]["flat_feature_exports_at_experimental_root"] == 0


def test_each_experimental_namespace_matches_machine_contract() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    for namespace, expected in contract["exports"].items():
        module = getattr(fq.experimental, namespace)
        assert list(module.__all__) == expected
        for name in expected:
            assert getattr(module, name) is not None


def test_flat_experimental_feature_routes_are_absent() -> None:
    for name in (
        "DynamicCircuit",
        "run_tebd",
        "from_qiskit",
        "JAXStatevectorShardState",
    ):
        assert name not in fq.experimental.__all__
        assert name not in dir(fq.experimental)
        try:
            getattr(fq.experimental, name)
        except AttributeError as exc:
            assert "domain namespace" in str(exc)
        else:  # pragma: no cover - fail-closed diagnostic
            raise AssertionError(f"flat experimental route unexpectedly exists: {name}")


def test_experimental_reorganization_does_not_change_stable_root() -> None:
    stable = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )

    assert sorted(fq.__all__) == stable["stable_exports"]
    assert CONTRACT.name not in fq.__all__
