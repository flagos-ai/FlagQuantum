from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "experimental-surface-v2-candidate.json"


def test_experimental_root_contains_only_domain_namespaces() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert list(fq.experimental.__all__) == contract["top_level_namespaces"]
    assert set(dir(fq.experimental)) >= set(contract["top_level_namespaces"])
    assert contract["rules"]["top_level_governance_changed"] is False


def test_each_experimental_namespace_matches_machine_contract() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    for namespace, expected in contract["discoverable_exports"].items():
        module = getattr(fq.experimental, namespace)
        assert list(module.__all__) == expected
        for name in expected:
            assert getattr(module, name) is not None


def test_internal_records_and_evidence_are_not_discoverable() -> None:
    hidden = {
        "distributed": {
            "DistributedEvidenceContract",
            "JAXStatevectorShardState",
            "execute_torch_distributed_statevector",
            "mps_site_kernel_stats",
        },
        "dynamic": {
            "DynamicConformanceCase",
            "DynamicExecutionResult",
            "run_dynamic_conformance",
        },
        "mps": {"MPSProductionPlan", "build_mps_release_artifact"},
        "numerics": {
            "SplitRealImagConformanceReport",
            "run_split_real_imag_conformance",
        },
        "simulation": {"TEBDResult"},
    }

    for namespace, names in hidden.items():
        module = getattr(fq.experimental, namespace)
        assert names.isdisjoint(module.__all__)
        assert names.isdisjoint(dir(module))
        for name in names:
            with pytest.raises(AttributeError):
                getattr(module, name)


def test_every_removed_v1_route_fails_closed() -> None:
    current = json.loads(CONTRACT.read_text(encoding="utf-8"))
    previous = json.loads(
        (ROOT / "contracts" / "experimental-namespace-v1-candidate.json").read_text(
            encoding="utf-8"
        )
    )

    for namespace, previous_names in previous["exports"].items():
        removed = set(previous_names) - set(current["discoverable_exports"][namespace])
        module = getattr(fq.experimental, namespace)
        for name in removed:
            assert name not in dir(module)
            with pytest.raises(AttributeError):
                getattr(module, name)


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
