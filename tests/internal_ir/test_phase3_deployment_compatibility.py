from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.executable_artifact import SealedExecutableArtifact
from flagquantum._compiler.runtime_abi import (
    ExecutionBinding,
    RuntimeResult,
    SubmissionReceipt,
)
from flagquantum.deployment.cloud import DeploymentPackage
from flagquantum.deployment.routing_evidence import DEPLOYMENT_PACKAGE_SCHEMA
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "tests/fixtures/internal_ir/phase3_deployment_compatibility.json"


def _profile() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_inventory_matches_current_legacy_and_phase3_boundaries() -> None:
    profile = _profile()
    legacy = profile["legacy_boundary"]
    phase3 = profile["phase3_boundary"]

    assert legacy["fields"] == [item.name for item in fields(DeploymentPackage)]
    assert legacy["schema"] == DEPLOYMENT_PACKAGE_SCHEMA
    assert phase3["artifact"].endswith(SealedExecutableArtifact.__name__)
    assert phase3["binding"].endswith(ExecutionBinding.__name__)
    assert phase3["receipt"].endswith(SubmissionReceipt.__name__)
    assert phase3["result"].endswith(RuntimeResult.__name__)


def test_mapping_separates_compilation_artifact_execution_and_display_ownership() -> (
    None
):
    mappings = {item["legacy"]: item for item in _profile()["field_mapping"]}

    assert set(mappings) == {
        "qasm or metadata.qcis",
        "ir",
        "shots",
        "backend provider and name",
        "backend semantic capabilities",
        "metadata.routing_evidence",
        "name",
        "metadata provider extensions",
    }
    assert mappings["ir"]["mapping_class"] == "requires_verified_recompile"
    assert mappings["shots"]["mapping_class"] == "execution_only"
    assert mappings["backend provider and name"]["mapping_class"] == (
        "execution_locator_only"
    )
    assert mappings["name"]["mapping_class"] == "nonsemantic_label"
    assert "never reuse" in mappings["qasm or metadata.qcis"]["identity_rule"]
    assert "never artifact" in mappings["shots"]["identity_rule"]


def test_compatibility_matrix_preserves_every_current_default() -> None:
    matrix = {item["case"]: item for item in _profile()["compatibility_matrix"]}

    legacy = matrix["existing DeploymentPackage through existing provider path"]
    assert legacy["current_behavior"] == "supported and authoritative"
    assert legacy["batch_g_effect"] == "none"
    assert all(
        item["batch_g_effect"]
        in {
            "none",
            "proposal only",
            "remains unsupported",
            "remains inactive",
            "remains unauthorized",
        }
        for item in matrix.values()
    )


def test_batch_g_has_zero_public_runtime_provider_and_default_path_impact() -> None:
    assert _profile()["status"] == "reference_mapping"
    assert public_api_snapshot.validate() == ()
    for name in (
        "SealedExecutableArtifact",
        "ExecutionBinding",
        "SubmissionReceipt",
        "RuntimeResult",
    ):
        assert not hasattr(fq, name)
