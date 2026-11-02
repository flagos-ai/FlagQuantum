"""Validate the corrected #899 corpus and the provenance of every rank payload."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/development"
pytestmark = pytest.mark.integration


def read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["measuring_source_committed"] is True
    assert len(payload["source_commit"]) == 40
    subprocess.run(
        ["git", "cat-file", "-e", f"{payload['source_commit']}^{{commit}}"],
        cwd=ROOT,
        check=True,
        timeout=15,
    )
    assert payload["command"].startswith("python benchmarks/")
    assert str(ROOT) not in payload["command"]
    assert payload["scalability_claim_allowed"] is False
    return payload


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_corrected_corpus_meets_its_declared_domains_and_bounds(dtype: str) -> None:
    payload = read_payload(
        ARTIFACTS / "representation_conformance_cpu_20261102" / f"{dtype}.json"
    )
    assert payload["conformance_measurement"] == "same_training_forward_actual_svd"
    assert payload["summary"]["incomplete_cells"] == []
    assert payload["skipped_configurations"] == []
    assert len(payload["cells"]) == 48
    assert len(payload["seeds"]) == 3
    assert {cell["mps_truncated"] for cell in payload["cells"]} == {False, True}
    boundaries = 0
    for cell in payload["cells"]:
        tolerances = cell["tolerances"]
        gradients = cell["gradients"]
        assert gradients["unsupported"] == {}
        assert gradients["forward_path"] == "grad_enabled_projected_stop_subspace"
        assert cell["energies"] == pytest.approx(
            gradients["forward_energies"], rel=1e-8
        )
        assert gradients["oracle"]["relative_l2_error"] < tolerances["oracle_relative"]
        assert cell["actual_svd"]["source"] == "actual_training_forward_svd"
        assert cell["actual_svd"]["near_degenerate"] == any(
            record["near_degenerate"] for record in cell["actual_svd"]["records"]
        )
        own_map = gradients["mps_own_map"]["derivative"]
        if dtype == "complex64":
            assert own_map["status"] == "unsupported_precision"
            assert own_map["checked"] == 0
            assert own_map["max_relative_error"] is None
        elif own_map["status"] == "domain_boundary":
            boundaries += 1
            assert own_map["max_relative_error"] is None
            assert (
                own_map["step_consistent"] is False
                or own_map["near_degenerate_perturbations"]
            )
        else:
            assert own_map["status"] == "step_consistent_probe"
            assert own_map["max_relative_error"] is not None
        if not cell["mps_truncated"]:
            for mode in ("mps", "tensor_network"):
                assert (
                    gradients["per_mode"][mode]["relative_l2_error"]
                    < tolerances["untruncated_gradient_relative"]
                )
                assert (
                    1 - cell["states"][mode]["fidelity_vs_statevector"]
                    < tolerances["untruncated_state_fidelity_deficit"]
                )
                # All three observables must be checked, not just the total energy.
                reference = cell["observables"]["statevector"]
                assert set(reference) == {"z:0", "x:1", "zz:0,1"}
                absolute_bound = 1e-5 if dtype == "complex64" else 1e-10
                assert cell["observables"][mode] == pytest.approx(
                    reference, abs=absolute_bound
                )
    if dtype == "complex128":
        assert boundaries > 0


@pytest.mark.parametrize("world", [2, 3, 4])
def test_every_placement_rank_is_reproducible_and_retains_runtime_evidence(
    world: int,
) -> None:
    folder = ARTIFACTS / "mps_placement_conformance_cpu_20261102"
    ranks = [
        read_payload(folder / f"world-{world}.rank{rank}.json") for rank in range(world)
    ]
    reference = ranks[0]
    assert reference["rendezvous"] == "file_store"
    assert f"--local-world-size {world}" in reference["command"]
    for rank, payload in enumerate(ranks):
        assert payload["rank"] == rank
        assert payload["parameters"] == reference["parameters"]
        assert payload["source_commit"] == reference["source_commit"]
        assert payload["tolerances"] == reference["tolerances"]
        for index, arm in enumerate(payload["arms"]):
            assert arm["sharded_gradient"] == pytest.approx(
                reference["arms"][index]["sharded_gradient"], abs=1e-12
            )
            assert sorted(
                wire for owner in arm["site_ownership"] for wire in owner
            ) == list(range(payload["n_qubits"]))
            assert arm["runtime"]["mps_backward_execution"] == "completed"
            assert arm["runtime"]["statevector_fallback"] is False
            assert arm["runtime"]["gradient_collective_bytes"] > 0
            assert arm["runtime"]["saved_tensor_bytes"] > 0
            assert arm["runtime"]["scalability_claim_allowed"] is False
    for arm in reference["arms"]:
        assert arm["value_relative_error"] < reference["tolerances"]["value_relative"]
        assert (
            arm["gradient"]["relative_l2_error"]
            < reference["tolerances"]["gradient_relative"]
        )
        assert (
            1 - arm["gradient"]["cosine_similarity"]
            < reference["tolerances"]["gradient_cosine_deficit"]
        )
        assert arm["truncated"] == (arm["max_bond"] == 2)
