from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "build_campaign_manifest.py"
SPEC = importlib.util.spec_from_file_location("sc27_campaign", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def references() -> dict[str, dict[str, str]]:
    return {
        f"n{n_wires}-seed{seed}": {
            "path": f"references/n{n_wires}-seed{seed}.json",
            "sha256": f"{n_wires - 28:x}" * 64,
        }
        for n_wires in range(28, 33)
        for seed in (41, 42, 43)
    }


def reference_catalog() -> dict:
    return {
        "schema": "flagquantum.sc27.correctness_reference_catalog.v1",
        "provider": "PennyLane Lightning-GPU",
        "gradient_method": "adjoint",
        "source_commit": "a" * 40,
        "container_digest": "sha256:" + "d" * 64,
        "references": references(),
    }


def manifest() -> dict:
    return MODULE.build_manifest(
        source_commit="a" * 40,
        core_container_digest="sha256:" + "b" * 64,
        pennylane_container_digest="sha256:" + "d" * 64,
        tqd_container_digest="sha256:" + "e" * 64,
        custatevec_container_digest="sha256:" + "f" * 64,
        cudaq_container_digest="sha256:" + "0" * 64,
        correctness_reference_catalog=reference_catalog(),
        output_root="/shared/results/sc27",
    )


def test_compiles_all_execution_ready_runs_without_duplicates() -> None:
    result = manifest()
    assert result["ready_run_count"] == 99
    ids = [run["id"] for run in result["runs"]]
    assert len(ids) == len(set(ids))
    assert {run["campaign"] for run in result["runs"]} == {
        "statevector_strong_scaling",
        "statevector_weak_scaling",
        "statevector_ablations",
        "statevector_capacity",
        "external_pennylane",
        "external_tqd",
        "external_custatevec_forward",
        "external_cudaq_capability",
        "heisenberg_pilot",
    }
    containers = result["containers"]
    assert {
        run["container_digest"]
        for run in result["runs"]
        if run["campaign"] == "external_pennylane"
    } == {containers["pennylane"]}
    assert {
        run["container_digest"]
        for run in result["runs"]
        if run["campaign"] == "external_tqd"
    } == {containers["tqd"]}
    by_id = {run["id"]: run for run in result["runs"]}
    weak_28 = by_id["statevector-weak-w1-run1"]
    weak_32 = by_id["statevector-weak-w16-run1"]
    assert weak_28["command"][weak_28["command"].index("--reference-json") + 1].endswith(
        "n28-seed41.json"
    )
    assert weak_32["command"][weak_32["command"].index("--reference-json") + 1].endswith(
        "n32-seed41.json"
    )


def test_multinode_and_ablation_commands_are_explicit() -> None:
    result = manifest()
    by_id = {run["id"]: run for run in result["runs"]}
    strong16 = by_id["statevector-strong-w16-run1"]
    assert strong16["resources"] == {
        "nodes": 2,
        "gpus_per_node": 8,
        "launch_mode": "multi_node_per_node",
    }
    assert "--node-rank={node_rank}" in strong16["command"]
    overlap = by_id["ablation-communication-overlap-off-run1"]["command"]
    assert overlap[overlap.index("--ablation-id") + 1] == "communication_overlap_off"
    assert "--disable-gradient-overlap" in overlap
    assert "full" not in overlap


def test_primary_science_remains_unschedulable_until_pilot_and_dmrg() -> None:
    result = manifest()
    assert result["blocked_campaigns"] == [
        {
            "campaign": "heisenberg_primary",
            "required_run_count": 15,
            "reason": (
                "requires sealed pilot selection, protocol freeze update, "
                "and sealed independent 256-site DMRG reference"
            ),
        }
    ]


def test_rejects_unpinned_source_or_container() -> None:
    with pytest.raises(ValueError, match="source commit"):
        MODULE.build_manifest(
            source_commit="main",
            core_container_digest="sha256:" + "b" * 64,
            pennylane_container_digest="sha256:" + "d" * 64,
            tqd_container_digest="sha256:" + "e" * 64,
            custatevec_container_digest="sha256:" + "f" * 64,
            cudaq_container_digest="sha256:" + "0" * 64,
            correctness_reference_catalog=reference_catalog(),
            output_root="/shared/results",
        )
