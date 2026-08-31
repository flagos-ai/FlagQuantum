import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]
ROOT = Path("benchmarks/results/local")


@pytest.mark.parametrize(
    ("world_size", "node_count", "dirty_name", "none_name"),
    (
        (
            4,
            1,
            "mps_training_32q_l2_b16_adam_4gpu_dirty_matched_20260805.json",
            "mps_training_32q_l2_b16_adam_4gpu_no_final_canonicalization_20260805.json",
        ),
        (
            8,
            1,
            "mps_training_32q_l2_b16_adam_8gpu_profile_baseline_20260805.json",
            "mps_training_32q_l2_b16_adam_8gpu_no_final_canonicalization_20260805.json",
        ),
        (
            16,
            2,
            "mps_training_32q_l2_b16_adam_16gpu_dirty_matched_20260805.json",
            "mps_training_32q_l2_b16_adam_16gpu_no_final_canonicalization_20260805.json",
        ),
    ),
)
def test_no_final_canonicalization_a800_evidence_is_complete(
    world_size, node_count, dirty_name, none_name
):
    dirty = json.loads((ROOT / dirty_name).read_text())
    none = json.loads((ROOT / none_name).read_text())
    assert dirty["world_size"] == none["world_size"] == world_size
    assert dirty["node_count"] == none["node_count"] == node_count
    # The 8-GPU baseline predates serialization of the CLI's default value.
    assert dirty["workload"].get("canonicalization_policy", "dirty") == "dirty"
    assert none["workload"]["canonicalization_policy"] == "none"
    assert dirty["workload_sha256"] != none["workload_sha256"]
    assert dirty["p50_seconds"] / none["p50_seconds"] >= 1.05
    assert (
        max(
            abs(actual - expected)
            for actual, expected in zip(none["losses"], dirty["losses"])
        )
        <= 2e-5
    )
    dirty_step = dirty["training"]["step_metrics"][-1]
    none_step = none["training"]["step_metrics"][-1]
    assert none_step["qr_factorization_count"] < dirty_step["qr_factorization_count"]
    assert none_step["reverse_python_dispatches"] < (
        dirty_step["reverse_python_dispatches"]
    )
    assert none["distribution_semantics"] == "sharded_across_ranks"
    assert none["scalability_claim_allowed"] is False
    assert none["release_gate_allowed"] is False
