from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_ablation_matrix.py"
SPEC = importlib.util.spec_from_file_location("sc27_ablation_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def full_payload() -> dict:
    return {
        "world_size": 8,
        "source_identity": {
            "commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
            "source_dirty": False,
        },
        "protocol": {"independent_run_index": 1, "warmup": 5, "repetitions": 20},
        "workload": {
            "n_wires": 31,
            "name": "full_width_linear_hea",
            "layers": 8,
            "gate_count": 488,
            "parameter_count": 248,
            "observable": "Z(15)",
            "dtype": "complex64",
            "seed": 41,
        },
        "training_step": {"samples_seconds": [1.0] * 20},
        "correctness": {"passed": True},
        "fallback_events": [],
        "ownership": {"optimizer_state": "owner_sharded_across_ranks"},
        "saved_forward_state_reused": True,
        "ablation": {
            "id": "full",
            "persistent_layout_enabled_observed": True,
            "gradient_collective_count_max": 8,
            "async_gradient_collective_count_max": 8,
            "overlapped_gradient_collective_count_max": 7,
            "optimizer_state_bytes_by_rank": [100] * 8,
            "checkpoint_strategy_observed": "reversible_adjoint",
            "profile_communication_seconds_max": 0.1,
            "profile_overlap_fraction_max": 0.5,
            "profile_gradient_all_reduce_overlap_fraction_max": 0.5,
        },
    }


def valid_matrix() -> list[dict]:
    full = full_payload()
    payloads = [full]
    for identifier in MODULE.IDS[1:]:
        item = copy.deepcopy(full)
        item["ablation"]["id"] = identifier
        if identifier == "persistent_layout_off":
            item["ablation"]["persistent_layout_enabled_observed"] = False
        elif identifier == "communication_overlap_off":
            item["ablation"]["async_gradient_collective_count_max"] = 0
            item["ablation"]["overlapped_gradient_collective_count_max"] = 0
            item["ablation"]["profile_overlap_fraction_max"] = 0.0
            item["ablation"][
                "profile_gradient_all_reduce_overlap_fraction_max"
            ] = 0.0
        elif identifier == "gradient_bucketing_off":
            item["ablation"]["gradient_collective_count_max"] = 248
        elif identifier == "unique_optimizer_ownership_off":
            item["ownership"]["optimizer_state"] = "replicated_across_ranks"
            item["ablation"]["optimizer_state_bytes_by_rank"] = [800] * 8
        elif identifier == "reverse_rematerialization":
            item["ablation"]["checkpoint_strategy_observed"] = "full_rematerialization"
            item["ablation"]["persistent_layout_enabled_observed"] = False
            item["saved_forward_state_reused"] = False
        payloads.append(item)
    return payloads


def test_accepts_complete_single_factor_matrix() -> None:
    result = MODULE.audit(valid_matrix())
    assert result["paper_ready_ablation_matrix"] is True
    assert result["reverse_rematerialization_is_structurally_coupled"] is True


def test_rejects_label_only_overlap_ablation() -> None:
    payloads = valid_matrix()
    overlap = next(
        item for item in payloads if item["ablation"]["id"] == "communication_overlap_off"
    )
    overlap["ablation"]["async_gradient_collective_count_max"] = 8
    result = MODULE.audit(payloads)
    assert result["paper_ready_ablation_matrix"] is False
    assert "mechanism:communication_overlap_off:async_collectives_present" in result[
        "blockers"
    ]
