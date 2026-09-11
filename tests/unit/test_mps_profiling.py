import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.internal.evidence.mps_critical_path_summary import aggregate
from flagquantum.runtime.executors.mps.profiling import (
    MPS_TRACE_COMMUNICATION_LABELS,
    MPS_TRACE_OPERATION_LABELS,
    build_mps_critical_path_report,
    workload_fingerprint,
)
from flagquantum.simulation.mps.factorization import _split_pair_matrix
from flagquantum.simulation.mps.models import MPSConfig


def _rank(rank, forward, reverse, optimizer, total):
    return {
        "rank": rank,
        "step_metrics": (
            {
                "step": 0,
                "forward_seconds": forward,
                "reverse_seconds": reverse,
                "optimizer_seconds": optimizer,
                "end_to_end_seconds": total,
                "boundary_forward_exchanges": 2,
                "boundary_reverse_exchanges": 2,
                "boundary_bytes": 128,
                "gradient_collective_count": 3,
                "gradient_collective_bytes": 24,
                "svd_activity": 1,
                "qr_activity": 0,
            },
        ),
    }


def test_report_reconciles_phases_and_classifies_messages():
    manifest = {"family": "latency", "sites": 8, "steps": 21}
    report = build_mps_critical_path_report(
        (_rank(0, 1.0, 2.0, 0.5, 3.6), _rank(1, 1.1, 2.0, 0.5, 3.8)),
        workload_manifest=manifest,
        environment_manifest={"backend": "gloo", "source_commit": "abc"},
        warmup_steps=0,
    )
    assert report["phase_reconciliation_passed"]
    assert report["workload_sha256"] == workload_fingerprint(manifest)
    assert report["message_totals_by_payload_class"]["boundary_forward_tensor"] == {
        "message_count": 8,
        "payload_bytes": 128,
    }
    assert (
        report["message_totals_by_payload_class"]["parameter_gradient_collective"][
            "message_count"
        ]
        == 6
    )
    assert report["rank_imbalance_by_step"]["0"][
        "idle_imbalance_seconds"
    ] == pytest.approx(0.2)
    assert not report["performance_claim_allowed"]
    assert (
        report["trace_contract"]["required_operation_labels"]
        == MPS_TRACE_OPERATION_LABELS
    )
    assert (
        report["trace_contract"]["required_communication_labels"]
        == MPS_TRACE_COMMUNICATION_LABELS
    )
    assert report["cold_setup_seconds_by_rank"]["0"] == {
        "process_group_seconds": 0.0,
        "communication_seconds": 0.0,
    }


def test_report_rejects_missing_rank_and_bad_inputs():
    with pytest.raises(ValueError, match="contiguous"):
        build_mps_critical_path_report(
            (_rank(1, 1, 1, 1, 3),),
            workload_manifest={},
            environment_manifest={},
            warmup_steps=0,
        )
    with pytest.raises(ValueError, match="warmup_steps"):
        build_mps_critical_path_report(
            (_rank(0, 1, 1, 1, 3),),
            workload_manifest={},
            environment_manifest={},
            warmup_steps=-1,
        )


def test_factorization_trace_labels_qr_and_svd():
    matrix = torch.randn(1, 4, 4, dtype=torch.complex64)
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        _split_pair_matrix(
            matrix, left_dim=2, right_dim=2, config=MPSConfig(max_bond=4)
        )
        _split_pair_matrix(
            matrix, left_dim=2, right_dim=2, config=MPSConfig(max_bond=2)
        )
    names = {event.key for event in profiler.key_averages()}
    assert "flagquantum::mps::qr" in names
    assert "flagquantum::mps::svd" in names


def _write_trace(path, labels):
    events = [
        {"ph": "X", "cat": "user_annotation", "name": label, "dur": 1000.0}
        for label in labels
    ]
    events.append({"ph": "X", "cat": "kernel", "name": "nccl:all_reduce", "dur": 1.0})
    path.write_text(__import__("json").dumps({"traceEvents": events}, indent=2))


def _aggregate_fixture(tmp_path):
    manifest = {
        "schema": "flagquantum.issue097.test_manifest.v1",
        "dtype": "complex64",
        "optimizer": "adam",
        "warmup_steps": 1,
        "retained_warm_steps": 2,
        "world_sizes": [1],
        "cases": {
            "small_latency": {"sites_per_rank": 2, "initial_bond": 1, "max_bond": 4}
        },
        "required_phase_labels": [
            label.rsplit("::", 1)[-1] for label in MPS_TRACE_OPERATION_LABELS
        ],
        "required_communication_labels": ["gradient_collective"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))
    trace_path = tmp_path / "rank-0.trace.json"
    _write_trace(
        trace_path,
        MPS_TRACE_OPERATION_LABELS + ("flagquantum::mps::gradient_collective",),
    )
    steps = [
        {
            "rank": 0,
            "step": step,
            "sample_class": "warm" if step >= 1 else "cold_or_warmup",
            "end_to_end_seconds": 1.0,
            "phase_seconds": {
                "forward": 0.6,
                "reverse_vjp": 0.3,
                "optimizer": 0.1,
                "unattributed": 0.0,
            },
        }
        for step in range(3)
    ]
    report = {
        "workload_manifest": {
            "manifest_schema": manifest["schema"],
            "case": "small_latency",
            "world_size": 1,
            "n_wires": 2,
            **manifest["cases"]["small_latency"],
            "steps": 3,
            "dtype": "complex64",
            "optimizer": "adam",
        },
        "world_size": 1,
        "warmup_steps": 1,
        "warm_sample_count_across_ranks": 2,
        "phase_reconciliation_passed": True,
        "maximum_reconciliation_relative_error": 0.0,
        "rank_step_samples": steps,
        "message_totals_by_payload_class": {},
        "environment_manifest": {
            "backend": "nccl",
            "device": "test",
            "world_size": 1,
            "local_world_size": 1,
            "node_count": 1,
            "rank_placement": [
                {"rank": 0, "local_rank": 0, "hostname": "test", "device": "cuda:0"}
            ],
            "critical_file_sha256": {"runtime": "abc"},
        },
        "trace_paths": [str(trace_path)],
        "cold_setup_seconds_by_rank": {
            "0": {"process_group_seconds": 0.1, "communication_seconds": 0.2}
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(__import__("json").dumps(report))
    return manifest_path, report_path, report


def test_aggregate_validates_trace_and_frozen_contract(tmp_path):
    manifest_path, report_path, _ = _aggregate_fixture(tmp_path)
    result = aggregate([report_path], manifest_path=manifest_path)
    assert result["baseline_gate_passed"]
    point = result["bottleneck_points"][0]
    assert (
        point["trace_operation_totals"]["flagquantum::mps::objective_adjoint"]["count"]
        == 1
    )


def test_aggregate_fails_closed_for_mismatched_or_missing_evidence(tmp_path):
    manifest_path, report_path, report = _aggregate_fixture(tmp_path)
    report["workload_manifest"]["max_bond"] = 999
    report["environment_manifest"].pop("rank_placement")
    report["trace_paths"] = [str(tmp_path / "missing.trace.json")]
    report_path.write_text(__import__("json").dumps(report))
    result = aggregate([report_path], manifest_path=manifest_path)
    assert not result["baseline_gate_passed"]
    assert {
        "workload_manifest_mismatch",
        "topology_environment_manifest_invalid",
        "rank_trace_missing",
    } <= set(result["blockers"])
