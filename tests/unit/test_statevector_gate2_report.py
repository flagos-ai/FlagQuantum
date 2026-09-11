import json
from copy import deepcopy

from benchmarks.statevector_gate2_report import build_report


def test_gate2_report_passes_only_matched_isolated_evidence(tmp_path):
    external = {
        "correctness": {"passed": True, "max_abs_error": 1e-7},
        "stability": {"comparison_claim_allowed": True},
        "world_size": 1,
        "external_baseline_isolation": {
            "purpose": "benchmark_only",
            "provider": "NVIDIA cuStateVec",
            "flagquantum_runtime_dependency": False,
        },
        "flagquantum": {"median_seconds": 2.0},
        "custatevec": {"median_seconds": 1.0},
        "speedup_custatevec_over_flagquantum": 2.0,
    }
    common = {
        "workload": {"n_wires": 28},
        "correctness": {"reference_gradient_absolute_error_max": 1e-7},
        "backward": {"median_seconds": 2.0},
        "end_to_end": {"median_seconds": 3.0},
        "backward_communication_bytes_per_rank_max": 100,
    }
    operator_off = deepcopy(common)
    operator_off["benchmark_protocol"] = {"triton_vjp_adjoint": False}
    operator_on = deepcopy(common)
    operator_on["benchmark_protocol"] = {"triton_vjp_adjoint": True}
    operator_on["backward"]["median_seconds"] = 1.0
    operator_on["end_to_end"]["median_seconds"] = 2.0
    communication_off = deepcopy(common)
    communication_off["benchmark_protocol"] = {"cross_shard_cx_pack": False}
    communication_on = deepcopy(common)
    communication_on["benchmark_protocol"] = {"cross_shard_cx_pack": True}
    communication_on["backward_communication_bytes_per_rank_max"] = 50
    communication_on["end_to_end"]["median_seconds"] = 2.0

    paths = {}
    for name, payload in {
        "external": external,
        "operator_off": operator_off,
        "operator_on": operator_on,
        "communication_off": communication_off,
        "communication_on": communication_on,
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    report = build_report(
        external_path=paths["external"],
        operator_off_path=paths["operator_off"],
        operator_on_path=paths["operator_on"],
        communication_off_path=paths["communication_off"],
        communication_on_path=paths["communication_on"],
    )
    assert report["passed"] is True
    assert report["blockers"] == []

    external["external_baseline_isolation"]["flagquantum_runtime_dependency"] = True
    paths["external"].write_text(json.dumps(external), encoding="utf-8")
    report = build_report(
        external_path=paths["external"],
        operator_off_path=paths["operator_off"],
        operator_on_path=paths["operator_on"],
        communication_off_path=paths["communication_off"],
        communication_on_path=paths["communication_on"],
    )
    assert report["passed"] is False
    assert "external_comparison_failed" in report["blockers"]
