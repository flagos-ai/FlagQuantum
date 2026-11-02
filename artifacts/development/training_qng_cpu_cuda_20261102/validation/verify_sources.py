"""Verify retained case outcomes against the committed producer and protocol bytes."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[4]
evidence = Path(__file__).resolve().parents[1]
checks = []
for directory in ("training-cpu", "qng-cpu", "qng-cuda"):
    path = evidence / directory
    if not path.exists():
        continue
    freeze = json.loads((path / "freeze.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    source = freeze["source"]
    commit = source["code_version"]
    for name, expected in source["producer_sha256"].items():
        code = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=root, timeout=15
        )
        assert hashlib.sha256(code).hexdigest() == expected, (directory, name)
    protocol = (
        "training_comparison_cpu_v1.json"
        if directory == "training-cpu"
        else f'qng_hva_ablation_{"cuda" if directory == "qng-cuda" else "cpu"}_v1.json'
    )
    code = subprocess.check_output(
        ["git", "show", f"{commit}:benchmarks/manifests/{protocol}"],
        cwd=root,
        timeout=15,
    )
    assert hashlib.sha256(code).hexdigest() == source["protocol_sha256"]
    assert json.loads(code) == freeze["protocol"]
    counts = {}
    for case in summary["cases"]:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
        result = json.loads((path / case["result"]).read_text())
        assert result["source"] == source
        assert result["status"] == case["status"]
        if case["status"] != "completed":
            assert result["status"] == "unsupported" and result["error"]
            continue
        assert result["parity"]["passed"]
        assert result["completed_steps"] == (60 if directory == "training-cpu" else 30)
        assert len(result["trace"]) == result["completed_steps"]
        assert result["process_peak_rss_bytes"] > 0
        assert result["total_measured_seconds"] >= result["training_seconds"]
        if directory == "training-cpu":
            assert (
                max(
                    result[key]
                    for key in (
                        "checkpoint_replay_max_parameter_error",
                        "checkpoint_replay_max_gradient_error",
                        "checkpoint_replay_max_optimizer_error",
                    )
                )
                <= 1e-10
            )
        elif directory == "qng-cuda":
            assert result["device_name"] and result["cuda_runtime"]
            assert result["cuda_peak_allocated_bytes"] > 0
            assert "fresh_private" in result["triton_cache_policy"]
    checks.append(
        {
            "campaign": directory,
            "source_commit": commit,
            "dirty": source["dirty"],
            "counts": counts,
            "producer_and_manifest_hashes_match_committed_source": True,
        }
    )
payload = {
    "schema": "flagquantum.training_qng_evidence_audit.v1",
    "report_renderer": {"source_commit": "bd8819e22e48c9826341ae64eaf514aa10cb5ff1"},
    "campaigns": checks,
}
(evidence / "validation/source-and-case-audit.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(json.dumps(payload, indent=2))
