"""Audit retained precision cases against committed source and raw samples."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]


def audit() -> dict[str, Any]:
    campaigns = []
    for name in ("cpu", "cuda", "mps"):
        directory = EVIDENCE / name
        if not directory.exists():
            continue
        freeze = json.loads((directory / "freeze.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        source = freeze["source"]
        commit = source["code_version"]
        assert source["dirty"] is False
        for path, expected in source["producer_sha256"].items():
            data = subprocess.check_output(
                ["git", "show", f"{commit}:{path}"], cwd=ROOT, timeout=15
            )
            assert hashlib.sha256(data).hexdigest() == expected, (name, path)
        data = subprocess.check_output(
            [
                "git",
                "show",
                f"{commit}:benchmarks/manifests/precision_frontier_v1.json",
            ],
            cwd=ROOT,
            timeout=15,
        )
        assert hashlib.sha256(data).hexdigest() == source["protocol_sha256"]
        assert json.loads(data) == freeze["protocol"]
        assert [row["case"] for row in summary["cases"]] == freeze["case_names"]
        assert summary["source"] == source
        counts: dict[str, int] = {}
        for row in summary["cases"]:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            result_path = directory / row["result"]
            if not result_path.exists():
                assert row["status"] in ("failed", "timeout")
                continue
            result = json.loads(result_path.read_text())
            assert (
                result["source"] == source and result["protocol"] == freeze["protocol"]
            )
            assert result["status"] == row["status"]
            if row["status"] != "completed":
                assert result["error"]
                continue
            timing, memory = result["timing"], result["memory"]
            assert len(timing["samples_seconds"]) == freeze["repeats"]
            assert timing["statistic"] == "median"
            assert timing["median_seconds"] == statistics.median(
                timing["samples_seconds"]
            )
            assert all(
                math.isfinite(value) and value > 0
                for value in timing["samples_seconds"]
            )
            hashes = result["output_word_hashes_sha256"]
            assert len(hashes) == freeze["repeats"] and all(
                len(value) == 64 for value in hashes
            )
            assert result["repeated_output_bit_identity"] == (len(set(hashes)) == 1)
            assert result["execution"]["output_device_and_dtype_verified"] is True
            assert len(result["output_state_real"]) == 1 << result["case"]["qubits"]
            assert len(result["output_state_imag"]) == len(result["output_state_real"])
            assert len(result["gradient"]) == 1
            assert memory["frontier_peak_memory_bytes"] == (
                memory["cuda_peak_allocated_bytes"]
                if name == "cuda"
                else (
                    memory["process_peak_rss_bytes_at_timing_end"]
                    if name == "cpu"
                    else None
                )
            )
            assert memory["state_payload_bytes"] == (1 << result["case"]["qubits"]) * (
                16
                if result["mode"] in ("fp64", "double_single_p3", "double_single_p4")
                else 8
            )
            assert set(result["observables"]) == {"Z0", "X1", "Z0Z1", "objective"}
            assert result["synthesis"]["coefficient_conversion_max_error"] == 0
        campaigns.append(
            {
                "campaign": name,
                "source_commit": commit,
                "counts": counts,
                "committed_producer_and_manifest_hashes_match": True,
                "raw_samples_identity_and_peak_memory_scopes_checked": True,
            }
        )
    discovery = json.loads((EVIDENCE / "validation/initial-discovery.json").read_text())
    assert (
        hashlib.sha256(
            (EVIDENCE / "validation" / discovery["archive"]).read_bytes()
        ).hexdigest()
        == discovery["archive_sha256"]
    )
    return {"schema": "flagquantum.precision_frontier_audit.v1", "campaigns": campaigns}


if __name__ == "__main__":
    payload = audit()
    (EVIDENCE / "validation/source-and-sample-audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
