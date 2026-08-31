from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_campaign.py"
SPEC = importlib.util.spec_from_file_location("sc27_campaign_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_reports_missing_campaign_artifacts(tmp_path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    manifest = {
        "source_commit": "a" * 40,
        "containers": {"cudaq": "sha256:" + "b" * 64},
        "correctness_reference_catalog": {
            "provider": "PennyLane Lightning-GPU",
            "gradient_method": "adjoint",
            "source_commit": "a" * 40,
            "container_digest": None,
        },
        "correctness_references": {
            "n31-seed41": {
                "path": str(reference),
                "sha256": digest(reference.read_bytes()),
            }
        },
        "ready_run_count": 1,
        "runs": [
            {
                "id": "missing",
                "campaign": "external_cudaq_capability",
                "structured_output": str(tmp_path / "missing.json"),
                "raw_log": str(tmp_path / "missing.log"),
                "container_digest": "sha256:" + "b" * 64,
                "dependencies": [],
            }
        ],
        "blocked_campaigns": [],
    }
    result = MODULE.audit(manifest)
    assert result["campaign_ready_for_promotion"] is False
    assert "artifact:missing:structured_output_missing" in result["blockers"]


def test_checks_actual_raw_log_bytes_not_only_payload_shape(tmp_path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    log = tmp_path / "run.log"
    log.write_text("original log\n")
    output = tmp_path / "run.json"
    payload = {
        "source_identity": {
            "commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
            "source_dirty": False,
            "raw_log_sha256": digest(log.read_bytes()),
        }
    }
    output.write_text(json.dumps(payload))
    manifest = {
        "source_commit": "a" * 40,
        "containers": {"cudaq": "sha256:" + "b" * 64},
        "correctness_reference_catalog": {
            "provider": "PennyLane Lightning-GPU",
            "gradient_method": "adjoint",
            "source_commit": "a" * 40,
            "container_digest": None,
        },
        "correctness_references": {
            "n31-seed41": {
                "path": str(reference),
                "sha256": digest(reference.read_bytes()),
            }
        },
        "ready_run_count": 1,
        "runs": [
            {
                "id": "cudaq-run",
                "campaign": "external_cudaq_capability",
                "structured_output": str(output),
                "raw_log": str(log),
                "container_digest": "sha256:" + "b" * 64,
                "dependencies": [],
            }
        ],
        "blocked_campaigns": [],
    }
    log.write_text("tampered log\n")
    result = MODULE.audit(manifest)
    assert "identity:cudaq-run:raw_log_sha256_mismatch" in result["blockers"]


def test_rejects_reference_sha_and_command_path_disagreement(tmp_path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    reference_sha = digest(reference.read_bytes())
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}")
    log = tmp_path / "run.log"
    log.write_text("sealed\n")
    output = tmp_path / "run.json"
    output.write_text(
        json.dumps(
            {
                "source_identity": {
                    "commit": "a" * 40,
                    "container_digest": "sha256:" + "b" * 64,
                    "source_dirty": False,
                    "raw_log_sha256": digest(log.read_bytes()),
                },
                "correctness": {"reference_artifact_sha256": reference_sha},
            }
        )
    )
    references = {
        f"n{n_wires}-seed{seed}": {
            "path": str(reference),
            "sha256": reference_sha,
        }
        for n_wires in range(28, 33)
        for seed in (41, 42, 43)
    }
    manifest = {
        "source_commit": "a" * 40,
        "containers": {
            "pennylane": "sha256:" + "b" * 64,
        },
        "correctness_reference_catalog": {
            "provider": "PennyLane Lightning-GPU",
            "gradient_method": "adjoint",
            "source_commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
        },
        "correctness_references": references,
        "ready_run_count": 2,
        "runs": [
            {
                "id": "mismatch",
                "campaign": "external_pennylane",
                "structured_output": str(output),
                "raw_log": str(log),
                "container_digest": "sha256:" + "b" * 64,
                "command": [
                    "python",
                    "runner.py",
                    "--correctness-reference",
                    str(wrong),
                ],
                "dependencies": [f"correctness-reference:{reference_sha}"],
            }
        ],
        "blocked_campaigns": [],
    }
    result = MODULE.audit(manifest)
    assert "dependency:mismatch:reference_command_mismatch" in result["blockers"]
