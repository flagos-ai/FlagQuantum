from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.plan_quota import build_quota_plan

pytestmark = pytest.mark.unit


def _config() -> dict[str, object]:
    return {
        "dataset": {
            "max_records": 640,
            "validation_ratio": 0.05,
            "test_ratio": 0.05,
        },
        "training": {
            "epochs": 20,
            "num_candidates": 4,
            "validation_steps": 3,
            "remote_call_budget_per_seed": 71269,
        },
        "generation": {
            "sequence_count": 32,
            "max_steps": 64,
            "num_candidates": 4,
            "portability_steps": 3,
        },
        "seeds": [1701, 1702, 1703],
        "remote_call_budget": 128,
    }


def test_quota_plan_exposes_components_and_declared_ceiling() -> None:
    plan = build_quota_plan(_config(), "a" * 64)

    assert plan["unit"] == "distinct_provider_task_submissions"
    assert plan["estimate_kind"] == "conservative_maximum"
    assert plan["includes_status_polls"] is False
    assert plan["system"] == {
        "host_count": 2,
        "estimated_max_calls_per_host": 10,
        "estimated_max_calls": 20,
        "declared_ceiling_per_host": 128,
        "declared_ceiling": 256,
    }
    assert plan["protein"]["breakdown_per_seed"] == {
        "selected_records": 640,
        "training_records": 576,
        "validation_records": 32,
        "test_records": 32,
        "structural_calls": 5,
        "training_calls_per_epoch": 3040,
        "training_calls": 60800,
        "baseline_calls_per_record": 66,
        "guided_calls_per_record": 261,
        "generation_calls": 10464,
        "total": 71269,
    }
    assert plan["protein"]["estimated_max_calls"] == 213807
    assert plan["portability"]["breakdown"] == {
        "initialization_calls": 5,
        "guided_step_calls": 12,
        "total": 17,
    }
    assert plan["totals"] == {
        "estimated_max_calls": 213846,
        "declared_ceiling": 214193,
        "budget_complete": True,
    }


def test_quota_plan_keeps_unapproved_protein_budget_unresolved() -> None:
    config = _config()
    config["training"]["remote_call_budget_per_seed"] = "<required>"  # type: ignore[index]

    plan = build_quota_plan(config, "b" * 64)

    assert plan["protein"]["declared_ceiling_per_seed"] is None
    assert plan["totals"]["declared_ceiling"] is None
    assert plan["totals"]["budget_complete"] is False


def test_quota_plan_rejects_an_unbound_config_identity() -> None:
    with pytest.raises(ValueError, match="config_sha256"):
        build_quota_plan(_config(), "not-a-digest")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        (None, "remote_call_budget", 16, "portability estimate"),
        ("training", "remote_call_budget_per_seed", 71268, "protein estimate"),
    ),
)
def test_quota_plan_rejects_a_ceiling_below_the_derived_estimate(
    section: str | None, field: str, value: int, message: str
) -> None:
    config = _config()
    target = config if section is None else config[section]
    target[field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        build_quota_plan(config, "c" * 64)


def test_cli_hashes_strict_config_and_prints_no_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "private-config.json"
    raw = json.dumps(_config(), sort_keys=True).encode()
    config_path.write_bytes(raw)
    config_path.chmod(0o600)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-m",
            "examples.qdiffusion_kaiwu.plan_quota",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["experiment_config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert str(tmp_path) not in completed.stdout


@pytest.mark.parametrize("unsafe_kind", ("public-file", "public-parent", "symlink"))
def test_cli_requires_private_anchored_config(tmp_path: Path, unsafe_kind: str) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    config_path = private_parent / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    config_path.chmod(0o600)
    argument = config_path
    if unsafe_kind == "public-file":
        config_path.chmod(0o644)
    elif unsafe_kind == "public-parent":
        private_parent.chmod(0o755)
    else:
        argument = private_parent / "config-link.json"
        argument.symlink_to(config_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-m",
            "examples.qdiffusion_kaiwu.plan_quota",
            str(argument),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "quota-plan configuration" in completed.stderr
