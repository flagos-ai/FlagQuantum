from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "paper/sc27/build_profile_preflight_manifest.py"
    spec = importlib.util.spec_from_file_location("sc27_profile_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _catalog(commit: str) -> dict:
    return {
        "schema": "flagquantum.sc27.correctness_reference_catalog.v1",
        "provider": "PennyLane Lightning-GPU",
        "gradient_method": "adjoint",
        "source_commit": commit,
        "references": {
            "n28-seed41": {
                "path": "/sealed/references/n28-seed41.json",
                "sha256": "b" * 64,
            }
        },
    }


def test_builds_one_profiled_production_smoke_run(tmp_path: Path) -> None:
    module = _module()
    commit = "a" * 40
    manifest = module.build_manifest(
        source_commit=commit,
        core_container_digest="sha256:" + "c" * 64,
        correctness_reference_catalog=_catalog(commit),
        output_root=str(tmp_path / "runs"),
    )
    assert manifest["run_count"] == 1
    run = manifest["run"]
    assert run["resources"]["gpus_per_node"] == 2
    command = run["command"]
    assert command[:4] == [
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "benchmarks/statevector_training_scaling.py",
    ]
    assert "--profile" in command
    assert command[command.index("--n-wires") + 1] == "28"
    assert command[command.index("--workload") + 1] == "full-width-linear"
    assert command[command.index("--optimizer") + 1] == "adam"
    assert run["audit_command"][1] == "paper/sc27/audit_profile_preflight.py"
    assert run["dependencies"] == ["correctness-reference:" + "b" * 64]


def test_rejects_reference_identity_drift(tmp_path: Path) -> None:
    module = _module()
    commit = "a" * 40
    catalog = _catalog("d" * 40)
    with pytest.raises(ValueError, match="source commit mismatch"):
        module.build_manifest(
            source_commit=commit,
            core_container_digest="sha256:" + "c" * 64,
            correctness_reference_catalog=catalog,
            output_root=str(tmp_path / "runs"),
        )
