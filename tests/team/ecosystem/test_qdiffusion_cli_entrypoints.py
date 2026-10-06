from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ENTRYPOINTS = (
    "a800_sampler_smoke",
    "assemble_acceptance",
    "build_environment_lock",
    "build_transfer_bundle",
    "plan_quota",
    "preflight_protein_artifacts",
    "qboson_live_smoke",
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_development_probe",
    "qdiffusion_system_live",
    "stream_development_evidence",
    "validate_acceptance",
    "verify_extracted_bundle",
    "verify_environment_lock",
    "verify_transfer_bundle",
)

PINNED_SDK_ENTRYPOINTS = (
    "qboson_live_smoke",
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_live",
)

LIVE_HOST_ENTRYPOINTS = (
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_live",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_documented_module_entrypoint_binds_current_checkout(
    tmp_path: Path, entrypoint: str
) -> None:
    repository = Path(__file__).parents[3]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository),
        "PYTHONNOUSERSITE": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            "-m",
            f"examples.qdiffusion_kaiwu.{entrypoint}",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize("entrypoint", PINNED_SDK_ENTRYPOINTS)
def test_live_entrypoint_rejects_unpinned_sdk_version_before_other_inputs(
    tmp_path: Path, entrypoint: str
) -> None:
    repository = Path(__file__).parents[3]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository),
        "PYTHONNOUSERSITE": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            "-m",
            f"examples.qdiffusion_kaiwu.{entrypoint}",
            "--expected-sdk-version",
            "1.4.1",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid choice: '1.4.1'" in completed.stderr


@pytest.mark.parametrize("entrypoint", LIVE_HOST_ENTRYPOINTS)
def test_live_entrypoint_binds_expected_hostname_to_frozen_host_identity(
    entrypoint: str,
) -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / f"{entrypoint}.py"
    ).read_text(encoding="utf-8")

    assert "expected_hostname !=" in source
    assert "config[\"host_identities\"]" in source
    assert (
        'parser.error("--expected-hostname differs from the frozen host identity")'
        in source
    )
    assert source.index("expected_hostname !=") < source.index("socket.gethostname()")
