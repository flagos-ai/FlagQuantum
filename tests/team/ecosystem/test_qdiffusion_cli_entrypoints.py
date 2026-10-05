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
    "build_transfer_bundle",
    "preflight_protein_artifacts",
    "qboson_live_smoke",
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_development_probe",
    "qdiffusion_system_live",
    "validate_acceptance",
    "verify_transfer_bundle",
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
