from pathlib import Path

import pytest

from tools.artifact_manifest import main as manifest_main
from tools.verify_distribution_artifacts import _forbidden

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_package_uses_one_dynamic_version_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "flagquantum.version.__version__"}' in pyproject
    assert '\nversion = "0.1.0"' not in pyproject


def test_ci_has_supported_python_core_and_optional_jax_jobs():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for version in ("3.10", "3.11", "3.12"):
        assert f'"{version}"' in workflow
    assert "cpu-core:" in workflow
    assert "jax-optional:" in workflow
    assert "distributed-cpu:" in workflow
    assert "verify_distribution_artifacts.py" in workflow
    assert "artifact_manifest.py" in workflow
    assert "Clean-install wheel" in workflow
    assert "Clean-install sdist" in workflow
    assert "native distributed runtime does not require JAX" in workflow
    local_gpu = (ROOT / ".github" / "workflows" / "local-gpu.yml").read_text(
        encoding="utf-8"
    )
    assert "one-gpu-local" in local_gpu
    assert "two-gpu-distributed-required" in local_gpu
    assert "--nproc-per-node=2" in local_gpu
    assert "hardware_run_manifest.py" in local_gpu
    scheduled = (ROOT / ".github" / "workflows" / "scheduled-hardware.yml").read_text(
        encoding="utf-8"
    )
    assert "gpu-scheduled" in scheduled
    assert "multinode-scheduled" in scheduled
    assert "gpu.log" in scheduled
    assert "gpu-count: [4, 8]" in scheduled


def test_distribution_quarantine_rejects_repo_only_members():
    assert _forbidden("flagquantum/__pycache__/module.pyc")
    assert _forbidden("source/tests/test_api.py")
    assert _forbidden("source/examples/data.parquet")
    assert not _forbidden("flagquantum/core/ir.py")


def test_artifact_manifest_records_checksum(tmp_path):
    artifact = tmp_path / "sample.whl"
    artifact.write_bytes(b"flagquantum")
    output = tmp_path / "manifest.json"

    assert manifest_main(["--output", str(output), str(artifact)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "flagquantum_artifact_manifest_v1" in text
    assert "sample.whl" in text
    assert "sha256" in text
