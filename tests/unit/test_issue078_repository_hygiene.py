import hashlib
import json
from pathlib import Path

import pytest
import tomllib

from tools.check_import_time import sample_imports
from tools.check_repository_hygiene import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    violations,
)
from tools.fetch_example_assets import verify_asset
from tools.verify_distribution_artifacts import _forbidden

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_repository_has_no_tracked_cache_dataset_model_or_oversized_artifact():
    assert (
        violations(
            ROOT,
            max_file_bytes=DEFAULT_MAX_FILE_BYTES,
            max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
        )
        == ()
    )


def test_dependency_groups_keep_core_minimal_and_ranges_executable():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["dependencies"] == ["torch>=2.5,<2.14"]
    extras = project["optional-dependencies"]
    assert extras["jax"] == ["jax>=0.10,<0.11"]
    assert all("jax" not in item for item in extras["dev"])
    assert all("datasets" not in item for item in extras["dev"])
    assert all("transformers" not in item for item in extras["dev"])


def test_dependency_policy_matches_ci_python_matrix():
    policy = tomllib.loads((ROOT / "dependency-policy.toml").read_text())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert policy["python"] == ["3.10", "3.11", "3.12"]
    for version in policy["python"]:
        assert f'"{version}"' in workflow
    assert policy["extras"]["providers"] == []


def test_example_assets_are_versioned_and_checksummed_not_tracked_payloads(tmp_path):
    manifest = json.loads((ROOT / "examples/assets/manifest.json").read_text())
    assert manifest["schema"] == "flagquantum_example_assets_v1"
    for asset in manifest["assets"]:
        assert asset["revision"] not in {"", "main", "latest"}
        assert len(asset["sha256"]) == 64
        assert asset["bytes"] > 0
    payload = b"tiny maintained fixture"
    path = tmp_path / "fixture.bin"
    path.write_bytes(payload)
    verify_asset(
        path,
        {
            "id": "fixture",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    assert not (ROOT / "examples/models").exists()


def test_distribution_quarantine_covers_models_datasets_and_generated_files():
    for member in (
        "flagquantum/__pycache__/x.pyc",
        "source/examples/data.parquet",
        "source/model.safetensors",
        "source/checkpoint.pt",
        "source/tests/test_api.py",
    ):
        assert _forbidden(member)
    assert not _forbidden("flagquantum/core/ir.py")


def test_lazy_import_budget_and_optional_jax_absence():
    samples = sample_imports(3)
    assert max(samples) < 0.5
