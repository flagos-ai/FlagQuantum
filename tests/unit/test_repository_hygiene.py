import hashlib
import json
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tools.check_dependency_policy import policy_errors
from tools.check_import_time import sample_imports
from tools.check_repository_hygiene import (
    DEFAULT_MAX_DEVELOPMENT_DOC_FILES,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_RESULT_BYTES,
    DEFAULT_MAX_RESULT_FILES,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_MAX_TRACKED_FILES,
    repository_metrics,
    violations,
)
from tools.fetch_example_assets import verify_asset
from tools.verify_distribution_artifacts import _forbidden

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def test_repository_has_no_tracked_cache_dataset_model_or_oversized_artifact():
    assert (
        violations(
            ROOT,
            max_file_bytes=DEFAULT_MAX_FILE_BYTES,
            max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
            max_tracked_files=DEFAULT_MAX_TRACKED_FILES,
            max_result_bytes=DEFAULT_MAX_RESULT_BYTES,
            max_result_files=DEFAULT_MAX_RESULT_FILES,
            max_development_doc_files=DEFAULT_MAX_DEVELOPMENT_DOC_FILES,
        )
        == ()
    )


def test_repository_budgets_are_ratchets_above_the_current_tree() -> None:
    metrics = repository_metrics(ROOT)
    assert metrics["tracked_files"] <= DEFAULT_MAX_TRACKED_FILES
    assert metrics["tracked_bytes"] <= DEFAULT_MAX_TOTAL_BYTES
    assert metrics["benchmark_result_files"] <= DEFAULT_MAX_RESULT_FILES
    assert metrics["benchmark_result_bytes"] <= DEFAULT_MAX_RESULT_BYTES
    assert metrics["development_doc_files"] <= DEFAULT_MAX_DEVELOPMENT_DOC_FILES


def test_repository_budgets_fail_closed_when_a_ceiling_is_exceeded() -> None:
    errors = violations(
        ROOT,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_total_bytes=0,
        max_tracked_files=0,
        max_result_bytes=0,
        max_result_files=0,
        max_development_doc_files=0,
    )
    assert any(error.startswith("tracked tree has") for error in errors)
    assert any(error.startswith("benchmark results has") for error in errors)
    assert any(error.startswith("development documentation has") for error in errors)


def test_evidence_archive_index_uses_immutable_verified_releases() -> None:
    payload = json.loads((ROOT / "docs/development/evidence-archives.json").read_text())
    assert payload["schema"] == "flagquantum_evidence_archives_v2"
    assert payload["archives"]

    for archive in payload["archives"]:
        tag = archive["release_tag"]
        assert re.fullmatch(r"evidence-\d{4}-\d{2}-\d{2}\.\d+", tag)
        assert archive["release_url"].endswith(f"/releases/tag/{tag}")
        assert f"/releases/download/{tag}/" in archive["archive_url"]
        assert f"/releases/download/{tag}/" in archive["manifest_url"]
        assert re.fullmatch(r"[0-9a-f]{40}", archive["source_commit"])
        assert re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
        assert archive["archive_bytes"] > 0
        assert archive["file_count"] > 0
        assert archive["uncompressed_bytes"] >= archive["archive_bytes"]
        roots = archive["roots"]
        assert roots
        assert roots == sorted(set(roots))
        assert set(roots) <= {"benchmarks/results", "docs/development"}


def test_repository_root_has_no_generated_benchmark_or_retired_logo() -> None:
    assert not (ROOT / "mps_benchmark").exists()
    assert not (ROOT / "sv_benckend_benchmark_local").exists()
    assert not (ROOT / "assets" / "logo.png").exists()
    assert (ROOT / "assets" / "logo_flagquantum.png").is_file()


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for document in ROOT.rglob("*.md"):
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local_target = target.split("#", 1)[0]
            if local_target and not (document.parent / local_target).exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "\n".join(broken)


def test_dependency_groups_keep_core_minimal_and_ranges_executable():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["dependencies"] == [
        "torch>=2.13,<2.15",
        "tomli>=2.0,<3; python_version < '3.11'",
    ]
    extras = project["optional-dependencies"]
    assert extras["jax"] == ["jax>=0.10,<0.11"]
    assert all("jax" not in item for item in extras["dev"])
    assert all("datasets" not in item for item in extras["dev"])
    assert all("transformers" not in item for item in extras["dev"])


def test_dependency_policy_matches_ci_python_matrix():
    policy = tomllib.loads((ROOT / "dependency-policy.toml").read_text())
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert policy["python"] == ["3.10", "3.11", "3.12"]
    for version in policy["python"]:
        assert f'"{version}"' in workflow
    assert policy_errors(policy, pyproject) == ()


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
