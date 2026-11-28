"""Source retries stay bounded, isolated and subject to the original verification."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _steps(workflow: str) -> list[list[dict[str, Any]]]:
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return [
        job.get("steps", [])
        for job in document["jobs"].values()
        if isinstance(job, dict)
    ]


@pytest.mark.parametrize("workflow,expected", [("ci.yml", 35), ("local-gpu.yml", 6)])
def test_every_source_handoff_has_one_bounded_independent_retry(
    workflow: str, expected: int
) -> None:
    count = 0
    for steps in _steps(workflow):
        ids = [step["id"] for step in steps if "id" in step]
        assert len(ids) == len(set(ids))
        for index, first in enumerate(steps):
            identifier = first.get("id", "")
            if identifier not in (
                "source-incremental-download",
                "source-full-download",
            ):
                continue
            count += 1
            role = identifier.removesuffix("-download")
            retry, promotion = steps[index + 1 : index + 3]
            assert first["uses"] == retry["uses"] == "actions/download-artifact@v8.0.2"
            assert first["timeout-minutes"] == (
                1 if role == "source-incremental" else 2
            )
            assert first["continue-on-error"] is True
            assert retry["timeout-minutes"] == 2
            assert not retry.get("continue-on-error", False)
            assert retry["id"] == role + "-retry"
            assert (
                retry["if"]
                == "${{ !cancelled() && steps."
                + identifier
                + ".outcome == 'failure' }}"
            )
            assert retry["with"]["name"] == first["with"]["name"]
            assert (
                retry["with"]["digest-mismatch"]
                == first["with"]["digest-mismatch"]
                == "error"
            )
            assert retry["with"]["path"] != first["with"]["path"]
            assert promotion["id"] == role + "-promote"
            assert (
                promotion["if"]
                == "${{ steps." + role + "-retry.outcome == 'success' }}"
            )
            assert promotion["env"]["FQ_CI_SOURCE_DIR"] == first["with"]["path"]
    assert count == expected


def _promotion(kind: str) -> str:
    script = next(
        step["run"]
        for steps in _steps("ci.yml")
        for step in steps
        if step.get("id") == f"source-{kind}-promote"
    )
    assert isinstance(script, str)
    return script


@pytest.mark.parametrize("kind", ["incremental", "full"])
@pytest.mark.parametrize("partial_exists", [False, True])
def test_successful_retry_replaces_only_the_canonical_input_directory(
    tmp_path: Path, kind: str, partial_exists: bool
) -> None:
    source = tmp_path / "source with spaces;$()"
    retry = source.with_name(source.name + f"-{kind}-retry")
    abandoned = source.with_name(source.name + f"-{kind}-abandoned")
    retry.mkdir()
    (retry / "source.sha").write_bytes(b"verified source identity\n")
    (retry / "source.pack").write_bytes(b"verified retry bytes\0\xff")
    if partial_exists:
        source.mkdir()
        (source / "source.pack").write_bytes(b"untrusted partial transfer")
        (source / "partial-only").write_bytes(b"never consumed")
    result = subprocess.run(
        ["bash", "-e", "-c", _promotion(kind)],
        env={**os.environ, "FQ_CI_SOURCE_DIR": str(source)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert set(p.name for p in source.iterdir()) == {"source.sha", "source.pack"}
    assert (source / "source.sha").read_bytes() == b"verified source identity\n"
    assert (source / "source.pack").read_bytes() == b"verified retry bytes\0\xff"
    assert not retry.exists()
    if partial_exists:
        assert (abandoned / "source.pack").read_bytes() == b"untrusted partial transfer"
        assert (abandoned / "partial-only").read_bytes() == b"never consumed"
    else:
        assert not abandoned.exists()


@pytest.mark.parametrize("kind", ["incremental", "full"])
def test_retry_does_not_overwrite_an_existing_abandoned_attempt(
    tmp_path: Path, kind: str
) -> None:
    source = tmp_path / "source"
    retry = source.with_name(source.name + f"-{kind}-retry")
    abandoned = source.with_name(source.name + f"-{kind}-abandoned")
    for directory, content in [
        (source, b"first"),
        (retry, b"retry"),
        (abandoned, b"retained"),
    ]:
        directory.mkdir()
        (directory / "source.pack").write_bytes(content)
    result = subprocess.run(
        ["bash", "-e", "-c", _promotion(kind)],
        env={**os.environ, "FQ_CI_SOURCE_DIR": str(source)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert (source / "source.pack").read_bytes() == b"first"
    assert (retry / "source.pack").read_bytes() == b"retry"
    assert (abandoned / "source.pack").read_bytes() == b"retained"


def _coverage_script(job: str, name: str) -> str:
    document = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    script = next(
        step["run"]
        for step in document["jobs"][job]["steps"]
        if step.get("name") == name
    )
    assert isinstance(script, str)
    return script


@pytest.mark.parametrize(
    "fault", ["none", "stale-source", "missing-pack", "empty-pack"]
)
def test_plan_bundle_keeps_initial_source_before_full_fallback(
    tmp_path: Path, fault: str
) -> None:
    temporary = tmp_path / "temporary with spaces;$()"
    source = temporary / "ci-source"
    plan = temporary / "coverage-plan"
    source.mkdir(parents=True)
    plan.mkdir()
    identity = "a" * 40
    (source / "source.sha").write_text(
        ("b" * 40 if fault == "stale-source" else identity) + "\n"
    )
    initial = b"original verified incremental pack\0\xff"
    if fault != "missing-pack":
        (source / "source.pack").write_bytes(b"" if fault == "empty-pack" else initial)
    (plan / "plan.json").write_bytes(b"unchanged collected plan")
    env = {**os.environ, "RUNNER_TEMP": str(temporary), "GITHUB_SHA": identity}
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            _coverage_script(
                "coverage-plan", "Preserve the verified incremental coverage bootstrap"
            ),
        ],
        env=env,
        capture_output=True,
        timeout=10,
    )
    if fault != "none":
        assert result.returncode != 0
        assert not (temporary / "coverage-bootstrap").exists()
        assert not (plan / "source.pack").exists()
        return
    assert result.returncode == 0, result.stderr
    (source / "source.pack").write_bytes(b"large full-source fallback")
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            _coverage_script(
                "coverage-plan",
                "Bundle the verified incremental source with the coverage plan",
            ),
        ],
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (plan / "source.pack").read_bytes() == initial
    assert (plan / "source.sha").read_text() == identity + "\n"
    assert (plan / "plan.json").read_bytes() == b"unchanged collected plan"


def test_bundled_plan_survives_full_source_retry_and_refuses_stale_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.ci.ci_coverage_plan import verify_plan, write_plan

    identity = "a" * 40
    monkeypatch.setenv("GITHUB_SHA", identity)
    temporary = tmp_path / "temporary with spaces;$()"
    source = temporary / "ci-source"
    node = "tests/unit/test_original.py::test_original"
    write_plan((node,), source, 1)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    (source / "source.sha").write_text(identity + "\n")
    (source / "source.pack").write_bytes(b"incremental source")
    env = {
        **os.environ,
        "RUNNER_TEMP": str(temporary),
        "GITHUB_SHA": identity,
        "FQ_CI_SOURCE_DIR": str(source),
    }
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            _coverage_script(
                "coverage-shard",
                "Preserve the verified coverage plan before source fallback",
            ),
        ],
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    retry = source.with_name("ci-source-full-retry")
    retry.mkdir()
    (retry / "source.sha").write_text(identity + "\n")
    (retry / "source.pack").write_bytes(b"complete full source")
    result = subprocess.run(
        ["bash", "-e", "-c", _promotion("full")],
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    plan = temporary / "coverage-plan"
    assert {p.name: p.read_bytes() for p in plan.iterdir()} == original
    assert verify_plan(plan, 0).read_text().splitlines() == [node]
    assert set(p.name for p in source.iterdir()) == {"source.sha", "source.pack"}
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    with pytest.raises(ValueError, match="source"):
        verify_plan(plan, 0)
