"""Pure RSS-failure evidence checks; no monitor probe or numerical worker."""

from __future__ import annotations

import gc
import json
import subprocess
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import collect_ftqc_phase_cost as controller
from tools import ftqc_monitor_evidence as evidence

pytestmark = pytest.mark.unit


def timeout() -> BaseException:
    try:
        raise subprocess.TimeoutExpired("ps", 0.05, output=b"private process rows")
    except subprocess.TimeoutExpired as error:
        return error


def test_record_is_bounded_scalar_only_and_never_accepts_checkpoint() -> None:
    result = evidence.collect_failure_identity(timeout())
    assert result["capture_status"] == "captured"
    assert result["checkpoint"] == "unknown"
    assert result["frames"] and len(result["frames"]) <= 16
    assert "private process rows" not in json.dumps(result)
    assert len(json.dumps(result).encode()) <= 16384
    assert all(len(row["filename_sha256"]) == 64 for row in result["frames"])


def test_unsupported_runtime_has_portable_frames_without_false_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence, "_supports_structure", lambda: False)
    result = evidence.collect_failure_identity(timeout())
    assert result["structure_schema"] is None
    assert result["structure_status"] == "unsupported_runtime_schema"
    assert result["frames"] and result["checkpoint"] == "unknown"
    assert all(row["structural_sha256"] is None for row in result["functions"])


def test_structure_hash_keeps_filename_separate_and_code_changes_visible() -> None:
    def fixture() -> int:
        return 1

    code = fixture.__code__
    if not evidence._supports_structure():
        with pytest.raises(ValueError, match="unsupported"):
            evidence._structure_digest(code)
        return
    assert evidence._structure_digest(code) == evidence._structure_digest(
        code.replace(co_filename="/another/private/path.py")
    )
    assert evidence._structure_digest(code) != evidence._structure_digest(
        code.replace(co_consts=(None, 2))
    )


def test_structure_scalar_refusal() -> None:
    def fixture() -> float:
        return 1.0

    if not evidence._supports_structure():
        with pytest.raises(ValueError, match="unsupported"):
            evidence._structure_digest(fixture.__code__)
        return
    with pytest.raises(ValueError, match="nonfinite"):
        evidence._structure_digest(fixture.__code__.replace(co_consts=(float("nan"),)))
    with pytest.raises(ValueError, match="bound"):
        evidence._structure_digest(fixture.__code__.replace(co_consts=(b"x" * 131073,)))


def test_capture_failure_is_explicit_and_contains_no_exception_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_value: str) -> str:
        raise ValueError("private failure")

    monkeypatch.setattr(evidence, "_filename_digest", broken)
    result = evidence.collect_failure_identity(timeout())
    assert result == evidence.unavailable_failure_identity()
    assert "private" not in json.dumps(result)


def test_traceback_bound_reports_truncation() -> None:
    def recur(depth: int) -> None:
        if depth:
            recur(depth - 1)
        raise subprocess.TimeoutExpired("ps", 0.05)

    try:
        recur(20)
    except subprocess.TimeoutExpired as error:
        result = evidence.collect_failure_identity(error)
    assert len(result["frames"]) == 16 and result["traceback_truncated"] is True
    assert result["checkpoint"] == "unknown"


def test_runtime_code_identity_uses_objects_without_alias_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def communicate() -> None:
        raise subprocess.TimeoutExpired("ps", 0.05)

    fake = SimpleNamespace(
        __file__=communicate.__code__.co_filename,
        Popen=SimpleNamespace(communicate=communicate),
    )
    monkeypatch.setattr(evidence, "subprocess", fake)
    try:
        communicate()
    except subprocess.TimeoutExpired as error:
        result = evidence.collect_failure_identity(error)
    row = next(
        frame for frame in result["frames"] if frame["function"] == "communicate"
    )
    assert row["matches_runtime_code_object"] is True
    assert row["filename_matches_runtime_module"] is True
    assert result["checkpoint"] == "unknown"
    assert result["structure_status"] in (
        "incomplete_runtime_identity",
        "unsupported_runtime_schema",
    )


def synthetic_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: bool,
    capture_failure: bool | BaseException = False,
    cleanup_failure: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    calls: list[str] = []

    class Process:
        pid = 4242
        returncode: int | None = None

        def poll(self) -> int | None:
            return None if fail else 0

    process = Process()

    def query(_group: int) -> int:
        calls.append("query")
        if fail:
            raise subprocess.TimeoutExpired("ps", 0.05)
        return 0

    def stop(_process: object) -> None:
        calls.append("cleanup")
        if cleanup_failure:
            raise OSError("cleanup failed")
        process.returncode = -15

    def capture(error: BaseException) -> evidence.FailureIdentity:
        calls.append("metadata")
        assert isinstance(error, subprocess.TimeoutExpired)
        if isinstance(capture_failure, BaseException):
            raise capture_failure
        if capture_failure:
            raise RuntimeError("metadata failed")
        return evidence.unavailable_failure_identity()

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(controller, "budget", lambda *_: None)
    monkeypatch.setattr(controller, "_process_group_rss", query)
    monkeypatch.setattr(controller, "_stop_group", stop)
    monkeypatch.setattr(controller, "collect_failure_identity", capture)
    directory = tmp_path / "attempt"
    directory.mkdir()
    report = controller.execute(["synthetic"], directory, tmp_path, float("inf"))
    return report, calls


@pytest.mark.parametrize("capture_failure", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_metadata_follows_cleanup_without_masking_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_failure: bool,
    cleanup_failure: bool,
) -> None:
    report, calls = synthetic_execute(
        tmp_path,
        monkeypatch,
        fail=True,
        capture_failure=capture_failure,
        cleanup_failure=cleanup_failure,
    )
    assert calls == ["query", "cleanup", "metadata"]
    assert report["status"] == "failed" and "TimeoutExpired" in report["diagnostic"]
    assert ("cleanup_error" in report) is cleanup_failure
    assert report["rss_failure_identity"]["checkpoint"] == "unknown"
    assert report["rss_query_deadline_seconds"] == 0.05
    assert report["rss_max_gap_limit_seconds"] == 0.1
    assert report["returncode"] == (None if cleanup_failure else -15)


def test_non_timeout_failure_has_no_new_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, calls = synthetic_execute(tmp_path, monkeypatch, fail=False)
    # The existing final-group/positive-RSS guard fails; it is not a query timeout.
    assert report["status"] == "failed" and calls == ["query", "cleanup"]
    assert "rss_failure_identity" not in report


def test_success_has_no_metadata_and_no_additional_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 4242
        returncode = 0
        polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else 0

    rss = iter([4096, 0])
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(controller, "budget", lambda *_: None)
    monkeypatch.setattr(controller, "_process_group_rss", lambda _: next(rss))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        controller,
        "collect_failure_identity",
        lambda _: pytest.fail("success metadata"),
    )
    directory = tmp_path / "attempt"
    directory.mkdir()
    report = controller.execute(["synthetic"], directory, tmp_path, float("inf"))
    assert report["status"] == "completed" and "rss_failure_identity" not in report
    with pytest.raises(StopIteration):
        next(rss)


def test_failure_reference_is_released_after_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WeakTimeout(subprocess.TimeoutExpired):
        pass

    error: BaseException | None = WeakTimeout("ps", 0.05)
    ref = weakref.ref(error)

    class Process:
        pid = 4242
        returncode = -15

        def poll(self) -> None:
            return None

    def query(_group: int) -> int:
        assert error is not None
        raise error

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(controller, "budget", lambda *_: None)
    monkeypatch.setattr(controller, "_process_group_rss", query)
    monkeypatch.setattr(controller, "_stop_group", lambda _: None)
    directory = tmp_path / "attempt"
    directory.mkdir()
    controller.execute(["synthetic"], directory, tmp_path, float("inf"))
    error = None
    gc.collect()
    assert ref() is None


def test_allowed_name_bound_preserves_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def wait(depth: int) -> None:
        if depth:
            wait(depth - 1)
        raise subprocess.TimeoutExpired("ps", 0.05)

    monkeypatch.setattr(
        evidence,
        "subprocess",
        SimpleNamespace(
            __file__=wait.__code__.co_filename,
            Popen=SimpleNamespace(wait=wait),
        ),
    )
    try:
        wait(12)
    except subprocess.TimeoutExpired as error:
        result = evidence.collect_failure_identity(error)
    assert sum(frame["function"] == "wait" for frame in result["frames"]) == 8
    assert result["traceback_truncated"] is True and result["checkpoint"] == "unknown"


def test_metadata_size_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    original = evidence._packed

    def oversized(value: object) -> bytes:
        return b"x" * 16385 if isinstance(value, dict) else original(value)

    monkeypatch.setattr(evidence, "_packed", oversized)
    assert (
        evidence.collect_failure_identity(timeout())
        == evidence.unavailable_failure_identity()
    )


def test_cancellation_keeps_cleanup_and_does_not_add_timeout_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[object] = []

    class Process:
        pid = 4242
        returncode = -15

        def poll(self) -> None:
            return None

    def cancelled(_group: int) -> int:
        raise SystemExit(143)

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(controller, "budget", lambda *_: None)
    monkeypatch.setattr(controller, "_process_group_rss", cancelled)
    monkeypatch.setattr(controller, "_stop_group", stopped.append)
    monkeypatch.setattr(
        controller,
        "collect_failure_identity",
        lambda _: pytest.fail("cancellation metadata"),
    )
    directory = tmp_path / "attempt"
    directory.mkdir()
    report = controller.execute(["synthetic"], directory, tmp_path, float("inf"))
    assert len(stopped) == 1 and report["status"] == "failed"
    assert "SystemExit" in report["diagnostic"] and "rss_failure_identity" not in report


def test_metadata_cancellation_cannot_replace_original_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, calls = synthetic_execute(
        tmp_path,
        monkeypatch,
        fail=True,
        capture_failure=KeyboardInterrupt(),
    )
    assert calls == ["query", "cleanup", "metadata"]
    assert report["status"] == "failed" and "TimeoutExpired" in report["diagnostic"]
    assert report["rss_failure_identity"]["capture_status"] == "capture_failed"
