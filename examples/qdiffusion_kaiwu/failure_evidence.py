"""Value-free failure classification for private QBoson evidence."""

from __future__ import annotations

from typing import Any

from flagquantum.remote.kaiwu import KaiwuSDKError


def redacted_failure_record(error: BaseException) -> dict[str, str]:
    """Classify an error without persisting its type name or message verbatim."""

    if isinstance(error, KeyboardInterrupt):
        category = "KeyboardInterrupt"
    elif isinstance(error, TimeoutError):
        category = "TimeoutError"
    elif isinstance(error, KaiwuSDKError):
        category = "KaiwuSDKError"
    elif isinstance(error, OSError):
        category = "OSError"
    elif isinstance(error, ValueError):
        category = "ValueError"
    elif isinstance(error, RuntimeError):
        category = "RuntimeError"
    else:
        category = "Exception"
    return {"type": category, "message": ""}


def apply_artifact_postflight(
    record: dict[str, Any], error: BaseException | None
) -> None:
    """Close a component through its existing failure fields after rehashing."""

    record["artifact_inputs_unchanged"] = error is None
    if error is None:
        return
    record["run_completed"] = False
    if record.get("failure") is None:
        record["failure"] = redacted_failure_record(error)


__all__ = ("apply_artifact_postflight", "redacted_failure_record")
