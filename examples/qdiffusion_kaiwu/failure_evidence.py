"""Value-free failure classification for private QBoson evidence."""

from __future__ import annotations

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


__all__ = ("redacted_failure_record",)
