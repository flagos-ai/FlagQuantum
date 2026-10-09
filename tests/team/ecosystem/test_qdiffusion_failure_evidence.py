from __future__ import annotations

import pytest

from examples.qdiffusion_kaiwu.failure_evidence import (
    apply_artifact_postflight,
    redacted_failure_record,
)
from flagquantum.remote.kaiwu import KaiwuSDKError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error", "category"),
    (
        (KeyboardInterrupt("credential-secret"), "KeyboardInterrupt"),
        (TimeoutError("credential-secret"), "TimeoutError"),
        (KaiwuSDKError("credential-secret"), "KaiwuSDKError"),
        (OSError("credential-secret"), "OSError"),
        (ValueError("credential-secret"), "ValueError"),
        (RuntimeError("credential-secret"), "RuntimeError"),
        (Exception("credential-secret"), "Exception"),
    ),
)
def test_failure_record_keeps_only_stable_category(
    error: BaseException, category: str
) -> None:
    assert redacted_failure_record(error) == {"type": category, "message": ""}


def test_failure_record_does_not_persist_dynamic_exception_type_or_message() -> None:
    secret_type = type("CredentialSecretError", (RuntimeError,), {})

    record = redacted_failure_record(secret_type("credential-secret"))

    assert record == {"type": "RuntimeError", "message": ""}
    assert "CredentialSecret" not in repr(record)
    assert "credential-secret" not in repr(record)


def test_successful_artifact_postflight_sets_only_unchanged_flag() -> None:
    record = {"run_completed": True, "failure": None}

    apply_artifact_postflight(record, None)

    assert record == {
        "run_completed": True,
        "failure": None,
        "artifact_inputs_unchanged": True,
    }


def test_failed_artifact_postflight_uses_closed_failure_fields() -> None:
    record = {"run_completed": True, "failure": None}

    apply_artifact_postflight(record, ValueError("credential-secret"))

    assert record == {
        "run_completed": False,
        "failure": {"type": "ValueError", "message": ""},
        "artifact_inputs_unchanged": False,
    }
    assert "credential-secret" not in repr(record)


def test_artifact_postflight_preserves_original_execution_failure() -> None:
    original = {"type": "TimeoutError", "message": ""}
    record = {"run_completed": False, "failure": original}

    apply_artifact_postflight(record, OSError("later-postflight-secret"))

    assert record["failure"] is original
    assert record["artifact_inputs_unchanged"] is False
