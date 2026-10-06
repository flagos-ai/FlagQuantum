from __future__ import annotations

import pytest

from examples.qdiffusion_kaiwu.failure_evidence import redacted_failure_record
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
