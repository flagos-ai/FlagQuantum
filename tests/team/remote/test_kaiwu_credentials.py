from __future__ import annotations

import pickle

import pytest

from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    resolve_kaiwu_credentials,
)

pytestmark = pytest.mark.unit


def test_explicit_credentials_are_validated_redacted_and_not_serializable() -> None:
    credentials = KaiwuCredentials(user_id=" user-123 ", sdk_code=" secret-code ")

    assert resolve_kaiwu_credentials(credentials) == ("user-123", "secret-code")
    assert "user-123" not in repr(credentials)
    assert "secret-code" not in repr(credentials)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(credentials)


@pytest.mark.parametrize(
    ("user_id", "sdk_code", "error"),
    (
        (1, "code", TypeError),
        ("user", 1, TypeError),
        ("", "code", ValueError),
        ("user", " ", ValueError),
        ("user\nname", "code", ValueError),
        ("user", "code\tvalue", ValueError),
        ("user\u200bname", "code", ValueError),
    ),
)
def test_explicit_credentials_reject_invalid_values(
    user_id: object, sdk_code: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        KaiwuCredentials(
            user_id=user_id,  # type: ignore[arg-type]
            sdk_code=sdk_code,  # type: ignore[arg-type]
        )


def test_explicit_credentials_do_not_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QBOSON_USER_ID", "environment-user")
    monkeypatch.setenv("QBOSON_SDK_CODE", "environment-secret")

    resolved = resolve_kaiwu_credentials(
        KaiwuCredentials(user_id="explicit-user", sdk_code="explicit-secret")
    )

    assert resolved == ("explicit-user", "explicit-secret")


def test_complete_dedicated_environment_pair_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QBOSON_USER_ID", " user-from-env ")
    monkeypatch.setenv("QBOSON_SDK_CODE", " secret-from-env ")
    monkeypatch.setenv("USER_ID", "ignored-generic-user")
    monkeypatch.setenv("SDK_CODE", "ignored-generic-secret")

    assert resolve_kaiwu_credentials() == ("user-from-env", "secret-from-env")


@pytest.mark.parametrize("present", ("user", "code", "neither"))
def test_missing_or_partial_environment_pair_fails_closed(
    monkeypatch: pytest.MonkeyPatch, present: str
) -> None:
    monkeypatch.delenv("QBOSON_USER_ID", raising=False)
    monkeypatch.delenv("QBOSON_SDK_CODE", raising=False)
    if present == "user":
        monkeypatch.setenv("QBOSON_USER_ID", "user")
    elif present == "code":
        monkeypatch.setenv("QBOSON_SDK_CODE", "code")

    with pytest.raises(RuntimeError) as error:
        resolve_kaiwu_credentials()

    assert "user" not in str(error.value).lower() or "USER_ID" in str(error.value)
    assert "code" not in str(error.value).lower() or "SDK_CODE" in str(error.value)


def test_empty_environment_values_fail_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QBOSON_USER_ID", " ")
    monkeypatch.setenv("QBOSON_SDK_CODE", "secret-value")

    with pytest.raises(RuntimeError) as error:
        resolve_kaiwu_credentials()

    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize(
    ("user_id", "sdk_code"),
    (("user\nname", "secret-value"), ("user", "secret\tvalue")),
)
def test_environment_credentials_reject_control_characters_without_echoing(
    monkeypatch: pytest.MonkeyPatch,
    user_id: str,
    sdk_code: str,
) -> None:
    monkeypatch.setenv("QBOSON_USER_ID", user_id)
    monkeypatch.setenv("QBOSON_SDK_CODE", sdk_code)

    with pytest.raises(RuntimeError, match="printable characters") as error:
        resolve_kaiwu_credentials()

    assert user_id not in str(error.value)
    assert sdk_code not in str(error.value)
