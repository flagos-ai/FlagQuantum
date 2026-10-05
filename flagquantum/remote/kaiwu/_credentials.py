"""Credential handling for Kaiwu SDK license initialization."""

from __future__ import annotations

import os
from typing import NoReturn


class KaiwuCredentials:
    """One in-memory QBoson user ID and SDK authorization-code pair.

    The SDK authorization code is a secret even though it is not conventionally
    named an API key.  This object cannot be pickled and never reveals either
    value through ``repr``.  Receipts must not serialize this object or the pair
    returned by :func:`resolve_kaiwu_credentials`.
    """

    __slots__ = ("_sdk_code", "_user_id")

    def __init__(self, *, user_id: str, sdk_code: str) -> None:
        if not isinstance(user_id, str) or not isinstance(sdk_code, str):
            raise TypeError("Kaiwu user_id and sdk_code must be strings")
        user_id, sdk_code = user_id.strip(), sdk_code.strip()
        if not user_id or not sdk_code:
            raise ValueError("Kaiwu user_id and sdk_code must not be empty")
        self._user_id = user_id
        self._sdk_code = sdk_code

    def __repr__(self) -> str:
        return "KaiwuCredentials(user_id='[REDACTED]', sdk_code='[REDACTED]')"

    def __reduce__(self) -> NoReturn:
        raise TypeError("KaiwuCredentials cannot be serialized")

    def _pair(self) -> tuple[str, str]:
        return self._user_id, self._sdk_code


def resolve_kaiwu_credentials(
    credentials: KaiwuCredentials | None = None,
) -> tuple[str, str]:
    """Resolve one complete pair from an explicit object or dedicated variables.

    FlagQuantum uses ``QBOSON_USER_ID`` and ``QBOSON_SDK_CODE`` rather than the
    generic names shown in some vendor examples.  Dedicated names avoid
    accidentally consuming unrelated application credentials.
    """

    if credentials is not None:
        if not isinstance(credentials, KaiwuCredentials):
            raise TypeError("credentials must be KaiwuCredentials or None")
        return credentials._pair()

    user_id = os.environ.get("QBOSON_USER_ID")
    sdk_code = os.environ.get("QBOSON_SDK_CODE")
    if user_id is None and sdk_code is None:
        raise RuntimeError(
            "Set both QBOSON_USER_ID and QBOSON_SDK_CODE or pass "
            "KaiwuCredentials explicitly"
        )
    if user_id is None or sdk_code is None:
        raise RuntimeError(
            "Set both QBOSON_USER_ID and QBOSON_SDK_CODE; partial Kaiwu "
            "credentials are not allowed"
        )
    user_id, sdk_code = user_id.strip(), sdk_code.strip()
    if not user_id or not sdk_code:
        raise RuntimeError("Kaiwu environment credentials must not be empty")
    return user_id, sdk_code
