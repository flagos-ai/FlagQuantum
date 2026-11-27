"""Credential discovery for Jiuding control-plane requests."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import NoReturn


class JiudingCredentials:
    """One in-memory Jiuding credential pair with a redacted representation.

    Use this object when credentials belong to a notebook or application session.
    FlagQuantum does not include it in remote-job receipts.
    """

    __slots__ = ("_access_key", "_secret_key")

    def __init__(self, *, access_key: str, secret_key: str) -> None:
        if not isinstance(access_key, str) or not isinstance(secret_key, str):
            raise TypeError("Jiuding access_key and secret_key must be strings")
        access_key, secret_key = access_key.strip(), secret_key.strip()
        if not access_key or not secret_key:
            raise ValueError("Jiuding access_key and secret_key must not be empty")
        self._access_key = access_key
        self._secret_key = secret_key

    def __repr__(self) -> str:
        return "JiudingCredentials(access_key='[REDACTED]', secret_key='[REDACTED]')"

    def __reduce__(self) -> NoReturn:
        raise TypeError("JiudingCredentials cannot be serialized")

    def _pair(self) -> tuple[str, str]:
        return self._access_key, self._secret_key


def resolve_jiuding_credentials(
    credentials: JiudingCredentials | None,
) -> tuple[str, str]:
    """Resolve explicit credentials or fall back to established discovery."""
    if credentials is not None:
        if not isinstance(credentials, JiudingCredentials):
            raise TypeError("credentials must be JiudingCredentials or None")
        return credentials._pair()
    return load_jiuding_credentials()


def load_jiuding_credentials() -> tuple[str, str]:
    """Load one complete credential pair without mixing sources."""
    ak = os.environ.get("JIUDING_AK")
    sk = os.environ.get("JIUDING_SK")
    if ak is not None or sk is not None:
        if ak is None or sk is None:
            raise RuntimeError(
                "Set both JIUDING_AK and JIUDING_SK; partial environment "
                "credentials are not allowed"
            )
    else:
        try:
            ak = _decode(Path("/etc/accesskey/user-ak"))
            sk = _decode(Path("/etc/accesskey/user-sk"))
        except (OSError, ValueError):
            raise RuntimeError(
                "Set both JIUDING_AK and JIUDING_SK, or use a Jiuding "
                "workspace with both injected credential files"
            ) from None
    ak, sk = ak.strip(), sk.strip()
    if not ak or not sk:
        raise RuntimeError("Jiuding credentials must not be empty")
    return ak, sk


def _decode(path: Path) -> str:
    encoded = b"".join(path.read_bytes().split())
    return base64.b64decode(encoded, validate=True).decode()
