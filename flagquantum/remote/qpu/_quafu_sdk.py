"""Private loading and credential boundary for the maintained Quafu SDK."""

from __future__ import annotations

import importlib
import os
from importlib import metadata
from types import ModuleType
from typing import Any

from ...errors import ValidationError

_DISTRIBUTION = "quafu-sdk"
_EXTRA = "quafu"


def _load_quafu_sdk() -> ModuleType:
    try:
        metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        raise RuntimeError(
            "Quafu execution requires the optional quafu-sdk package; "
            "install FlagQuantum with the 'quafu' extra."
        ) from None

    try:
        sdk = importlib.import_module("quafu")
    except Exception as exc:
        if type(exc).__name__ == "ConflictError":
            raise RuntimeError(
                "quafu-sdk cannot share an environment with pyquafu because "
                "both install the 'quafu' package; use a clean environment."
            ) from None
        raise
    if not callable(getattr(sdk, "Client", None)):
        raise RuntimeError(
            "The installed 'quafu' package is not the supported quafu-sdk; "
            "use a clean environment and install FlagQuantum with the "
            f"'{_EXTRA}' extra."
        )
    return sdk


def _submission_api_key(explicit: str | None) -> str:
    value = explicit if explicit is not None else os.getenv("QUAFU_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "Quafu submission requires api_key=... or QUAFU_API_KEY; "
            "FlagQuantum does not start interactive SDK login."
        )
    return value.strip()


class _QuafuSdkAdapter:
    """Keep SDK clients and their native objects inside the Remote boundary."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def for_discovery(
        cls,
        *,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> _QuafuSdkAdapter:
        if client is not None:
            return cls(client)
        sdk = _load_quafu_sdk()
        return cls(sdk.Client(base_url=base_url))

    @classmethod
    def for_submission(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> _QuafuSdkAdapter:
        if client is not None:
            if api_key is not None or base_url is not None:
                raise TypeError(
                    "An injected Quafu SDK client cannot be combined with "
                    "api_key or base_url"
                )
            return cls(client)
        key = _submission_api_key(api_key)
        sdk = _load_quafu_sdk()
        return cls(sdk.Client(api_key=key, base_url=base_url))


__all__: tuple[str, ...] = ()
