"""Lazy, fail-closed initialization of the proprietary Kaiwu SDK."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

from ._credentials import KaiwuCredentials, resolve_kaiwu_credentials


class KaiwuSDKError(RuntimeError):
    """Base error for the proprietary Kaiwu SDK boundary."""


class KaiwuSDKUnavailableError(KaiwuSDKError):
    """Raised when a compatible Kaiwu SDK cannot be imported."""


class KaiwuSDKVersionError(KaiwuSDKError):
    """Raised when the installed SDK differs from the pinned lane."""


class KaiwuLicenseInitializationError(KaiwuSDKError):
    """Raised without vendor exception text when license initialization fails."""


@dataclass(frozen=True)
class KaiwuSDKEnvironment:
    """Non-secret identity of one initialized Kaiwu SDK environment."""

    sdk_version: str
    python_version: str


def _load_kaiwu_module() -> ModuleType:
    try:
        return import_module("kaiwu")
    except ImportError:
        raise KaiwuSDKUnavailableError(
            "Kaiwu SDK is unavailable; install the approved Python 3.10 wheel"
        ) from None


def _preflight_kaiwu_sdk(*, expected_version: str) -> ModuleType:
    """Return one pinned SDK module after credential-free structural checks."""

    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError("expected_version must be a non-empty string")
    if sys.version_info[:2] != (3, 10):
        raise KaiwuSDKUnavailableError(
            "Kaiwu SDK requires the pinned Python 3.10 compatibility lane"
        )

    module = _load_kaiwu_module()
    installed_version = getattr(module, "__version__", None)
    if not isinstance(installed_version, str) or not installed_version:
        raise KaiwuSDKVersionError("Kaiwu SDK does not expose a usable version")
    if installed_version != expected_version.strip():
        raise KaiwuSDKVersionError(
            f"Kaiwu SDK version mismatch: expected {expected_version.strip()!r}, "
            f"observed {installed_version!r}"
        )

    license_module = getattr(module, "license", None)
    initializer = getattr(license_module, "init", None)
    if not callable(initializer):
        raise KaiwuSDKUnavailableError("Kaiwu SDK does not expose license.init")
    return module


def _initialize_preflighted_kaiwu_license(
    module: ModuleType,
    credentials: KaiwuCredentials | None,
    *,
    expected_version: str,
) -> KaiwuSDKEnvironment:
    """Initialize a module already checked by :func:`_preflight_kaiwu_sdk`."""

    installed_version = getattr(module, "__version__", None)
    license_module = getattr(module, "license", None)
    initializer = getattr(license_module, "init", None)
    if installed_version != expected_version.strip() or not callable(initializer):
        raise KaiwuSDKUnavailableError("Kaiwu SDK changed after local preflight")
    user_id, sdk_code = resolve_kaiwu_credentials(credentials)
    try:
        initializer(user_id=user_id, sdk_code=sdk_code)
    except Exception:
        raise KaiwuLicenseInitializationError(
            "Kaiwu license initialization failed; vendor details were redacted"
        ) from None
    return KaiwuSDKEnvironment(
        sdk_version=installed_version,
        python_version=platform.python_version(),
    )


def initialize_kaiwu_license(
    credentials: KaiwuCredentials | None = None,
    *,
    expected_version: str,
) -> KaiwuSDKEnvironment:
    """Initialize one pinned SDK license without retaining or exposing secrets.

    Current Kaiwu documentation supports Python 3.10 only. Version, runtime,
    and license-entrypoint checks occur before credential resolution and before
    ``license.init``. Vendor exception text is intentionally discarded because
    it may echo the supplied authorization values.
    """

    module = _preflight_kaiwu_sdk(expected_version=expected_version)
    return _initialize_preflighted_kaiwu_license(
        module, credentials, expected_version=expected_version
    )
