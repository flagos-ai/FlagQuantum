from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import flagquantum.remote.kaiwu.sdk as sdk_module
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuLicenseInitializationError,
    KaiwuSDKUnavailableError,
    KaiwuSDKVersionError,
    initialize_kaiwu_license,
)

pytestmark = pytest.mark.unit


def _supported_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk_module.sys, "version_info", (3, 10, 18))


def test_runtime_fails_before_import_or_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = Mock(side_effect=AssertionError("SDK import must not occur"))
    resolver = Mock(side_effect=AssertionError("credentials must not be resolved"))
    monkeypatch.setattr(sdk_module, "import_module", importer)
    monkeypatch.setattr(sdk_module, "resolve_kaiwu_credentials", resolver)
    monkeypatch.setattr(sdk_module.sys, "version_info", (3, 12, 0))

    with pytest.raises(KaiwuSDKUnavailableError, match="Python 3.10"):
        initialize_kaiwu_license(expected_version="1.3.1")

    importer.assert_not_called()
    resolver.assert_not_called()


def test_missing_sdk_has_owned_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _supported_python(monkeypatch)
    monkeypatch.setattr(
        sdk_module, "import_module", Mock(side_effect=ImportError("vendor path"))
    )

    with pytest.raises(KaiwuSDKUnavailableError, match="approved Python 3.10 wheel"):
        initialize_kaiwu_license(expected_version="1.3.1")


def test_version_mismatch_fails_before_credentials_or_license(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _supported_python(monkeypatch)
    initializer = Mock()
    module = SimpleNamespace(
        __version__="1.4.1", license=SimpleNamespace(init=initializer)
    )
    resolver = Mock(side_effect=AssertionError("credentials must not be resolved"))
    monkeypatch.setattr(sdk_module, "import_module", Mock(return_value=module))
    monkeypatch.setattr(sdk_module, "resolve_kaiwu_credentials", resolver)

    with pytest.raises(KaiwuSDKVersionError, match="version mismatch"):
        initialize_kaiwu_license(expected_version="1.3.1")

    initializer.assert_not_called()
    resolver.assert_not_called()


def test_license_initialization_uses_explicit_pair_and_returns_only_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _supported_python(monkeypatch)
    initializer = Mock()
    module = SimpleNamespace(
        __version__="1.3.1", license=SimpleNamespace(init=initializer)
    )
    monkeypatch.setattr(sdk_module, "import_module", Mock(return_value=module))
    credentials = KaiwuCredentials(user_id="user-secret", sdk_code="code-secret")

    environment = initialize_kaiwu_license(
        credentials,
        expected_version="1.3.1",
    )

    initializer.assert_called_once_with(user_id="user-secret", sdk_code="code-secret")
    assert environment.sdk_version == "1.3.1"
    assert environment.python_version
    assert "user-secret" not in repr(environment)
    assert "code-secret" not in repr(environment)


def test_vendor_failure_discards_exception_text_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _supported_python(monkeypatch)
    initializer = Mock(
        side_effect=RuntimeError("failed for user-secret using code-secret")
    )
    module = SimpleNamespace(
        __version__="1.3.1", license=SimpleNamespace(init=initializer)
    )
    monkeypatch.setattr(sdk_module, "import_module", Mock(return_value=module))
    credentials = KaiwuCredentials(user_id="user-secret", sdk_code="code-secret")

    with pytest.raises(KaiwuLicenseInitializationError) as caught:
        initialize_kaiwu_license(credentials, expected_version="1.3.1")

    assert "user-secret" not in str(caught.value)
    assert "code-secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_missing_license_initializer_fails_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _supported_python(monkeypatch)
    resolver = Mock(side_effect=AssertionError("credentials must not be resolved"))
    monkeypatch.setattr(
        sdk_module,
        "import_module",
        Mock(return_value=SimpleNamespace(__version__="1.3.1")),
    )
    monkeypatch.setattr(sdk_module, "resolve_kaiwu_credentials", resolver)

    with pytest.raises(KaiwuSDKUnavailableError, match="license.init"):
        initialize_kaiwu_license(expected_version="1.3.1")

    resolver.assert_not_called()


@pytest.mark.parametrize(
    "module",
    (
        SimpleNamespace(
            __version__="1.4.1",
            license=SimpleNamespace(init=lambda **kwargs: None),
        ),
        SimpleNamespace(__version__="1.3.1"),
    ),
)
def test_preflighted_module_change_fails_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
    module: SimpleNamespace,
) -> None:
    resolver = Mock(side_effect=AssertionError("credentials must not be resolved"))
    monkeypatch.setattr(sdk_module, "resolve_kaiwu_credentials", resolver)

    with pytest.raises(KaiwuSDKUnavailableError, match="changed after local preflight"):
        sdk_module._initialize_preflighted_kaiwu_license(
            module,
            None,
            expected_version="1.3.1",
        )

    resolver.assert_not_called()
