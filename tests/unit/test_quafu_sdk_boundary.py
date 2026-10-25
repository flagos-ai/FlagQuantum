from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flagquantum.errors import ValidationError
from flagquantum.remote.qpu import _quafu_sdk

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_quafu_sdk_is_optional_and_lazily_imported() -> None:
    source = (ROOT / "flagquantum/remote/qpu/_quafu_sdk.py").read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert "quafu" not in imported


def test_importing_flagquantum_does_not_import_quafu_sdk() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import flagquantum, sys; assert 'quafu' not in sys.modules",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_injected_client_needs_no_sdk_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object()
    monkeypatch.setattr(
        _quafu_sdk,
        "_load_quafu_sdk",
        lambda: pytest.fail("injected clients must not load quafu-sdk"),
    )

    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=client)

    assert adapter._client is client


def test_submission_refuses_missing_key_before_loading_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.setattr(
        _quafu_sdk,
        "_load_quafu_sdk",
        lambda: pytest.fail("credential validation must run before SDK loading"),
    )

    with pytest.raises(ValidationError, match="does not start interactive SDK login"):
        _quafu_sdk._QuafuSdkAdapter.for_submission()


def test_submission_passes_explicit_key_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    client = object()

    def client_factory(**kwargs: object) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(
        _quafu_sdk,
        "_load_quafu_sdk",
        lambda: SimpleNamespace(Client=client_factory),
    )

    adapter = _quafu_sdk._QuafuSdkAdapter.for_submission(
        api_key="qf_test_key",
        base_url="https://example.invalid",
    )

    assert adapter._client is client
    assert calls == [
        {
            "api_key": "qf_test_key",
            "base_url": "https://example.invalid",
        }
    ]


def test_injected_submission_client_cannot_mix_configuration() -> None:
    with pytest.raises(TypeError, match="cannot be combined"):
        _quafu_sdk._QuafuSdkAdapter.for_submission(
            client=object(), api_key="qf_test_key"
        )


def test_loader_reports_namespace_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    class ConflictError(RuntimeError):
        pass

    monkeypatch.setattr(_quafu_sdk.metadata, "version", lambda _: "0.2.7")
    monkeypatch.setattr(
        _quafu_sdk.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(ConflictError()),
    )

    with pytest.raises(RuntimeError, match="cannot share an environment with pyquafu"):
        _quafu_sdk._load_quafu_sdk()
