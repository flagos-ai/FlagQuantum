from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flagquantum.errors import ValidationError
from flagquantum.remote.qpu import QuafuProvider, _quafu_sdk

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


class _ReadOnlyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def devices(self) -> list[dict[str, object]]:
        self.calls.append(("devices", None))
        return [
            {
                "name": "Baihua",
                "kind": "real",
                "status": "online",
                "n_qubits": 156,
                "queue": 4,
                "limits": {"max_shots": 8192},
            },
            {
                "name": "sim",
                "kind": "sim",
                "status": "online",
                "n_qubits": 24,
                "queue": None,
                "limits": {"max_shots": 8192},
            },
        ]

    def metadata(self, target: str) -> dict[str, object]:
        self.calls.append(("metadata", target))
        width = 156 if target == "Baihua" else 24
        return {
            "n_qubits": width,
            "basis_gates": ["h", "rx", "ry", "rz", "cz"],
            "limits": {"max_qubits": width, "max_shots": 8192},
            "calibration_id": "cal-1" if target == "Baihua" else None,
        }

    def device_calibration(self, device: str) -> SimpleNamespace:
        self.calls.append(("device_calibration", device))
        return SimpleNamespace(raw=_calibration())

    def calibration(self, calibration_id: str) -> SimpleNamespace:
        self.calls.append(("calibration", calibration_id))
        return SimpleNamespace(raw=_calibration())


def _calibration() -> dict[str, object]:
    return {
        "id": "cal-1",
        "chip": "Baihua",
        "calibrated_at": "2026-10-25T10:00:00+08:00",
        "qubits": [{"index": 0, "t1_us": 70.5}],
        "couplers": [],
    }


def test_sdk_discovery_normalizes_profiles_without_native_objects() -> None:
    client = _ReadOnlyClient()
    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=client)
    provider = QuafuProvider(_sdk_adapter=adapter)

    devices = provider.list_devices()

    assert [(item.name, item.n_qubits) for item in devices] == [
        ("Baihua", 156),
        ("sim", 24),
    ]
    assert devices[0].basis_gates == ("h", "rx", "ry", "rz", "cz")
    assert devices[0].is_simulator is False
    assert devices[1].is_simulator is True
    assert devices[0].metadata == {
        "source": "quafu-sdk",
        "kind": "real",
        "limits": {"max_qubits": 156, "max_shots": 8192},
        "status": "online",
        "queue": 4,
        "calibration_id": "cal-1",
    }
    assert client.calls == [
        ("devices", None),
        ("metadata", "Baihua"),
        ("metadata", "sim"),
    ]


def test_default_read_path_builds_credential_free_sdk_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ReadOnlyClient()
    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=client)
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> _quafu_sdk._QuafuSdkAdapter:
        calls.append(kwargs)
        return adapter

    monkeypatch.setattr(_quafu_sdk._QuafuSdkAdapter, "for_discovery", factory)
    provider = QuafuProvider(task_server_url="https://quafu.test/api/v1")

    devices = provider.list_devices()

    assert devices[0].name == "Baihua"
    assert calls == [{"base_url": "https://quafu.test"}]


def test_sdk_calibration_is_copied_and_identity_checked() -> None:
    client = _ReadOnlyClient()
    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=client)
    provider = QuafuProvider(_sdk_adapter=adapter)

    current = provider.fetch_calibration("Baihua")
    historical = provider.fetch_calibration("Baihua", calibration_id="cal-1")

    assert current == _calibration()
    assert historical == _calibration()
    assert isinstance(current, dict)
    assert client.calls == [
        ("device_calibration", "Baihua"),
        ("calibration", "cal-1"),
    ]


def test_sdk_discovery_rejects_missing_authoritative_width() -> None:
    class MissingWidthClient(_ReadOnlyClient):
        def devices(self) -> list[dict[str, object]]:
            return [{"name": "Baihua", "kind": "real"}]

        def metadata(self, target: str) -> dict[str, object]:
            return {"basis_gates": ["cz"], "limits": {}}

    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=MissingWidthClient())
    provider = QuafuProvider(_sdk_adapter=adapter)

    with pytest.raises(RuntimeError, match="has no positive n_qubits"):
        provider.list_devices(n_qubits=2)


def test_sdk_boundary_rejects_non_plain_payloads() -> None:
    class NativeValue:
        pass

    class InvalidClient(_ReadOnlyClient):
        def devices(self) -> list[dict[str, object]]:
            return [{"name": "Baihua", "native": NativeValue()}]

    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=InvalidClient())

    with pytest.raises(RuntimeError, match="unsupported NativeValue data"):
        adapter.devices()
