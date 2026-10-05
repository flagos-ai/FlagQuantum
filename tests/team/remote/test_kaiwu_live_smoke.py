from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import qboson_live_smoke as smoke_module
from examples.qdiffusion_kaiwu.qboson_live_smoke import (
    _write_private_json,
    run_live_smoke,
)
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytestmark = pytest.mark.unit


class _CompletedClient:
    def __init__(self, *, expose_provider_identity: bool) -> None:
        self.expose_provider_identity = expose_provider_identity
        self.submissions = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submissions += 1
        suffix = str(self.submissions) if self.expose_provider_identity else None
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"provider-{suffix}" if suffix else None,
            provider_target="SPQC-test" if suffix else None,
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = (1, -1)
        count = receipt.requested_samples
        samples = tuple(sample for _ in range(count))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt=receipt,
            samples=samples,
            energies=tuple(energy for _ in samples),
            raw_status="Completed",
            metadata={
                "fallback_occurred": False,
                "provider_result_schema": {
                    "available": False,
                    "reason": "test_client",
                },
            },
        )


def test_live_smoke_runs_both_modes_without_overclaiming() -> None:
    client = _CompletedClient(expose_provider_identity=False)

    record = run_live_smoke(
        client=client,
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="a" * 64,
    )

    assert client.submissions == 2
    assert [task["task_mode"] for task in record["tasks"]] == [
        "optimization",
        "sampling",
    ]
    assert record["live_provider_smoke_passed"] is True
    assert record["provider_identity_complete"] is False
    assert record["hardware_acceptance"] is False
    assert record["transport"] == "injected_test"
    assert record["real_provider_evidence"] is False
    assert record["qboson_hardware_used"] is False
    assert record["fallback_occurred"] is False
    assert record["environment_lock_sha256"] == "a" * 64
    assert all(
        task["provider_result_schema"] == {"available": False, "reason": "test_client"}
        for task in record["tasks"]
    )


def test_injected_live_smoke_cannot_claim_hardware_with_complete_identity() -> None:
    record = run_live_smoke(
        client=_CompletedClient(expose_provider_identity=True),
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="b" * 64,
    )

    assert record["provider_identity_complete"] is True
    assert record["live_provider_smoke_passed"] is True
    assert record["hardware_acceptance"] is False


def test_live_smoke_requires_exact_sdk_client_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_module, "KaiwuSDKClient", _CompletedClient)
    record = run_live_smoke(
        client=_CompletedClient(expose_provider_identity=True),
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="c" * 64,
    )

    assert record["transport"] == "kaiwu_cim"
    assert record["real_provider_evidence"] is True
    assert record["qboson_hardware_used"] is True
    assert record["hardware_acceptance"] is True


def test_live_smoke_rejects_invalid_environment_lock_digest() -> None:
    with pytest.raises(ValueError, match="environment_lock_sha256"):
        run_live_smoke(
            client=_CompletedClient(expose_provider_identity=True),
            task_prefix="smoke",
            project_no="CPQC-test",
            timeout=1.0,
            poll_interval=0.01,
            environment_lock_sha256="not-a-digest",
        )


def test_private_record_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    payload = {"secret": "not-a-real-credential"}

    _write_private_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(path, payload)


@pytest.mark.parametrize(
    ("credential", "payload"),
    (
        (
            "sdk-code-secret",
            {"provider_result_schema": {"fields": ["sdk-code-secret"]}},
        ),
        (
            'sdk"code\\secret\nline',
            {"nested": [{"failure": 'sdk"code\\secret\nline'}]},
        ),
        ("user-id-secret", {"provider-user-id-secret": True}),
    ),
)
def test_private_record_rejects_credentials_before_file_creation(
    tmp_path: Path, credential: str, payload: dict[str, object]
) -> None:
    path = tmp_path / "smoke.json"

    with pytest.raises(RuntimeError, match="credentials"):
        _write_private_json(
            path,
            payload,
            forbidden_values=("unrelated-secret", credential),
        )

    assert not path.exists()


def test_live_smoke_verifies_environment_before_client_initialization() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qboson_live_smoke.py"
    ).read_text(encoding="utf-8")

    assert "--environment-lock" in source
    assert source.index("verify_environment_lock(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("resolve_kaiwu_credentials()") < source.index(
        "client = KaiwuSDKClient("
    )
    assert "type(client) is KaiwuSDKClient" in source
    assert 'if not payload["hardware_acceptance"]:' in source
    assert "raise SystemExit(1)" in source


def test_live_smoke_cli_writes_diagnostic_then_exits_nonzero_when_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _CLIClient(_CompletedClient):
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            super().__init__(expose_provider_identity=False)

    output = tmp_path / "smoke.json"
    monkeypatch.setattr(smoke_module, "KaiwuSDKClient", _CLIClient)
    monkeypatch.setattr(
        smoke_module,
        "verify_environment_lock",
        lambda path: ({}, "d" * 64),
    )
    monkeypatch.setattr(
        smoke_module,
        "resolve_kaiwu_credentials",
        lambda: ("test-user", "test-sdk-code"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qboson_live_smoke",
            "--checkpoint-dir",
            str(tmp_path),
            "--environment-lock",
            str(tmp_path / "lock.json"),
            "--output",
            str(output),
            "--project-no",
            "CPQC-test",
            "--task-prefix",
            "smoke",
            "--acknowledge-provider-cost",
            smoke_module.ACKNOWLEDGEMENT,
        ],
    )

    with pytest.raises(SystemExit) as raised:
        smoke_module.main()

    assert raised.value.code == 1
    assert (
        json.loads(output.read_text(encoding="utf-8"))["hardware_acceptance"] is False
    )
