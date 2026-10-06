from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from examples.qdiffusion_kaiwu import (
    qdiffusion_portability_replay_live as portability_module,
)
from examples.qdiffusion_kaiwu.qdiffusion_portability_replay_live import (
    _verified_checkpoint,
    run_portability_replay,
)
from examples.qdiffusion_kaiwu.stable_source_tree import capture_regular_file
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytestmark = pytest.mark.unit


class _CompletedClient:
    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id="fake-task",
            provider_target="SPQC-fake",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = tuple(1 for _ in matrix)
        samples = tuple(sample for _ in range(receipt.requested_samples))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt,
            samples,
            tuple(energy for _ in samples),
            "Completed",
            {"fallback_occurred": False},
        )


class _Generator:
    def __init__(self, sampler: Any) -> None:
        self.energy_model = SimpleNamespace(sampler=sampler)
        self.sampler = sampler
        self.tokenizer = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["ACDE"])

    def eval(self) -> _Generator:
        return self

    def to(self, device: torch.device) -> _Generator:
        del device
        return self

    def objective(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del batch
        self.sampler.solve(
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        return {"energy_objective": torch.tensor([0.25])}

    def generate(self, target: torch.Tensor, *, max_steps: int) -> torch.Tensor:
        del max_steps
        self.sampler.solve(
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        return target


class _FailingGenerator(_Generator):
    def generate(self, target: torch.Tensor, *, max_steps: int) -> torch.Tensor:
        del target, max_steps
        raise RuntimeError("local failure after provider use")


def _config() -> dict[str, Any]:
    return {
        "requested_samples": 10,
        "remote_call_budget": 20,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
        "training": {"freeze_proposal": True},
        "dataset": {"max_length": 256, "sha256": "1" * 64},
        "checkpoint": {"sha256": "2" * 64},
        "tokenizer": {"sha256": "3" * 64},
        "evaluation_model": {"sha256": "4" * 64},
        "generation": {
            "sequence_count": 1,
            "num_candidates": 4,
            "proposal_temperature": 0.3,
            "proposal_noise_scale": 1.0,
            "energy_temperature": 1.25,
            "disable_resample": False,
            "resample_ratio": 0.2,
            "resample_top_p": 0.9,
            "portability_training_seed": 1701,
            "portability_fixture_index": 0,
            "portability_steps": 3,
        },
    }


def test_verified_checkpoint_requires_primary_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "best_epoch_3.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trained")
    checkpoint.chmod(0o600)
    digest = hashlib.sha256(b"trained").hexdigest()
    record = {
        "trained_energy_checkpoint_name": checkpoint.name,
        "trained_energy_checkpoint_sha256": digest,
    }

    assert _verified_checkpoint(tmp_path, checkpoint, record) == digest

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest differs"):
        _verified_checkpoint(tmp_path, checkpoint, record)


def test_verified_checkpoint_rejects_public_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "best_epoch_3.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trained")
    checkpoint.chmod(0o644)
    record = {
        "trained_energy_checkpoint_name": checkpoint.name,
        "trained_energy_checkpoint_sha256": hashlib.sha256(b"trained").hexdigest(),
    }

    with pytest.raises(ValueError, match="must be owner-only"):
        _verified_checkpoint(tmp_path, checkpoint, record)


def test_portability_replay_runs_bounded_slice_without_false_acceptance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained")
    builder = SimpleNamespace(
        build_qdiffusion=lambda **kwargs: _Generator(kwargs["bm_sampler"])
    )
    runtime = SimpleNamespace(
        load_trained_energy_weights=lambda *args: {},
        seed_torch=lambda seed: None,
        encode_sequence=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
    )
    io_module = SimpleNamespace(read_fasta_records=lambda path: [("protein", "ACDE")])

    record = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=_CompletedClient(),
        config=_config(),
        config_sha256="a" * 64,
        artifact_preflight_sha256="e" * 64,
        training_record_sha256="b" * 64,
        test_fasta=tmp_path / "test.fasta",
        base_checkpoint=tmp_path / "dplm",
        trained_checkpoint=checkpoint,
        trained_checkpoint_sha256=hashlib.sha256(b"trained").hexdigest(),
        execution_host="jp-a800-172",
        observed_hostname="host-172",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="c" * 40,
        plugin_revision="d" * 40,
        source_preflight_sha256="f" * 64,
        transfer_manifest_sha256="0" * 64,
        environment_lock_sha256="1" * 64,
        sdk_version="1.3.1",
        project_no="project",
        task_prefix="replay",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        device=torch.device("cpu"),
        real_provider_transport=False,
    )

    assert record["run_completed"] is True
    assert record["remote_call_count"] == 1
    assert record["fixture"]["token_constraints_passed"] is True
    assert (
        record["trained_energy_checkpoint_sha256"]
        == hashlib.sha256(b"trained").hexdigest()
    )
    assert record["transport"] == "injected_test"
    assert record["pinned_sdk_client"] is False
    assert record["source_preflight_sha256"] == "f" * 64
    assert record["transfer_manifest_sha256"] == "0" * 64
    assert record["environment_lock_sha256"] == "1" * 64
    assert record["acceptance"]["portability"] == "fail"


def test_portability_replay_rejects_checkpoint_change_during_weight_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained")
    builder = SimpleNamespace(
        build_qdiffusion=lambda **kwargs: _Generator(kwargs["bm_sampler"])
    )

    def mutate_checkpoint(*args: object) -> None:
        del args
        checkpoint.write_bytes(b"changed")

    runtime = SimpleNamespace(
        load_trained_energy_weights=mutate_checkpoint,
        seed_torch=lambda seed: None,
        encode_sequence=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
    )
    io_module = SimpleNamespace(read_fasta_records=lambda path: [("protein", "ACDE")])

    record = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=_CompletedClient(),
        config=_config(),
        config_sha256="a" * 64,
        artifact_preflight_sha256="e" * 64,
        training_record_sha256="b" * 64,
        test_fasta=tmp_path / "test.fasta",
        base_checkpoint=tmp_path / "dplm",
        trained_checkpoint=checkpoint,
        trained_checkpoint_sha256=hashlib.sha256(b"trained").hexdigest(),
        execution_host="jp-a800-172",
        observed_hostname="host-172",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="c" * 40,
        plugin_revision="d" * 40,
        source_preflight_sha256="f" * 64,
        transfer_manifest_sha256="0" * 64,
        environment_lock_sha256="1" * 64,
        sdk_version="1.3.1",
        project_no="project",
        task_prefix="replay-checkpoint-change",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        device=torch.device("cpu"),
        real_provider_transport=False,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "ValueError"
    assert record["remote_call_count"] == 0
    assert record["acceptance"]["portability"] == "fail"


def test_portability_replay_rejects_frozen_input_change_during_build(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained")
    frozen_paths = {
        "dataset": tmp_path / "dataset.fasta",
        "base_checkpoint": tmp_path / "base.pt",
        "tokenizer": tmp_path / "tokenizer.json",
        "evaluation_model": tmp_path / "esm2.pt",
    }
    for name, path in frozen_paths.items():
        path.write_bytes(name.encode())
    artifact_snapshots = {
        name: capture_regular_file(path, label=name)
        for name, path in frozen_paths.items()
    }

    def build_and_mutate(**kwargs: object) -> _Generator:
        frozen_paths["base_checkpoint"].write_bytes(b"changed")
        return _Generator(kwargs["bm_sampler"])

    builder = SimpleNamespace(build_qdiffusion=build_and_mutate)
    runtime = SimpleNamespace(
        load_trained_energy_weights=lambda *args: {},
        seed_torch=lambda seed: None,
        encode_sequence=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
    )
    io_module = SimpleNamespace(
        read_fasta_records=lambda path: [("protein", "ACDE")]
    )

    record = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=_CompletedClient(),
        config=_config(),
        config_sha256="a" * 64,
        artifact_preflight_sha256="e" * 64,
        training_record_sha256="b" * 64,
        test_fasta=tmp_path / "test.fasta",
        base_checkpoint=frozen_paths["base_checkpoint"],
        trained_checkpoint=checkpoint,
        trained_checkpoint_sha256=hashlib.sha256(b"trained").hexdigest(),
        execution_host="jp-a800-172",
        observed_hostname="host-172",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="c" * 40,
        plugin_revision="d" * 40,
        source_preflight_sha256="f" * 64,
        transfer_manifest_sha256="0" * 64,
        environment_lock_sha256="1" * 64,
        sdk_version="1.3.1",
        project_no="project",
        task_prefix="replay-input-change",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        device=torch.device("cpu"),
        real_provider_transport=False,
        artifact_snapshots=artifact_snapshots,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "ValueError"
    assert record["remote_call_count"] == 0
    assert record["acceptance"]["portability"] == "fail"


def test_portability_replay_sdk_subclass_cannot_claim_real_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _InjectedSubclass(_CompletedClient):
        pass

    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained")
    builder = SimpleNamespace(
        build_qdiffusion=lambda **kwargs: _Generator(kwargs["bm_sampler"])
    )
    runtime = SimpleNamespace(
        load_trained_energy_weights=lambda *args: {},
        seed_torch=lambda seed: None,
        encode_sequence=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
    )
    io_module = SimpleNamespace(read_fasta_records=lambda path: [("protein", "ACDE")])
    monkeypatch.setattr(portability_module, "KaiwuSDKClient", _CompletedClient)

    record = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=_InjectedSubclass(),
        config=_config(),
        config_sha256="a" * 64,
        artifact_preflight_sha256="e" * 64,
        training_record_sha256="b" * 64,
        test_fasta=tmp_path / "test.fasta",
        base_checkpoint=tmp_path / "dplm",
        trained_checkpoint=checkpoint,
        trained_checkpoint_sha256=hashlib.sha256(b"trained").hexdigest(),
        execution_host="jp-a800-172",
        observed_hostname="host-172",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="c" * 40,
        plugin_revision="d" * 40,
        source_preflight_sha256="f" * 64,
        transfer_manifest_sha256="0" * 64,
        environment_lock_sha256="1" * 64,
        sdk_version="1.3.1",
        project_no="project",
        task_prefix="replay-subclass",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        device=torch.device("cpu"),
        real_provider_transport=True,
    )

    assert record["run_completed"] is True
    assert record["transport"] == "injected_test"
    assert record["pinned_sdk_client"] is False
    assert record["qboson_hardware_used"] is False
    assert record["real_provider_evidence"] is False
    assert record["acceptance"]["portability"] == "fail"


def test_portability_replay_preserves_proven_provider_use_after_local_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained")
    builder = SimpleNamespace(
        build_qdiffusion=lambda **kwargs: _FailingGenerator(kwargs["bm_sampler"])
    )
    runtime = SimpleNamespace(
        load_trained_energy_weights=lambda *args: {},
        seed_torch=lambda seed: None,
        encode_sequence=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
    )
    io_module = SimpleNamespace(read_fasta_records=lambda path: [("protein", "ACDE")])
    monkeypatch.setattr(portability_module, "KaiwuSDKClient", _CompletedClient)

    record = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=_CompletedClient(),
        config=_config(),
        config_sha256="a" * 64,
        artifact_preflight_sha256="e" * 64,
        training_record_sha256="b" * 64,
        test_fasta=tmp_path / "test.fasta",
        base_checkpoint=tmp_path / "dplm",
        trained_checkpoint=checkpoint,
        trained_checkpoint_sha256=hashlib.sha256(b"trained").hexdigest(),
        execution_host="jp-a800-172",
        observed_hostname="host-172",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="c" * 40,
        plugin_revision="d" * 40,
        source_preflight_sha256="f" * 64,
        transfer_manifest_sha256="0" * 64,
        environment_lock_sha256="1" * 64,
        sdk_version="1.3.1",
        project_no="project",
        task_prefix="replay-failed-after-provider-use",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        device=torch.device("cpu"),
        real_provider_transport=True,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "RuntimeError"
    assert record["remote_call_count"] == 1
    assert record["pinned_sdk_client"] is True
    assert record["qboson_hardware_used"] is True
    assert record["real_provider_evidence"] is True
    assert record["acceptance"]["portability"] == "fail"


def test_portability_replay_receipt_only_timeout_does_not_claim_provider_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _PendingClient(_CompletedClient):
        def query_status(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> str:
            del receipt, matrix
            return "Pending"

    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained")
    builder = SimpleNamespace(
        build_qdiffusion=lambda **kwargs: _Generator(kwargs["bm_sampler"])
    )
    runtime = SimpleNamespace(
        load_trained_energy_weights=lambda *args: {},
        seed_torch=lambda seed: None,
        encode_sequence=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
    )
    io_module = SimpleNamespace(read_fasta_records=lambda path: [("protein", "ACDE")])
    monkeypatch.setattr(portability_module, "KaiwuSDKClient", _PendingClient)

    record = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=_PendingClient(),
        config=_config(),
        config_sha256="a" * 64,
        artifact_preflight_sha256="e" * 64,
        training_record_sha256="b" * 64,
        test_fasta=tmp_path / "test.fasta",
        base_checkpoint=tmp_path / "dplm",
        trained_checkpoint=checkpoint,
        trained_checkpoint_sha256=hashlib.sha256(b"trained").hexdigest(),
        execution_host="jp-a800-172",
        observed_hostname="host-172",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="c" * 40,
        plugin_revision="d" * 40,
        source_preflight_sha256="f" * 64,
        transfer_manifest_sha256="0" * 64,
        environment_lock_sha256="1" * 64,
        sdk_version="1.3.1",
        project_no="project",
        task_prefix="replay-receipt-only-timeout",
        requested_samples=10,
        timeout=0.0,
        poll_interval=0.01,
        device=torch.device("cpu"),
        real_provider_transport=True,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "TimeoutError"
    assert record["remote_call_count"] == 1
    assert record["provider_identity_complete"] is True
    assert record["returned_samples"] is None
    assert record["qboson_hardware_used"] is False
    assert record["real_provider_evidence"] is False


def test_replay_source_preflights_before_credentials_and_requires_cost_ack() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_portability_replay_live.py"
    ).read_text(encoding="utf-8")

    assert "ACKNOWLEDGEMENT" in source
    assert source.index("_load_training_record(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("preflight_artifacts_with_snapshots(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index('artifact_preflight.get("config_sha256")') < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("load_source_preflight(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("verify_frozen_environment_lock(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("verify_approved_kaiwu_distribution(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index(
        "validate_private_json_output_path(args.artifact_preflight_output)"
    ) < source.index("resolve_kaiwu_credentials()")
    assert source.index("validate_private_json_output_path(args.output)") < (
        source.index("resolve_kaiwu_credentials()")
    )
    assert source.index("validate_private_directory(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index('os.environ.pop("QBOSON_USER_ID", None)') < source.index(
        "client = KaiwuSDKClient("
    )
    assert source.index('os.environ.pop("QBOSON_SDK_CODE", None)') < source.index(
        "client = KaiwuSDKClient("
    )
    assert 'role != "portability_replay"' in source
    assert "not a second training run" in source
