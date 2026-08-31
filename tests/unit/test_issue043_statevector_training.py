"""Single-rank training, optimizer ownership and recovery contracts."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd
from flagquantum.runtime.backends.statevector.training import (
    DistributedTrainingError,
    _validate_checkpoint_generations,
    train_distributed_statevector,
)

pytestmark = pytest.mark.unit


def _circuit(theta_value=0.43, phi_value=-0.21):
    theta = torch.tensor(theta_value, requires_grad=True)
    phi = torch.tensor(phi_value, requires_grad=True)
    circuit = fq.Circuit(2).ry(0, theta).rxx(0, 1, phi).rz(1, theta)
    return circuit, (theta, phi)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_multi_step_training_matches_ordinary_local_torch_optimizer(optimizer):
    circuit, parameters = _circuit()
    reference, reference_parameters = _circuit()
    optimizer_cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.Adam
    reference_optimizer = optimizer_cls(reference_parameters, lr=0.03)
    expected_losses = []
    for _ in range(4):
        reference_optimizer.zero_grad()
        state = reference.state(refresh=True)
        probabilities = state.abs().square()
        loss = (probabilities[:, :2].sum() - probabilities[:, 2:].sum()).sum()
        loss.backward()
        reference_optimizer.step()
        expected_losses.append(float(loss.detach()))

    result = fqxd.train_distributed_statevector(
        circuit, steps=4, optimizer=optimizer, lr=0.03
    )
    assert result.losses == pytest.approx(expected_losses, abs=3e-5)
    for actual, expected in zip(parameters, reference_parameters):
        assert float(actual.detach()) == pytest.approx(
            float(expected.detach()), abs=3e-5
        )
    assert all(item.optimizer_state_local for item in result.ownership)
    assert (
        result.summary()["optimizer_state_ownership_semantics"]
        == "single_device_fast_path"
    )


def test_resume_rejects_changed_workload_contract(tmp_path: Path):
    circuit, _ = _circuit()
    train_distributed_statevector(
        circuit, steps=1, optimizer="adam", lr=0.02, checkpoint_dir=tmp_path
    )
    changed, _ = _circuit(theta_value=0.44)
    with pytest.raises(DistributedTrainingError, match="checkpoint_contract_mismatch"):
        train_distributed_statevector(
            changed,
            steps=2,
            optimizer="adam",
            lr=0.02,
            checkpoint_dir=tmp_path,
            resume=True,
        )


def test_resume_rejects_checkpoint_without_integrity_metadata(tmp_path: Path):
    circuit, _ = _circuit()
    train_distributed_statevector(circuit, steps=1, checkpoint_dir=tmp_path)
    (tmp_path / "rank-0.pt.sha256").unlink()
    resumed, _ = _circuit()
    with pytest.raises(
        DistributedTrainingError, match="checkpoint_integrity_metadata_missing"
    ):
        train_distributed_statevector(
            resumed, steps=2, checkpoint_dir=tmp_path, resume=True
        )


def test_resume_rejects_corrupted_checkpoint_bytes(tmp_path: Path):
    circuit, _ = _circuit()
    train_distributed_statevector(circuit, steps=1, checkpoint_dir=tmp_path)
    path = tmp_path / "rank-0.pt"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(data)
    resumed, _ = _circuit()
    with pytest.raises(DistributedTrainingError, match="checkpoint_integrity_mismatch"):
        train_distributed_statevector(
            resumed, steps=2, checkpoint_dir=tmp_path, resume=True
        )


def test_resume_uses_restricted_checkpoint_deserialization(tmp_path: Path):
    circuit, _ = _circuit()
    train_distributed_statevector(circuit, steps=1, checkpoint_dir=tmp_path)
    path = tmp_path / "rank-0.pt"
    torch.save({"unsafe_custom_object": Path("not-allowlisted")}, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "rank-0.pt.sha256").write_text(checksum + "\n")
    resumed, _ = _circuit()
    with pytest.raises(
        DistributedTrainingError, match="checkpoint_deserialization_failed"
    ):
        train_distributed_statevector(
            resumed, steps=2, checkpoint_dir=tmp_path, resume=True
        )


def test_resume_rejects_mixed_rank_checkpoint_generations():
    with pytest.raises(
        DistributedTrainingError, match="checkpoint_generation_mismatch"
    ):
        _validate_checkpoint_generations((2, 3), rank=0)


def test_checkpoint_resume_matches_uninterrupted_seeded_run(tmp_path: Path):
    uninterrupted, uninterrupted_parameters = _circuit()
    full = train_distributed_statevector(
        uninterrupted, steps=4, optimizer="adam", lr=0.02
    )

    first, _ = _circuit()
    partial = train_distributed_statevector(
        first,
        steps=2,
        optimizer="adam",
        lr=0.02,
        checkpoint_dir=tmp_path,
    )
    resumed, resumed_parameters = _circuit()
    continuation = train_distributed_statevector(
        resumed,
        steps=4,
        optimizer="adam",
        lr=0.02,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert partial.completed_steps == 2
    assert continuation.start_step == 2
    assert continuation.losses == pytest.approx(full.losses[2:], abs=3e-5)
    for actual, expected in zip(resumed_parameters, uninterrupted_parameters):
        assert float(actual.detach()) == pytest.approx(
            float(expected.detach()), abs=3e-5
        )


def test_resume_rejects_target_before_completed_checkpoint(tmp_path: Path):
    circuit, _ = _circuit()
    train_distributed_statevector(circuit, steps=2, checkpoint_dir=tmp_path)
    resumed, _ = _circuit()
    with pytest.raises(DistributedTrainingError, match="resume_step_regression"):
        train_distributed_statevector(
            resumed, steps=1, checkpoint_dir=tmp_path, resume=True
        )


def test_oom_preflight_cancel_and_fault_are_structured():
    circuit, _ = _circuit()
    with pytest.raises(DistributedTrainingError, match="oom_preflight"):
        train_distributed_statevector(circuit, steps=1, memory_budget_bytes=1)
    with pytest.raises(DistributedTrainingError, match="cancelled"):
        train_distributed_statevector(circuit, steps=1, cancelled=lambda: True)
    with pytest.raises(DistributedTrainingError, match="injected_rank_failure"):
        train_distributed_statevector(circuit, steps=1, fault_rank=0)
    with pytest.raises(ValueError, match="fault_rank must be"):
        train_distributed_statevector(circuit, steps=1, fault_rank=1)


def test_progress_proves_useful_work_and_all_required_phases(tmp_path: Path):
    circuit, _ = _circuit()
    result = train_distributed_statevector(
        circuit, steps=2, checkpoint_dir=tmp_path, rematerialization_interval=2
    )
    phases = {item.phase for item in result.progress}
    assert {"forward", "backward", "optimizer", "checkpoint", "teardown"} <= phases
    assert {
        "execution_segment",
        "gate_block",
        "collective",
        "backward_segment",
        "optimizer_step",
    } <= phases
    assert any(item.useful_work_launched for item in result.progress)
    assert result.completed_steps == 2
    summary = result.summary()
    assert summary["local_world_size"] == 1
    assert summary["node_count"] == 1
    assert summary["peak_memory_bytes"] > 0
    assert summary["communication_bytes"] == 0


def test_backward_operation_is_actively_bounded(monkeypatch):
    import flagquantum.runtime.backends.statevector.training as training

    circuit, _ = _circuit()
    original = training.execute_torch_distributed_statevector_reverse
    calls = 0

    def delayed(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls > 1:

            class DelayedResult:
                def backward(self):
                    time.sleep(1.0)

                def __getattr__(self, name):
                    return getattr(result, name)

            return DelayedResult()
        return result

    import time

    monkeypatch.setattr(
        training, "execute_torch_distributed_statevector_reverse", delayed
    )
    with pytest.raises(DistributedTrainingError, match="backward_timeout"):
        training.train_distributed_statevector(circuit, steps=1, timeout_seconds=0.05)
