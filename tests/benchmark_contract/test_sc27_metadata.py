from __future__ import annotations

from types import SimpleNamespace

import torch

import benchmarks.sc27_metadata as metadata
from benchmarks.sc27_metadata import (
    build_optimizer,
    canonical_sha256,
    cuda_profile_metrics,
    optimizer_state_bytes,
    reference_errors,
    tensor_bytes,
)


def test_canonical_sha256_ignores_mapping_order() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_tensor_and_adam_state_bytes_are_measured() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    assert tensor_bytes([value]) == 8
    optimizer = build_optimizer("adam", [value], learning_rate=0.01)
    assert optimizer is not None
    value.sum().backward()
    optimizer.step()
    assert optimizer_state_bytes(optimizer) > 0


def test_none_optimizer_has_no_state() -> None:
    value = torch.tensor(1.0, requires_grad=True)
    optimizer = build_optimizer("none", [value], learning_rate=0.01)
    assert optimizer is None
    assert optimizer_state_bytes(optimizer) == 0


def test_reference_errors_bind_and_compare_reference(tmp_path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text(
        '{"correctness":{"initial_value":0.5,"initial_gradients":[0.1,0.2]}}'
    )
    result = reference_errors(
        value=0.5,
        gradients=[0.1, 0.2],
        reference_path=reference,
    )
    assert result["passed"] is True
    assert len(result["reference_artifact_sha256"]) == 64


def test_reference_errors_fail_on_parameter_count_mismatch(tmp_path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text(
        '{"correctness":{"initial_value":0.5,"initial_gradients":[0.1]}}'
    )
    result = reference_errors(
        value=0.5,
        gradients=[0.1, 0.2],
        reference_path=reference,
    )
    assert result["passed"] is False
    assert result["parameter_count_matches"] is False


def test_gpu_identity_resolves_cuda_visible_physical_device(monkeypatch) -> None:
    properties = SimpleNamespace(
        uuid="seven-without-canonical-prefix",
        pci_bus_id=81,
        name="NVIDIA A800-SXM4-80GB",
        total_memory=80 * (1 << 30),
        major=8,
        minor=0,
        multi_processor_count=108,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setattr(
        metadata,
        "_command_output",
        lambda command, cwd=None: (
            "3, GPU-three, 0000:31:00.0, NVIDIA A800-SXM4-80GB, 81920, "
            "400.00, Enabled, Default, 1410, 1593\n"
            "7, GPU-seven, 0000:81:00.0, NVIDIA A800-SXM4-80GB, 81920, "
            "400.00, Enabled, Default, 1410, 1593"
        ),
    )
    result = metadata.gpu_identity(1)
    assert result["gpu_uuid"] == "GPU-seven"
    assert result["pci_bus_id"] == "0000:81:00.0"
    assert result["identity_complete"] is True


def test_topology_snapshot_is_content_addressed(monkeypatch) -> None:
    monkeypatch.setattr(
        metadata, "_command_output", lambda command, cwd=None: "GPU0 GPU1\nGPU0 X NV8"
    )
    result = metadata.topology_snapshot()
    assert result["captured"] is True
    assert len(result["nvidia_smi_topology_sha256"]) == 64


def test_source_identity_uses_validated_container_environment(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(metadata, "_command_output", lambda command, cwd=None: None)
    monkeypatch.setenv("FQ_SC27_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("FQ_SC27_SOURCE_DIRTY", "false")
    assert metadata.source_commit(tmp_path) == "a" * 40
    assert metadata.source_dirty(tmp_path) is False


def test_source_identity_rejects_invalid_container_commit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(metadata, "_command_output", lambda command, cwd=None: None)
    monkeypatch.setenv("FQ_SC27_SOURCE_COMMIT", "not-a-commit")
    assert metadata.source_commit(tmp_path) is None


def test_cuda_profile_metrics_measure_union_and_overlap() -> None:
    cuda = torch.autograd.DeviceType.CUDA
    events = [
        SimpleNamespace(
            device_type=cuda,
            name="ncclKernel_AllReduce",
            time_range=SimpleNamespace(start=0.0, end=10.0),
        ),
        SimpleNamespace(
            device_type=cuda,
            name="compute_kernel",
            time_range=SimpleNamespace(start=5.0, end=15.0),
        ),
    ]
    profile = SimpleNamespace(events=lambda: events)
    result = cuda_profile_metrics(
        profile, wall_seconds=1.0, logical_communication_bytes=100
    )
    assert result["communication_device_seconds_union"] == 10e-6
    assert result["communication_compute_overlap_seconds"] == 5e-6
    assert result["exposed_communication_device_seconds"] == 5e-6
    assert result["overlap_fraction_of_communication"] == 0.5
    assert result["gradient_all_reduce_kernel_count"] == 1
    assert result["gradient_all_reduce_overlap_fraction"] == 0.5
