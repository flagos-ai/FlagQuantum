"""Fail-closed public contracts for ISSUE-052."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.backends.mps.reverse import _qr_forward
from flagquantum.runtime.backends.mps.training import (
    MPSStepMetrics,
    MPSTrainingError,
    ShardedMPSTrainingResult,
)
from flagquantum.runtime.backends.mps.training_engine import (
    _checkpoint_capacity_error,
    _checkpoint_start_policy_error,
    _initial_state_contract,
    _load_checkpoint,
    _parameter_broadcast_buckets,
    _prune_checkpoint_generations,
    _resolve_compile_site_kernels,
    _save_checkpoint,
)
from flagquantum.runtime.distributed.models import (
    DistributedShardPlan,
    ShardedMPSState,
    TorchDistributedContext,
)

pytestmark = pytest.mark.unit


def test_initial_state_contract_hashes_reused_tensor_once(monkeypatch):
    tensor = torch.ones((1, 4, 2, 4), dtype=torch.complex64)
    initial = {wire: tensor for wire in range(10)}
    ownership = (tuple(range(10)),)
    calls = 0
    original_numpy = torch.Tensor.numpy

    def counted_numpy(value):
        nonlocal calls
        calls += 1
        return original_numpy(value)

    monkeypatch.setattr(torch.Tensor, "numpy", counted_numpy)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        "flagquantum.runtime.backends.mps.training_engine.all_gather_json",
        lambda payload: (payload,),
    )
    fingerprint, shapes = _initial_state_contract(
        fq.Circuit(10).to_ir(), initial, 1, ownership
    )

    assert len(fingerprint) == 64
    assert len(shapes) == 10
    assert calls == 1


def test_parameter_broadcasts_are_bucketed_by_owner_and_dtype():
    parameters = tuple(torch.zeros((), dtype=torch.float64) for _ in range(1000))
    owners = tuple(index % 8 for index in range(len(parameters)))

    buckets = _parameter_broadcast_buckets(
        parameters, owners, max_bucket_bytes=25 * 1024 * 1024, world_size=8
    )

    assert len(buckets) == 1
    assert tuple(map(len, buckets[0].indices_by_owner)) == (125,) * 8
    assert buckets[0].input_buffer.numel() == 125
    assert buckets[0].output_buffer.numel() == 1000
    assert len(buckets[0].pack_targets) == 125
    assert len(buckets[0].unpack_targets) == 875
    assert buckets[0].payload_bytes == 8000


def _circuit():
    theta = torch.tensor(0.31, requires_grad=True)
    return fq.Circuit(3).ry(0, theta).rxx(0, 1, theta), theta


def _write_test_checkpoint(root, parameter):
    return _save_checkpoint(
        root,
        rank=0,
        world_size=1,
        step=3,
        owned_indices=(0,),
        parameters=(parameter,),
        optimizer=None,
        contract={"test": "mps-checkpoint-integrity"},
        bond_layout={0: (1, 1, 2, 1)},
    )


def _load_test_checkpoint(root, parameter):
    return _load_checkpoint(
        root,
        rank=0,
        world_size=1,
        owned_indices=(0,),
        parameters=(parameter,),
        optimizer=None,
        contract={"test": "mps-checkpoint-integrity"},
        bond_layout={0: (1, 1, 2, 1)},
        checkpoint_path=root / "rank-0-step-3.pt",
    )


def test_mps_checkpoint_persists_integrity_metadata_and_restores(tmp_path):
    parameter = torch.tensor(0.31, requires_grad=True)
    path = _write_test_checkpoint(tmp_path, parameter)
    assert path.with_suffix(".pt.sha256").is_file()
    parameter.data.fill_(9.0)
    assert _load_test_checkpoint(tmp_path, parameter) == 3
    assert float(parameter.detach()) == pytest.approx(0.31)


def test_mps_checkpoint_rejects_missing_integrity_metadata(tmp_path):
    parameter = torch.tensor(0.31, requires_grad=True)
    path = _write_test_checkpoint(tmp_path, parameter)
    path.with_suffix(".pt.sha256").unlink()
    with pytest.raises(MPSTrainingError, match="integrity metadata missing"):
        _load_test_checkpoint(tmp_path, parameter)


def test_mps_checkpoint_rejects_corrupted_bytes_before_deserialization(tmp_path):
    parameter = torch.tensor(0.31, requires_grad=True)
    path = _write_test_checkpoint(tmp_path, parameter)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(data)
    with pytest.raises(MPSTrainingError, match="integrity mismatch"):
        _load_test_checkpoint(tmp_path, parameter)


def test_mps_checkpoint_rejects_non_tensor_pickle_even_with_matching_checksum(
    tmp_path,
):
    parameter = torch.tensor(0.31, requires_grad=True)
    path = _write_test_checkpoint(tmp_path, parameter)
    torch.save({"unsafe_custom_object": Path("not-allowlisted")}, path)
    path.with_suffix(".pt.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    with pytest.raises(MPSTrainingError, match="deserialization failed"):
        _load_test_checkpoint(tmp_path, parameter)


def test_mps_checkpoint_validates_parameter_schema_before_live_state_mutation(
    tmp_path,
):
    parameter = torch.tensor(0.31, requires_grad=True)
    path = _write_test_checkpoint(tmp_path, parameter)
    payload = torch.load(path, weights_only=True)
    payload["parameters"][0] = torch.tensor([7.0, 8.0])
    torch.save(payload, path)
    path.with_suffix(".pt.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    before = parameter.detach().clone()
    with pytest.raises(MPSTrainingError, match="shape or dtype mismatch"):
        _load_test_checkpoint(tmp_path, parameter)
    torch.testing.assert_close(parameter.detach(), before)


def test_checkpoint_retention_bounds_storage_and_preserves_committed_generation(
    tmp_path,
):
    for step in range(1, 6):
        path = tmp_path / f"rank-0-step-{step}.pt"
        path.write_bytes(str(step).encode())
        path.with_suffix(".pt.sha256").write_text("checksum\n")
    deleted = _prune_checkpoint_generations(
        tmp_path,
        rank=0,
        committed_step=4,
        keep_generations=2,
    )
    assert {path.name for path in tmp_path.glob("*.pt")} == {
        "rank-0-step-3.pt",
        "rank-0-step-4.pt",
    }
    assert len(deleted) == 6


def test_checkpoint_capacity_preflight_fails_before_allocation_with_audit_values():
    assert (
        _checkpoint_capacity_error(
            minimum_free_bytes=4096,
            estimated_generation_bytes=2048,
            reserve_bytes=1024,
        )
        is None
    )
    error = _checkpoint_capacity_error(
        minimum_free_bytes=2047,
        estimated_generation_bytes=1024,
        reserve_bytes=1024,
    )
    assert error is not None
    assert "minimum_free_bytes=2047" in error
    assert "required_bytes=2048" in error


def test_checkpoint_start_policy_requires_explicit_resume_or_overwrite():
    error = _checkpoint_start_policy_error(
        committed_manifest_exists=True,
        resume=False,
        allow_overwrite=False,
    )
    assert error is not None
    assert "resume=True" in error
    assert (
        _checkpoint_start_policy_error(
            committed_manifest_exists=True,
            resume=True,
            allow_overwrite=False,
        )
        is None
    )
    assert (
        _checkpoint_start_policy_error(
            committed_manifest_exists=True,
            resume=False,
            allow_overwrite=True,
        )
        is None
    )
    assert "requires a committed" in _checkpoint_start_policy_error(
        committed_manifest_exists=False,
        resume=True,
        allow_overwrite=False,
    )


def test_site_kernel_auto_policy_preserves_cpu_and_short_cuda_fast_paths():
    circuit, _ = _circuit()
    ir = circuit.to_ir()
    assert _resolve_compile_site_kernels(
        None, ir=ir, device=torch.device("cpu"), steps=100
    ) == (False, "auto_cpu_eager_fast_path")
    assert _resolve_compile_site_kernels(
        None, ir=ir, device=torch.device("cuda"), steps=2
    ) == (False, "auto_short_training_avoids_compile_warmup")
    assert _resolve_compile_site_kernels(
        None,
        ir=ir,
        device=torch.device("cuda"),
        steps=100,
        checkpointing=True,
    ) == (False, "auto_checkpoint_compatibility_uses_eager")


def test_site_kernel_auto_policy_compiles_repeated_compatible_cuda_training():
    circuit, _ = _circuit()
    assert _resolve_compile_site_kernels(
        None, ir=circuit.to_ir(), device=torch.device("cuda"), steps=3
    ) == (True, "auto_cuda_repeated_training")


def test_site_kernel_auto_policy_fails_open_to_eager_for_reversed_rotations():
    theta = torch.tensor(0.31, requires_grad=True)
    ir = fq.Circuit(2).rxx(1, 0, theta).to_ir()
    assert _resolve_compile_site_kernels(
        None, ir=ir, device=torch.device("cuda"), steps=3
    ) == (False, "auto_incompatible_wire_order_uses_eager")


def test_single_rank_sharded_mps_gather_avoids_object_collective(monkeypatch):
    mps = fq.MPSState.zero(1)
    context = TorchDistributedContext(
        rank=0,
        world_size=1,
        local_rank=0,
        backend="gloo",
        device=torch.device("cpu"),
        initialized=True,
    )
    sharded = ShardedMPSState(
        n_wires=1,
        bsz=1,
        config=mps.config,
        local_tensors={0: mps.tensors[0]},
        shards=(
            DistributedShardPlan(
                rank=0,
                world_size=1,
                wires=(0,),
                left_boundary=None,
                right_boundary=None,
            ),
        ),
        context=context,
    )

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda *_args, **_kwargs: pytest.fail("single rank must not gather objects"),
    )

    gathered = sharded.gather_tensors()

    assert tuple(gathered) == (0,)
    torch.testing.assert_close(gathered[0], mps.tensors[0])


def test_requires_explicit_distributed_lifecycle():
    circuit, _ = _circuit()
    with pytest.raises(MPSTrainingError, match="requires torch.distributed"):
        fq.train_distributed_mps(circuit, steps=1)


def test_result_never_promotes_execution_without_speedup_and_capacity_evidence():
    step = MPSStepMetrics(
        step=0,
        loss=0.1,
        forward_seconds=1.0,
        reverse_seconds=1.0,
        optimizer_seconds=0.1,
        end_to_end_seconds=2.1,
        training_compute_seconds=2.0,
        diagnostics_seconds=0.1,
        static_qr_metadata_records=2,
        dynamic_metadata_broadcasts=0,
        reverse_segment_cache_hit=True,
        gradient_bucket_cache_hit=True,
        layer_halo_message_count=1,
        layer_halo_payload_bytes=64,
        layer_halo_intra_node_bytes=64,
        layer_halo_inter_node_bytes=0,
        layer_halo_wait_seconds=0.01,
        owned_site_work=1,
        owned_bond_work=1,
        boundary_exchanges=1,
        svd_activity=1,
        qr_activity=1,
        peak_memory_bytes=128,
        useful_work_completed=True,
        gradient_bucket_count=2,
        gradient_bucket_fill_ratio=0.5,
        reverse_tape_segments=4,
        fused_reverse_segments=2,
        reverse_autograd_grad_invocations=3,
        reverse_python_dispatches=4,
        qr_factorization_count=1,
        svd_factorization_count=1,
    )
    summary = ShardedMPSTrainingResult(
        losses=(0.1,),
        completed_steps=1,
        start_step=0,
        rank=0,
        world_size=2,
        local_world_size=2,
        optimizer="adam",
        ownership=(),
        steps=(step,),
        checkpoint_files=(),
        memory_growth_bytes=0,
        suspected_memory_leak=False,
    ).summary()
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["correctness_gate_passed"] is True
    assert summary["speedup_gate_passed"] is False
    assert summary["capacity_gate_passed"] is False
    assert summary["scalability_claim_allowed"] is False
    assert summary["checkpoint_schema"] == "sharded_mps_training_checkpoint_v2"
    assert summary["checkpoint_integrity"] == "sha256_atomic_file_and_metadata"
    assert (
        summary["checkpoint_commit_protocol"]
        == "immutable_rank_shards_atomic_manifest_v1"
    )
    assert summary["checkpoint_retention_generations"] == 2
    assert summary["checkpoint_storage_semantics"] == "local_filesystem"
    assert summary["checkpoint_deserialization_policy"] == "torch_weights_only"
    assert summary["checkpoint_writer_lease"] == "exclusive_atomic_file_v1"
    assert summary["checkpoint_writer_lease_break_allowed"] is False
    assert summary["checkpoint_writer_lease_stale_seconds"] == 3600.0
    assert summary["checkpoint_writer_lease_heartbeat_count"] == 0
    step_summary = summary["step_metrics"][0]
    assert step_summary["gradient_bucket_count"] == 2
    assert step_summary["reverse_tape_segments"] == 4
    assert step_summary["reverse_autograd_grad_invocations"] == 3
    assert step_summary["qr_factorization_count"] == 1


def test_idle_rank_and_memory_growth_are_explicit_blockers():
    summary = ShardedMPSTrainingResult(
        losses=(),
        completed_steps=0,
        start_step=0,
        rank=1,
        world_size=2,
        local_world_size=2,
        optimizer="sgd",
        ownership=(),
        steps=(),
        checkpoint_files=(),
        memory_growth_bytes=1024,
        suspected_memory_leak=True,
    ).summary()
    assert "multi_step_memory_growth_suspected" in summary["blockers"]
    assert "rank_has_no_completed_site_or_bond_work" in summary["blockers"]
    assert summary["rank_useful_work"] is False


@pytest.mark.parametrize("right_dim", [1, 2, 3])
def test_mps_qr_forward_reconstructs_tiny_batched_factorization(right_dim):
    torch.manual_seed(52)
    left = torch.randn(7, 2, 2, right_dim, dtype=torch.complex64)
    right = torch.randn(7, right_dim, 2, 3, dtype=torch.complex64)

    q, updated_right = _qr_forward(left, right)

    expected = torch.einsum("blsm,bmtr->blstr", left, right)
    actual = torch.einsum("blsm,bmtr->blstr", q, updated_right)
    flat_q = q.reshape(7, 4, right_dim)
    gram = torch.einsum("bmi,bmj->bij", flat_q.conj(), flat_q)
    identity = torch.eye(right_dim, dtype=gram.dtype).expand_as(gram)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gram, identity, rtol=2e-5, atol=2e-5)
