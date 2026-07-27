from __future__ import annotations

import pytest
import torch

from flagquantum.runtime.backends.mps.factorization import (
    FactorizationMemorySnapshot,
    FactorizationWorkspacePolicy,
    MPSFactorizationMemoryError,
    estimate_rxx_factorization_working_set,
    plan_rxx_factorization_microbatch,
)
from flagquantum.simulation.mps import (
    MPSConfig,
    _split_pair_matrix,
    _split_pair_matrix_bucket,
)

pytestmark = pytest.mark.unit


def _samples(*, device: str = "cpu"):
    return (
        torch.empty((1, 4, 2, 8), dtype=torch.complex64, device=device),
        torch.empty((1, 8, 2, 4), dtype=torch.complex64, device=device),
        torch.empty((1, 4, 4), dtype=torch.complex64, device=device),
    )


def _provider(available: int):
    return lambda _device: FactorizationMemorySnapshot(
        free_bytes=available,
        total_bytes=available,
        allocated_bytes=0,
        reserved_bytes=0,
    )


def test_outputs_and_solver_workspace_reduce_capacity_with_same_inputs() -> None:
    left, right, matrix = _samples()
    inputs_only = FactorizationWorkspacePolicy(
        minimum_headroom_bytes=0,
        nccl_reserve_bytes=0,
        allocator_safety_margin_bytes=0,
        solver_workspace_pair_multiplier=0.0,
        maximum_chunk_size=16,
    )
    workspace_aware = FactorizationWorkspacePolicy(
        minimum_headroom_bytes=0,
        nccl_reserve_bytes=0,
        allocator_safety_margin_bytes=0,
        solver_workspace_pair_multiplier=8.0,
        maximum_chunk_size=16,
    )
    small = estimate_rxx_factorization_working_set(
        left, right, matrix, policy=inputs_only, requires_grad=False
    )
    estimate_rxx_factorization_working_set(
        left, right, matrix, policy=workspace_aware, requires_grad=False
    )
    available = small.bytes_for_chunk(4)
    first = plan_rxx_factorization_microbatch(
        left,
        right,
        matrix,
        requested_chunk_size=4,
        policy=inputs_only,
        memory_provider=_provider(available),
    )
    second = plan_rxx_factorization_microbatch(
        left,
        right,
        matrix,
        requested_chunk_size=4,
        policy=workspace_aware,
        memory_provider=_provider(available),
    )
    assert first.selected_chunk_size == 4
    assert second.selected_chunk_size < first.selected_chunk_size
    assert second.working_set.input_bytes_per_item == small.input_bytes_per_item
    assert second.working_set.solver_workspace_bytes_per_item > 0


def test_memory_plan_downshifts_and_preserves_declared_headroom() -> None:
    left, right, matrix = _samples()
    policy = FactorizationWorkspacePolicy(
        minimum_headroom_bytes=1_000,
        nccl_reserve_bytes=100,
        allocator_safety_margin_bytes=100,
        solver_workspace_pair_multiplier=1.0,
        maximum_chunk_size=8,
    )
    working = estimate_rxx_factorization_working_set(
        left, right, matrix, policy=policy, requires_grad=False
    )
    free = policy.minimum_headroom_bytes + working.bytes_for_chunk(2)
    decision = plan_rxx_factorization_microbatch(
        left,
        right,
        matrix,
        requested_chunk_size=8,
        policy=policy,
        memory_provider=_provider(free),
    )
    assert decision.selected_chunk_size == 2
    assert decision.downshifted is True
    assert decision.selected_working_set_bytes <= decision.available_working_bytes
    assert free - decision.selected_working_set_bytes >= policy.minimum_headroom_bytes


def test_high_bond_defaults_to_serial_and_record_is_tensor_free() -> None:
    left = torch.empty((1, 1024, 2, 1024), dtype=torch.complex64, device="meta")
    right = torch.empty((1, 1024, 2, 1024), dtype=torch.complex64, device="meta")
    matrix = torch.empty((1, 4, 4), dtype=torch.complex64, device="meta")
    decision = plan_rxx_factorization_microbatch(
        left,
        right,
        matrix,
        requested_chunk_size=8,
        memory_provider=_provider(100 * (1 << 30)),
    )
    assert decision.selected_chunk_size == 1
    assert decision.maximum_by_policy == 1
    record = decision.as_dict()

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(record)


def test_factorization_rejects_before_allocation_when_one_item_cannot_fit() -> None:
    left, right, matrix = _samples()
    policy = FactorizationWorkspacePolicy(
        minimum_headroom_bytes=10_000,
        nccl_reserve_bytes=1_000,
        allocator_safety_margin_bytes=1_000,
    )
    with pytest.raises(
        MPSFactorizationMemoryError,
        match=r"rejected before allocation.*requested_bytes=.*shape=.*dtype=torch.complex64",
    ):
        plan_rxx_factorization_microbatch(
            left,
            right,
            matrix,
            requested_chunk_size=1,
            policy=policy,
            memory_provider=_provider(10_000),
        )


@pytest.mark.parametrize("max_bond", (None, 2))
def test_batched_factorization_matches_individual_pair_splits(max_bond) -> None:
    torch.manual_seed(108)
    matrices = torch.randn(3, 2, 4, 4, dtype=torch.complex64)
    config = MPSConfig(max_bond=max_bond, cutoff=0.0)
    actual = _split_pair_matrix_bucket(matrices, left_dim=2, right_dim=2, config=config)
    expected = tuple(
        _split_pair_matrix(matrix, left_dim=2, right_dim=2, config=config)
        for matrix in matrices
    )
    for matrix, (left, right, info), (ref_left, ref_right, ref_info) in zip(
        matrices, actual, expected
    ):
        reconstructed = torch.einsum("blsm,bmtr->blstr", left, right).reshape(2, 4, 4)
        reference = torch.einsum("blsm,bmtr->blstr", ref_left, ref_right).reshape(
            2, 4, 4
        )
        torch.testing.assert_close(reconstructed, reference, atol=2e-5, rtol=2e-5)
        assert info["method"] == ref_info["method"]
        assert info["rank"] == ref_info["rank"]
        assert info["discarded_weight"] == pytest.approx(
            ref_info["discarded_weight"], abs=2e-6
        )
        if int(info["rank"]) < int(info["original_rank"]):
            singular_values = torch.linalg.svdvals(matrix)
            rank = int(info["rank"])
            expected_gap = torch.min(
                torch.abs(singular_values[:, rank - 1] - singular_values[:, rank])
            ).item()
            assert info["singular_value_gap"] == pytest.approx(expected_gap, abs=2e-6)


def test_regular_pair_split_uses_one_batched_qr_launch(monkeypatch) -> None:
    import flagquantum.simulation.mps_factorization as factorization

    calls = []
    original = torch.linalg.qr

    def counted_qr(matrix, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, **kwargs)

    monkeypatch.setattr(factorization.torch.linalg, "qr", counted_qr)
    matrix = torch.randn(7, 4, 4, dtype=torch.complex64)

    left, right, info = _split_pair_matrix(
        matrix,
        left_dim=2,
        right_dim=2,
        config=MPSConfig(max_bond=None, cutoff=0.0),
    )

    assert calls == [(7, 4, 4)]
    assert left.shape == (7, 2, 2, 4)
    assert right.shape == (7, 4, 2, 2)
    assert info["method"] == "qr"


def test_regular_pair_split_uses_one_batched_svd_launch(monkeypatch) -> None:
    import flagquantum.simulation.mps_factorization as factorization

    calls = []
    original = factorization._cuda_svd

    def counted_svd(matrix, *, driver):
        calls.append(tuple(matrix.shape))
        return original(matrix, driver=driver)

    monkeypatch.setattr(factorization, "_cuda_svd", counted_svd)
    matrix = torch.randn(7, 4, 4, dtype=torch.complex64)

    left, right, info = _split_pair_matrix(
        matrix,
        left_dim=2,
        right_dim=2,
        config=MPSConfig(max_bond=2, cutoff=0.0),
    )

    assert calls == [(7, 4, 4)]
    assert left.shape == (7, 2, 2, 2)
    assert right.shape == (7, 2, 2, 2)
    assert info["method"] == "svd"
