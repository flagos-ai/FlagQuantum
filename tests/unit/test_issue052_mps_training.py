"""Fail-closed public contracts for ISSUE-052."""

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.backends.mps.reverse import _qr_forward
from flagquantum.runtime.backends.mps.training import (
    MPSStepMetrics,
    MPSTrainingError,
    ShardedMPSTrainingResult,
)
from flagquantum.runtime.distributed.models import (
    DistributedShardPlan,
    ShardedMPSState,
    TorchDistributedContext,
)

pytestmark = pytest.mark.unit


def _circuit():
    theta = torch.tensor(0.31, requires_grad=True)
    return fq.Circuit(3).ry(0, theta).rxx(0, 1, theta), theta


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
        owned_site_work=1,
        owned_bond_work=1,
        boundary_exchanges=1,
        svd_activity=1,
        qr_activity=1,
        peak_memory_bytes=128,
        useful_work_completed=True,
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
