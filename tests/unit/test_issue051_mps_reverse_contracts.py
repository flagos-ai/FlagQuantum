import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flagquantum.runtime.backends.mps.records import (
    MPSReverseCheckpointPolicy,
    MPSReverseContractError,
    MPSReverseTape,
    MPSReverseTapeRecord,
)
from flagquantum.runtime.backends.mps.reverse import (
    _qr_forward,
    _reverse_execution_segments,
    _static_exact_qr_record,
    _validate_svd_gaps,
)
from flagquantum.runtime.backends.mps.reverse_planning import (
    cached_mps_reverse_segments,
    clear_mps_reverse_segment_cache,
    mps_reverse_segment_cache_stats,
    plan_mps_canonicalization_bonds,
)
from flagquantum.runtime.backends.mps.reverse_transport import (
    all_reduce_reverse_layer_records,
    begin_reverse_layer_halo_prefetch,
    finish_reverse_layer_halo_prefetch,
)

pytestmark = pytest.mark.unit


def test_disjoint_layer_metadata_uses_one_fixed_tensor_collective(monkeypatch):
    calls = 0

    def all_reduce(value, op):
        nonlocal calls
        calls += 1
        assert value.shape == (2, 21)
        assert op == torch.distributed.ReduceOp.SUM

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    reference = torch.zeros((), dtype=torch.complex64)
    payloads = {
        index: {
            "input_shapes": ((1, 2, 2, 4), (1, 4, 2, 2)),
            "output_shapes": ((1, 2, 2, 3), (1, 3, 2, 2)),
            "split_info": {
                "rank": 3,
                "original_rank": 4,
                "discarded_weight": 1e-6,
            },
        }
        for index in (7, 8)
    }
    decoded = all_reduce_reverse_layer_records(payloads, ((7, 0), (8, 0)), reference)
    assert calls == 1
    assert set(decoded) == {7, 8}
    assert all(item["split_info"]["rank"] == 3 for item in decoded.values())


def test_none_canonicalization_policy_skips_final_sweep():
    assert plan_mps_canonicalization_bonds(8, (1, 3, 5), "none") == ()


def test_mps_reverse_observable_functions_are_defined_once() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "flagquantum/runtime/backends/mps/reverse_z_observables.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert len(names) == len(set(names))


def _record(
    index: int, *, gap: float | None = None, discarded_weight: float = 0.0
) -> MPSReverseTapeRecord:
    return MPSReverseTapeRecord(
        operation_id=f"operation-{index}",
        forward_sequence=index,
        reverse_sequence=0,
        kind="two_site",
        wires=(0, 1),
        compute_owner=0,
        owner_ranks=(0, 1),
        communication_peer=1,
        communication_sequence=index,
        input_shapes=((1, 1, 2, 2), (1, 2, 2, 1)),
        output_shapes=((1, 1, 2, 2), (1, 2, 2, 1)),
        parameter_indices=(0,),
        discarded_weight=discarded_weight,
        kept_rank=1,
        original_rank=2,
        singular_value_gap=gap,
        rematerialize=True,
    )


def test_reverse_tape_identity_and_sequence_are_deterministic():
    policy = MPSReverseCheckpointPolicy(
        max_saved_bytes=1024, rematerialization_interval=1
    )
    first = MPSReverseTape.build(
        [_record(0), _record(1)], saved_tensor_bytes=64, checkpoint_policy=policy
    )
    second = MPSReverseTape.build(
        [_record(0), _record(1)], saved_tensor_bytes=64, checkpoint_policy=policy
    )
    assert first.identity == second.identity
    assert [item.reverse_sequence for item in reversed(first.records)] == [0, 1]
    first.validate_distributed()


def test_checkpoint_policy_is_bounded_and_versioned():
    with pytest.raises(ValueError):
        MPSReverseCheckpointPolicy(max_saved_bytes=0)
    policy = MPSReverseCheckpointPolicy()
    assert policy.version == "mps_reverse_checkpoint_v3"
    assert policy.save_two_site_factorizations is False
    assert policy.max_saved_bytes > 0
    with pytest.raises(ValueError, match="factorization"):
        MPSReverseCheckpointPolicy(max_saved_factorization_bytes=-1)
    with pytest.raises(ValueError, match="not implemented"):
        MPSReverseCheckpointPolicy(rematerialization_interval=2)


def test_complex_qr_pair_pullback_passes_gradcheck_and_preserves_state():
    generator = torch.Generator().manual_seed(51)
    left = torch.randn(
        1, 2, 2, 3, dtype=torch.complex128, generator=generator
    ).requires_grad_()
    right = torch.randn(
        1, 3, 2, 2, dtype=torch.complex128, generator=generator
    ).requires_grad_()

    assert torch.autograd.gradcheck(
        _qr_forward, (left, right), eps=1e-6, atol=2e-5, rtol=2e-4
    )
    output_left, output_right = _qr_forward(left, right)
    before = torch.einsum("blsm,bmtr->blstr", left, right)
    after = torch.einsum("blsm,bmtr->blstr", output_left, output_right)
    torch.testing.assert_close(after, before, atol=1e-10, rtol=1e-10)


def test_static_exact_qr_record_derives_shapes_without_owner_metadata():
    record = _static_exact_qr_record(
        (16, 3, 2, 5), (16, 5, 2, 4), max_bond=None, cutoff=0.0
    )
    assert record == {
        "input_shapes": ((16, 3, 2, 5), (16, 5, 2, 4)),
        "output_shapes": ((16, 3, 2, 6), (16, 6, 2, 4)),
        "split_info": {
            "rank": 6,
            "original_rank": 6,
            "discarded_weight": 0.0,
        },
    }


@pytest.mark.parametrize(("max_bond", "cutoff"), ((5, 0.0), (None, 1e-8), (5, 1e-8)))
def test_static_exact_qr_record_preserves_dynamic_truncation_metadata(max_bond, cutoff):
    assert (
        _static_exact_qr_record(
            (16, 3, 2, 5),
            (16, 5, 2, 4),
            max_bond=max_bond,
            cutoff=cutoff,
        )
        is None
    )


def test_thin_qr_pullback_passes_gradcheck_and_rejects_rank_deficiency():
    generator = torch.Generator().manual_seed(510)
    left = torch.randn(
        1, 2, 2, 2, dtype=torch.complex128, generator=generator
    ).requires_grad_()
    right = torch.randn(
        1, 2, 2, 1, dtype=torch.complex128, generator=generator
    ).requires_grad_()
    assert torch.autograd.gradcheck(
        _qr_forward, (left, right), eps=1e-6, atol=3e-5, rtol=3e-4
    )
    deficient = torch.zeros_like(left)
    with pytest.raises(MPSReverseContractError, match="rank deficient"):
        _qr_forward(deficient, right)


def test_full_state_and_sequence_contracts_fail_closed():
    tape = MPSReverseTape.build(
        [_record(0)],
        saved_tensor_bytes=64,
        checkpoint_policy=MPSReverseCheckpointPolicy(),
    )
    broken = MPSReverseTape(
        records=(
            MPSReverseTapeRecord(
                **{**tape.records[0].summary(), "reverse_sequence": 4}
            ),
        ),
        identity=tape.identity,
        saved_tensor_bytes=tape.saved_tensor_bytes,
        checkpoint_policy=tape.checkpoint_policy,
    )
    with pytest.raises(MPSReverseContractError, match="sequence"):
        broken.validate_distributed()


def test_near_degenerate_truncation_boundary_fails_closed():
    tape = MPSReverseTape.build(
        [_record(0, gap=1e-10, discarded_weight=1e-6)],
        saved_tensor_bytes=64,
        checkpoint_policy=MPSReverseCheckpointPolicy(),
    )
    with pytest.raises(MPSReverseContractError, match="degenerate.*operation-0"):
        _validate_svd_gaps(tape, 1e-7)
    _validate_svd_gaps(tape, 1e-12)


def test_degenerate_zero_weight_tail_is_not_a_physical_truncation():
    tape = MPSReverseTape.build(
        [_record(0, gap=0.0, discarded_weight=1e-16)],
        saved_tensor_bytes=64,
        checkpoint_policy=MPSReverseCheckpointPolicy(),
    )
    _validate_svd_gaps(tape, 1e-7)


def test_owner_local_reverse_segments_stop_at_dependencies_and_boundaries():
    records = []
    for index, (wire, owner, kind, peer) in enumerate(
        (
            (0, 0, "one_site", None),
            (1, 0, "one_site", None),
            (1, 0, "one_site", None),
            (2, 0, "two_site", 1),
            (3, 1, "one_site", None),
            (4, 1, "one_site", None),
        )
    ):
        base = _record(index)
        records.append(
            MPSReverseTapeRecord(
                **{
                    **base.summary(),
                    "kind": kind,
                    "wires": (wire,) if kind == "one_site" else (wire, wire + 1),
                    "compute_owner": owner,
                    "owner_ranks": (owner,) if peer is None else (owner, peer),
                    "communication_peer": peer,
                    "input_shapes": ((2, 1, 2, 1),),
                    "output_shapes": ((2, 1, 2, 1),),
                }
            )
        )
    tape = MPSReverseTape.build(
        records, saved_tensor_bytes=1, checkpoint_policy=MPSReverseCheckpointPolicy()
    )
    segments = _reverse_execution_segments(tape, fuse_owner_local=True)
    assert [len(item) for item in segments] == [2, 1, 1, 2]
    assert all(len({record.compute_owner for record in item}) == 1 for item in segments)
    assert len(_reverse_execution_segments(tape, fuse_owner_local=False)) == 6


def test_reverse_segment_cache_is_bounded_and_maps_current_tape_records():
    clear_mps_reverse_segment_cache()
    first_tape = MPSReverseTape.build(
        [_record(0), _record(1)],
        saved_tensor_bytes=64,
        checkpoint_policy=MPSReverseCheckpointPolicy(),
    )
    second_tape = MPSReverseTape.build(
        [_record(0), _record(1)],
        saved_tensor_bytes=128,
        checkpoint_policy=MPSReverseCheckpointPolicy(),
    )
    first, first_hit = cached_mps_reverse_segments(first_tape, fuse_owner_local=True)
    second, second_hit = cached_mps_reverse_segments(second_tape, fuse_owner_local=True)
    assert first_hit is False
    assert second_hit is True
    assert first == second
    assert all(
        record is second_tape.records[record.forward_sequence]
        for segment in second
        for record in segment
    )
    assert mps_reverse_segment_cache_stats() == {
        "entries": 1,
        "max_entries": 128,
        "hits": 1,
        "misses": 1,
    }


def test_layer_halo_prefetch_reports_inter_node_payload_and_wait(monkeypatch):
    class CompletedWork:
        def wait(self, *, timeout):
            assert timeout.total_seconds() > 0
            return True

    tensor = torch.zeros(1, 1, 2, 1, dtype=torch.complex64)
    state = SimpleNamespace(
        rank=0,
        world_size=2,
        local_tensors={0: tensor},
        owner=lambda wire: wire,
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setattr(torch.distributed, "irecv", object())
    monkeypatch.setattr(
        torch.distributed,
        "P2POp",
        lambda operation, value, peer: (operation, value, peer),
    )
    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda operations: [CompletedWork() for _ in operations],
    )
    prefetch, indices = begin_reverse_layer_halo_prefetch(
        [(7, None, 0)], state, {1: (1, 1, 2, 1)}
    )
    assert indices == frozenset({7})
    assert prefetch.message_count == 1
    assert prefetch.payload_bytes == tensor.numel() * tensor.element_size()
    assert prefetch.intra_node_payload_bytes == 0
    assert prefetch.inter_node_payload_bytes == prefetch.payload_bytes
    assert finish_reverse_layer_halo_prefetch(prefetch) >= 0.0
