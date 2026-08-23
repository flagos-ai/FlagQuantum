"""Execute a shared-value dynamic reverse DAG on two ranks."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.tensor_network import (
    DistributedTNContractionDAG,
    DistributedTNContractionRecord,
    DistributedTNMeshGroupCache,
    DistributedTNRematerializationProvider,
    DistributedTNValueLayout,
    distributed_dag,
    execute_dynamic_tn_reverse_segment,
    execute_explicit_tn_reverse_dag,
    execute_partial_mesh_forward_pair,
    load_dynamic_tn_reverse_checkpoint,
    partition_tn_tensor_for_partial_mesh,
    plan_dynamic_tn_reverse_segment,
    plan_explicit_tn_reverse_dag,
    plan_partial_mesh_tn_layout,
    save_dynamic_tn_reverse_checkpoint,
)


def _shared_dag() -> DistributedTNContractionDAG:
    specs = (
        ("input:0", None, (0, 1), (4, 4)),
        ("input:1", None, (1, 2), (4, 4)),
        ("input:2", None, (2, 3), (4, 2)),
        ("input:3", None, (2, 4), (4, 2)),
        ("intermediate:0", "operation:0", (0, 2), (4, 4)),
        ("intermediate:1", "operation:1", (0, 3), (4, 2)),
        ("intermediate:2", "operation:2", (0, 4), (4, 2)),
        ("intermediate:3", "operation:3", (3, 4), (2, 2)),
    )
    values = tuple(
        DistributedTNValueLayout(
            value_id=value_id,
            producer_id=producer_id,
            labels=labels,
            shape=shape,
            dtype="torch.float64",
            nbytes=8 * int(torch.tensor(shape).prod()),
            semantics="replicated_small" if producer_id is None else "unique_owner",
            owner_ranks=(0, 1) if producer_id is None else (0,),
        )
        for value_id, producer_id, labels, shape in specs
    )
    operations = (
        DistributedTNContractionRecord(
            "operation:0",
            0,
            ("input:0", "input:1"),
            "intermediate:0",
            (0,),
            (0, 2),
            (4, 4),
            1,
            16,
        ),
        DistributedTNContractionRecord(
            "operation:1",
            1,
            ("intermediate:0", "input:2"),
            "intermediate:1",
            (0,),
            (0, 3),
            (4, 2),
            1,
            8,
        ),
        DistributedTNContractionRecord(
            "operation:2",
            2,
            ("intermediate:0", "input:3"),
            "intermediate:2",
            (0,),
            (0, 4),
            (4, 2),
            1,
            8,
        ),
        DistributedTNContractionRecord(
            "operation:3",
            3,
            ("intermediate:1", "intermediate:2"),
            "intermediate:3",
            (0,),
            (3, 4),
            (2, 2),
            1,
            4,
        ),
    )
    dag = DistributedTNContractionDAG(
        version=distributed_dag.TN_DAG_VERSION,
        identity="pending",
        world_size=2,
        objective="memory",
        small_tensor_replication_bytes=4096,
        values=values,
        operations=operations,
        communication_edges=(),
        output_value_id="intermediate:3",
    )
    object.__setattr__(
        dag, "identity", distributed_dag._dag_identity(dag._identity_payload())
    )
    dag.validate()
    return dag


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != 2:
        raise RuntimeError("shared DAG reverse runtime requires two ranks")

    dag = _shared_dag()
    generator = torch.Generator(device=device).manual_seed(9173)
    inputs = {
        "input:0": torch.randn(
            4, 4, generator=generator, device=device, dtype=torch.float64
        ),
        "input:1": torch.randn(
            4, 4, generator=generator, device=device, dtype=torch.float64
        ),
        "input:2": torch.randn(
            4, 2, generator=generator, device=device, dtype=torch.float64
        ),
        "input:3": torch.randn(
            4, 2, generator=generator, device=device, dtype=torch.float64
        ),
    }
    tape = dict(inputs)
    tape["intermediate:0"] = torch.einsum("ab,bc->ac", tape["input:0"], tape["input:1"])
    tape["intermediate:1"] = torch.einsum(
        "ac,cd->ad", tape["intermediate:0"], tape["input:2"]
    )
    tape["intermediate:2"] = torch.einsum(
        "ac,ce->ae", tape["intermediate:0"], tape["input:3"]
    )
    tape["intermediate:3"] = torch.einsum(
        "ad,ae->de", tape["intermediate:1"], tape["intermediate:2"]
    )
    reverse = plan_explicit_tn_reverse_dag(dag)
    reference = execute_explicit_tn_reverse_dag(dag, reverse, tape)
    expect_plan_mismatch = os.environ.get("FQ_EXPECT_PLAN_MISMATCH") == "1"
    expect_tape_mismatch = os.environ.get("FQ_EXPECT_TAPE_MISMATCH") == "1"
    checkpoint_stage = os.environ.get("FQ_CHECKPOINT_STAGE", "")
    use_forward_provider = os.environ.get("FQ_USE_FORWARD_PROVIDER") == "1"
    reject_provider_budget = os.environ.get("FQ_REJECT_FORWARD_PROVIDER_BUDGET") == "1"
    checkpoint_directory = os.environ.get("FQ_CHECKPOINT_DIR")
    if checkpoint_stage in {"save", "resume"} and not checkpoint_directory:
        raise RuntimeError("checkpoint stage requires FQ_CHECKPOINT_DIR")
    segment = plan_dynamic_tn_reverse_segment(
        dag,
        reverse,
        start_output_value_id=dag.output_value_id,
        source_mesh_labels=(5,),
        destination_mesh_labels=(6,),
        mesh_shape=(2,),
        max_records=(3 if expect_plan_mismatch and dist.get_rank() == 1 else 4),
    )
    if expect_tape_mismatch and dist.get_rank() == 1:
        del tape["input:3"]
    if use_forward_provider:
        del tape["intermediate:0"]
        del tape["intermediate:1"]
    forward_provider = None
    rematerialization_context = (
        DistributedTNMeshGroupCache(mesh_labels=(5,), mesh_shape=(2,))
        if use_forward_provider
        else nullcontext(None)
    )
    stage_saved = False
    try:
        with (
            rematerialization_context as rematerialization_cache,
            DistributedTNMeshGroupCache(
                mesh_labels=(6,), mesh_shape=(2,)
            ) as group_cache,
        ):
            if use_forward_provider:
                assert rematerialization_cache is not None
                probe_values = (
                    DistributedTNValueLayout(
                        "probe:left",
                        None,
                        (5, 0),
                        (2, 3),
                        "torch.float64",
                        48,
                        "replicated_small",
                        (0, 1),
                    ),
                    DistributedTNValueLayout(
                        "probe:right",
                        None,
                        (5, 1),
                        (2, 4),
                        "torch.float64",
                        64,
                        "replicated_small",
                        (0, 1),
                    ),
                    DistributedTNValueLayout(
                        "probe:output",
                        "probe:operation",
                        (0, 1),
                        (3, 4),
                        "torch.float64",
                        96,
                        "unique_owner",
                        (0,),
                    ),
                )
                probe_layouts = tuple(
                    plan_partial_mesh_tn_layout(
                        value, mesh_labels=(5,), mesh_shape=(2,)
                    )
                    for value in probe_values
                )
                probe_left = torch.arange(
                    6, dtype=torch.float64, device=device
                ).reshape(2, 3)
                probe_right = torch.arange(
                    8, dtype=torch.float64, device=device
                ).reshape(2, 4)
                probe = execute_partial_mesh_forward_pair(
                    partition_tn_tensor_for_partial_mesh(
                        probe_left, probe_layouts[0], rank=dist.get_rank()
                    ),
                    probe_layouts[0],
                    partition_tn_tensor_for_partial_mesh(
                        probe_right, probe_layouts[1], rank=dist.get_rank()
                    ),
                    probe_layouts[1],
                    probe_layouts[2],
                    group_cache=rematerialization_cache,
                )
                torch.testing.assert_close(
                    probe.value,
                    torch.einsum("ka,kb->ab", probe_left, probe_right),
                )
                if probe.subgroup_collective_count != 1:
                    raise RuntimeError(
                        "partial-mesh forward did not reduce a lost mesh axis"
                    )
                forward_provider = DistributedTNRematerializationProvider(
                    dag,
                    tape,
                    mesh_labels=(5,),
                    mesh_shape=(2,),
                    group_cache=rematerialization_cache,
                    max_transient_bytes=(100 if reject_provider_budget else 192),
                )
            if checkpoint_stage == "resume":
                checkpoint = load_dynamic_tn_reverse_checkpoint(
                    segment,
                    checkpoint_directory,
                    device=device,
                )
                actual = execute_dynamic_tn_reverse_segment(
                    dag,
                    reverse,
                    segment,
                    tape,
                    None,
                    group_cache=group_cache,
                    resume_from=checkpoint,
                    source_forward_provider=forward_provider,
                )
            else:
                checkpoint = execute_dynamic_tn_reverse_segment(
                    dag,
                    reverse,
                    segment,
                    tape,
                    torch.ones_like(tape[dag.output_value_id]),
                    group_cache=group_cache,
                    max_records=2,
                    source_forward_provider=forward_provider,
                )
                if expect_plan_mismatch or expect_tape_mismatch:
                    raise RuntimeError("rank-divergent reverse input was not rejected")
                if checkpoint.completed or checkpoint.next_record_index != 2:
                    raise RuntimeError("dynamic reverse checkpoint boundary is invalid")
            if checkpoint_directory and checkpoint_stage != "resume":
                writer_conflict = os.environ.get("FQ_PRECREATE_CHECKPOINT_LOCK") == "1"
                if writer_conflict and dist.get_rank() == 0:
                    os.makedirs(checkpoint_directory, exist_ok=True)
                    with open(
                        os.path.join(checkpoint_directory, ".checkpoint.lock"),
                        "x",
                        encoding="utf-8",
                    ) as stream:
                        stream.write("competing-writer")
                dist.barrier()
                save_dynamic_tn_reverse_checkpoint(
                    checkpoint, segment, checkpoint_directory
                )
                if checkpoint_stage == "save":
                    stage_saved = True
                    actual = None
                else:
                    del checkpoint
                    corrupt_checkpoint = os.environ.get("FQ_CORRUPT_CHECKPOINT") == "1"
                    if corrupt_checkpoint and dist.get_rank() == 0:
                        with open(
                            os.path.join(checkpoint_directory, "rank-00001.pt"),
                            "ab",
                        ) as stream:
                            stream.write(b"corrupt")
                    remove_commit = os.environ.get("FQ_REMOVE_CHECKPOINT_COMMIT") == "1"
                    if remove_commit and dist.get_rank() == 0:
                        os.unlink(os.path.join(checkpoint_directory, "COMMITTED"))
                    dist.barrier()
                    checkpoint = load_dynamic_tn_reverse_checkpoint(
                        segment,
                        checkpoint_directory,
                        device=device,
                    )
                    if corrupt_checkpoint:
                        raise RuntimeError("corrupt checkpoint was not rejected")
            if checkpoint_stage != "resume" and not stage_saved:
                actual = execute_dynamic_tn_reverse_segment(
                    dag,
                    reverse,
                    segment,
                    tape,
                    None,
                    group_cache=group_cache,
                    resume_from=checkpoint,
                    source_forward_provider=forward_provider,
                )
    except RuntimeError as error:
        plan_failed_closed = expect_plan_mismatch and "differs across ranks" in str(
            error
        )
        tape_failed_closed = expect_tape_mismatch and "tensor preflight failed" in str(
            error
        )
        checkpoint_failed_closed = os.environ.get(
            "FQ_CORRUPT_CHECKPOINT"
        ) == "1" and "integrity check" in str(error)
        uncommitted_failed_closed = os.environ.get(
            "FQ_REMOVE_CHECKPOINT_COMMIT"
        ) == "1" and "not durably committed" in str(error)
        writer_conflict_failed_closed = os.environ.get(
            "FQ_PRECREATE_CHECKPOINT_LOCK"
        ) == "1" and "writer lock is already held" in str(error)
        provider_budget_failed_closed = (
            reject_provider_budget and "forward provider failed" in str(error)
        )
        if (
            not plan_failed_closed
            and not tape_failed_closed
            and not checkpoint_failed_closed
            and not uncommitted_failed_closed
            and not writer_conflict_failed_closed
            and not provider_budget_failed_closed
        ):
            raise
        print(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "segment_consensus_fail_closed": plan_failed_closed,
                    "tensor_preflight_fail_closed": tape_failed_closed,
                    "checkpoint_corruption_fail_closed": (checkpoint_failed_closed),
                    "uncommitted_checkpoint_fail_closed": (uncommitted_failed_closed),
                    "checkpoint_writer_conflict_fail_closed": (
                        writer_conflict_failed_closed
                    ),
                    "rematerialization_budget_fail_closed": (
                        provider_budget_failed_closed
                    ),
                }
            ),
            flush=True,
        )
        dist.destroy_process_group()
        return
    if stage_saved:
        print(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "durable_checkpoint_stage_saved": True,
                }
            ),
            flush=True,
        )
        dist.destroy_process_group()
        return
    for value_id, expected in reference.input_cotangents.items():
        torch.testing.assert_close(actual.cotangents[value_id], expected)
    if actual.accumulated_cotangent_count != 1:
        raise RuntimeError("shared intermediate cotangent was not accumulated")
    if actual.pending_cotangent_contributions:
        raise RuntimeError("shared DAG left incomplete cotangent contributions")
    print(
        json.dumps(
            {
                "rank": dist.get_rank(),
                "shared_dag_reverse_passed": True,
                "executed_reverse_ids": actual.executed_reverse_ids,
                "accumulated_cotangent_count": (actual.accumulated_cotangent_count),
                "pending_cotangent_contribution_count": 0,
                "rank_consensus_validated": actual.rank_consensus_validated,
                "rank_tensor_preflight_validated": (
                    actual.rank_tensor_preflight_validated
                ),
                "checkpoint_resume_passed": True,
                "checkpoint_record_index": checkpoint.next_record_index,
                "completed_after_resume": actual.completed,
                "durable_checkpoint_reload_passed": bool(
                    checkpoint_directory and checkpoint_stage != "save"
                ),
                "process_group_restart_resume_passed": (checkpoint_stage == "resume"),
                "forward_provider_rematerialization_passed": (
                    use_forward_provider
                    and actual.rematerialized_forward_value_count == 3
                ),
                "rematerialized_forward_value_count": (
                    actual.rematerialized_forward_value_count
                ),
                "rematerialization_operation_count": (
                    forward_provider.operation_count if use_forward_provider else 0
                ),
                "rematerialization_peak_transient_bytes": (
                    forward_provider.peak_transient_bytes if use_forward_provider else 0
                ),
                "rematerialization_released_transient_values": (
                    forward_provider.released_transient_value_count
                    if use_forward_provider
                    else 0
                ),
                "partial_mesh_forward_reduction_passed": (use_forward_provider),
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
