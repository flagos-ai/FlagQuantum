#!/usr/bin/env python3
"""Measure forward, backward, and optimizer end-to-end statevector scaling."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402, I001
from flagquantum.runtime.backends.statevector.forward import (  # noqa: E402
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.backends.statevector.reverse import (  # noqa: E402
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)
from flagquantum.runtime.mps_training import (  # noqa: E402
    OwnerShardedParameterSynchronizer,
)

SCHEMA = "flagquantum.statevector.training_scaling.v1"


def build_capacity_workload(
    n_wires: int, device: torch.device
) -> tuple[fq.Circuit, tuple[torch.Tensor, ...], dict[str, Any]]:
    theta = torch.tensor(0.23, device=device, requires_grad=True)
    phi = torch.tensor(-0.37, device=device, requires_grad=True)
    circuit = fq.Circuit(n_wires, device=device)
    circuit.ry(n_wires - 1, theta)
    circuit.cx(n_wires - 1, n_wires - 2).ry(n_wires - 2, phi)
    return (
        circuit,
        (theta, phi),
        {
            "name": "analytic_capacity_microbenchmark",
            "layers": 1,
            "gate_count": 3,
            "active_wire_count": 2,
            "parameter_count": 2,
            "entanglement": "single_cnot",
            "validation": "closed_form_value_and_gradient",
        },
    )


def build_full_width_workload(
    n_wires: int,
    device: torch.device,
    *,
    layers: int,
    seed: int,
    entanglement: str,
) -> tuple[fq.Circuit, tuple[torch.Tensor, ...], dict[str, Any]]:
    """Build a deterministic full-width HEA with explicit entanglement topology."""

    if layers <= 0:
        raise ValueError("layers must be positive")
    parameters = tuple(
        torch.tensor(
            0.11 + 0.013 * ((seed + layer * n_wires + wire) % 37),
            device=device,
            requires_grad=True,
        )
        for layer in range(layers)
        for wire in range(n_wires)
    )
    circuit = fq.Circuit(n_wires, device=device)
    cursor = 0
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.ry(wire, parameters[cursor])
            cursor += 1
        # Alternate the traversal direction so repeated layers do not privilege
        # one end of the logical line. Every wire is already active through RY;
        # the optional closing edge is retained as an explicit stress workload.
        if entanglement == "brickwork":
            if layer % 2 == 0:
                edges = [
                    (wire, wire + 1)
                    for parity in (0, 1)
                    for wire in range(parity, n_wires - 1, 2)
                ]
            else:
                edges = [
                    (wire + 1, wire)
                    for parity in (1, 0)
                    for wire in range(n_wires - 2 - parity, -1, -2)
                ]
        elif layer % 2 == 0:
            edges = [(wire, wire + 1) for wire in range(n_wires - 1)]
        else:
            edges = [(wire + 1, wire) for wire in range(n_wires - 2, -1, -1)]
        if entanglement == "ring":
            if layer % 2 == 0:
                edges.append((n_wires - 1, 0))
            else:
                edges.append((0, n_wires - 1))
        elif entanglement not in {"linear", "brickwork"}:
            raise ValueError("entanglement must be 'linear', 'ring', or 'brickwork'")
        for control, target in edges:
            circuit.cx(control, target)
    return (
        circuit,
        parameters,
        {
            "name": f"full_width_{entanglement}_hea",
            "layers": layers,
            "gate_count": (
                (2 * n_wires if entanglement == "ring" else 2 * n_wires - 1) * layers
            ),
            "active_wire_count": n_wires,
            "parameter_count": n_wires * layers,
            "entanglement": f"directed_cnot_{entanglement}",
            "initialization": "deterministic_nonzero_ry_angles",
            "seed": seed,
            "validation": "dense_autograd_small_scale_plus_distributed_invariants",
        },
    )


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _elapsed_seconds(started: float, device: torch.device) -> float:
    _sync(device)
    return time.perf_counter() - started


def _global_max(values: list[float], device: torch.device) -> list[float]:
    packed = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return [float(value) for value in packed.cpu()]


def _release_phase_storage(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    _sync(device)


def _analytic(theta: float, phi: float) -> tuple[float, tuple[float, float]]:
    value = math.cos(theta) * math.cos(phi)
    return value, (
        -math.sin(theta) * math.cos(phi),
        -math.cos(theta) * math.sin(phi),
    )


def _timing(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.repetitions < 3 or args.warmup < 0:
        raise ValueError("repetitions >= 3 and warmup >= 0 required")
    if args.reverse_chunk_amplitudes <= 0 or args.reverse_chunk_amplitudes & (
        args.reverse_chunk_amplitudes - 1
    ):
        raise ValueError("reverse chunk amplitudes must be a positive power of two")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["FQ_STATEVECTOR_COMM_AWARE_LAYOUT"] = (
        "0" if args.wire_layout == "canonical" else "1"
    )
    os.environ["FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES"] = str(
        args.reverse_chunk_amplitudes
    )
    os.environ["FQ_STATEVECTOR_REVERSE_EXCHANGE_WORKSPACE"] = (
        "0" if args.disable_reverse_workspace else "1"
    )
    os.environ["FQ_STATEVECTOR_TRITON_VJP_ADJOINT"] = (
        "1" if args.enable_triton_vjp_adjoint else "0"
    )
    os.environ["FQ_STATEVECTOR_FUSED_VJP_PIPELINE"] = (
        "1" if args.enable_fused_vjp_pipeline else "0"
    )
    os.environ["FQ_STATEVECTOR_CROSS_SHARD_CX_PACK"] = (
        "1"
        if (
            args.enable_cross_shard_cx_pack
            or args.enable_forward_cross_shard_cx_pack
        )
        else "0"
    )
    os.environ["FQ_STATEVECTOR_REVERSE_CROSS_SHARD_CX_PACK"] = (
        "1" if args.enable_cross_shard_cx_pack else "0"
    )
    os.environ["FQ_STATEVECTOR_TOPOLOGY_AWARE_RANK_BITS"] = (
        "0" if args.disable_topology_aware_rank_bits else "1"
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if args.workload == "capacity":
            circuit, parameters, workload_metadata = build_capacity_workload(
                args.n_wires, device
            )
            checkpoint_policy = StatevectorCheckpointPolicy()
            observable_wire = args.n_wires - 2
        else:
            entanglement = {
                "full-width-ring": "ring",
                "full-width-brickwork": "brickwork",
            }.get(args.workload, "linear")
            circuit, parameters, workload_metadata = build_full_width_workload(
                args.n_wires,
                device,
                layers=args.layers,
                seed=args.seed,
                entanglement=entanglement,
            )
            checkpoint_policy = StatevectorCheckpointPolicy(
                strategy="reversible_adjoint"
            )
            observable_wire = args.n_wires // 2
        owners = tuple(index % world for index in range(len(parameters)))
        owned = [p for p, owner in zip(parameters, owners) if owner == rank]
        optimizer = torch.optim.Adam(owned, lr=args.learning_rate) if owned else None
        parameter_synchronizer = OwnerShardedParameterSynchronizer(
            parameters=parameters,
            owners=owners,
            rank=rank,
            world_size=world,
        )

        def reverse_pass(*, update: bool) -> dict[str, Any]:
            for parameter in parameters:
                parameter.grad = None
            dist.barrier()
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            end_to_end_started = time.perf_counter()
            differentiable_forward_started = time.perf_counter()
            reverse = execute_torch_distributed_statevector_reverse(
                circuit,
                observable_wire=observable_wire,
                checkpoint_policy=checkpoint_policy,
                device=device,
            )
            differentiable_forward = _elapsed_seconds(
                differentiable_forward_started, device
            )
            backward_started = time.perf_counter()
            reverse.backward()
            backward = _elapsed_seconds(backward_started, device)
            reverse_summary = reverse.summary()
            before = tuple(float(parameter.detach().cpu()) for parameter in parameters)
            gradients = tuple(
                float(parameter.grad.detach().cpu()) for parameter in parameters
            )
            value = float(reverse.value.detach().cpu())
            if args.workload == "capacity":
                expected_value, expected_gradients = _analytic(*before)
                value_error = abs(value - expected_value)
                gradient_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(gradients, expected_gradients)
                )
            else:
                expected_value = None
                expected_gradients = None
                value_error = None
                gradient_error = None
            optimizer_started = time.perf_counter()
            if update and optimizer is not None:
                optimizer.step()
            if update and world > 1:
                if args.disable_batched_parameter_sync:
                    for parameter, owner in zip(parameters, owners):
                        dist.broadcast(parameter.data, src=owner)
                else:
                    parameter_synchronizer.synchronize()
            optimizer_seconds = _elapsed_seconds(optimizer_started, device)
            end_to_end = _elapsed_seconds(end_to_end_started, device)
            (
                differentiable_forward,
                backward,
                optimizer_seconds,
                end_to_end,
                peak_memory,
            ) = _global_max(
                [
                    differentiable_forward,
                    backward,
                    optimizer_seconds,
                    end_to_end,
                    float(torch.cuda.max_memory_allocated(device)),
                ],
                device,
            )
            return {
                "distribution_semantics": reverse_summary[
                    "distribution_semantics"
                ],
                "forward_distribution_semantics": reverse_summary[
                    "forward_distribution_semantics"
                ],
                "backward_distribution_semantics": reverse_summary[
                    "backward_distribution_semantics"
                ],
                "gradient_distribution": reverse_summary["gradient_distribution"],
                "differentiable_forward_seconds": differentiable_forward,
                "backward_seconds": backward,
                "optimizer_seconds": optimizer_seconds,
                "end_to_end_seconds": end_to_end,
                "peak_memory_bytes": int(peak_memory),
                "value": value,
                "parameters": before,
                "expected_value": expected_value,
                "gradients": gradients,
                "expected_gradients": expected_gradients,
                "value_absolute_error": value_error,
                "gradient_absolute_error_max": gradient_error,
                "value_finite": math.isfinite(value),
                "gradients_finite": all(math.isfinite(item) for item in gradients),
                "backward_communication_bytes": reverse.summary()[
                    "backward_communication_bytes"
                ],
                "backward_communication_count": reverse_summary[
                    "backward_communication_count"
                ],
                "backward_intra_node_communication_count": reverse_summary[
                    "backward_intra_node_communication_count"
                ],
                "backward_intra_node_communication_bytes": reverse_summary[
                    "backward_intra_node_communication_bytes"
                ],
                "backward_inter_node_communication_count": reverse_summary[
                    "backward_inter_node_communication_count"
                ],
                "backward_inter_node_communication_bytes": reverse_summary[
                    "backward_inter_node_communication_bytes"
                ],
                "peak_backward_scratch_bytes": reverse_summary[
                    "peak_backward_scratch_bytes"
                ],
                "checkpoint_policy": reverse_summary["checkpoint_policy"],
                "saved_forward_state_reused": reverse_summary[
                    "saved_forward_state_reused"
                ],
                "backward_exchange_chunk_amplitudes": reverse_summary[
                    "backward_exchange_chunk_amplitudes"
                ],
                "backward_exchange_chunk_bytes": reverse_summary[
                    "backward_exchange_chunk_bytes"
                ],
                "backward_exchange_workspace_allocation_count": reverse_summary[
                    "backward_exchange_workspace_allocation_count"
                ],
                "backward_exchange_workspace_reuse_count": reverse_summary[
                    "backward_exchange_workspace_reuse_count"
                ],
                "backward_exchange_workspace_reserved_bytes": reverse_summary[
                    "backward_exchange_workspace_reserved_bytes"
                ],
                "backward_exchange_pipeline_prefetch_count": reverse_summary[
                    "backward_exchange_pipeline_prefetch_count"
                ],
            }

        forward_samples: list[float] = []
        forward_peaks: list[int] = []
        forward_communication_bytes: list[int] = []
        forward_communication_counts: list[int] = []
        forward_distributed_gate_counts: list[int] = []
        forward_intra_node_counts: list[int] = []
        forward_intra_node_bytes: list[int] = []
        forward_inter_node_counts: list[int] = []
        forward_inter_node_bytes: list[int] = []
        for _ in range(args.repetitions):
            dist.barrier()
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.no_grad():
                forward = execute_torch_distributed_statevector(
                    circuit,
                    device=device,
                    wire_layout=args.wire_layout.replace("-", "_"),
                    preferred_local_wires=(observable_wire,),
                )
            forward_seconds = _elapsed_seconds(started, device)
            forward_summary = forward.summary()
            (
                forward_seconds,
                forward_peak,
                communication_bytes,
                communication_count,
                distributed_gate_count,
                intra_node_count,
                intra_node_bytes,
                inter_node_count,
                inter_node_bytes,
            ) = _global_max(
                [
                    forward_seconds,
                    float(torch.cuda.max_memory_allocated(device)),
                    float(forward_summary["communication_bytes"]),
                    float(forward_summary["communication_count"]),
                    float(forward_summary["distributed_gate_count"]),
                    float(forward_summary["intra_node_communication_count"]),
                    float(forward_summary["intra_node_communication_bytes"]),
                    float(forward_summary["inter_node_communication_count"]),
                    float(forward_summary["inter_node_communication_bytes"]),
                ],
                device,
            )
            forward_samples.append(forward_seconds)
            forward_peaks.append(int(forward_peak))
            forward_communication_bytes.append(int(communication_bytes))
            forward_communication_counts.append(int(communication_count))
            forward_distributed_gate_counts.append(int(distributed_gate_count))
            forward_intra_node_counts.append(int(intra_node_count))
            forward_intra_node_bytes.append(int(intra_node_bytes))
            forward_inter_node_counts.append(int(inter_node_count))
            forward_inter_node_bytes.append(int(inter_node_bytes))
            del forward
            _release_phase_storage(device)

        for _ in range(args.warmup):
            reverse_pass(update=False)
            _release_phase_storage(device)

        measured = []
        for _ in range(args.repetitions):
            measured.append(reverse_pass(update=True))
            _release_phase_storage(device)

        final_parameters = tuple(
            float(parameter.detach().cpu()) for parameter in parameters
        )
        measured_initial = measured[0]
        local_record = {
            "rank": rank,
            "hostname": platform.node(),
            "local_rank": local_rank,
            "final_parameters": final_parameters,
            "initial_parameters": measured_initial["parameters"],
            "initial_value": measured_initial["value"],
            "initial_gradients": measured_initial["gradients"],
            "end_to_end_samples_seconds": tuple(
                row["end_to_end_seconds"] for row in measured
            ),
            "backward_samples_seconds": tuple(
                row["backward_seconds"] for row in measured
            ),
            "peak_memory_bytes": max(row["peak_memory_bytes"] for row in measured),
            "backward_communication_bytes": max(
                row["backward_communication_bytes"] for row in measured
            ),
        }
        records: list[dict[str, Any] | None] = [None] * world
        dist.all_gather_object(records, local_record)
        if rank != 0:
            return None
        complete_records = [record for record in records if record is not None]
        rank_placement = [
            {
                "rank": record["rank"],
                "hostname": record["hostname"],
                "local_rank": record["local_rank"],
            }
            for record in complete_records
        ]
        numeric_value_errors = [
            row["value_absolute_error"]
            for row in measured
            if row["value_absolute_error"] is not None
        ]
        numeric_gradient_errors = [
            row["gradient_absolute_error_max"]
            for row in measured
            if row["gradient_absolute_error_max"] is not None
        ]
        max_value_error = max(numeric_value_errors, default=None)
        max_gradient_error = max(numeric_gradient_errors, default=None)
        finite_results = all(
            row["value_finite"] and row["gradients_finite"] for row in measured
        )
        parameter_spread = max(
            max(
                abs(a - b) for a, b in zip(record["final_parameters"], final_parameters)
            )
            for record in complete_records
        )
        reference_value_error = None
        reference_gradient_error = None
        reference_world_size = None
        if args.reference_json is not None:
            reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
            reference_workload = reference.get("workload", {})
            if (
                reference_workload.get("n_wires") != args.n_wires
                or reference_workload.get("name") != workload_metadata["name"]
                or reference_workload.get("layers") != workload_metadata["layers"]
            ):
                raise ValueError("reference JSON workload does not match this run")
            reference_correctness = reference.get("correctness", {})
            reference_gradients = reference_correctness.get("initial_gradients")
            reference_value = reference_correctness.get("initial_value")
            if reference_gradients is None or reference_value is None:
                raise ValueError(
                    "reference JSON must contain correctness.initial_value and "
                    "correctness.initial_gradients"
                )
            reference_value_error = abs(measured_initial["value"] - reference_value)
            reference_gradient_error = max(
                abs(actual - expected)
                for actual, expected in zip(
                    measured_initial["gradients"], reference_gradients, strict=True
                )
            )
            reference_world_size = reference.get("world_size")
        reference_matches = (
            reference_value_error is None
            or (
                reference_value_error <= args.atol
                and reference_gradient_error is not None
                and reference_gradient_error <= args.atol
            )
        )
        payload = {
            "schema_version": SCHEMA,
            "benchmark": "statevector_differentiable_training_scaling",
            "artifact_class": "measured_development_run",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "backend": "nccl",
            "world_size": world,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world)),
            "node_count": len({record["hostname"] for record in complete_records}),
            "rank_placement": rank_placement,
            "rank_measurements": complete_records,
            "local_memory_bytes_by_rank": [
                int(record["peak_memory_bytes"]) for record in complete_records
            ],
            "distribution_semantics": forward_summary["distribution_semantics"],
            "forward_distribution_semantics": forward_summary[
                "distribution_semantics"
            ],
            "backward_distribution_semantics": measured_initial[
                "backward_distribution_semantics"
            ],
            "gradient_distribution": measured_initial["gradient_distribution"],
            "workload": {
                "n_wires": args.n_wires,
                **workload_metadata,
                "observable": f"Z({observable_wire})",
                "dtype": "complex64",
                "optimizer": "adam",
                "learning_rate": args.learning_rate,
                "local_amplitudes": 2**args.n_wires // world,
            },
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "forward": {
                **_timing(forward_samples),
                "logical_to_physical_wires": forward_summary[
                    "logical_to_physical_wires"
                ],
                "peak_memory_bytes_max": max(forward_peaks),
                "communication_bytes_per_rank_max": max(forward_communication_bytes),
                "communication_count_max": max(forward_communication_counts),
                "distributed_gate_count": max(forward_distributed_gate_counts),
                "intra_node_communication_count_max": max(forward_intra_node_counts),
                "intra_node_communication_bytes_per_rank_max": max(
                    forward_intra_node_bytes
                ),
                "inter_node_communication_count_max": max(forward_inter_node_counts),
                "inter_node_communication_bytes_per_rank_max": max(
                    forward_inter_node_bytes
                ),
            },
            "differentiable_forward": _timing(
                [row["differentiable_forward_seconds"] for row in measured]
            ),
            "backward": _timing([row["backward_seconds"] for row in measured]),
            "optimizer": _timing([row["optimizer_seconds"] for row in measured]),
            "end_to_end": {
                **_timing([row["end_to_end_seconds"] for row in measured]),
                "peak_memory_bytes_max": max(
                    row["peak_memory_bytes"] for row in measured
                ),
            },
            "correctness": {
                "passed": finite_results
                and parameter_spread == 0.0
                and reference_matches
                and (
                    max_value_error is None
                    or (
                        max_value_error <= args.atol
                        and max_gradient_error is not None
                        and max_gradient_error <= args.atol
                    )
                ),
                "absolute_tolerance": args.atol,
                "value_absolute_error_max": max_value_error,
                "gradient_absolute_error_max": max_gradient_error,
                "final_parameter_spread": parameter_spread,
                "final_parameters": final_parameters,
                "initial_parameters": measured_initial["parameters"],
                "initial_value": measured_initial["value"],
                "initial_gradients": measured_initial["gradients"],
                "reference_world_size": reference_world_size,
                "reference_value_absolute_error": reference_value_error,
                "reference_gradient_absolute_error_max": reference_gradient_error,
                "all_values_and_gradients_finite": finite_results,
                "validation_method": workload_metadata["validation"],
            },
            "backward_communication_bytes_per_rank_max": max(
                row["backward_communication_bytes"] for row in measured
            ),
            "backward_communication_count_max": max(
                row["backward_communication_count"] for row in measured
            ),
            "backward_intra_node_communication_count_max": max(
                row["backward_intra_node_communication_count"] for row in measured
            ),
            "backward_intra_node_communication_bytes_per_rank_max": max(
                row["backward_intra_node_communication_bytes"] for row in measured
            ),
            "backward_inter_node_communication_count_max": max(
                row["backward_inter_node_communication_count"] for row in measured
            ),
            "backward_inter_node_communication_bytes_per_rank_max": max(
                row["backward_inter_node_communication_bytes"] for row in measured
            ),
            "communication_bytes": max(forward_communication_bytes)
            + max(row["backward_communication_bytes"] for row in measured),
            "communication_bytes_scope": "forward_plus_backward_per_rank_maxima",
            "intra_node_communication_bytes": max(forward_intra_node_bytes)
            + max(
                row["backward_intra_node_communication_bytes"] for row in measured
            ),
            "inter_node_communication_bytes": max(forward_inter_node_bytes)
            + max(
                row["backward_inter_node_communication_bytes"] for row in measured
            ),
            "peak_backward_scratch_bytes_max": max(
                row["peak_backward_scratch_bytes"] for row in measured
            ),
            "backward_exchange_chunk_amplitudes": measured[0][
                "backward_exchange_chunk_amplitudes"
            ],
            "backward_exchange_chunk_bytes": measured[0][
                "backward_exchange_chunk_bytes"
            ],
            "backward_exchange_workspace_allocation_count_max": max(
                row["backward_exchange_workspace_allocation_count"] for row in measured
            ),
            "backward_exchange_workspace_reuse_count_max": max(
                row["backward_exchange_workspace_reuse_count"] for row in measured
            ),
            "backward_exchange_workspace_reserved_bytes_max": max(
                row["backward_exchange_workspace_reserved_bytes"] for row in measured
            ),
            "backward_exchange_pipeline_prefetch_count_max": max(
                row["backward_exchange_pipeline_prefetch_count"] for row in measured
            ),
            "checkpoint_policy": measured[0]["checkpoint_policy"],
            "saved_forward_state_reused": all(
                row["saved_forward_state_reused"] for row in measured
            ),
            "benchmark_protocol": {
                "timing_scope": "barrier_to_global_max_cuda_synchronized",
                "wire_layout": args.wire_layout,
                "topology_aware_rank_bits": (
                    not args.disable_topology_aware_rank_bits
                ),
                "triton_vjp_adjoint": args.enable_triton_vjp_adjoint,
                "fused_vjp_pipeline": args.enable_fused_vjp_pipeline,
                "cross_shard_cx_pack": (
                    args.enable_cross_shard_cx_pack
                    or args.enable_forward_cross_shard_cx_pack
                ),
                "reverse_cross_shard_cx_pack": args.enable_cross_shard_cx_pack,
                "reverse_exchange_workspace": not args.disable_reverse_workspace,
                "initial_state": "dense_storage_zero_basis_then_all_wire_rotations",
                "statistics": "median_with_coefficient_of_variation",
                "sample_count": args.repetitions,
                "parameter_synchronization": (
                    "per_parameter_owner_broadcast"
                    if args.disable_batched_parameter_sync
                    else "owner_masked_packed_all_reduce"
                ),
                "parameter_synchronization_collectives_per_step": (
                    len(parameters)
                    if world > 1 and args.disable_batched_parameter_sync
                    else int(world > 1)
                ),
                "full_state_materialization": False,
                "communication_tier_accounting": (
                    "peer_to_peer_logical_send_bytes_by_torchrun_node; "
                    "collective_reductions_remain_in_total_only"
                ),
            },
            "hardware": {
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "validation_blockers": [],
        }
        if not payload["correctness"]["passed"]:
            payload["validation_blockers"].append("analytic_correctness_failed")
        if not reference_matches:
            payload["validation_blockers"].append("cross_world_size_reference_mismatch")
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(payload, sort_keys=True), flush=True)
        if payload["validation_blockers"]:
            raise SystemExit(2)
        return payload
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument(
        "--workload",
        choices=(
            "capacity",
            "full-width-linear",
            "full-width-ring",
            "full-width-brickwork",
        ),
        default="capacity",
    )
    parser.add_argument(
        "--wire-layout",
        choices=("canonical", "communication-aware"),
        default="communication-aware",
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--reverse-chunk-amplitudes", type=int, default=1 << 22)
    parser.add_argument("--disable-reverse-workspace", action="store_true")
    parser.add_argument("--enable-triton-vjp-adjoint", action="store_true")
    parser.add_argument("--enable-fused-vjp-pipeline", action="store_true")
    parser.add_argument("--enable-cross-shard-cx-pack", action="store_true")
    parser.add_argument("--enable-forward-cross-shard-cx-pack", action="store_true")
    parser.add_argument("--disable-topology-aware-rank-bits", action="store_true")
    parser.add_argument("--disable-batched-parameter-sync", action="store_true")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
