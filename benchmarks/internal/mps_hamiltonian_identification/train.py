"""Recover a non-uniform 1D Hamiltonian from differentiable MPS observations.

The site-sharded observation path is experimental development evidence.  It is
not a production-support or release-certification claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum.runtime.executors.mps.records as fqxm  # noqa: E402
from flagquantum.runtime.executors.mps.forward import (  # noqa: E402
    execute_torch_distributed_mps_forward,
)
from flagquantum.runtime.executors.mps.reverse import (  # noqa: E402
    execute_torch_distributed_mps_reverse,
    site_sharded_z_zz_observations,
)
from flagquantum.simulation.mps.site_kernels import (  # noqa: E402
    reset_site_kernel_stats,
    site_kernel_stats,
)

from core import (  # noqa: E402
    build_variable_time_batched_trotter_circuit,
    make_probes,
    predict_batch,
    relative_parameter_error,
    smooth_couplings,
)


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def _distributed_device(requested: str) -> tuple[torch.device, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "auto":
        requested = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    return device, rank, world, local_rank


def _all_reduce_gradients(parameters: tuple[torch.Tensor, ...], world: int) -> None:
    if world == 1:
        return
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _probe_batches(indices: tuple[int, ...], probes, batch_size: int) -> tuple[tuple[int, ...], ...]:
    grouped: dict[int, list[int]] = {}
    for index in indices:
        grouped.setdefault(probes[index].time_steps, []).append(index)
    batches = []
    for time_depth in sorted(grouped, reverse=True):
        values = grouped[time_depth]
        batches.extend(tuple(values[start : start + batch_size]) for start in range(0, len(values), batch_size))
    return tuple(batches)


def _sharded_observations(coupling, field, probes, observation_sites, args, device):
    circuit = build_variable_time_batched_trotter_circuit(
        coupling, field, probes, dt=args.dt, device=device
    )
    forward = execute_torch_distributed_mps_forward(
        circuit, device=device, max_bond=args.max_bond, cutoff=args.cutoff,
        rebalance_threshold=float("inf"),
        global_error_budget=args.discarded_weight_tolerance,
        error_budget_policy="enforce",
        truncation_gradient_policy=args.gradient_policy,
        compile_site_kernels=args.site_sharded and args.compile_site_kernels,
    )
    return site_sharded_z_zz_observations(
        forward.shard_state, observation_sites,
        compiled=args.site_sharded and args.compile_observables,
    ), forward.summary()


def _mse_terms(targets, observation_sites, n_wires):
    terms = [({wire: "z"}, targets[:, position]) for position, wire in enumerate(observation_sites)]
    offset = len(observation_sites)
    for wire in observation_sites:
        if wire + 1 < n_wires:
            terms.append(({wire: "z", wire + 1: "z"}, targets[:, offset]))
            offset += 1
    return tuple(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=512)
    parser.add_argument("--n-initial-states", type=int, default=8)
    parser.add_argument("--time-steps", type=_parse_ints, default=(1, 2, 3, 4))
    parser.add_argument("--validation-initial-states", type=int, default=0)
    parser.add_argument("--validation-time-steps", type=_parse_ints, default=(5,))
    parser.add_argument("--observation-stride", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument(
        "--reverse-max-saved-gib", type=float, default=24.0,
        help="Per-rank explicit reverse-tape bound for site-sharded training.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.0,
        help="MPS truncation cutoff. Exact QR/autograd (0) is the stable training default; positive cutoffs use complex SVD.",
    )
    parser.add_argument(
        "--gradient-policy",
        choices=("exact", "approximate"),
        default="exact",
        help="Gradient contract for site-sharded truncation. Approximate gradients are accepted only within --discarded-weight-tolerance.",
    )
    parser.add_argument(
        "--discarded-weight-tolerance",
        type=float,
        default=0.0,
        help="Maximum cumulative discarded weight per optimizer step (and per teacher forward microbatch).",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-loss", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--heartbeat-probes", type=int, default=1)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print rank-0 loss every N steps; 0 prints approximately ten times per run.",
    )
    parser.add_argument("--quiet-rank-heartbeats", action="store_true")
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument(
        "--compile-brickwork",
        action="store_true",
        help="Use shape-bucketed torch.compile kernels for the RY/RXX brickwork sweep.",
    )
    parser.add_argument(
        "--site-sharded",
        action="store_true",
        help="Shard one batched MPS by contiguous sites across all torchrun ranks.",
    )
    parser.add_argument(
        "--compile-site-kernels",
        action="store_true",
        help="Compile audited rank-local RY/RXX buckets while preserving NCCL boundary execution.",
    )
    parser.add_argument(
        "--replicated-capacity-baseline",
        action="store_true",
        help="Run the complete probe set on every rank for an explicit non-scalable capacity baseline.",
    )
    parser.add_argument(
        "--compile-observables",
        action="store_true",
        help="Use the experimental real-channel compiled Z/ZZ environment scan (requires --compile-brickwork).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--single-step-acceptance",
        action="store_true",
        help="Fail-closed one-step gate for backward, optimizer update, peak memory, and discarded-weight budget.",
    )
    parser.add_argument(
        "--peak-memory-budget-gib",
        type=float,
        default=38.0,
        help="Per-rank CUDA allocated-memory ceiling used by --single-step-acceptance.",
    )
    parser.add_argument("--output", type=Path, default=Path("mps-system-id.json"))
    args = parser.parse_args()
    if args.compile_observables and not (
        args.compile_brickwork or (args.site_sharded and args.compile_site_kernels)
    ):
        parser.error("--compile-observables requires --compile-brickwork or site-sharded compiled kernels")
    if args.compile_site_kernels and not args.site_sharded:
        parser.error("--compile-site-kernels requires --site-sharded")
    if args.site_sharded and args.replicated_capacity_baseline:
        parser.error("--site-sharded and --replicated-capacity-baseline are mutually exclusive")
    if args.gradient_policy == "approximate" and not args.site_sharded:
        parser.error("--gradient-policy approximate requires --site-sharded")
    if args.single_step_acceptance and (not args.site_sharded or args.steps != 1):
        parser.error("--single-step-acceptance requires --site-sharded and --steps 1")
    if args.gradient_policy == "exact" and args.discarded_weight_tolerance != 0:
        parser.error("exact gradient policy requires --discarded-weight-tolerance 0")
    if args.gradient_policy == "exact" and args.cutoff > 0:
        parser.error("positive --cutoff requires --gradient-policy approximate")
    if args.n_qubits < 2 or args.observation_stride < 1 or args.steps < 1 or args.heartbeat_probes < 1 or args.probe_batch_size < 1 or args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0 or args.validation_initial_states < 0 or args.log_every < 0 or args.reverse_max_saved_gib <= 0 or args.discarded_weight_tolerance < 0 or args.peak_memory_budget_gib <= 0:
        raise SystemExit("n-wires >= 2, observation-stride >= 1, and steps >= 1 are required")

    device, rank, world, local_rank = _distributed_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    truth_j, truth_h = smooth_couplings(args.n_qubits, device=device)
    probes = make_probes(
        args.n_qubits,
        n_initial_states=args.n_initial_states,
        time_steps=args.time_steps,
        seed=args.seed,
    )
    observation_sites = tuple(range(0, args.n_qubits, args.observation_stride))
    if args.site_sharded:
        if world < 2:
            raise SystemExit("--site-sharded requires torchrun with at least two ranks")
        reset_site_kernel_stats()
        batches = _probe_batches(tuple(range(len(probes))), probes, args.probe_batch_size)
        targets = {}
        teacher_records = []
        teacher_discarded_weight = 0.0
        with torch.no_grad():
            for batch_indices in batches:
                value, record = _sharded_observations(
                    truth_j, truth_h,
                    tuple(probes[index] for index in batch_indices),
                    observation_sites, args, device,
                )
                targets[batch_indices] = value.detach()
                teacher_records.append(record)
                teacher_discarded_weight += float(record["truncation_error"])
        estimate_j = torch.full_like(truth_j, 0.62, requires_grad=True)
        estimate_h = torch.full_like(truth_h, 0.52, requires_grad=True)
        parameters = (estimate_j, estimate_h)
        owners = (0, 1 % world)
        owned = [parameter for parameter, owner in zip(parameters, owners) if owner == rank]
        optimizer = torch.optim.Adam(owned, lr=args.lr) if owned else None
        history = []
        best_loss, best_step, stale_steps = float("inf"), 0, 0
        improvement_anchor = float("inf")
        best_parameters = None
        stop_reason = "maximum_steps_reached"
        optimizer_steps = 0
        started = time.perf_counter()
        communication_bytes = 0
        step_seconds = []
        step_acceptance_records = []
        for step in range(args.steps):
            step_started = time.perf_counter()
            parameters_before = tuple(parameter.detach().clone() for parameter in parameters)
            for parameter in parameters:
                parameter.grad = None
            step_loss = torch.zeros((), device=device)
            step_discarded_weight = 0.0
            reverse_summaries = []
            for batch_indices in batches:
                batch_probes = tuple(probes[index] for index in batch_indices)
                circuit = build_variable_time_batched_trotter_circuit(
                    estimate_j, estimate_h, batch_probes, dt=args.dt, device=device
                )
                reverse = execute_torch_distributed_mps_reverse(
                    circuit,
                    observable_terms=_mse_terms(targets[batch_indices], observation_sites, args.n_qubits),
                    device=device, max_bond=args.max_bond, cutoff=args.cutoff,
                    gradient_policy=args.gradient_policy,
                    gradient_tolerance=args.discarded_weight_tolerance,
                    checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
                        max_saved_bytes=int(args.reverse_max_saved_gib * (1 << 30))
                    ),
                    compile_site_kernels=args.compile_site_kernels,
                    compile_observables=args.compile_observables,
                )
                before = [None if p.grad is None else p.grad.clone() for p in parameters]
                reverse.backward()
                reverse_summary = reverse.summary()
                reverse_summaries.append(reverse_summary)
                step_discarded_weight += float(reverse.discarded_weight)
                weight = len(batch_indices) / len(probes)
                for parameter, previous in zip(parameters, before):
                    base = torch.zeros_like(parameter) if previous is None else previous
                    parameter.grad.copy_(base + (parameter.grad - base) * weight)
                step_loss += reverse.value.detach() * weight
                communication_bytes += sum(
                    sum(math.prod(shape) * 8 for shape in record.input_shapes + record.output_shapes)
                    for record in reverse.tape.records if record.communication_peer is not None
                )
            if step_discarded_weight > args.discarded_weight_tolerance:
                raise RuntimeError(
                    "optimizer-step discarded weight exceeded tolerance: "
                    f"{step_discarded_weight:.9g} > {args.discarded_weight_tolerance:.9g}"
                )
            local_gradient_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            )
            local_owned_gradient_nonzero = all(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                for parameter in owned
            )
            if optimizer is not None:
                optimizer.step()
            for parameter, owner in zip(parameters, owners):
                dist.broadcast(parameter.data, src=owner)
            optimizer_steps += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - step_started)
            parameter_update_max_abs = tuple(
                float((parameter.detach() - previous).abs().max().cpu())
                for parameter, previous in zip(parameters, parameters_before)
            )
            local_step_acceptance = {
                "rank": rank,
                "backward_completed": all(
                    item["mps_backward_execution"] == "completed"
                    for item in reverse_summaries
                ),
                "gradient_finite": local_gradient_finite,
                "owned_gradient_nonzero": local_owned_gradient_nonzero,
                "optimizer_update_max_abs": parameter_update_max_abs,
                "optimizer_updated": all(value > 0 for value in parameter_update_max_abs),
                "discarded_weight": step_discarded_weight,
                "discarded_weight_tolerance": args.discarded_weight_tolerance,
                "discarded_weight_budget_satisfied": (
                    step_discarded_weight <= args.discarded_weight_tolerance
                ),
                "reverse_summaries": reverse_summaries,
            }
            gathered_step_acceptance = [None] * world
            dist.all_gather_object(gathered_step_acceptance, local_step_acceptance)
            step_acceptance_records.append(gathered_step_acceptance)
            value = float(step_loss.cpu())
            history.append(value)
            if value < best_loss:
                best_loss, best_step = value, step + 1
                best_parameters = tuple(parameter.detach().clone() for parameter in parameters)
                checkpoint = args.output.with_suffix(args.output.suffix + f".best.rank-{rank}.pt")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"step": best_step, "loss": best_loss, "owned_parameters": {
                    index: parameters[index].detach().cpu() for index, owner in enumerate(owners) if owner == rank
                }, "optimizer": None if optimizer is None else optimizer.state_dict()}, checkpoint)
            if value < improvement_anchor - args.early_stopping_min_delta:
                improvement_anchor = value
                stale_steps = 0
            else:
                stale_steps += 1
            log_every = args.log_every or max(1, args.steps // 10)
            if rank == 0 and (
                step == 0 or step + 1 == args.steps or (step + 1) % log_every == 0
            ):
                print(json.dumps({"step": step + 1, "loss": value}), flush=True)
            if args.early_stopping_loss is not None and value <= args.early_stopping_loss:
                stop_reason = "target_loss_reached"
                break
            if args.early_stopping_patience and stale_steps >= args.early_stopping_patience:
                stop_reason = "patience_exhausted"
                break
        assert best_parameters is not None
        for parameter, best in zip(parameters, best_parameters):
            parameter.data.copy_(best)
        elapsed = time.perf_counter() - started
        validation = None
        if args.validation_initial_states:
            validation_probes = make_probes(args.n_qubits, n_initial_states=args.validation_initial_states,
                time_steps=args.validation_time_steps, seed=args.seed + 1)
            validation_sum = torch.zeros((), device=device)
            validation_count = 0
            with torch.no_grad():
                for indices in _probe_batches(tuple(range(len(validation_probes))), validation_probes, args.probe_batch_size):
                    selected = tuple(validation_probes[index] for index in indices)
                    truth, _ = _sharded_observations(truth_j, truth_h, selected, observation_sites, args, device)
                    prediction, _ = _sharded_observations(estimate_j, estimate_h, selected, observation_sites, args, device)
                    validation_sum += (prediction - truth).square().mean(dim=-1).sum()
                    validation_count += len(indices)
            validation = {"probe_count": validation_count, "mse": float((validation_sum / validation_count).cpu()),
                "seed": args.seed + 1, "time_steps": args.validation_time_steps,
                "execution": "site_sharded_observation_scan"}
        rank_record = {
            **teacher_records[-1],
            "device": str(device),
            "hostname": socket.gethostname(),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else None
            ),
            "allocated_memory_bytes": (
                int(torch.cuda.memory_allocated(device))
                if device.type == "cuda" else None
            ),
            "compiled_site_kernels": site_kernel_stats(),
        }
        records = [None] * world
        dist.all_gather_object(records, rank_record)
        peak_memory_budget_bytes = int(args.peak_memory_budget_gib * (1 << 30))
        peak_memory_bytes = max((item["peak_memory_bytes"] or 0) for item in records)
        final_step_acceptance = step_acceptance_records[-1]
        compile_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        acceptance_checks = {
            "cuda_accelerator_all_ranks": all(
                item["peak_memory_bytes"] is not None for item in records
            ),
            "backward_completed_all_ranks": all(
                item["backward_completed"] for item in final_step_acceptance
            ),
            "gradients_finite_all_ranks": all(
                item["gradient_finite"] for item in final_step_acceptance
            ),
            "owned_gradients_nonzero_all_ranks": all(
                item["owned_gradient_nonzero"] for item in final_step_acceptance
            ),
            "optimizer_updated_all_parameters": all(
                item["optimizer_updated"] for item in final_step_acceptance
            ),
            "peak_memory_within_budget": peak_memory_bytes <= peak_memory_budget_bytes,
            "discarded_weight_within_budget_all_ranks": all(
                item["discarded_weight_budget_satisfied"]
                for item in final_step_acceptance
            ),
            "persistent_compile_cache_configured": bool(compile_cache),
        }
        acceptance_passed = all(acceptance_checks.values())
        if rank == 0:
            payload = {
                "schema": "flagquantum.mps_hamiltonian_identification.site_sharded.v2",
                "task": "nonuniform_1d_hamiltonian_identification",
                "n_qubits": args.n_qubits,
                "probe_count": len(probes),
                "time_steps": args.time_steps,
                "observation_sites": observation_sites,
                "observable_term_count": targets[batches[0]].shape[-1],
                "training_steps": len(history),
                "optimizer_steps": optimizer_steps,
                "probe_microbatches_per_step": len(batches),
                "gradient_policy": args.gradient_policy,
                "discarded_weight_tolerance": args.discarded_weight_tolerance,
                "teacher_discarded_weight": teacher_discarded_weight,
                "kernel_execution": (
                    "torch_compile_rank_local_buckets"
                    if args.compile_site_kernels else "eager_rank_local"
                ),
                "observable_execution": (
                    "torch_compile_real_channel_rank_scan"
                    if args.compile_observables else "eager_complex_rank_scan"
                ),
                "cold_step_seconds": step_seconds[0],
                "warm_step_seconds": step_seconds[1:],
                "loss_history": history,
                "final_loss": history[-1],
                "seconds": elapsed,
                "seconds_per_completed_step": elapsed / len(history),
                "coupling_relative_error": relative_parameter_error(
                    estimate_j, truth_j
                ),
                "field_relative_error": relative_parameter_error(estimate_h, truth_h),
                "world_size": world,
                "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", str(world))),
                "node_count": len({item["hostname"] for item in records}),
                "distribution_semantics": "sharded_across_ranks",
                "mps_state_sharded_across_ranks": True,
                "teacher_target_generation": "site_sharded_observation_scan",
                "teacher_full_mps_materialization": False,
                "full_learned_mps_materialization": False,
                "capacity_claim_allowed": False,
                "capacity_blockers": ["production_gpu_capacity_baseline_not_attached"],
                "early_stopping": {"patience": args.early_stopping_patience, "min_delta": args.early_stopping_min_delta,
                    "target_loss": args.early_stopping_loss, "stop_reason": stop_reason,
                    "best_step": best_step, "best_loss": best_loss,
                    "patience_improvement_anchor": improvement_anchor},
                "best_checkpoints": [
                    str(args.output.with_suffix(args.output.suffix + f".best.rank-{item}.pt"))
                    for item in range(world)
                ],
                "held_out_validation": validation,
                "communication": {"backend": str(dist.get_backend()), "measured_logical_boundary_bytes": communication_bytes},
                "gpu_evidence": {
                    "accelerator": device.type == "cuda",
                    "peak_memory_bytes_by_rank": [item["peak_memory_bytes"] for item in records],
                    "rank_owned_tensor_bytes": [item["rank_ownership"]["local_tensor_bytes"] for item in records],
                },
                "single_step_acceptance": {
                    "requested": args.single_step_acceptance,
                    "passed": acceptance_passed,
                    "checks": acceptance_checks,
                    "peak_memory_bytes": peak_memory_bytes,
                    "peak_memory_budget_bytes": peak_memory_budget_bytes,
                    "persistent_compile_cache": compile_cache,
                    "rank_records": final_step_acceptance,
                },
                "parameters": {
                    "true_coupling": truth_j.detach().cpu().tolist(),
                    "learned_coupling": estimate_j.detach().cpu().tolist(),
                    "true_field": truth_h.detach().cpu().tolist(),
                    "learned_field": estimate_h.detach().cpu().tolist(),
                },
                "rank_records": records,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(
                json.dumps({"output": str(args.output), "final_loss": history[-1]}),
                flush=True,
            )
        dist.destroy_process_group()
        if args.single_step_acceptance and not acceptance_passed:
            raise RuntimeError(
                "single-step acceptance failed: "
                + json.dumps(acceptance_checks, sort_keys=True)
            )
        return
    local_indices = (
        tuple(range(len(probes)))
        if args.replicated_capacity_baseline
        else tuple(range(rank, len(probes), world))
    )
    if not local_indices:
        raise SystemExit(f"rank {rank} owns no probes; reduce world size or add probes")

    local_batches = _probe_batches(local_indices, probes, args.probe_batch_size)
    end_to_end_started = time.perf_counter()
    target_started = time.perf_counter()
    with torch.no_grad():
        targets = {}
        for position, batch_indices in enumerate(local_batches, start=1):
            batch_probes = tuple(probes[index] for index in batch_indices)
            targets[batch_indices] = predict_batch(
                truth_j,
                truth_h,
                batch_probes,
                observation_sites=observation_sites,
                dt=args.dt,
                max_bond=args.max_bond,
                cutoff=args.cutoff,
                device=device,
                compiled_brickwork=args.compile_brickwork,
                compiled_observables=args.compile_observables,
            ).detach()
            if not args.quiet_rank_heartbeats and (position % args.heartbeat_probes == 0 or position == len(local_batches)):
                print(json.dumps({"phase": "target_generation", "rank": rank, "completed_batches": position, "total_batches": len(local_batches), "batch_size": len(batch_indices)}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    target_generation_seconds = time.perf_counter() - target_started

    estimate_j = torch.full_like(truth_j, 0.62, requires_grad=True)
    estimate_h = torch.full_like(truth_h, 0.52, requires_grad=True)
    optimizer = torch.optim.Adam((estimate_j, estimate_h), lr=args.lr)
    history: list[float] = []
    started = time.perf_counter()
    best_loss = float("inf")
    best_step = 0
    improvement_anchor = float("inf")
    stale_steps = 0
    stop_reason = "maximum_steps_reached"
    for step in range(args.steps):
        optimizer.zero_grad()
        local_sum = torch.zeros((), device=device)
        for position, batch_indices in enumerate(local_batches, start=1):
            batch_probes = tuple(probes[index] for index in batch_indices)
            prediction = predict_batch(
                estimate_j,
                estimate_h,
                batch_probes,
                observation_sites=observation_sites,
                dt=args.dt,
                max_bond=args.max_bond,
                cutoff=args.cutoff,
                device=device,
                compiled_brickwork=args.compile_brickwork,
                compiled_observables=args.compile_observables,
            )
            per_probe_loss = torch.mean((prediction - targets[batch_indices]) ** 2, dim=-1)
            batch_loss_sum = per_probe_loss.sum()
            local_sum = local_sum + batch_loss_sum.detach()
            # Backward one probe at a time so a 512/1024-qubit run does not
            # retain every MPS tape simultaneously.
            (batch_loss_sum / len(probes)).backward()
            if not args.quiet_rank_heartbeats and (position % args.heartbeat_probes == 0 or position == len(local_batches)):
                print(json.dumps({"phase": "training", "rank": rank, "step": step + 1, "completed_batches": position, "total_batches": len(local_batches), "batch_size": len(batch_indices)}), flush=True)
        # SUM-reduced gradients of per-rank probe sums divided by the global
        # probe count equal the global mean, including uneven rank shards.
        _all_reduce_gradients((estimate_j, estimate_h), world)
        optimizer.step()
        global_loss = local_sum.detach()
        if world > 1:
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        value = float((global_loss / len(probes)).cpu())
        history.append(value)
        if value < best_loss:
            best_loss = value
            best_step = step + 1
        if value < improvement_anchor - args.early_stopping_min_delta:
            improvement_anchor = value
            stale_steps = 0
        else:
            stale_steps += 1
        log_every = args.log_every or max(1, args.steps // 10)
        if rank == 0 and (step == 0 or step + 1 == args.steps or (step + 1) % log_every == 0):
            print(json.dumps({"step": step + 1, "loss": value}), flush=True)
        if args.early_stopping_loss is not None and value <= args.early_stopping_loss:
            stop_reason = "target_loss_reached"
            break
        if args.early_stopping_patience and stale_steps >= args.early_stopping_patience:
            stop_reason = "patience_exhausted"
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    validation = None
    if args.validation_initial_states:
        validation_probes = make_probes(
            args.n_qubits,
            n_initial_states=args.validation_initial_states,
            time_steps=args.validation_time_steps,
            seed=args.seed + 1,
        )
        validation_indices = tuple(range(rank, len(validation_probes), world))
        validation_batches = _probe_batches(
            validation_indices, validation_probes, args.probe_batch_size
        )
        time_values = tuple(sorted(set(args.validation_time_steps)))
        time_position = {value: index for index, value in enumerate(time_values)}
        loss_sums = torch.zeros(len(time_values), device=device)
        loss_counts = torch.zeros(len(time_values), device=device)
        validation_started = time.perf_counter()
        with torch.no_grad():
            for batch_indices in validation_batches:
                batch_probes = tuple(validation_probes[index] for index in batch_indices)
                truth = predict_batch(
                    truth_j, truth_h, batch_probes,
                    observation_sites=observation_sites, dt=args.dt,
                    max_bond=args.max_bond, cutoff=args.cutoff, device=device,
                    compiled_brickwork=args.compile_brickwork,
                    compiled_observables=args.compile_observables,
                )
                prediction = predict_batch(
                    estimate_j, estimate_h, batch_probes,
                    observation_sites=observation_sites, dt=args.dt,
                    max_bond=args.max_bond, cutoff=args.cutoff, device=device,
                    compiled_brickwork=args.compile_brickwork,
                    compiled_observables=args.compile_observables,
                )
                losses = torch.mean((prediction - truth) ** 2, dim=-1)
                position = time_position[batch_probes[0].time_steps]
                loss_sums[position] += losses.sum()
                loss_counts[position] += losses.numel()
        if world > 1:
            dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_counts, op=dist.ReduceOp.SUM)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_time = loss_sums / loss_counts.clamp_min(1)
        validation = {
            "probe_count": len(validation_probes),
            "initial_state_count": args.validation_initial_states,
            "seed": args.seed + 1,
            "time_steps": time_values,
            "observation_sites": observation_sites,
            "mse": float((loss_sums.sum() / loss_counts.sum()).cpu()),
            "mse_by_time_step": {
                str(value): float(per_time[index].cpu())
                for index, value in enumerate(time_values)
            },
            "seconds": time.perf_counter() - validation_started,
        }
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    placement = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": str(device),
        "owned_probe_indices": local_indices,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "allocated_memory_bytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
    }
    placements = [placement]
    if world > 1:
        placements = [None] * world
        dist.all_gather_object(placements, placement)
    if rank == 0:
        payload = {
            "schema": "flagquantum.mps_hamiltonian_identification.v1",
            "task": "nonuniform_1d_hamiltonian_identification",
            "model": "sum_i J_i X_iX_i+1 + sum_i h_i Y_i",
            "n_qubits": args.n_qubits,
            "probe_count": len(probes),
            "probe_batch_size": args.probe_batch_size,
            "brickwork_execution": "torch_compile_shape_bucket_fused" if args.compile_brickwork else "circuit_interpreter_eager",
            "observable_execution": "torch_compile_real_channel_scan" if args.compile_observables else "batched_complex_environment_scan",
            "time_steps": args.time_steps,
            "observation_sites": observation_sites,
            "observable_count_per_probe": len(observation_sites) + sum(wire + 1 < args.n_qubits for wire in observation_sites),
                "max_bond": args.max_bond,
                "reverse_max_saved_bytes": int(args.reverse_max_saved_gib * (1 << 30)),
            "cutoff": args.cutoff,
            "requested_training_steps": args.steps,
            "training_steps": len(history),
            "learning_rate": args.lr,
            "early_stopping": {
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "target_loss": args.early_stopping_loss,
                "stop_reason": stop_reason,
                "best_step": best_step,
                "best_loss": best_loss,
                "patience_improvement_anchor": improvement_anchor,
            },
            "initial_loss": history[0],
            "final_loss": history[-1],
            "coupling_relative_error": relative_parameter_error(estimate_j, truth_j),
            "field_relative_error": relative_parameter_error(estimate_h, truth_h),
            "seconds": elapsed,
            "target_generation_seconds": target_generation_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "seconds_per_completed_step": elapsed / len(history),
            "loss_history": history,
            "held_out_validation": validation,
            "world_size": world,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", str(world))),
            "node_count": len({item["hostname"] for item in placements}),
            "rank_placement": placements,
            "communication": {
                "backend": str(dist.get_backend()) if world > 1 else "none",
                "gradient_all_reduce_count": args.steps * 2 if world > 1 else 0,
                "gradient_all_reduce_bytes_per_step": (estimate_j.numel() + estimate_h.numel()) * estimate_j.element_size() if world > 1 else 0,
            },
            "distribution_semantics": (
                "replicated_per_rank"
                if world > 1 and args.replicated_capacity_baseline
                else "observable_term_parallel"
                if world > 1
                else "single_device_fast_path"
            ),
            "mps_state_sharded_across_ranks": False,
            "scalability_claim_allowed": False,
            "scalability_blockers": ["mps_state_is_not_sharded_across_ranks"],
            "claim_boundary": (
                "every rank executes the same complete probe workload; capacity cannot expand"
                if args.replicated_capacity_baseline
                else "probes are distributed; each probe owns a complete MPS on one device"
            ),
            "parameters": {
                "true_coupling": truth_j.detach().cpu().tolist(),
                "learned_coupling": estimate_j.detach().cpu().tolist(),
                "true_field": truth_h.detach().cpu().tolist(),
                "learned_field": estimate_h.detach().cpu().tolist(),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "final_loss": history[-1]}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
