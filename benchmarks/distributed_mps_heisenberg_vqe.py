"""Scientific, site-sharded MPS VQE capacity probe for a Heisenberg chain.

This benchmark trains one logical open-boundary MPS.  It is deliberately a
single case per torchrun invocation: a capacity sweep should launch a fresh
process group for every point so that an OOM cannot contaminate later points.
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
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd
import flagquantum.experimental.mps as fqxm
from benchmarks.sc27_metadata import (
    driver_version,
    gpu_identity,
    source_identity,
    topology_snapshot,
)
from examples.distributed_mps.variable_bond_capacity_8gpu import bond_dimensions
from flagquantum.ops import set_global_precision
from flagquantum.runtime.backends.mps.forward import (
    _initial_ownership,
    cost_aware_mps_ownership,
    gate_aligned_cost_aware_mps_ownership,
    mps_factorization_site_costs,
)


def heisenberg_terms(
    n_sites: int, *, anisotropy: float, field: float
) -> tuple[tuple[dict[int, str], float], ...]:
    """Open XX + YY + delta ZZ chain, optionally with a uniform Z field."""
    terms: list[tuple[dict[int, str], float]] = []
    for left in range(n_sites - 1):
        terms.extend(
            (
                ({left: "x", left + 1: "x"}, 1.0),
                ({left: "y", left + 1: "y"}, 1.0),
                ({left: "z", left + 1: "z"}, anisotropy),
            )
        )
    terms.extend(({wire: "z"}, field) for wire in range(n_sites) if field)
    return tuple(terms)


def exact_heisenberg_ground_energy(
    n_sites: int, *, anisotropy: float, field: float
) -> float:
    """Return the open-chain ground energy by sparse exact diagonalization."""
    import numpy as np
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh

    dimension = 1 << n_sites
    basis = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    for left in range(n_sites - 1):
        left_bit = (basis >> left) & 1
        right_bit = (basis >> (left + 1)) & 1
        unlike = left_bit != right_bit
        diagonal += anisotropy * np.where(unlike, -1.0, 1.0)
        rows = basis[unlike]
        row_blocks.append(rows)
        column_blocks.append(rows ^ (3 << left))
    if field:
        for wire in range(n_sites):
            diagonal += field * (1.0 - 2.0 * ((basis >> wire) & 1))
    rows = np.concatenate(row_blocks)
    columns = np.concatenate(column_blocks)
    off_diagonal = coo_matrix(
        (np.full(rows.size, 2.0), (rows, columns)),
        shape=(dimension, dimension),
    ).tocsr()
    hamiltonian = off_diagonal + diags(diagonal, format="csr")
    return float(eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0])


def make_parameters(
    count: int,
    device: torch.device,
    seed: int,
    scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        (
            scale * torch.randn((), generator=generator, device=device, dtype=dtype)
        ).requires_grad_()
        for _ in range(count)
    )


def hva_circuit(
    n_sites: int,
    depth: int,
    parameters: tuple[torch.Tensor, ...],
    parameterization: str,
) -> fq.Circuit:
    """Even/odd XX, YY and ZZ layers acting on the supplied Néel-like MPS."""
    circuit = fq.Circuit(n_sites, device=parameters[0].device)
    if parameterization == "shared_parity":
        for layer in range(depth):
            offset = 6 * layer
            for axis, gate in enumerate((circuit.rxx, circuit.ryy, circuit.rzz)):
                for parity in (0, 1):
                    theta = parameters[offset + 2 * axis + parity]
                    for left in range(parity, n_sites - 1, 2):
                        gate(left, left + 1, theta=theta)
    else:
        per_layer = 3 * (n_sites - 1) + (
            n_sites if parameterization == "bond_resolved_phase" else 0
        )
        for layer in range(depth):
            offset = per_layer * layer
            for axis, gate in enumerate((circuit.rxx, circuit.ryy, circuit.rzz)):
                for left in range(n_sites - 1):
                    theta = parameters[offset + axis * (n_sites - 1) + left]
                    gate(left, left + 1, theta=theta)
            if parameterization == "bond_resolved_phase":
                phase_offset = offset + 3 * (n_sites - 1)
                for wire in range(n_sites):
                    circuit.rz(wire, theta=parameters[phase_offset + wire])
    return circuit


def rank_owned_neel_mps(
    n_sites: int,
    bond: int,
    noise: float,
    device: torch.device,
    seed: int,
    real_dtype: torch.dtype,
    ownership: tuple[tuple[int, ...], ...],
) -> dict[int, torch.Tensor]:
    """A reproducible weakly perturbed Neel MPS warm start.

    Bond-one product states produce exact zero singular-value degeneracies in
    early capped SVDs.  A tiny bond-space perturbation removes that numerical
    pathology without replacing the physically motivated Neel reference by a
    random highly-entangled state.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    tensors = {}
    for wire in ownership[rank]:
        left = 1 if wire == 0 else bond
        right = 1 if wire == n_sites - 1 else bond
        generator = torch.Generator(device=device).manual_seed(seed + 104729 * wire)
        real = torch.randn(
            1, left, 2, right, generator=generator, device=device, dtype=real_dtype
        )
        imag = torch.randn(
            1, left, 2, right, generator=generator, device=device, dtype=real_dtype
        )
        tensor = noise * torch.complex(real, imag) / math.sqrt(max(1, left * right))
        tensor[0, 0, wire % 2, 0] += 1.0
        tensors[wire] = tensor
    return tensors


def rank_owned_isometric_mps(
    n_sites: int,
    bond: int,
    device: torch.device,
    seed: int,
    real_dtype: torch.dtype,
    ownership: tuple[tuple[int, ...], ...],
) -> dict[int, torch.Tensor]:
    """Topology-invariant random left-isometric MPS, keyed by global site."""
    rank, world = dist.get_rank(), dist.get_world_size()
    bonds = bond_dimensions(n_sites, bond)
    tensors: dict[int, torch.Tensor] = {}
    for wire in ownership[rank]:
        left, right = bonds[wire], bonds[wire + 1]
        generator = torch.Generator(device=device).manual_seed(seed + 104729 * wire)
        real = torch.randn(
            2 * left, right, device=device, generator=generator, dtype=real_dtype
        )
        imag = torch.randn(
            2 * left, right, device=device, generator=generator, dtype=real_dtype
        )
        q, _ = torch.linalg.qr(torch.complex(real, imag), mode="reduced")
        tensors[wire] = q.reshape(1, left, 2, right).contiguous()
    return tensors


def rank_owned_dimer_singlet_mps(
    n_sites: int,
    device: torch.device,
    real_dtype: torch.dtype,
    ownership: tuple[tuple[int, ...], ...],
) -> dict[int, torch.Tensor]:
    """Exact product of nearest-neighbor singlets with alternating 2/1 bonds."""

    if n_sites % 2:
        raise ValueError("dimer_singlet requires an even number of sites")
    rank, world = dist.get_rank(), dist.get_world_size()
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    tensors: dict[int, torch.Tensor] = {}
    for wire in ownership[rank]:
        if wire % 2 == 0:
            tensor = torch.zeros(1, 1, 2, 2, dtype=complex_dtype, device=device)
            tensor[0, 0, 0, 0] = 1 / math.sqrt(2)
            tensor[0, 0, 1, 1] = -1 / math.sqrt(2)
        else:
            tensor = torch.zeros(1, 2, 2, 1, dtype=complex_dtype, device=device)
            tensor[0, 0, 1, 0] = 1
            tensor[0, 1, 0, 0] = 1
        tensors[wire] = tensor
    return tensors


def _largest_realized_bond(summary: dict) -> int:
    metric_ranks = [
        int(step["largest_realized_bond"])
        for step in summary["step_metrics"]
        if step.get("largest_realized_bond") is not None
    ]
    if metric_ranks:
        return max(metric_ranks)
    ranks = [
        int(update["kept_rank"])
        for step in summary["step_metrics"]
        for update in step["bond_updates"]
        if update.get("kept_rank") is not None
    ]
    return max(ranks, default=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True, help="HVA layer count")
    parser.add_argument("--max-bond", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=1e-8)
    parser.add_argument("--anisotropy", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--learning-rate-decay", type=float, default=1.0)
    parser.add_argument(
        "--optimizer", choices=("adam", "sgd", "adam_lbfgs"), default="adam"
    )
    parser.add_argument("--lbfgs-start-step", type=int, default=None)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=0.8)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--record-parameter-gradients", action="store_true")
    parser.add_argument("--gradient-tolerance", type=float, default=1.0)
    parser.add_argument("--checkpoint-budget-gib", type=float, default=1.0)
    parser.add_argument(
        "--factorization-budget-gib",
        type=float,
        default=None,
        help="optional pool for saved SVD/QR graphs; exhaustion triggers recomputation",
    )
    parser.add_argument(
        "--save-two-site-factorizations",
        action="store_true",
        help="spend checkpoint memory to retain forward QR/SVD split graphs",
    )
    parser.add_argument("--profile-trace-dir", type=Path, default=None)
    parser.add_argument(
        "--partition-policy",
        choices=("equal_sites", "cost_aware", "gate_aligned_cost_aware"),
        default="equal_sites",
    )
    parser.add_argument("--seed", type=int, default=260717)
    parser.add_argument("--parameter-scale", type=float, default=0.02)
    parser.add_argument("--initial-bond", type=int, default=2)
    parser.add_argument("--initial-noise", type=float, default=1e-4)
    parser.add_argument(
        "--parameterization",
        choices=("shared_parity", "bond_resolved", "bond_resolved_phase"),
        default="shared_parity",
    )
    parser.add_argument(
        "--exact-reference-max-sites",
        type=int,
        default=20,
        help="compute a sparse exact ground energy at or below this system size",
    )
    parser.add_argument("--convergence-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--initial-state",
        choices=("neel_like", "random_isometric", "dimer_singlet"),
        default="random_isometric",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="disable torch.compile site/observable kernels for deterministic profiling",
    )
    parser.add_argument(
        "--disable-halo-prefetch",
        action="store_true",
        help="disable compiled-layer asynchronous boundary halo prefetch for A/B profiling",
    )
    parser.add_argument(
        "--svd-driver",
        choices=("gesvd", "gesvdj", "gesvda"),
        default="gesvd",
        help="CUDA SVD driver; gesvda is approximate and intended for performance probes",
    )
    parser.add_argument("--independent-run-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--container-digest")
    parser.add_argument("--raw-log-sha256")
    parser.add_argument(
        "--reference-energy",
        type=float,
        help="independent reference energy for systems above exact-reference-max-sites",
    )
    parser.add_argument(
        "--reference-artifact-sha256",
        help="SHA256 of the independent reference solver's sealed raw artifact",
    )
    parser.add_argument(
        "--reference-method",
        choices=("independent_dmrg",),
        default="independent_dmrg",
    )
    args = parser.parse_args()
    real_dtype = torch.float64 if args.precision == "float64" else torch.float32
    set_global_precision(
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    if args.n_sites < 2 or args.depth < 1 or args.max_bond < 1 or args.steps < 1:
        parser.error("n-sites >= 2 and depth, max-bond, steps >= 1 are required")
    if not 1 <= args.initial_bond <= args.max_bond or args.initial_noise <= 0:
        parser.error("1 <= initial-bond <= max-bond and initial-noise > 0 are required")
    if not 0.0 < args.learning_rate_decay <= 1.0:
        parser.error("learning-rate-decay must be in (0, 1]")
    if (args.reference_energy is None) != (args.reference_artifact_sha256 is None):
        parser.error(
            "reference-energy and reference-artifact-sha256 must be supplied together"
        )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        timeout=timedelta(minutes=30),
        device_id=device if backend == "nccl" else None,
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.n_sites < world:
        raise SystemExit(
            "n-sites must be at least world size so every rank owns a site"
        )
    predicted_bonds = tuple(bond_dimensions(args.n_sites, args.initial_bond))
    predicted_site_costs = mps_factorization_site_costs(predicted_bonds)
    if args.partition_policy == "equal_sites":
        site_ownership = _initial_ownership(args.n_sites, world)
    elif args.partition_policy == "cost_aware":
        site_ownership = cost_aware_mps_ownership(predicted_bonds, world)
    else:
        site_ownership = gate_aligned_cost_aware_mps_ownership(
            predicted_bonds, world
        )

    parameter_count = (
        6 * args.depth
        if args.parameterization == "shared_parity"
        else (
            3 * (args.n_sites - 1)
            + (args.n_sites if args.parameterization == "bond_resolved_phase" else 0)
        )
        * args.depth
    )
    parameters = make_parameters(
        parameter_count, device, args.seed, args.parameter_scale, real_dtype
    )
    circuit = hva_circuit(args.n_sites, args.depth, parameters, args.parameterization)
    terms = heisenberg_terms(args.n_sites, anisotropy=args.anisotropy, field=args.field)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    status, error, result = "passed", None, None
    profiler = None
    if args.profile_trace_dir is not None:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profiler = profile(activities=activities, record_shapes=False)
        profiler.start()
    try:
        result = fqxd.train_distributed_mps(
            circuit,
            steps=args.steps,
            hamiltonian_terms=terms,
            optimizer=args.optimizer,
            lr=args.learning_rate,
            lr_decay=args.learning_rate_decay,
            lbfgs_start_step=args.lbfgs_start_step,
            lbfgs_lr=args.lbfgs_learning_rate,
            lbfgs_history_size=args.lbfgs_history_size,
            device=device,
            max_bond=args.max_bond,
            cutoff=args.cutoff,
            gradient_policy="approximate" if args.cutoff else "exact",
            gradient_tolerance=args.gradient_tolerance,
            compile_site_kernels=device.type == "cuda" and not args.disable_compile,
            prefetch_layer_halos=not args.disable_halo_prefetch,
            compile_observables=device.type == "cuda" and not args.disable_compile,
            svd_driver=args.svd_driver,
            record_parameter_gradients=args.record_parameter_gradients,
            initial_mps_tensors=(
                rank_owned_isometric_mps(
                    args.n_sites,
                    args.initial_bond,
                    device,
                    args.seed,
                    real_dtype,
                    site_ownership,
                )
                if args.initial_state == "random_isometric"
                else (
                    rank_owned_dimer_singlet_mps(
                        args.n_sites, device, real_dtype, site_ownership
                    )
                    if args.initial_state == "dimer_singlet"
                    else rank_owned_neel_mps(
                        args.n_sites,
                        args.initial_bond,
                        args.initial_noise,
                        device,
                        args.seed,
                        real_dtype,
                        site_ownership,
                    )
                )
            ),
            initial_mps_left_canonical=args.initial_state == "random_isometric",
            site_ownership=site_ownership,
            reverse_checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
                max_saved_bytes=int(args.checkpoint_budget_gib * (1 << 30)),
                max_saved_factorization_bytes=(
                    None
                    if args.factorization_budget_gib is None
                    else int(args.factorization_budget_gib * (1 << 30))
                ),
                save_two_site_factorizations=args.save_two_site_factorizations,
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if profiler is not None:
            profiler.stop()
            args.profile_trace_dir.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(
                str(args.profile_trace_dir / f"rank-{rank}.json")
            )
    except torch.OutOfMemoryError as exc:
        status, error = "cuda_oom", str(exc).splitlines()[0]
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status, error = "cuda_oom", str(exc).splitlines()[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = None if result is None else result.summary()
    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "device": str(device),
        "gpu_identity": gpu_identity(local_rank) if device.type == "cuda" else None,
        "topology": topology_snapshot() if device.type == "cuda" else None,
        "status": status,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "owned_sites": [
            site_ownership[rank][0],
            site_ownership[rank][-1] + 1,
        ],
        "predicted_partition_cost": sum(
            predicted_site_costs[wire] for wire in site_ownership[rank]
        ),
        "training": summary,
        "largest_realized_bond": (
            0 if summary is None else _largest_realized_bond(summary)
        ),
    }
    records: list[dict | None] = [None] * world
    dist.all_gather_object(records, local)
    if rank == 0:
        passed = all(
            record is not None and record["status"] == "passed" for record in records
        )
        rank_placement = [
            {
                "rank": record["rank"],
                "hostname": record["hostname"],
                "node_id": 0,
                "local_rank": record["local_rank"],
                "device_id": record["device"],
                **(record["gpu_identity"] or {}),
                "topology": record["topology"],
            }
            for record in records
            if record is not None
        ]
        rank_ownership = [
            {
                "rank": record["rank"],
                "wire_start": record["owned_sites"][0],
                "wire_stop": record["owned_sites"][1],
                "predicted_partition_cost": record["predicted_partition_cost"],
            }
            for record in records
            if record is not None
        ]
        local_memory_bytes_by_rank = [
            int(record["peak_memory_bytes"]) for record in records if record is not None
        ]
        unique_updates: dict[str, dict] = {}
        if passed:
            for record in records:
                for step in record["training"]["step_metrics"]:
                    for update in step["bond_updates"]:
                        unique_updates.setdefault(update["operation_id"], update)
        communication_bytes = 2 * sum(
            int(update["payload_bytes"]) for update in unique_updates.values()
        )
        exact_energy = (
            exact_heisenberg_ground_energy(
                args.n_sites, anisotropy=args.anisotropy, field=args.field
            )
            if passed and args.n_sites <= args.exact_reference_max_sites
            else None
        )
        reference_energy = (
            exact_energy if exact_energy is not None else args.reference_energy
        )
        reference_method = (
            "sparse_exact_diagonalization"
            if exact_energy is not None
            else (args.reference_method if args.reference_energy is not None else None)
        )
        best_energy = (
            min(records[0]["training"]["losses"])
            if passed and records[0] is not None
            else None
        )
        relative_energy_error = (
            None
            if reference_energy is None or best_energy is None
            else abs(best_energy - reference_energy) / abs(reference_energy)
        )
        convergence_certified = (
            relative_energy_error is not None
            and relative_energy_error <= args.convergence_tolerance
        )
        workload_identity = {
            "name": "heisenberg_xxz_vqe",
            "n_sites": args.n_sites,
            "depth": args.depth,
            "max_bond": args.max_bond,
            "cutoff": args.cutoff,
            "anisotropy": args.anisotropy,
            "field": args.field,
            "initial_state": args.initial_state,
            "parameterization": args.parameterization,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "seed": args.seed,
            "world_size": world,
        }
        step_metrics_by_rank = [
            record["training"]["step_metrics"] if record and record["training"] else []
            for record in records
        ]
        optimizer_bytes_by_rank = [
            max((int(step["optimizer_memory_bytes"]) for step in steps), default=0)
            for steps in step_metrics_by_rank
        ]
        communication_bytes_by_rank = [
            sum(
                int(step["layer_halo_payload_bytes"])
                + int(step["boundary_bytes"])
                + int(step["gradient_collective_bytes"])
                + int(step["optimizer_collective_bytes"])
                for step in steps
            )
            for steps in step_metrics_by_rank
        ]
        discarded_weight_by_step = [
            max(float(rank_steps[index]["discarded_weight"]) for rank_steps in step_metrics_by_rank)
            for index in range(args.steps)
        ] if passed else []
        step_seconds = (
            [
                max(
                    float(rank_steps[index]["end_to_end_seconds"])
                    for rank_steps in step_metrics_by_rank
                )
                for index in range(args.steps)
            ]
            if passed
            else []
        )
        target_reached_step = None
        time_to_target_seconds = None
        if reference_energy is not None and passed:
            cumulative = 0.0
            for index, (energy, seconds) in enumerate(
                zip(records[0]["training"]["losses"], step_seconds, strict=True),
                start=1,
            ):
                cumulative += seconds
                error = abs(float(energy) - reference_energy) / abs(reference_energy)
                if error <= args.convergence_tolerance:
                    target_reached_step = index
                    time_to_target_seconds = cumulative
                    break
        payload = {
            "schema": "flagquantum.distributed_mps_heisenberg_vqe.v1",
            "evidence_source": "measured_runtime",
            "artifact_classification": "capacity_probe_non_release",
            "workload": "open_boundary_antiferromagnetic_xxz_heisenberg_vqe",
            "ansatz": "heisenberg_hamiltonian_variational_ansatz_even_odd_xx_yy_zz",
            "n_sites": args.n_sites,
            "qubits": args.n_sites,
            "ansatz_depth": args.depth,
            "two_qubit_gate_depth": 6 * args.depth,
            "two_qubit_gate_count": 3 * args.depth * (args.n_sites - 1),
            "parameter_count": parameter_count,
            "parameter_scale": args.parameter_scale,
            "parameterization": args.parameterization,
            "hamiltonian_term_count": len(terms),
            "anisotropy": args.anisotropy,
            "field": args.field,
            "exact_ground_energy": exact_energy,
            "reference": {
                "method": reference_method,
                "energy": reference_energy,
                "artifact_sha256": (
                    args.reference_artifact_sha256
                    if exact_energy is None
                    else None
                ),
                "independent_of_flagquantum": exact_energy is None
                and args.reference_energy is not None,
            },
            "best_variational_energy": best_energy,
            "absolute_energy_error": (
                None
                if reference_energy is None or best_energy is None
                else best_energy - reference_energy
            ),
            "relative_energy_error": relative_energy_error,
            "energy_error_per_site": (
                None
                if reference_energy is None or best_energy is None
                else abs(best_energy - reference_energy) / args.n_sites
            ),
            "convergence_tolerance": args.convergence_tolerance,
            "convergence_certified": convergence_certified,
            "target_reached": target_reached_step is not None,
            "target_reached_step": target_reached_step,
            "time_to_target_seconds": time_to_target_seconds,
            "optimizer_step_seconds": step_seconds,
            "requested_max_bond": args.max_bond,
            "initial_bond": args.initial_bond,
            "initial_noise": args.initial_noise,
            "initial_state": args.initial_state,
            "partition_policy": args.partition_policy,
            "partition_cost_model": (
                "dense_two_site_svd_proxy_v1"
                if args.partition_policy == "cost_aware"
                else "equal_site_count_v1"
            ),
            "cutoff": args.cutoff,
            "gradient_tolerance": args.gradient_tolerance,
            "checkpoint_budget_bytes": int(args.checkpoint_budget_gib * (1 << 30)),
            "factorization_budget_bytes": (
                None
                if args.factorization_budget_gib is None
                else int(args.factorization_budget_gib * (1 << 30))
            ),
            "save_two_site_factorizations": args.save_two_site_factorizations,
            "compile_site_kernels": device.type == "cuda" and not args.disable_compile,
            "layer_halo_prefetch": (
                device.type == "cuda"
                and not args.disable_compile
                and not args.disable_halo_prefetch
            ),
            "compile_observables": device.type == "cuda" and not args.disable_compile,
            "svd_driver": args.svd_driver,
            "numerical_mode": (
                "fast_approximate_svd" if args.svd_driver == "gesvda" else "standard_svd"
            ),
            "compiled_layer_budget_reservation": (
                args.svd_driver == "gesvda" and not args.disable_compile
            ),
            "static_heisenberg_mpo_carry_descriptors": True,
            "steps": args.steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "learning_rate_decay": args.learning_rate_decay,
            "optimizer": args.optimizer,
            "lbfgs_start_step": args.lbfgs_start_step,
            "lbfgs_learning_rate": args.lbfgs_learning_rate,
            "lbfgs_history_size": args.lbfgs_history_size,
            "precision": args.precision,
            "optimization_trace": (
                []
                if not passed or records[0] is None
                else [
                    {
                        "optimizer_step": index,
                        "energy": float(energy),
                        "reference_energy": reference_energy,
                        "reference_method": reference_method,
                        "absolute_error": (
                            None
                            if reference_energy is None
                            else abs(float(energy) - reference_energy)
                        ),
                        "relative_error": (
                            None
                            if reference_energy is None
                            else abs(float(energy) - reference_energy)
                            / abs(reference_energy)
                        ),
                        "optimizer_stage": records[0]["training"]["step_metrics"][
                            index
                        ]["optimizer_stage"],
                        "learning_rate": records[0]["training"]["step_metrics"][index][
                            "learning_rate"
                        ],
                        "objective_evaluations": records[0]["training"]["step_metrics"][
                            index
                        ]["objective_evaluations"],
                    }
                    for index, energy in enumerate(records[0]["training"]["losses"])
                ]
            ),
            "world_size": world,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", str(world))),
            "node_count": len({record["hostname"] for record in records if record}),
            "backend": backend,
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device_name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else platform.processor()
            ),
            "distribution_semantics": (
                "sharded_across_ranks" if world > 1 else "single_device_fast_path"
            ),
            "claim_evidence_type": "production_runtime",
            "rank_placement": rank_placement,
            "rank_ownership": rank_ownership,
            "source_identity": source_identity(
                repo_root=ROOT,
                workload=workload_identity,
                container_digest=args.container_digest,
                raw_log_sha256=args.raw_log_sha256,
            ),
            "protocol": {
                "independent_run_index": args.independent_run_index,
                "seed": args.seed,
                "retains_failed_runs": True,
                "compile_reported_separately": True,
                "target_relative_energy_error": args.convergence_tolerance,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else platform.processor()
                ),
                "driver_version": driver_version(),
            },
            "ownership": {
                "primal_state": "sharded_across_ranks" if world > 1 else "single_device",
                "adjoint_state": "sharded_across_ranks" if world > 1 else "single_device",
                "parameter_gradient": "owner_sharded_across_ranks" if world > 1 else "single_device",
                "optimizer_state": "owner_sharded_across_ranks" if world > 1 else "single_device",
                "full_mps_materialized": False,
                "optimizer_bytes_per_rank": optimizer_bytes_by_rank,
            },
            "local_memory_bytes_by_rank": local_memory_bytes_by_rank,
            "memory_plan": {
                "status": "measured",
                "local_memory_bytes_by_rank": local_memory_bytes_by_rank,
            },
            "communication_bytes": communication_bytes,
            "communication_bytes_by_rank": communication_bytes_by_rank,
            "discarded_weight_by_step": discarded_weight_by_step,
            "communication_plan": {
                "status": "production_executed" if passed else "failed",
                "communication_backend": backend,
                "communication_bytes": communication_bytes,
                "boundary_operation_count": len(unique_updates),
                "topology_tier": "intra_node",
            },
            "full_mps_materialization": False,
            "rank_records": records,
            "completed": passed,
            "fallback_events": [],
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": [
                "capacity_probe_has_no_matched_single_gpu_oom_baseline",
            ]
            + (
                []
                if convergence_certified
                else ["convergence_and_energy_accuracy_gate_not_certified"]
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "completed": passed}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
