"""Distributed 42q ADAPT-VQE TN hero benchmark with real NCCL collectives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

import flagquantum as fq
try:
    from adapt_vqe_tn import (
        build_circuit,
        commutator_gradient_hamiltonian,
        deserialize_slicing_plan,
        hamiltonian,
        operator_pool,
        reslice_binary_axes,
    )
except ModuleNotFoundError:
    from benchmarks.development.adapt_vqe_tn import (
        build_circuit,
        commutator_gradient_hamiltonian,
        deserialize_slicing_plan,
        hamiltonian,
        operator_pool,
        reslice_binary_axes,
    )
from flagquantum.runtime.backends.tensor_network.distributed_sliced_reverse import (
    execute_distributed_sliced_tn_explicit_reverse,
)
from flagquantum.runtime.backends.tensor_network.sliced_tasks import (
    plan_distributed_tn_slice_tasks,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--extra-entanglers", type=int, default=12)
    parser.add_argument(
        "--initial-state",
        choices=("mean_field", "zero", "legacy_h_ry"),
        default="mean_field",
    )
    parser.add_argument("--pool-limit", type=int, default=16)
    parser.add_argument("--adapt-iterations", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--optimization-steps", type=int, default=50)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-intermediate-gib", type=float, default=2.0)
    parser.add_argument("--binary-target-slices", type=int, default=16)
    parser.add_argument("--plan-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def limited_pool(rows: int, cols: int, limit: int):
    pool = operator_pool(rows, cols)
    if limit < len(pool):
        indices = tuple(
            round(index * (len(pool) - 1) / max(1, limit - 1))
            for index in range(limit)
        )
        pool = tuple(pool[index] for index in indices)
    return pool


def main() -> None:
    args = arguments()
    if args.adapt_iterations <= 0 or args.optimization_steps <= 0:
        raise ValueError("adapt iterations and optimization steps must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.reset_peak_memory_stats(device)

    def progress(stage: str) -> None:
        print(f"[rank={rank}] {stage}", flush=True)

    pool = limited_pool(args.rows, args.cols, args.pool_limit)
    target = hamiltonian(args.rows, args.cols)
    artifact = json.loads(args.plan_artifact.read_text(encoding="utf-8"))
    initial_slicing = deserialize_slicing_plan(
        artifact["plans"]["0"]["slicing"]
    )
    initial_slicing.validate_economics()
    if initial_slicing.n_slices < world_size:
        raise ValueError("initial slicing plan has fewer slices than ranks")

    def circuit(operators, parameters):
        return build_circuit(
            args.rows,
            args.cols,
            args.cycles,
            args.extra_entanglers,
            operators,
            parameters,
            args.initial_state,
        )

    def distributed_value_and_gradient(plan, slicing, parameters):
        tasks = plan_distributed_tn_slice_tasks(
            slicing,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        return execute_distributed_sliced_tn_explicit_reverse(
            plan,
            slicing,
            tasks,
            parameters,
            gradient_reduction="all_reduce",
        )

    dist.barrier()
    progress("initial_build_start")
    torch.cuda.synchronize(device)
    started = perf_counter()
    empty = torch.empty(0, dtype=torch.float64, device=device)
    initial_circuit = circuit((), empty)
    initial_plan = fq.build_tensor_network_hamiltonian_expectation(
        initial_circuit, target
    )
    progress("initial_plan_done")
    initial_result = distributed_value_and_gradient(
        initial_plan, initial_slicing, ()
    )
    progress("initial_execute_done")

    local_screening_seconds = 0.0
    screening_collective_seconds = 0.0
    target_peak_elements = int(
        args.max_intermediate_gib * 2**30 // torch.tensor([], dtype=torch.complex128).element_size()
    )
    selected_indices: list[int] = []
    parameters = torch.empty(0, dtype=torch.float64, device=device)
    adapt_records = []
    reverse_result = None
    selected_slicing = initial_slicing
    local_gradients = torch.zeros(len(pool), dtype=torch.float64, device=device)
    for adapt_iteration in range(1, args.adapt_iterations + 1):
        current_circuit = circuit(
            tuple(pool[index] for index in selected_indices), parameters
        )
        local_gradients = torch.zeros(len(pool), dtype=torch.float64, device=device)
        available = [index for index in range(len(pool)) if index not in selected_indices]
        for index in available:
            if index % world_size != rank:
                continue
            progress(f"adapt_{adapt_iteration}_screening_{index}_start")
            screening_started = perf_counter()
            observable = commutator_gradient_hamiltonian(pool[index], target)
            if observable is not None:
                plan = fq.build_tensor_network_hamiltonian_expectation(
                    current_circuit, observable
                )
                slicing = plan.cotengra_slicing_plan(
                    target_peak_elements=target_peak_elements,
                    max_repeats=1,
                    minimize="write",
                    methods=("greedy",),
                    seed=0,
                )
                slicing.validate_economics()
                local_gradients[index] = plan.contract_slicing_plan(slicing).real.sum()
            torch.cuda.synchronize(device)
            local_screening_seconds += perf_counter() - screening_started
            progress(f"adapt_{adapt_iteration}_screening_{index}_done")
        screening_collective_started = perf_counter()
        dist.all_reduce(local_gradients, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        screening_collective_seconds += perf_counter() - screening_collective_started
        selected_index = max(available, key=lambda index: abs(float(local_gradients[index])))
        selected_gradient = float(local_gradients[selected_index])
        selected_indices.append(selected_index)
        parameters = torch.cat(
            (parameters.detach(), torch.zeros(1, dtype=torch.float64, device=device))
        ).requires_grad_(True)
        selected_operators = tuple(pool[index] for index in selected_indices)
        selected_circuit = circuit(selected_operators, parameters)
        selected_plan = fq.build_tensor_network_hamiltonian_expectation(
            selected_circuit, target
        )
        progress(f"adapt_{adapt_iteration}_selected_plan_done")
        selected_slicing = selected_plan.cotengra_slicing_plan(
            target_peak_elements=2**60,
            max_repeats=1,
            minimize="write",
            methods=("greedy",),
            seed=0,
        )
        selected_slicing = reslice_binary_axes(
            selected_plan,
            selected_slicing,
            target_slices=args.binary_target_slices,
        )
        selected_slicing.validate_economics()
        if selected_slicing.n_slices < world_size:
            raise ValueError("selected slicing plan has fewer slices than ranks")
        first_moment = torch.zeros_like(parameters)
        second_moment = torch.zeros_like(parameters)
        optimization_history = []
        optimizer_gradients = []
        previous_energy = None
        for step in range(1, args.optimization_steps + 1):
            selected_circuit = circuit(selected_operators, parameters)
            selected_plan = fq.build_tensor_network_hamiltonian_expectation(
                selected_circuit, target
            )
            reverse_result = distributed_value_and_gradient(
                selected_plan, selected_slicing, (parameters,)
            )
            energy = float(reverse_result.value.real.sum())
            gradient = reverse_result.parameter_gradients[0]
            optimization_history.append(energy)
            optimizer_gradients.append(gradient.detach().cpu().tolist())
            progress(
                f"adapt_{adapt_iteration}_optimizer_{step}_done energy={energy:.15g}"
            )
            if previous_energy is not None and abs(energy - previous_energy) <= args.energy_tolerance:
                break
            previous_energy = energy
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * torch.square(gradient)
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            parameters = (
                parameters.detach()
                - args.learning_rate
                * corrected_first
                / (torch.sqrt(corrected_second) + 1e-8)
            ).requires_grad_(True)
        assert reverse_result is not None
        adapt_records.append(
            {
                "adapt_iteration": adapt_iteration,
                "selected_pool_index": selected_index,
                "selected_gradient": selected_gradient,
                "pool_gradients": local_gradients.cpu().tolist(),
                "optimization_steps_completed": len(optimization_history),
                "optimization_history": optimization_history,
                "optimizer_gradients": optimizer_gradients,
                "parameters": parameters.detach().cpu().tolist(),
            }
        )
    final_circuit = circuit(
        tuple(pool[index] for index in selected_indices), parameters.detach()
    )
    final_plan = fq.build_tensor_network_hamiltonian_expectation(final_circuit, target)
    final_result = distributed_value_and_gradient(final_plan, selected_slicing, ())
    progress("final_execute_done")

    dist.barrier()
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    local_metrics = torch.tensor(
        [
            elapsed,
            local_screening_seconds,
            screening_collective_seconds,
            reverse_result.local_execution_seconds,
            reverse_result.collective_seconds,
            float(torch.cuda.max_memory_allocated(device)),
            float(reverse_result.local_task_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local_metrics) for _ in range(world_size)]
    dist.all_gather(gathered, local_metrics)
    if rank == 0:
        rank_metrics = [item.cpu().tolist() for item in gathered]
        payload = {
            "schema_version": 1,
            "workload": "qubit_adapt_vqe_grid_tfim_distributed",
            "implementation": "FlagQuantum",
            "distribution_semantics": "candidate_partition_plus_rank_owned_slices",
            "scalability_claim_allowed": False,
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": world_size // local_world_size,
            "rows": args.rows,
            "cols": args.cols,
            "qubits": args.rows * args.cols,
            "cycles": args.cycles,
            "extra_entanglers": args.extra_entanglers,
            "initial_state": args.initial_state,
            "dtype": "complex128",
            "parameter_dtype": "float64",
            "learning_rate": args.learning_rate,
            "optimization_steps_requested": args.optimization_steps,
            "adapt_iterations_requested": args.adapt_iterations,
            "adapt_iterations_completed": len(adapt_records),
            "energy_tolerance": args.energy_tolerance,
            "operator_pool_size": len(pool),
            "initial_slice_count": initial_slicing.n_slices,
            "selected_slice_count": selected_slicing.n_slices,
            "selected_pool_indices": selected_indices,
            "initial_energy": float(initial_result.value.real.sum()),
            "parameters": parameters.detach().cpu().tolist(),
            "final_energy": float(final_result.value.real.sum()),
            "iterations": adapt_records,
            "execution_seconds": max(item[0] for item in rank_metrics),
            "peak_cuda_allocated_bytes": int(max(item[5] for item in rank_metrics)),
            "rank_metrics": [
                {
                    "rank": index,
                    "execution_seconds": values[0],
                    "screening_seconds": values[1],
                    "screening_collective_seconds": values[2],
                    "reverse_seconds": values[3],
                    "reverse_collective_seconds": values[4],
                    "peak_cuda_allocated_bytes": int(values[5]),
                    "reverse_local_task_count": int(values[6]),
                }
                for index, values in enumerate(rank_metrics)
            ],
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
