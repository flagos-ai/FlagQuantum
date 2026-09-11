"""Benchmark rank-owned sliced TN forward and reverse with NCCL."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
from collections import Counter
from dataclasses import replace
from math import prod
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.distributed_optimizer import (
    execute_rank_owned_tn_sgd_step,
)
from flagquantum.runtime.executors.tensor_network.distributed_sliced_reverse import (
    execute_distributed_sliced_tn_explicit_reverse,
)
from flagquantum.runtime.executors.tensor_network.sliced_reverse import (
    estimate_sliced_tn_full_tape_bytes,
    plan_sliced_tn_checkpoint_memory,
)
from flagquantum.runtime.executors.tensor_network.sliced_tasks import (
    plan_distributed_tn_slice_tasks,
)
from flagquantum.simulation.tensor_network.contraction import (
    _canonicalize_unit_extent_nodes,
    _cost_for_sliced_labels,
    _pair_steps_from_dynamic_path,
    _slice_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkSlicingPlan


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=18)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--slice-label-count", type=int, default=6)
    parser.add_argument(
        "--sliced-labels",
        type=str,
        help="Comma-separated explicit TN labels; overrides --slice-label-count.",
    )
    parser.add_argument("--grid-rows", type=int)
    parser.add_argument("--grid-cols", type=int)
    parser.add_argument("--grid-cycles", type=int, default=7)
    parser.add_argument("--max-intermediate-gib", type=float)
    parser.add_argument("--max-full-tape-gib", type=float)
    parser.add_argument("--checkpoint-gib", type=float)
    parser.add_argument(
        "--planner",
        choices=("native", "cotengra"),
        default="native",
    )
    parser.add_argument("--cotengra-max-repeats", type=int, default=16)
    parser.add_argument("--planner-cache", type=Path)
    parser.add_argument("--resolved-planner-cache", type=Path)
    parser.add_argument(
        "--cotengra-tree-path",
        type=Path,
        help="Import TensorCircuit-NG/cotengra tree-data and map it to this network.",
    )
    parser.add_argument(
        "--target-slices",
        type=int,
        help="Increase an imported path to at least this many slices.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--compiled-reverse", action="store_true")
    parser.add_argument("--deferred-parameter-pullback", action="store_true")
    parser.add_argument("--slice-batch-size", type=int, default=1)
    return parser.parse_args()


def _import_cotengra_tree_path(
    expectation, path: Path
) -> tuple[TensorNetworkSlicingPlan, tuple[int, ...]]:
    """Map a cotengra tree-data pickle onto the canonical FlagQuantum labels."""
    with path.open("rb") as stream:
        tree_data = pickle.load(stream)
    nodes, output_labels = _canonicalize_unit_extent_nodes(
        expectation.nodes, expectation.output_labels
    )
    external_inputs = tuple(tuple(labels) for labels in tree_data["inputs"])
    if len(external_inputs) != len(nodes):
        raise ValueError("external cotengra path has a different node count")
    if len(nodes) != 468:
        raise ValueError("external node ordering is not the supported 40q grid layout")
    node_order = tuple(
        list(range(428, 468))
        + list(range(0, 40))
        + list(range(40, 80))
        + list(range(80, 214))
        + list(range(214, 254))
        + list(range(254, 294))
        + list(range(294, 428))
    )
    reordered_nodes = tuple(nodes[index] for index in node_order)
    if tuple(map(len, external_inputs)) != tuple(
        len(node.labels) for node in reordered_nodes
    ):
        raise ValueError("external cotengra path has a different node-rank layout")

    dims = {
        label: int(node.tensor.shape[index])
        for node in reordered_nodes
        for index, label in enumerate(node.labels)
    }
    local_by_incidence: dict[tuple[tuple[int, ...], int], list[int]] = {}
    for label in dims:
        incidence = tuple(
            index for index, node in enumerate(reordered_nodes) if label in node.labels
        )
        local_by_incidence.setdefault((incidence, dims[label]), []).append(label)
    label_map = {}
    for label, size in tree_data["size_dict"].items():
        incidence = tuple(
            index for index, labels in enumerate(external_inputs) if label in labels
        )
        candidates = local_by_incidence.get((incidence, int(size)), [])
        if not candidates:
            raise ValueError("external cotengra path hypergraph does not match")
        label_map[label] = candidates.pop()
    mapped_output = tuple(label_map[label] for label in tree_data["output"])
    if mapped_output != tuple(output_labels):
        raise ValueError("external cotengra path output does not match")
    for external_label, size in tree_data["size_dict"].items():
        if dims[label_map[external_label]] != int(size):
            raise ValueError("external cotengra path dimensions do not match")
    sliced_labels = tuple(
        label_map[label] for label in tree_data["sliced_inds"]
    )
    assignments = {label: 0 for label in sliced_labels}
    steps = _pair_steps_from_dynamic_path(
        _slice_nodes(reordered_nodes, assignments), output_labels, tree_data["path"]
    )
    slice_shape = tuple(dims[label] for label in sliced_labels)
    n_slices = prod(slice_shape) if slice_shape else 1
    per_slice_cost = sum(step.estimated_cost for step in steps)
    peak_size = max((step.intermediate_size for step in steps), default=1)
    baseline = _cost_for_sliced_labels(
        reordered_nodes, output_labels, (), contraction_strategy="quality_multistart"
    )["estimated_cost"]
    element_size = max(node.tensor.element_size() for node in reordered_nodes)
    return TensorNetworkSlicingPlan(
        sliced_labels=sliced_labels,
        slice_shape=slice_shape,
        n_slices=n_slices,
        per_slice_cost=per_slice_cost,
        total_estimated_cost=per_slice_cost * n_slices,
        peak_size=peak_size,
        target_peak_size=peak_size,
        baseline_estimated_cost=baseline,
        recomputation_factor=float(per_slice_cost * n_slices) / float(baseline),
        budget_satisfied=True,
        element_size_bytes=element_size,
        peak_bytes=peak_size * element_size,
        target_peak_bytes=peak_size * element_size,
        contraction_path=steps,
        contraction_path_source="tensorcircuit-ng:cotengra-tree-data",
        canonicalize_unit_extent_labels=True,
    ), node_order


def _build_expectation(
    parameters: tuple[torch.Tensor, ...],
    *,
    qubits: int,
    grid_rows: int | None,
    grid_cols: int | None,
    grid_cycles: int,
    layers: int,
    device: torch.device,
):
    circuit = fq.Circuit(qubits, device=device, dtype=torch.complex128)
    for qubit, parameter in enumerate(parameters):
        circuit.ry(qubit, theta=parameter)
    if grid_rows is not None and grid_cols is not None:
        for cycle in range(grid_cycles):
            if cycle % 2 == 0:
                for row in range(grid_rows):
                    for col in range(grid_cols - 1):
                        circuit.cx(
                            row * grid_cols + col,
                            row * grid_cols + col + 1,
                        )
            else:
                for row in range(grid_rows - 1):
                    for col in range(grid_cols):
                        circuit.cx(
                            row * grid_cols + col,
                            (row + 1) * grid_cols + col,
                        )
    else:
        for layer in range(layers):
            for qubit in range(layer % 2, qubits - 1, 2):
                circuit.cx(qubit, qubit + 1)
    ket_plan = fq.build_tensor_network(
        circuit,
        device=device,
        dtype=torch.complex128,
    )
    expectation = fq.build_tensor_network_expectation(
        ket_plan,
        z=list(range(qubits)),
    )
    expectation = replace(
        expectation,
        nodes=tuple(
            replace(
                node,
                tensor=node.tensor.to(
                    device=device,
                    dtype=torch.complex128,
                ),
            )
            for node in expectation.nodes
        ),
    )
    node_dtypes = {node.tensor.dtype for node in expectation.nodes}
    if node_dtypes != {torch.complex128}:
        raise RuntimeError(
            f"TN NCCL benchmark requires uniform complex128 nodes, got {node_dtypes}"
        )
    return expectation


def _audit_evidence_fields(
    rank_payloads: list[dict[str, object] | None],
    *,
    world_size: int,
) -> dict[str, object]:
    """Lift executed rank evidence into the common scalability audit schema."""

    ranks = tuple(item for item in rank_payloads if item is not None)
    semantics = (
        "single_device_fast_path" if world_size == 1 else "sharded_across_ranks"
    )
    fields: dict[str, object] = {
        "claim_evidence_type": "production_runtime",
        "distribution_semantics": semantics,
        "forward_distribution_semantics": semantics,
        "backward_distribution_semantics": semantics,
    }
    if world_size == 1:
        return fields
    fields.update(
        {
            "rank_placement": [
                {
                    "rank": int(item["rank"]),
                    "local_rank": int(item["local_rank"]),
                    "hostname": str(item["hostname"]),
                }
                for item in ranks
            ],
            "rank_shards": [
                {
                    "rank": int(item["rank"]),
                    "owned_slice_count": int(item["local_task_count"]),
                    "task_plan_identity": str(item["task_plan_identity"]),
                }
                for item in ranks
            ],
            "rank_peak_memory_bytes": [
                int(item["cuda_peak_allocated_bytes"]) for item in ranks
            ],
            "communication_plan": {
                "collective_backend": "nccl",
                "collective_count_per_rank": [
                    int(item["collective_count"]) for item in ranks
                ],
                "collective_payload_bytes_per_rank": [
                    int(item["collective_payload_bytes_per_rank"])
                    for item in ranks
                ],
                "physical_route": "topology_dependent",
            },
        }
    )
    return fields


def main() -> None:
    arguments = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if arguments.training_steps <= 0:
        raise ValueError("--training-steps must be positive")

    grid_enabled = arguments.grid_rows is not None or arguments.grid_cols is not None
    if grid_enabled:
        if arguments.grid_rows is None or arguments.grid_cols is None:
            raise ValueError("--grid-rows and --grid-cols must be provided together")
        qubits = int(arguments.grid_rows) * int(arguments.grid_cols)
    else:
        qubits = int(arguments.qubits)
    parameters = tuple(
        torch.tensor(
            0.1 * (qubit + 1),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        for qubit in range(qubits)
    )
    expectation = _build_expectation(
        parameters,
        qubits=qubits,
        grid_rows=arguments.grid_rows,
        grid_cols=arguments.grid_cols,
        grid_cycles=int(arguments.grid_cycles),
        layers=int(arguments.layers),
        device=device,
    )
    planning_start = perf_counter()
    if arguments.cotengra_tree_path is not None:
        if rank == 0:
            imported, node_order = _import_cotengra_tree_path(
                expectation, arguments.cotengra_tree_path
            )
            shared_slicing = [(imported, node_order)]
        else:
            shared_slicing = [None]
        dist.broadcast_object_list(shared_slicing, src=0)
        imported_payload = shared_slicing[0]
        if imported_payload is None:
            raise RuntimeError("rank 0 did not import the cotengra tree path")
        slicing, node_order = imported_payload
        canonical_nodes, canonical_output_labels = _canonicalize_unit_extent_nodes(
            expectation.nodes, expectation.output_labels
        )
        expectation = replace(
            expectation,
            nodes=tuple(canonical_nodes[index] for index in node_order),
            output_labels=canonical_output_labels,
        )
    elif arguments.planner == "cotengra":
        if arguments.sliced_labels:
            raise ValueError("--sliced-labels cannot be combined with cotengra")
        if arguments.max_intermediate_gib is None:
            raise ValueError("cotengra planner requires --max-intermediate-gib")
        shared_slicing: list[object | None] = [None]
        if rank == 0:
            resolved_cache_hit = bool(
                arguments.resolved_planner_cache
                and arguments.resolved_planner_cache.exists()
            )
            if resolved_cache_hit:
                with arguments.resolved_planner_cache.open("rb") as stream:
                    shared_slicing[0] = pickle.load(stream)
            elif arguments.planner_cache and arguments.planner_cache.exists():
                with arguments.planner_cache.open("rb") as stream:
                    shared_slicing[0] = pickle.load(stream)
            else:
                target_bytes = int(
                    float(arguments.max_intermediate_gib) * 1024**3
                )
                element_size = max(
                    node.tensor.element_size() for node in expectation.nodes
                )
                shared_slicing[0] = expectation.cotengra_slicing_plan(
                    target_peak_elements=target_bytes // int(element_size),
                    max_repeats=int(arguments.cotengra_max_repeats),
                )
                if arguments.planner_cache:
                    arguments.planner_cache.parent.mkdir(parents=True, exist_ok=True)
                    with arguments.planner_cache.open("wb") as stream:
                        pickle.dump(shared_slicing[0], stream)
            if arguments.target_slices is not None and not resolved_cache_hit:
                shared_slicing[0] = expectation.reslice_external_plan(
                    shared_slicing[0],
                    target_slices=int(arguments.target_slices),
                )
                if arguments.resolved_planner_cache:
                    arguments.resolved_planner_cache.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    with arguments.resolved_planner_cache.open("wb") as stream:
                        pickle.dump(shared_slicing[0], stream)
        dist.broadcast_object_list(shared_slicing, src=0)
        slicing = shared_slicing[0]
        if slicing is None:
            raise RuntimeError("rank 0 did not produce a cotengra slicing plan")
    elif arguments.sliced_labels:
        explicit_labels = tuple(
            int(item.strip())
            for item in arguments.sliced_labels.split(",")
            if item.strip()
        )
        if not explicit_labels:
            raise ValueError("--sliced-labels must contain at least one label")
        slicing = expectation.slicing_plan(sliced_labels=explicit_labels)
    elif arguments.max_intermediate_gib is not None:
        slicing = expectation.slicing_plan(
            max_intermediate_bytes=int(
                float(arguments.max_intermediate_gib) * 1024**3
            )
        )
    else:
        counts = Counter(label for node in expectation.nodes for label in node.labels)
        labels = tuple(
            label
            for label, count in counts.items()
            if count >= 2 and label not in expectation.output_labels
        )[: arguments.slice_label_count]
        if len(labels) != arguments.slice_label_count:
            raise RuntimeError("TN benchmark could not find enough slice labels")
        slicing = expectation.slicing_plan(sliced_labels=labels)
    tasks = plan_distributed_tn_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=local_world_size,
    )
    planning_seconds = perf_counter() - planning_start
    if slicing.n_slices < world_size:
        raise RuntimeError("TN benchmark requires at least one slice per rank")
    local_tasks = tuple(task for task in tasks.tasks if task.owner_rank == rank)
    estimated_full_tape_bytes = estimate_sliced_tn_full_tape_bytes(
        expectation,
        slicing,
        assignments=local_tasks[0].assignments,
    )
    tape_budget_gib = (
        arguments.max_full_tape_gib
        if arguments.max_full_tape_gib is not None
        else arguments.max_intermediate_gib
    )
    tape_budget_bytes = (
        None
        if tape_budget_gib is None
        else int(float(tape_budget_gib) * 1024**3)
    )
    checkpoint_budget_bytes = (
        None
        if arguments.checkpoint_gib is None
        else int(float(arguments.checkpoint_gib) * 1024**3)
    )
    checkpoint_memory = (
        None
        if checkpoint_budget_bytes is None
        else plan_sliced_tn_checkpoint_memory(
            expectation,
            slicing,
            checkpoint_budget_bytes=checkpoint_budget_bytes,
            assignments=local_tasks[0].assignments,
        )
    )
    tape_preflight_passed = (
        tape_budget_bytes is None
        or (
            checkpoint_memory.predicted_working_set_bytes
            if checkpoint_memory is not None
            else estimated_full_tape_bytes
        )
        <= tape_budget_bytes
    )
    tape_preflights: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(
        tape_preflights,
        {
            "rank": rank,
            "estimated_full_tape_bytes": estimated_full_tape_bytes,
            "full_tape_budget_bytes": tape_budget_bytes,
            "checkpoint_budget_bytes": checkpoint_budget_bytes,
            "checkpoint_memory_plan": (
                None
                if checkpoint_memory is None
                else checkpoint_memory.summary(include_value_ids=False)
            ),
            "passed": tape_preflight_passed,
        },
    )
    if not all(bool(item["passed"]) for item in tape_preflights if item):
        if rank == 0:
            payload = {
                "schema_version": 1,
                "workload": (
                    f"{arguments.grid_rows}x{arguments.grid_cols}_"
                    f"{arguments.grid_cycles}c_full_tape_preflight"
                ),
                "world_size": world_size,
                "local_world_size": local_world_size,
                "node_count": max(1, world_size // local_world_size),
                "dtype": str(expectation.nodes[0].tensor.dtype).removeprefix(
                    "torch."
                ),
                "slice_count": slicing.n_slices,
                "sliced_labels": slicing.sliced_labels,
                "contraction_path_source": slicing.contraction_path_source,
                "contraction_path_steps": len(slicing.contraction_path),
                "per_slice_estimated_flops": slicing.per_slice_cost,
                "per_slice_peak_bytes": slicing.peak_bytes,
                "rank_preflights": tape_preflights,
                "execution_started": False,
                "capacity_preflight_passed": False,
                "capacity_blocker": (
                    "checkpointed_reverse_working_set_exceeds_rank_memory_budget"
                    if checkpoint_budget_bytes is not None
                    else "explicit_reverse_full_tape_exceeds_rank_memory_budget"
                ),
                "distribution_semantics": "planned_rank_owned_tn_slices",
                "scalability_claim_allowed": False,
            }
            if arguments.output is not None:
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                arguments.output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    if arguments.preflight_only:
        if rank == 0:
            payload = {
                "schema_version": 1,
                "workload": (
                    f"{arguments.grid_rows}x{arguments.grid_cols}_"
                    f"{arguments.grid_cycles}c_checkpoint_preflight"
                ),
                "world_size": world_size,
                "local_world_size": local_world_size,
                "node_count": max(1, world_size // local_world_size),
                "dtype": str(expectation.nodes[0].tensor.dtype).removeprefix(
                    "torch."
                ),
                "slice_count": slicing.n_slices,
                "per_slice_estimated_flops": slicing.per_slice_cost,
                "total_estimated_flops": slicing.total_estimated_cost,
                "per_slice_peak_bytes": slicing.peak_bytes,
                "rank_preflights": tape_preflights,
                "execution_started": False,
                "capacity_preflight_passed": True,
                "distribution_semantics": "planned_rank_owned_tn_slices",
                "scalability_claim_allowed": False,
            }
            if arguments.output is not None:
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                arguments.output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    warmup_value = torch.zeros(1, dtype=torch.complex128, device=device)
    dist.all_reduce(warmup_value)
    for parameter in parameters:
        warmup_gradient = torch.zeros_like(parameter)
        dist.all_reduce(warmup_gradient)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    torch.cuda.synchronize(device)
    execution_start = perf_counter()
    correctness = None
    step_records: list[dict[str, object]] = []
    result = None
    optimizer = None
    for step in range(arguments.training_steps):
        if step > 0:
            expectation = _build_expectation(
                parameters,
                qubits=qubits,
                grid_rows=arguments.grid_rows,
                grid_cols=arguments.grid_cols,
                grid_cycles=int(arguments.grid_cycles),
                layers=int(arguments.layers),
                device=device,
            )
            step_slicing = slicing
            tasks = plan_distributed_tn_slice_tasks(
                step_slicing,
                world_size=world_size,
                local_world_size=local_world_size,
            )
        else:
            step_slicing = slicing
        step_start = perf_counter()
        result = execute_distributed_sliced_tn_explicit_reverse(
            expectation,
            step_slicing,
            tasks,
            parameters,
            checkpoint_budget_bytes=checkpoint_budget_bytes,
            compiled_reverse=arguments.compiled_reverse,
            deferred_parameter_pullback=arguments.deferred_parameter_pullback,
            slice_batch_size=arguments.slice_batch_size,
            gradient_reduction=(
                "owner_reduce" if arguments.skip_reference else "all_reduce"
            ),
        )
        if rank == 0 and not arguments.skip_reference and step == 0:
            reference = expectation.contract(strategy="greedy")
            reference_gradients = torch.autograd.grad(reference.real, parameters)
            value_error = float((result.value - reference).detach().abs().max())
            gradient_error = max(
                float((actual - expected).detach().abs().max())
                for actual, expected in zip(
                    result.parameter_gradients,
                    reference_gradients,
                )
            )
            correctness = {
                "value_max_abs_error": value_error,
                "gradient_max_abs_error": gradient_error,
                "passed": value_error <= 1e-9 and gradient_error <= 1e-8,
                "checked_step": 0,
            }
        gradient_l2_squared = torch.stack(
            tuple(
                gradient.detach().abs().square().reshape(())
                for gradient in result.parameter_gradients
            )
        ).sum()
        if result.gradient_aggregation_semantics == "reduce_to_parameter_owner":
            dist.all_reduce(gradient_l2_squared, op=dist.ReduceOp.SUM)
        gradient_l2 = float(
            gradient_l2_squared.sqrt()
        )
        optimizer = execute_rank_owned_tn_sgd_step(
            parameters,
            result.parameter_gradients,
            learning_rate=arguments.learning_rate,
        )
        torch.cuda.synchronize(device)
        step_records.append(
            {
                "step": step,
                "execution_seconds": perf_counter() - step_start,
                "value_real": float(result.value.real.detach()),
                "value_imag": float(result.value.imag.detach()),
                "gradient_l2": gradient_l2,
                "parameter_checksum_after": float(
                    torch.stack(
                        tuple(parameter.detach() for parameter in parameters)
                    ).sum()
                ),
                "optimizer_seconds": optimizer.execution_seconds,
                "output_finite": bool(torch.isfinite(result.value).all()),
                "gradients_finite": all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in result.parameter_gradients
                ),
            }
        )
    assert result is not None
    assert optimizer is not None
    torch.cuda.synchronize(device)
    dist.barrier()
    execution_seconds = perf_counter() - execution_start
    local_payload = {
        **result.summary(),
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "execution_seconds": execution_seconds,
        "planning_seconds": planning_seconds,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "optimizer": optimizer.summary(),
        "step_records": step_records,
    }
    rank_payloads: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(rank_payloads, local_payload)
    node_count = len(
        {
            str(item["hostname"])
            for item in rank_payloads
            if item is not None
        }
    )
    parameter_consistency_passed = all(
        max(
            float(item["step_records"][step]["parameter_checksum_after"])
            for item in rank_payloads
            if item is not None
        )
        - min(
            float(item["step_records"][step]["parameter_checksum_after"])
            for item in rank_payloads
            if item is not None
        )
        <= 1e-12
        for step in range(arguments.training_steps)
    )
    if not parameter_consistency_passed:
        raise RuntimeError("optimizer parameters diverged across ranks")

    if rank == 0 and not arguments.skip_reference:
        audit_fields = _audit_evidence_fields(rank_payloads, world_size=world_size)
        payload = {
            "schema_version": 1,
            "workload": (
                (
                    f"{arguments.grid_rows}x{arguments.grid_cols}_"
                    f"{arguments.grid_cycles}c"
                )
                if grid_enabled
                else f"{qubits}q_{arguments.layers}l"
            )
            + f"_{slicing.n_slices}s_explicit_gradient",
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": node_count,
            "dtype": str(expectation.nodes[0].tensor.dtype).removeprefix("torch."),
            "parameter_count": len(parameters),
            "sliced_labels": slicing.sliced_labels,
            "slice_count": slicing.n_slices,
            "per_slice_estimated_flops": slicing.per_slice_cost,
            "total_estimated_flops": slicing.total_estimated_cost,
            "per_slice_peak_bytes": slicing.peak_bytes,
            "contraction_path_source": slicing.contraction_path_source,
            "contraction_path_steps": len(slicing.contraction_path),
            "rank_results": rank_payloads,
            "max_execution_seconds": max(
                float(item["execution_seconds"])
                for item in rank_payloads
                if item is not None
            ),
            "max_cuda_peak_allocated_bytes": max(
                int(item["cuda_peak_allocated_bytes"])
                for item in rank_payloads
                if item is not None
            ),
            "correctness": correctness,
            **audit_fields,
            "gradient_distribution_semantics": result.summary()[
                "gradient_distribution_semantics"
            ],
            "local_gradient_contribution_semantics": (
                "rank_owned_slice_contributions"
            ),
            "gradient_aggregation_semantics": (
                result.gradient_aggregation_semantics
            ),
            "gradient_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership": optimizer.ownership,
            "optimizer": optimizer.summary(),
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_ownership": optimizer.ownership,
            "parameter_distribution_after_step": "replicated_for_next_forward",
            "training_step_count": arguments.training_steps,
            "step_records": step_records,
            "parameter_consistency_passed": parameter_consistency_passed,
            "scalability_claim_allowed": False,
            "claim_boundary": (
                "Development hardware evidence; release audit and a "
                "single-device capacity failure are pending."
            ),
        }
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, sort_keys=True), flush=True)
    elif rank == 0:
        audit_fields = _audit_evidence_fields(rank_payloads, world_size=world_size)
        payload = {
            "schema_version": 1,
            "workload": (
                f"{arguments.grid_rows}x{arguments.grid_cols}_"
                f"{arguments.grid_cycles}c_{slicing.n_slices}s_capacity"
            ),
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": node_count,
            "dtype": str(expectation.nodes[0].tensor.dtype).removeprefix("torch."),
            "parameter_count": len(parameters),
            "sliced_labels": slicing.sliced_labels,
            "slice_count": slicing.n_slices,
            "per_slice_estimated_flops": slicing.per_slice_cost,
            "total_estimated_flops": slicing.total_estimated_cost,
            "per_slice_peak_bytes": slicing.peak_bytes,
            "rank_results": rank_payloads,
            "max_execution_seconds": max(
                float(item["execution_seconds"])
                for item in rank_payloads
                if item is not None
            ),
            "max_cuda_peak_allocated_bytes": max(
                int(item["cuda_peak_allocated_bytes"])
                for item in rank_payloads
                if item is not None
            ),
            "output_finite": bool(torch.isfinite(result.value).all()),
            "gradients_finite": all(
                bool(torch.isfinite(gradient).all())
                for gradient in result.parameter_gradients
            ),
            "reference_skipped": True,
            **audit_fields,
            "gradient_distribution_semantics": result.summary()[
                "gradient_distribution_semantics"
            ],
            "local_gradient_contribution_semantics": (
                "rank_owned_slice_contributions"
            ),
            "gradient_aggregation_semantics": (
                result.gradient_aggregation_semantics
            ),
            "gradient_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership": optimizer.ownership,
            "optimizer": optimizer.summary(),
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_ownership": optimizer.ownership,
            "parameter_distribution_after_step": "replicated_for_next_forward",
            "training_step_count": arguments.training_steps,
            "step_records": step_records,
            "parameter_consistency_passed": parameter_consistency_passed,
            "scalability_claim_allowed": False,
            "claim_boundary": (
                "Capacity execution evidence; numerical semantics are covered "
                "by separate reference-sized distributed tests."
            ),
        }
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
