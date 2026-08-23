"""Fair TensorCircuit-NG baseline for the FlagQuantum 36q TN benchmark."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

import jax
from jax.experimental import multihost_utils

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.experimental import DistributedContractor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=9)
    parser.add_argument("--grid-cycles", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=2**28)
    parser.add_argument("--path-repeats", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path-input", type=Path)
    parser.add_argument("--path-output", type=Path)
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--coordinator-address")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--local-device-ids")
    return parser.parse_args()


def _block(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def main() -> None:
    args = _arguments()
    if args.num_processes > 1:
        if not args.coordinator_address:
            raise ValueError("--coordinator-address is required for multi-host JAX")
        local_device_ids = (
            None
            if not args.local_device_ids
            else tuple(int(item) for item in args.local_device_ids.split(","))
        )
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=local_device_ids,
        )
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    available = jax.devices()
    if args.devices < 1 or args.devices > len(available):
        raise ValueError(f"requested {args.devices} devices, found {len(available)}")
    devices = available[: args.devices]
    qubits = args.grid_rows * args.grid_cols
    params = jnp.asarray([0.1 * (q + 1) for q in range(qubits)], dtype=jnp.float64)

    def nodes_fn(theta):
        circuit = tc.Circuit(qubits)
        for qubit in range(qubits):
            circuit.ry(qubit, theta=theta[qubit])
        for cycle in range(args.grid_cycles):
            if cycle % 2 == 0:
                for row in range(args.grid_rows):
                    for col in range(args.grid_cols - 1):
                        circuit.cnot(
                            row * args.grid_cols + col,
                            row * args.grid_cols + col + 1,
                        )
            else:
                for row in range(args.grid_rows - 1):
                    for col in range(args.grid_cols):
                        circuit.cnot(
                            row * args.grid_cols + col,
                            (row + 1) * args.grid_cols + col,
                        )
        ops = tuple((tc.gates.z(), [q]) for q in range(qubits))
        return circuit.expectation_before(*ops, reuse=False)

    planning_start = perf_counter()
    if args.path_input:
        contractor = DistributedContractor.from_path(
            str(args.path_input), nodes_fn, devices=devices, params=params
        )
    else:
        contractor = DistributedContractor(
            nodes_fn,
            params,
            devices=devices,
            cotengra_options={
                "slicing_reconf_opts": {"target_size": args.target_size},
                "max_repeats": args.path_repeats,
                "minimize": "write",
                "parallel": False,
                "progbar": False,
            },
        )
    if args.path_output:
        args.path_output.parent.mkdir(parents=True, exist_ok=True)
        tree = contractor.tree
        tree_data = {
            "inputs": tree.inputs,
            "output": tree.output,
            "size_dict": tree.size_dict,
            "path": tree.get_path(),
            "sliced_inds": tree.sliced_inds,
        }
        with args.path_output.open("wb") as stream:
            pickle.dump(tree_data, stream)
    planning_seconds = perf_counter() - planning_start

    if args.planning_only:
        stats = contractor.tree.contract_stats()
        payload = {
            "schema_version": 1,
            "implementation": "TensorCircuit-NG 1.8.0 DistributedContractor",
            "workload": f"{args.grid_rows}x{args.grid_cols}_{args.grid_cycles}c_global_z",
            "qubits": qubits,
            "dtype": "complex128",
            "devices": args.devices,
            "planning_only": True,
            "planning_seconds": planning_seconds,
            "slices": int(contractor.tree.nslices),
            "tree_flops": float(stats["flops"]),
            "tree_write": float(stats["write"]),
            "tree_max_size_elements": int(stats["size"]),
            "target_size_elements": args.target_size,
            "path_repeats": args.path_repeats,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output and jax.process_index() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        if jax.process_index() == 0:
            print(rendered)
        if args.num_processes > 1:
            multihost_utils.sync_global_devices("planning_done")
            jax.distributed.shutdown()
        return

    initial_params = params
    first_start = perf_counter()
    value, grad = contractor.value_and_grad(initial_params)
    compiled_update = initial_params - args.learning_rate * grad
    _block((value, grad, compiled_update))
    first_step_seconds = perf_counter() - first_start

    warmup_seconds = []
    for _ in range(max(0, args.warmup_steps - 1)):
        started = perf_counter()
        value, grad = contractor.value_and_grad(initial_params)
        compiled_update = initial_params - args.learning_rate * grad
        _block((value, grad, compiled_update))
        warmup_seconds.append(perf_counter() - started)

    params = initial_params
    step_seconds = []
    for _ in range(args.timed_steps):
        started = perf_counter()
        value, grad = contractor.value_and_grad(params)
        params = params - args.learning_rate * grad
        _block((value, grad, params))
        step_seconds.append(perf_counter() - started)

    stats = contractor.tree.contract_stats()
    payload = {
        "schema_version": 1,
        "implementation": "TensorCircuit-NG 1.8.0 DistributedContractor",
        "workload": f"{args.grid_rows}x{args.grid_cols}_{args.grid_cycles}c_global_z",
        "qubits": qubits,
        "dtype": "complex128",
        "parameter_dtype": "float64",
        "gradient": "exact_jax_reverse_mode",
        "optimizer": "sgd",
        "devices": args.devices,
        "global_device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
        "host": os.uname().nodename,
        "planning_seconds": planning_seconds,
        "first_step_seconds": first_step_seconds,
        "warmup_step_seconds": warmup_seconds,
        "timed_step_seconds": step_seconds,
        "steady_step_seconds": sum(step_seconds) / len(step_seconds),
        "value_real": float(jnp.real(value)),
        "gradient_l2": float(jnp.linalg.norm(grad)),
        "slices": int(contractor.tree.nslices),
        "tree_flops": float(stats["flops"]),
        "tree_write": float(stats["write"]),
        "tree_max_size_elements": int(stats["size"]),
        "target_size_elements": args.target_size,
        "path_repeats": args.path_repeats,
        "parameter_and_gradient_layout": "replicated_on_all_devices",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output and jax.process_index() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if jax.process_index() == 0:
        print(rendered)
    if args.num_processes > 1:
        multihost_utils.sync_global_devices("benchmark_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
