"""Profile FlagQuantum JAX pmap/shard_map statevector backward.

This benchmark is for a single logical statevector sharded across JAX devices.
It is not the older rank-local replicated JAX quantum-kernel benchmark.

Reproduce on local CPU with simulated JAX devices:

  PowerShell:
    $env:XLA_FLAGS="--xla_force_host_platform_device_count=8"
    python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 8 --n-wires 8 --layers 2 --iters 10 --warmup 2 --json-output benchmarks/results/statevector_pmap_all_to_all_cpu8.json

  Bash:
    XLA_FLAGS=--xla_force_host_platform_device_count=8 python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 8 --n-wires 8 --layers 2 --iters 10 --warmup 2 --json-output benchmarks/results/statevector_pmap_all_to_all_cpu8.json

Reproduce on one 4-GPU node:

    python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 4 --n-wires 8 --layers 2 --iters 50 --warmup 10 --json-output benchmarks/results/statevector_pmap_all_to_all_4gpu.json

Reproduce on one 8-GPU node:

    python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 8 --n-wires 10 --layers 2 --iters 50 --warmup 10 --json-output benchmarks/results/statevector_pmap_all_to_all_8gpu.json

Reproduce shard_map mesh backward on one 4-GPU node or 4 simulated CPU devices:

    python benchmarks/statevector_pmap_all_to_all_profile.py --backward-backend shard_map \
        --transport-pattern pair_exchange --world-size 4 --n-wires 8 --layers 2 \
        --iters 20 --warmup 5 \
        --json-output benchmarks/results/statevector_shard_map_pair_exchange_4gpu.json

Reproduce on two 8-GPU nodes with one Python process per node:

    # node 0
    python benchmarks/statevector_pmap_all_to_all_profile.py --jax-distributed \
        --jax-coordinator-address <node0-ip>:12355 --jax-num-processes 2 --jax-process-id 0 \
        --jax-local-device-ids 0,1,2,3,4,5,6,7 \
        --world-size 16 --n-wires 12 --layers 2 --iters 50 --warmup 10 \
        --json-output benchmarks/results/statevector_pmap_all_to_all_2node16gpu.json

    # node 1
    python benchmarks/statevector_pmap_all_to_all_profile.py --jax-distributed \
        --jax-coordinator-address <node0-ip>:12355 --jax-num-processes 2 --jax-process-id 1 \
        --jax-local-device-ids 0,1,2,3,4,5,6,7 \
        --world-size 16 --n-wires 12 --layers 2 --iters 50 --warmup 10

The pmap executor can use visible local devices in one process, or global
devices after JAX multi-process initialization. The shard_map executor currently
supports local/pair-exchange transport on a single-process mesh and fails
closed for multi-sharded-wire all-to-all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch


REPRO_COMMANDS = """
CPU 8-device local pmap:
  PowerShell:
    $env:XLA_FLAGS="--xla_force_host_platform_device_count=8"
    python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 8 --n-wires 8 --layers 2 --iters 10 --warmup 2 --json-output benchmarks/results/statevector_pmap_all_to_all_cpu8.json

  Bash:
    XLA_FLAGS=--xla_force_host_platform_device_count=8 python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 8 --n-wires 8 --layers 2 --iters 10 --warmup 2 --json-output benchmarks/results/statevector_pmap_all_to_all_cpu8.json

Single-node 4-GPU pmap:
  python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 4 --n-wires 8 --layers 2 --iters 50 --warmup 10 --json-output benchmarks/results/statevector_pmap_all_to_all_4gpu.json

Single-node 8-GPU pmap:
  python benchmarks/statevector_pmap_all_to_all_profile.py --world-size 8 --n-wires 10 --layers 2 --iters 50 --warmup 10 --json-output benchmarks/results/statevector_pmap_all_to_all_8gpu.json

Single-node 4-GPU shard_map pair-exchange:
  python benchmarks/statevector_pmap_all_to_all_profile.py --backward-backend shard_map --transport-pattern pair_exchange --world-size 4 --n-wires 8 --layers 2 --iters 20 --warmup 5 --json-output benchmarks/results/statevector_shard_map_pair_exchange_4gpu.json

Two-node 16-GPU pmap, one Python process per node:
  node 0:
    python benchmarks/statevector_pmap_all_to_all_profile.py --jax-distributed --jax-coordinator-address <node0-ip>:12355 --jax-num-processes 2 --jax-process-id 0 --jax-local-device-ids 0,1,2,3,4,5,6,7 --world-size 16 --n-wires 12 --layers 2 --iters 50 --warmup 10 --json-output benchmarks/results/statevector_pmap_all_to_all_2node16gpu.json
  node 1:
    python benchmarks/statevector_pmap_all_to_all_profile.py --jax-distributed --jax-coordinator-address <node0-ip>:12355 --jax-num-processes 2 --jax-process-id 1 --jax-local-device-ids 0,1,2,3,4,5,6,7 --world-size 16 --n-wires 12 --layers 2 --iters 50 --warmup 10
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0


def _rank_bits(world_size: int) -> int:
    if not _is_power_of_two(int(world_size)):
        raise ValueError("--world-size must be a power of two for qubit-address pmap all-to-all.")
    return int(math.log2(int(world_size)))


def _sharded_wires(n_wires: int, world_size: int) -> tuple[int, ...]:
    bits = _rank_bits(int(world_size))
    if int(n_wires) <= bits:
        raise ValueError("n_wires must be larger than log2(world_size) so each shard has local amplitudes.")
    return tuple(range(int(n_wires) - bits, int(n_wires)))


def _parameter_count(n_wires: int, world_size: int, layers: int, *, transport_pattern: str) -> int:
    sharded = _sharded_wires(int(n_wires), int(world_size))
    per_layer = 2 * int(n_wires)
    if str(transport_pattern) == "all_to_all":
        per_layer += max(0, len(sharded) - 1)
        per_layer += 1
        if len(sharded) >= 3:
            per_layer += 1
    return int(layers) * per_layer


def _build_sharded_transport_circuit(
    fq: Any,
    params: Any,
    *,
    n_wires: int,
    world_size: int,
    layers: int,
    transport_pattern: str,
) -> Any:
    sharded = _sharded_wires(int(n_wires), int(world_size))
    circuit = fq.Circuit(int(n_wires))
    cursor = 0
    for _layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.ry(wire, theta=params[cursor])
            cursor += 1
            circuit.rz(wire, theta=params[cursor])
            cursor += 1
        if str(transport_pattern) == "all_to_all":
            for left, right in zip(sharded[:-1], sharded[1:]):
                circuit.rxx(left, right, theta=params[cursor])
                cursor += 1
                circuit.cx(left, right)
            circuit.cx(sharded[0], sharded[-1])
            circuit.ry(sharded[-1], theta=params[cursor])
            cursor += 1
            if len(sharded) >= 3:
                circuit.ryy(sharded[0], sharded[-1], theta=params[cursor])
                cursor += 1
        elif str(transport_pattern) == "pair_exchange":
            if int(n_wires) >= 2:
                circuit.cx(0, 1)
        else:
            raise ValueError("transport_pattern must be 'all_to_all' or 'pair_exchange'.")
    return circuit


def _time_call(fn: Any, *, warmup: int, iters: int) -> tuple[Any, float, float]:
    last = None
    first_seconds = None
    for index in range(max(0, int(warmup))):
        start = time.perf_counter()
        last = fn()
        elapsed = time.perf_counter() - start
        if index == 0:
            first_seconds = elapsed
    times = []
    for _ in range(max(1, int(iters))):
        start = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - start)
    assert last is not None
    return last, float(sum(times) / len(times)), float(first_seconds if first_seconds is not None else times[0])


def _block_result(result: Any) -> Any:
    import jax

    jax.block_until_ready(result.value)
    jax.block_until_ready(result.gradient)
    return result


def _reference_value_and_grad(
    fq: Any,
    params: torch.Tensor,
    *,
    n_wires: int,
    world_size: int,
    layers: int,
    transport_pattern: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta: Any) -> Any:
        return _build_sharded_transport_circuit(
            fq,
            theta,
            n_wires=n_wires,
            world_size=world_size,
            layers=layers,
            transport_pattern=transport_pattern,
        )

    kernel = fq.compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=int(n_wires),
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    value = kernel(reference_params)
    value.backward()
    assert reference_params.grad is not None
    return value.detach(), reference_params.grad.detach()


def _environment(jax: Any) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "jax": getattr(jax, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }


def _write_json_output(path: str, text: str) -> None:
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--complex-bytes", type=int, choices=(8, 16), default=8)
    parser.add_argument("--distributed-profile", choices=("production", "development"), default="production")
    parser.add_argument("--backward-backend", choices=("pmap", "shard_map"), default="pmap")
    parser.add_argument("--transport-pattern", choices=("all_to_all", "pair_exchange"), default="all_to_all")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--max-reference-wires", type=int, default=14)
    parser.add_argument("--loss-atol", type=float, default=1e-5)
    parser.add_argument("--grad-atol", type=float, default=1e-5)
    parser.add_argument("--require-scalability", action="store_true")
    parser.add_argument("--jax-distributed", action="store_true")
    parser.add_argument("--jax-coordinator-address", default="")
    parser.add_argument("--jax-num-processes", type=int, default=None)
    parser.add_argument("--jax-process-id", type=int, default=None)
    parser.add_argument("--jax-local-device-ids", default="")
    parser.add_argument("--jax-cluster-detection-method", default="")
    parser.add_argument("--jax-initialization-timeout", type=int, default=300)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    import flagquantum as fq

    should_initialize_jax = (
        bool(args.jax_distributed)
        or args.jax_num_processes is not None
        or bool(args.jax_coordinator_address)
        or bool(os.environ.get("FQ_JAX_NUM_PROCESSES"))
        or bool(os.environ.get("JAX_NUM_PROCESSES"))
    )
    jax_init_summary = None
    if should_initialize_jax:
        jax_init_summary = fq.initialize_jax_distributed(
            coordinator_address=args.jax_coordinator_address or None,
            num_processes=args.jax_num_processes,
            process_id=args.jax_process_id,
            local_device_ids=args.jax_local_device_ids or None,
            cluster_detection_method=args.jax_cluster_detection_method or None,
            initialization_timeout=int(args.jax_initialization_timeout),
        )

    import jax

    local_devices = tuple(jax.local_devices())
    global_devices = tuple(jax.devices())
    if len(global_devices) < int(args.world_size):
        raise RuntimeError(
            f"JAX pmap all-to-all profile requires at least {int(args.world_size)} global JAX devices, "
            f"but only {len(global_devices)} are visible. For CPU smoke, set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=<world_size> before launching Python."
        )
    if not local_devices:
        raise RuntimeError("Every participating JAX process must own at least one local device.")
    sharded = _sharded_wires(int(args.n_wires), int(args.world_size))
    param_count = _parameter_count(
        int(args.n_wires),
        int(args.world_size),
        int(args.layers),
        transport_pattern=str(args.transport_pattern),
    )
    dtype = torch.float64 if int(args.complex_bytes) == 16 else torch.float32
    params = torch.linspace(-0.37, 0.41, steps=param_count, dtype=dtype)

    def build(theta: Any) -> Any:
        return _build_sharded_transport_circuit(
            fq,
            theta,
            n_wires=int(args.n_wires),
            world_size=int(args.world_size),
            layers=int(args.layers),
            transport_pattern=str(args.transport_pattern),
        )

    def run_profile() -> Any:
        return _block_result(
            fq.jax_sharded_statevector_parameter_value_and_grad(
                build,
                params,
                n_wires=int(args.n_wires),
                world_size=int(args.world_size),
                observable="z_sum",
                backward_backend=str(args.backward_backend),
                distributed_profile=str(args.distributed_profile),
                jax_backend=str(args.backward_backend),
                complex_bytes=int(args.complex_bytes),
                jit=False,
            )
        )

    result, avg_seconds, first_seconds = _time_call(run_profile, warmup=int(args.warmup), iters=int(args.iters))
    result_summary = result.summary()
    comparison: dict[str, Any]
    reference_payload: dict[str, Any] | None = None
    if args.skip_reference or int(args.n_wires) > int(args.max_reference_wires):
        comparison = {
            "status": "reference_skipped",
            "reason": "reference skipped by flag or n_wires exceeds max_reference_wires",
        }
    else:
        ref_value, ref_grad = _reference_value_and_grad(
            fq,
            params,
            n_wires=int(args.n_wires),
            world_size=int(args.world_size),
            layers=int(args.layers),
            transport_pattern=str(args.transport_pattern),
        )
        value = result.torch_value(like=params)
        grad = result.torch_gradient(like=params)
        loss_abs_error = float(torch.max(torch.abs(value - ref_value)).item())
        grad_max_abs_error = float(torch.max(torch.abs(grad - ref_grad)).item())
        status = (
            "ok"
            if loss_abs_error <= float(args.loss_atol) and grad_max_abs_error <= float(args.grad_atol)
            else "precision_mismatch"
        )
        comparison = {
            "status": status,
            "loss_abs_error": loss_abs_error,
            "grad_max_abs_error": grad_max_abs_error,
            "loss_atol": float(args.loss_atol),
            "grad_atol": float(args.grad_atol),
        }
        reference_payload = {
            "backend": "flagquantum_jax_rank_local_reference",
            "loss": float(ref_value.detach().cpu().item()),
            "gradient_norm": float(torch.linalg.vector_norm(ref_grad.detach().cpu()).item()),
        }

    payload = {
        "benchmark": "statevector_pmap_all_to_all_profile",
        "run_commands": REPRO_COMMANDS,
        "configuration": {
            "world_size": int(args.world_size),
            "n_wires": int(args.n_wires),
            "layers": int(args.layers),
            "sharded_wires": sharded,
            "parameter_count": int(param_count),
            "complex_bytes": int(args.complex_bytes),
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "distributed_profile": str(args.distributed_profile),
            "backward_backend": str(args.backward_backend),
            "transport_pattern": str(args.transport_pattern),
            "jax_distributed": bool(should_initialize_jax),
        },
        "environment": _environment(jax),
        "jax_distributed_initialization": jax_init_summary,
        "jax_process_count": int(jax.process_count()),
        "jax_process_index": int(jax.process_index()),
        "jax_local_device_count": len(local_devices),
        "jax_global_device_count": len(global_devices),
        "claim_evidence_type": result_summary.get("claim_evidence_type", "development_smoke"),
        "release_gate_allowed": bool(result_summary.get("release_gate_allowed", False)),
        "distribution_semantics": result_summary["distribution_semantics"],
        "scalability_claim_allowed": bool(result_summary["scalability_claim_allowed"]),
        "state_partition": result_summary["state_partition"],
        "sharded_wires": result_summary["sharded_wires"],
        "rank_coordinates": result_summary["rank_coordinates"],
        "world_size": result_summary["world_size"],
        "local_world_size": result_summary["local_world_size"],
        "node_count": result_summary["node_count"],
        "rank_shards": result_summary["rank_shards"],
        "local_memory_bytes_by_rank": result_summary["local_memory_bytes_by_rank"],
        "communication_bytes": result_summary["estimated_transfer_bytes"],
        "estimated_transfer_bytes": result_summary["estimated_transfer_bytes"],
        "intra_node_communication_bytes": result_summary["intra_node_communication_bytes"],
        "inter_node_communication_bytes": result_summary["inter_node_communication_bytes"],
        "peak_intermediate_bytes": max(result_summary["local_memory_bytes_by_rank"], default=0),
        "single_gpu_expected_oom": False,
        "capacity_baseline_device": "not_measured",
        "capacity_failure_reason": "not_measured",
        "optimizer_update_semantics": "not_measured",
        "training_step_count": 0,
        "gradient_distribution_semantics": result_summary.get("gradient_distribution_semantics", "incomplete"),
        "flagquantum_sharded_statevector": {
            "avg_seconds": avg_seconds,
            "first_loss_grad_seconds": first_seconds,
            "loss": float(result.torch_value(like=params).detach().cpu().item()),
            "gradient_norm": float(torch.linalg.vector_norm(result.torch_gradient(like=params).detach().cpu()).item()),
            "summary": result_summary,
        },
        "reference": reference_payload,
        "comparison": comparison,
    }
    payload["conclusion"] = {
        "headline": (
            f"FlagQuantum JAX {args.backward_backend} statevector backward is precision-aligned and sharded across devices."
            if comparison["status"] in {"ok", "reference_skipped"} and payload["release_gate_allowed"]
            else f"FlagQuantum JAX {args.backward_backend} statevector backward is not yet promotable for this run."
        ),
        "recommended_claim": (
            f"One logical {int(args.n_wires)}-qubit statevector was sharded across {int(args.world_size)} JAX "
            f"{args.backward_backend} device(s) with {args.transport_pattern} transport; steady loss+gradient averaged "
            f"{avg_seconds:.6g}s, estimated transfer bytes={result_summary['estimated_transfer_bytes']}."
        ),
    }
    if str(args.backward_backend) == "pmap":
        payload["flagquantum_pmap"] = payload["flagquantum_sharded_statevector"]
    payload = fq.attach_distributed_scalability_audit(payload)
    if args.require_scalability:
        fq.require_distributed_scalability(payload)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if int(jax.process_index()) == 0:
        print(text)
    if args.json_output and int(jax.process_index()) == 0:
        _write_json_output(args.json_output, text)


if __name__ == "__main__":
    main()
