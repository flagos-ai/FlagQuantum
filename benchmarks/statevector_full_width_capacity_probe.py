#!/usr/bin/env python3
"""Record sealed single-GPU OOM evidence for the frozen SC27 35-qubit step."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.sc27_metadata import (  # noqa: E402
    canonical_sha256,
    driver_version,
    gpu_identity,
    optimizer_state_bytes,
    source_identity,
    tensor_bytes,
    topology_snapshot,
)
from benchmarks.statevector_training_scaling import (  # noqa: E402
    build_full_width_workload,
)
from flagquantum.runtime.backends.statevector.reverse import (  # noqa: E402
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)

SCHEMA = "flagquantum.statevector.full_width_capacity.v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=35)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--independent-run-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--raw-log-sha256")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    frozen = (args.n_wires, args.layers, args.seed, args.learning_rate)
    if frozen != (35, 1, 20260722, 0.01):
        parser.error("SC27 capacity baseline is frozen at 35q/depth1/seed20260722/lr0.01")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    circuit, parameters, metadata = build_full_width_workload(
        args.n_wires,
        device,
        layers=args.layers,
        seed=args.seed,
        entanglement="linear",
    )
    observable = f"Z({args.n_wires // 2})"
    semantic_workload = {
        "n_wires": args.n_wires,
        "name": metadata["name"],
        "layers": metadata["layers"],
        "gate_count": metadata["gate_count"],
        "active_wire_count": metadata["active_wire_count"],
        "parameter_count": metadata["parameter_count"],
        "entanglement": metadata["entanglement"],
        "observable": observable,
        "dtype": "complex64",
        "seed": args.seed,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
    }
    semantic_sha256 = canonical_sha256(semantic_workload)
    workload = {**semantic_workload, "semantic_sha256": semantic_sha256}
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    before = tuple(float(parameter.detach().cpu()) for parameter in parameters)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    observed_oom = False
    error_message = None
    value = None
    gradients = None
    optimizer_seconds = None
    reverse_summary = None
    try:
        result = execute_torch_distributed_statevector_reverse(
            circuit,
            observable_wire=args.n_wires // 2,
            checkpoint_policy=StatevectorCheckpointPolicy(
                strategy="reversible_adjoint"
            ),
            device=device,
        )
        result.backward()
        reverse_summary = result.summary()
        value = float(result.value.detach().cpu())
        gradients = tuple(
            float(parameter.grad.detach().cpu()) for parameter in parameters
        )
        optimizer_started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started
    except torch.cuda.OutOfMemoryError as error:
        observed_oom = True
        error_message = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        torch.cuda.empty_cache()
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        observed_oom = True
        error_message = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    after = tuple(float(parameter.detach().cpu()) for parameter in parameters)
    completion_correct = bool(
        not observed_oom
        and value is not None
        and gradients is not None
        and math.isfinite(value)
        and all(math.isfinite(item) for item in gradients)
        and after != before
    )
    expectation_met = observed_oom
    payload = {
        "schema": SCHEMA,
        "benchmark": "statevector_full_width_capacity_single_gpu",
        "artifact_class": "measured_sc27_capacity_case",
        "world_size": 1,
        "node_count": 1,
        "rank_placement": [
            {
                "rank": 0, "local_rank": 0, "hostname": platform.node(),
                "device": "cuda:0", **gpu_identity(0),
                "topology": topology_snapshot(),
            }
        ],
        "source_identity": source_identity(
            repo_root=REPO_ROOT,
            workload=semantic_workload,
            container_digest=args.container_digest,
            raw_log_sha256=args.raw_log_sha256,
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "driver_version": driver_version(),
        },
        "protocol": {
            "independent_run_index": args.independent_run_index,
            "warmup": 0,
            "repetitions": 1,
            "gradient_method": "reversible_adjoint",
            "optimizer": "adam",
            "learning_rate": args.learning_rate,
        },
        "workload": workload,
        "expected_outcome": "cuda_oom",
        "observed_outcome": "cuda_oom" if observed_oom else "completion",
        "expectation_met": expectation_met,
        "single_device_oom_observed": observed_oom,
        "oom_type": "cuda_out_of_memory" if observed_oom else None,
        "oom_message": error_message,
        "elapsed_seconds": elapsed,
        "value": value,
        "gradients": gradients,
        "reverse_summary": reverse_summary,
        "optimizer": {
            "name": "adam",
            "executed": not observed_oom,
            "step_seconds": optimizer_seconds,
            "state_bytes": optimizer_state_bytes(optimizer),
            "parameters_changed": after != before,
        },
        "ownership": {
            "primal_state": "single_device_fast_path",
            "adjoint_state": "single_device_fast_path",
            "parameter_gradient": "single_device",
            "optimizer_state": "single_device",
            "full_quantum_state_materialized": False,
            "parameter_bytes": tensor_bytes(parameters),
            "gradient_bytes": tensor_bytes(
                parameter.grad for parameter in parameters if parameter.grad is not None
            ),
            "optimizer_bytes": optimizer_state_bytes(optimizer),
        },
        "correctness": {
            "completion_numerically_valid": completion_correct,
            "value_and_gradients_finite": completion_correct,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "fallback_events": [],
        "release_gate_allowed": False,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not expectation_met:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
