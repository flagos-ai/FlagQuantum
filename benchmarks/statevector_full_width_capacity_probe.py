#!/usr/bin/env python3
"""Record matched single-GPU capacity evidence for a full-width HEA step."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.statevector_training_scaling import (  # noqa: E402
    build_full_width_workload,
)
from flagquantum.runtime.backends.statevector.reverse import (  # noqa: E402
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--expect-oom", action="store_true")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    circuit, parameters, metadata = build_full_width_workload(
        args.n_wires,
        device,
        layers=args.layers,
        seed=args.seed,
        entanglement="linear",
    )
    workload = {
        "n_wires": args.n_wires,
        "layers": args.layers,
        "seed": args.seed,
        "dtype": "complex64",
        "observable_wire": args.n_wires // 2,
        **metadata,
    }
    workload_sha256 = hashlib.sha256(
        json.dumps(workload, sort_keys=True).encode()
    ).hexdigest()
    started = time.perf_counter()
    error_message = None
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
        optimizer = torch.optim.Adam(parameters, lr=0.05)
        optimizer.step()
        torch.cuda.synchronize(device)
        observed_oom = False
    except torch.cuda.OutOfMemoryError as error:
        observed_oom = True
        error_message = str(error)
    expectation_met = observed_oom == args.expect_oom
    payload = {
        "schema_version": "flagquantum.statevector.full_width_capacity.v1",
        "artifact_class": "measured_development_capacity_probe",
        "workload": workload,
        "workload_sha256": workload_sha256,
        "expected_outcome": "cuda_oom" if args.expect_oom else "completion",
        "observed_outcome": "cuda_oom" if observed_oom else "completion",
        "expectation_met": expectation_met,
        "single_device_oom_observed": observed_oom,
        "oom_type": "OutOfMemoryError" if observed_oom else None,
        "oom_message": error_message,
        "elapsed_seconds": time.perf_counter() - started,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
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
