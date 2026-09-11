"""Probe one end-to-end VQE training step in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.vqe_triton_runtime import _build_module, _hamiltonian  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "eager"), required=True)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--local-depth", type=int, default=32)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    fused = args.backend == "triton"
    os.environ["FQ_TRITON_SINGLE_QUBIT_LOOP"] = "1" if fused else "0"
    torch.manual_seed(17)
    initial = 0.05 * torch.randn(
        args.local_depth, args.n_wires, 2, device="cuda"
    )
    payload = {
        "benchmark": "flagquantum_vqe_triton_capacity",
        "benchmark_evidence_class": "local",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "distribution_semantics": "single_device_fast_path",
        "device": torch.cuda.get_device_name(),
        "backend": args.backend,
        "n_wires": args.n_wires,
        "local_rx_rz_depth": args.local_depth,
        "status": "pending",
    }
    try:
        module = _build_module(
            initial,
            n_wires=args.n_wires,
            local_depth=args.local_depth,
            hamiltonian=_hamiltonian(args.n_wires),
        )
        optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = module().sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        payload.update(
            status="success",
            step_seconds=time.perf_counter() - started,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
            energy=float(loss.detach()),
            runtime=dict(module.execute().runtime),
        )
    except torch.cuda.OutOfMemoryError as error:
        payload.update(
            status="oom",
            error_type=type(error).__name__,
            error=str(error),
            peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
