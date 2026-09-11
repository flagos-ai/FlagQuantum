"""Measure only the JAX leg of the end-to-end VQE comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.vqe_triton_runtime import (  # noqa: E402
    _build_module,
    _hamiltonian,
    _train,
    _warmup,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=20)
    parser.add_argument("--local-depth", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(17)
    initial = 0.05 * torch.randn(
        args.local_depth, args.n_wires, 2, device="cuda"
    )
    module = _build_module(
        initial,
        n_wires=args.n_wires,
        local_depth=args.local_depth,
        hamiltonian=_hamiltonian(args.n_wires),
        backend="jax",
    )
    startup_seconds = _warmup(module)
    result = _train(
        module,
        iterations=args.iterations,
        lr=args.lr,
        label="jax",
    )
    payload = {
        "benchmark": "flagquantum_vqe_jax_runtime_leg",
        "benchmark_evidence_class": "local",
        "distribution_semantics": "single_device_fast_path",
        "device": torch.cuda.get_device_name(),
        "physical_gpu": args.physical_gpu,
        "torch_version": torch.__version__,
        "n_wires": args.n_wires,
        "local_rx_rz_depth": args.local_depth,
        "parameters": int(initial.numel()),
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "timing_boundary": {
            "training_curve": "post_warmup_steady_state",
            "compile_and_first_execution_excluded": True,
            "startup_seconds": startup_seconds,
            "startup_measurement_note": (
                "One first forward+backward execution; may use persistent compiler caches."
            ),
        },
        "jax": result,
        "jax_runtime": dict(module.execute().runtime),
        "limitations": [
            "single-GPU local engineering evidence",
            "JAX GPU memory is not observable by torch.cuda.max_memory_allocated",
        ],
    }
    encoded = json.dumps(payload, indent=2)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
