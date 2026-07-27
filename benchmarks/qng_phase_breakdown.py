"""Profile exact block-QNG phases before selecting a Triton fusion boundary."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

import flagquantum.algorithms as fqa
from flagquantum.algorithms.optimization import OptimizationStage
from flagquantum.ops import get_global_precision, set_global_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    device = torch.device(args.device)
    previous = get_global_precision()
    set_global_precision(torch.complex128)
    try:
        count = fqa.heisenberg_hva_parameter_count(args.n_wires, args.depth)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        initial = 0.02 * torch.randn(
            count, generator=generator, device=device, dtype=torch.float64
        )
        hamiltonian = fqa.heisenberg_chain_hamiltonian(args.n_wires)
        result = fqa.run_hybrid_vqe(
            lambda parameters: fqa.heisenberg_hva(
                args.n_wires,
                args.depth,
                parameters,
                parameterization="bond_resolved_phase",
                initial_state="dimer_singlet",
            ),
            initial,
            hamiltonian,
            stages=(
                OptimizationStage(
                    "quantum",
                    "qng",
                    args.steps,
                    lr=0.03,
                    damping=1e-3,
                    block_size=count // args.depth,
                ),
            ),
        )
        exact = float(hamiltonian.ground_energy())
        records = []
        cumulative = 0.0
        cumulative_evaluations = 0
        for record in result.records:
            cumulative += record.wall_time_seconds
            cumulative_evaluations += record.evaluations
            row = asdict(record)
            row.update(
                {
                    "optimizer_step": record.step + 1,
                    "cumulative_wall_time_seconds": cumulative,
                    "cumulative_circuit_evaluations": cumulative_evaluations,
                    "relative_error": (record.loss - exact) / abs(exact),
                }
            )
            records.append(row)
        payload = {
            "schema": "flagquantum.qng_phase_breakdown.v1",
            "execution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "backend": "pytorch_exact_statevector",
            "jacobian_mode": "vectorized_reverse",
            "triton_qng_enabled": False,
            "cold_compile_included": False,
            "n_wires": args.n_wires,
            "depth": args.depth,
            "parameter_count": count,
            "precision": "complex128",
            "device": str(device),
            "device_name": platform.processor() if device.type == "cpu" else torch.cuda.get_device_name(device),
            "seed": args.seed,
            "exact_ground_energy": exact,
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "total_seconds": cumulative,
            "mean_metric_fraction": sum(
                r["quantum_metric_seconds"] / r["wall_time_seconds"] for r in records
            ) / len(records),
        }, indent=2))
    finally:
        set_global_precision(previous)


if __name__ == "__main__":
    main()
