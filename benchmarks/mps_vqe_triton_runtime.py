"""End-to-end MPS VQE: FlagQuantum Triton two-site fusion versus eager."""

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

import flagquantum as fq  # noqa: E402


def _build_module(
    initial: torch.Tensor,
    *,
    n_wires: int,
    layers: int,
    batch_size: int,
    max_bond: int,
    cutoff: float,
) -> fq.Module:
    def ansatz(parameters: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(n_wires, bsz=batch_size, device=parameters.device)
        for layer in range(layers):
            for wire in range(n_wires):
                circuit.ry(wire, theta=parameters[:, layer, wire, 0])
                circuit.rz(wire, theta=parameters[:, layer, wire, 1])
            parity = layer % 2
            for wire in range(parity, n_wires - 1, 2):
                circuit.cx(wire, wire + 1)
        return circuit

    return fq.Module(
        ansatz,
        tuple(initial.shape),
        init=initial.detach().clone(),
        device=initial.device,
        hamiltonian=fq.zz_chain_hamiltonian(
            n_wires, coupling=-1.0, field=0.1
        ),
        policy=fq.RuntimePolicy(
            backend="pytorch",
            mode="mps",
            observable="hamiltonian",
            mps_max_bond=max_bond,
            mps_cutoff=cutoff,
        ),
    )


def _step(module: fq.Module, optimizer: torch.optim.Optimizer) -> tuple:
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    result = module.execute()
    loss = result.value.sum()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return (
        time.perf_counter() - started,
        float(loss.detach()),
        int(torch.cuda.max_memory_allocated()),
        dict(result.runtime),
    )


def _train(
    module: fq.Module,
    *,
    backend: str,
    iterations: int,
    learning_rate: float,
) -> dict:
    os.environ["FQ_TRITON_MPS_TWO_SITE"] = "1" if backend == "triton" else "0"
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    cumulative = []
    energies = []
    peak_memory = []
    max_bonds = []
    truncation_errors = []
    triton_regions = []
    eager_regions = []
    svd_gradient_methods = []
    fixed_rank_qr_regions = []
    elapsed = 0.0
    for iteration in range(iterations):
        duration, energy, peak, runtime = _step(module, optimizer)
        elapsed += duration
        cumulative.append(elapsed)
        energies.append(energy)
        peak_memory.append(peak)
        max_bonds.append(int(runtime["max_bond"]))
        truncation_errors.append(float(runtime["truncation_error"]))
        triton_regions.append(int(runtime["triton_mps_two_site_regions"]))
        eager_regions.append(int(runtime["eager_mps_two_site_regions"]))
        svd_gradient_methods.append(str(runtime["svd_gradient_method"]))
        fixed_rank_qr_regions.append(int(runtime["fixed_rank_qr_regions"]))
        print(
            f"heartbeat backend={backend} iteration={iteration + 1}/{iterations} "
            f"seconds={duration:.3f} peak_gib={peak / 2**30:.3f} "
            f"max_bond={max_bonds[-1]} triton_regions={triton_regions[-1]} "
            f"truncation={truncation_errors[-1]:.3e}",
            flush=True,
        )
    return {
        "cumulative_seconds": cumulative,
        "energies": energies,
        "peak_memory_bytes": peak_memory,
        "max_bonds": max_bonds,
        "truncation_errors": truncation_errors,
        "triton_two_site_regions": triton_regions,
        "eager_two_site_regions": eager_regions,
        "svd_gradient_methods": svd_gradient_methods,
        "fixed_rank_qr_regions": fixed_rank_qr_regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=128)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=1e-6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--backend", choices=("triton", "eager", "both"), default="both")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(23)
    initial = 0.05 * torch.randn(
        args.batch_size, args.layers, args.n_wires, 2, device="cuda"
    )
    selected = ("triton", "eager") if args.backend == "both" else (args.backend,)
    results = {}
    for backend in selected:
        module = _build_module(
            initial,
            n_wires=args.n_wires,
            layers=args.layers,
            batch_size=args.batch_size,
            max_bond=args.max_bond,
            cutoff=args.cutoff,
        )
        results[backend] = _train(
            module,
            backend=backend,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
    payload = {
        "benchmark": "flagquantum_mps_vqe_triton_runtime",
        "benchmark_evidence_class": "local",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "distribution_semantics": "single_device_fast_path",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "task": "end_to_end_mps_vqe_training",
        "n_wires": args.n_wires,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "max_bond": args.max_bond,
        "cutoff": args.cutoff,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "hamiltonian": "-sum ZZ + 0.1 sum Z",
        "ansatz": "independent batched RY/RZ plus alternating nearest-neighbor CX brickwork",
        **results,
    }
    if args.backend == "both":
        payload["speedup_eager_over_triton"] = (
            results["eager"]["cumulative_seconds"][-1]
            / results["triton"]["cumulative_seconds"][-1]
        )
        payload["final_energy_abs_delta"] = abs(
            results["eager"]["energies"][-1]
            - results["triton"]["energies"][-1]
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
