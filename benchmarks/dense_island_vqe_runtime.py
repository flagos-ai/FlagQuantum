"""Batch-one Dense-Island versus gatewise MPS VQE-step benchmark."""

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
from flagquantum.simulation.dense_island import (  # noqa: E402
    DenseIslandPlan,
    DenseIslandState,
)


def _ry(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(
        torch.complex64
    )


def _rz(angle: torch.Tensor) -> torch.Tensor:
    phase = angle / 2
    zero = torch.exp(-1j * phase)
    one = torch.exp(1j * phase)
    return torch.diag(torch.stack((zero, one))).to(torch.complex64)


CX = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=torch.complex64,
)


def _dense_island_loss(
    parameters: torch.Tensor, *, island_width: int, max_bond: int
) -> tuple[torch.Tensor, dict[str, object]]:
    layers, n_wires, _ = parameters.shape
    plan = DenseIslandPlan.equal_width(
        n_wires, island_width=island_width, max_bond=max_bond
    )
    state = DenseIslandState.zero(plan, device=parameters.device)
    cx = CX.to(parameters.device)
    for layer in range(layers):
        for wire in range(n_wires):
            state.apply_local(_ry(parameters[layer, wire, 0]), (wire,))
            state.apply_local(_rz(parameters[layer, wire, 1]), (wire,))
        for wire in range(layer % 2, n_wires - 1, 2):
            state.apply_local(cx, (wire, wire + 1))
    value = state.expectation_z_zz_chain(
        z_weights=[0.1] * n_wires, zz_weights=[-1.0] * (n_wires - 1)
    ).sum()
    return value, state.summary()


def _mps_loss(parameters: torch.Tensor, *, max_bond: int) -> tuple[torch.Tensor, dict]:
    layers, n_wires, _ = parameters.shape
    circuit = fq.Circuit(n_wires, device=parameters.device)
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.ry(wire, theta=parameters[layer, wire, 0])
            circuit.rz(wire, theta=parameters[layer, wire, 1])
        for wire in range(layer % 2, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    state = fq.run_mps(circuit, max_bond=max_bond, cutoff=0.0)
    value = fq.zz_chain_hamiltonian(
        n_wires, coupling=-1.0, field=0.1
    ).expectation(state).sum()
    return value, state.summary()


def _run(parameters: torch.Tensor, backend: str, args: argparse.Namespace) -> dict:
    optimizer = torch.optim.Adam([parameters], lr=0.02)
    durations, energies, peaks, summaries = [], [], [], []
    for iteration in range(args.iterations):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if backend == "dense_island":
            loss, summary = _dense_island_loss(
                parameters,
                island_width=args.island_width,
                max_bond=args.max_bond,
            )
        else:
            loss, summary = _mps_loss(parameters, max_bond=args.max_bond)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        durations.append(duration)
        energies.append(float(loss.detach()))
        peaks.append(int(torch.cuda.max_memory_allocated()))
        summaries.append(summary)
        print(
            f"heartbeat backend={backend} iteration={iteration + 1}/{args.iterations} "
            f"seconds={duration:.3f} energy={energies[-1]:.7f} "
            f"peak_gib={peaks[-1] / 2**30:.3f}",
            flush=True,
        )
    return {
        "step_seconds": durations,
        "energies": energies,
        "peak_memory_bytes": peaks,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--island-width", type=int, default=4)
    parser.add_argument("--max-bond", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    os.environ["FQ_TRITON_MPS_TWO_SITE"] = "0"
    torch.manual_seed(31)
    initial = 0.05 * torch.randn(
        args.layers, args.n_wires, 2, device="cuda"
    )
    dense_parameters = initial.detach().clone().requires_grad_(True)
    mps_parameters = initial.detach().clone().requires_grad_(True)
    dense = _run(dense_parameters, "dense_island", args)
    mps = _run(mps_parameters, "mps", args)
    payload = {
        "benchmark": "dense_island_vqe_runtime",
        "evidence_class": "local_non_release",
        "device": torch.cuda.get_device_name(),
        "batch_size": 1,
        "n_wires": args.n_wires,
        "layers": args.layers,
        "island_width": args.island_width,
        "max_bond": args.max_bond,
        "iterations": args.iterations,
        "observable": "-sum(ZZ)+0.1sum(Z), shared environments",
        "dense_island": dense,
        "mps": mps,
        "final_energy_abs_delta": abs(dense["energies"][-1] - mps["energies"][-1]),
        "steady_speedup_mps_over_dense_island": (
            sum(mps["step_seconds"][1:]) / sum(dense["step_seconds"][1:])
            if args.iterations > 1
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
