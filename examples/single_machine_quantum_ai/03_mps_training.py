"""Local MPS quantum AI training example.

This demonstrates a larger local circuit route. The training objective pushes a
hardware-efficient ansatz toward a low-energy nearest-neighbor ZZ Hamiltonian.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
from common import (  # noqa: E402
    configure_jax_compilation_cache,
    exact_ground_energy,
    jax_available,
    print_kv,
    print_section,
    print_table,
    print_training_summary,
    speedup,
    sync_if_needed,
    time_value_and_grad,
)


def build_ansatz(theta: torch.Tensor, *, n_wires: int, layers: int, device: str) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    cursor = 0
    for _ in range(layers):
        for wire in range(n_wires):
            circuit.ry(wire, theta=theta[cursor])
            cursor += 1
            circuit.rz(wire, theta=theta[cursor])
            cursor += 1
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def make_jax_mps_kernel(
    parameters: torch.Tensor,
    *,
    n_wires: int,
    layers: int,
    device: str,
    hamiltonian: fq.Hamiltonian,
    max_bond: int,
) -> fq.JAXQuantumKernel:
    return fq.compile_quantum_kernel(
        lambda theta: build_ansatz(theta, n_wires=n_wires, layers=layers, device=device),
        parameters.detach(),
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=n_wires,
        hamiltonian=hamiltonian,
        max_bond=max_bond,
        jit=True,
    )


def value_and_grad_once(loss_fn, parameters: torch.Tensor, *, device: str) -> tuple[float, float, float]:
    params = parameters.detach().clone().requires_grad_(True)
    sync_if_needed(device)
    start = time.perf_counter()
    loss = loss_fn(params)
    grad = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)[0]
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return elapsed, float(loss.detach()), float(grad.detach().norm())


def parse_wire_list(text: str) -> list[int]:
    wires = []
    for item in text.split(","):
        stripped = item.strip()
        if stripped:
            wires.append(int(stripped))
    if not wires:
        raise ValueError("--scale-report must contain at least one wire count.")
    return wires


def run_scale_report(args: argparse.Namespace) -> None:
    cache_report = configure_jax_compilation_cache(args.jax_cache_dir)
    has_jax, jax_error = jax_available()
    if not has_jax:
        raise SystemExit(f"JAX backend requested for scale report but unavailable: {jax_error}.")

    rows = []
    for n_wires in parse_wire_list(args.scale_report):
        torch.manual_seed(19)
        n_params = fq.hardware_efficient_parameter_count(n_wires, args.layers)
        parameters = (0.15 * torch.randn(n_params, device=args.device)).requires_grad_(True)
        hamiltonian = fq.zz_chain_hamiltonian(n_wires, coupling=-1.0, field=0.1)

        compile_start = time.perf_counter()
        kernel = make_jax_mps_kernel(
            parameters,
            n_wires=n_wires,
            layers=args.layers,
            device=args.device,
            hamiltonian=hamiltonian,
            max_bond=args.max_bond,
        )
        compile_setup_seconds = time.perf_counter() - compile_start

        def loss_fn(theta: torch.Tensor) -> torch.Tensor:
            return kernel(theta).sum()

        first_seconds, first_loss, first_grad_norm = value_and_grad_once(loss_fn, parameters, device=args.device)
        steady = time_value_and_grad(
            loss_fn,
            parameters,
            iters=args.scale_iters,
            warmup=args.scale_warmup,
            device=args.device,
        )
        rows.append(
            {
                "n_wires": n_wires,
                "params": n_params,
                "max_bond": args.max_bond,
                "compile_setup_s": compile_setup_seconds,
                "first_loss_grad_s": first_seconds,
                "steady_loss_grad_s": steady["avg_seconds"],
                "loss": steady["loss"],
                "grad_norm": steady["grad_norm"],
                "fastpath": kernel.summary().get("hamiltonian_fastpath"),
            }
        )

    print_section("MPS JAX Scale Report")
    print_kv(
        {
            "layers": args.layers,
            "max_bond": args.max_bond,
            "device": args.device,
            "scale_warmup": args.scale_warmup,
            "scale_iters": args.scale_iters,
            "jax_cache_enabled": cache_report.get("enabled"),
            "jax_cache_dir": cache_report.get("cache_dir"),
            "note": "first_loss_grad_s includes JAX compilation; steady_loss_grad_s is after warmup.",
        }
    )
    print_table(
        rows,
        columns=[
            "n_wires",
            "params",
            "max_bond",
            "compile_setup_s",
            "first_loss_grad_s",
            "steady_loss_grad_s",
            "loss",
            "grad_norm",
            "fastpath",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--backend", choices=("jax", "torch"), default="jax")
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--reference", choices=("auto", "small_exact", "none"), default="auto")
    parser.add_argument("--small-exact-max-wires", type=int, default=12)
    parser.add_argument("--scale-report", default=None, help="Comma-separated wire counts, for example 60,120,300,600.")
    parser.add_argument("--scale-iters", type=int, default=3)
    parser.add_argument("--scale-warmup", type=int, default=1)
    parser.add_argument(
        "--jax-cache-dir",
        default=None,
        help="Optional persistent JAX compilation cache directory, for example .fq_jax_cache.",
    )
    args = parser.parse_args()

    if args.scale_report:
        run_scale_report(args)
        return

    cache_report = configure_jax_compilation_cache(args.jax_cache_dir)

    torch.manual_seed(19)
    n_params = fq.hardware_efficient_parameter_count(args.n_qubits, args.layers)
    parameters = (0.15 * torch.randn(n_params, device=args.device)).requires_grad_(True)
    optimizer = torch.optim.Adam([parameters], lr=args.lr)
    hamiltonian = fq.zz_chain_hamiltonian(args.n_qubits, coupling=-1.0, field=0.1)

    def build(theta: torch.Tensor) -> fq.Circuit:
        return build_ansatz(theta, n_wires=args.n_qubits, layers=args.layers, device=args.device)

    def native_mps_loss(theta: torch.Tensor) -> torch.Tensor:
        mps = fq.run_mps(build(theta), max_bond=args.max_bond)
        return hamiltonian.expectation(mps).sum()

    has_jax, jax_error = jax_available()
    if args.backend == "jax" and not has_jax:
        raise SystemExit(f"JAX backend requested but unavailable: {jax_error}. Use --backend torch to run the native path.")
    jax_kernel = None
    if has_jax:
        jax_kernel = make_jax_mps_kernel(
            parameters,
            n_wires=args.n_qubits,
            layers=args.layers,
            device=args.device,
            hamiltonian=hamiltonian,
            max_bond=args.max_bond,
        )

    def training_loss(theta: torch.Tensor) -> torch.Tensor:
        if args.backend == "jax":
            return jax_kernel(theta).sum()
        return native_mps_loss(theta)

    exact_energy = (
        exact_ground_energy(hamiltonian, args.n_qubits)
        if args.reference != "none" and args.n_qubits <= args.small_exact_max_wires
        else None
    )
    initial = float(training_loss(parameters).detach())
    for step in range(args.steps):
        optimizer.zero_grad()
        loss = training_loss(parameters)
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (step + 1) % max(1, args.steps // 5) == 0:
            print({"step": step + 1, "energy": float(loss.detach())})

    trained_mps = fq.run_mps(build(parameters.detach()), max_bond=args.max_bond)
    native_speed = None
    if args.compare_torch:
        native_speed = time_value_and_grad(
            native_mps_loss,
            parameters.detach(),
            iters=args.bench_iters,
            device=args.device,
        )
    jax_speed = time_value_and_grad(
        lambda theta: jax_kernel(theta).sum(),
        parameters.detach(),
        iters=args.bench_iters,
        device=args.device,
    ) if jax_kernel is not None else None
    final_energy = float(training_loss(parameters).detach())
    print_training_summary(
        title="Single-Machine Quantum AI (MPS)",
        example="single_machine_mps_training",
        metrics={
            "theoretical_ground_energy": exact_energy,
            "reference": (
                f"exact diagonalization <= {args.small_exact_max_wires} wires"
                if exact_energy is not None
                else ("disabled" if args.reference == "none" else "n/a for this wire count")
            ),
            "jax_cache": cache_report.get("cache_dir") if cache_report.get("enabled") else "disabled",
            "training_backend": args.backend,
            "initial_energy": initial,
            "final_energy": final_energy,
            "gap_to_theory": None if exact_energy is None else final_energy - exact_energy,
        },
        speed_compare={
            "pytorch_native_mps": native_speed if native_speed is not None else {"status": "unavailable", "reason": "run with --compare-torch"},
            "jax_kernel_mps": jax_speed if jax_speed is not None else {"status": "unavailable", "reason": jax_error},
            "speedup_jax_over_pytorch": speedup(
                native_speed["avg_seconds"] if native_speed else None,
                jax_speed["avg_seconds"] if jax_speed else None,
            ),
        },
        model_summary=trained_mps.summary(),
    )


if __name__ == "__main__":
    main()
