"""FlagQuantum flagship single-machine JAX-MPS training benchmark.

Repro commands
--------------
Quick smoke:

    python benchmarks/flagship_mps_training.py --cases dimer:20,hardware:8 \
        --steps 2 --iters 1 --warmup 0

Structured 1000q low-bond MPS:

    python benchmarks/flagship_mps_training.py --cases dimer:1000 \
        --steps 100 --iters 10 --warmup 3 --jax-cache-dir .fq_jax_cache \
        --json-output benchmarks/results/flagship_mps_1000q_dimer_cpu_jax.json

Hardware-efficient low-bond MPS scale:

    python benchmarks/flagship_mps_training.py --cases hardware:60,hardware:300,hardware:600,hardware:1000 \
        --layers 2 --max-bond 4 --steps 0 --iters 3 --warmup 1 --jax-cache-dir .fq_jax_cache \
        --json-output benchmarks/results/flagship_mps_hardware_scale_cpu_jax.json

This benchmark is a single-device fast-path benchmark. It does not claim
distributed sharded scalability.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_machine_quantum_ai"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import flagquantum as fq  # noqa: E402
from common import configure_jax_compilation_cache, jax_available, sync_if_needed, time_value_and_grad  # noqa: E402
from single_machine_quantum_ai_compat import (  # noqa: E402
    dimer_loss,
    dimer_target,
    init_dimer_teacher,
    init_dimer_trainable,
)


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _write_json(path_text: str, payload: dict[str, Any]) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_hardware_ansatz(theta: torch.Tensor, *, n_wires: int, layers: int, device: str) -> fq.Circuit:
    circuit = fq.Circuit(int(n_wires), device=device)
    cursor = 0
    for _layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.ry(wire, theta=theta[cursor])
            cursor += 1
            circuit.rz(wire, theta=theta[cursor])
            cursor += 1
        for wire in range(int(n_wires) - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def _value_and_grad_once(loss_fn: Any, params: torch.Tensor, *, device: str) -> dict[str, Any]:
    trial = params.detach().clone().requires_grad_(True)
    sync_if_needed(device)
    start = time.perf_counter()
    loss = loss_fn(trial)
    grad = torch.autograd.grad(loss, trial, retain_graph=False, create_graph=False)[0]
    sync_if_needed(device)
    return {
        "seconds": time.perf_counter() - start,
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad.detach().norm().cpu()),
        "grad_isfinite": bool(torch.isfinite(grad).all().item()),
    }


def _train(loss_fn: Any, params: torch.Tensor, *, steps: int, lr: float, device: str) -> dict[str, Any]:
    if int(steps) <= 0:
        return {"steps": 0, "initial_loss": None, "final_loss": None}
    trainable = params.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([trainable], lr=float(lr))
    initial_loss = None
    final_loss = None
    start = time.perf_counter()
    for step in range(int(steps)):
        optimizer.zero_grad()
        loss = loss_fn(trainable)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
    sync_if_needed(device)
    return {
        "steps": int(steps),
        "seconds": time.perf_counter() - start,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "parameters_after_training": trainable.detach(),
    }


def _run_dimer_case(n_wires: int, args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    if int(n_wires) % 2:
        raise ValueError("dimer cases require an even n_wires.")
    teacher = init_dimer_teacher(int(n_wires) // 2, device=device)
    target = dimer_target(teacher)
    params = init_dimer_trainable(teacher)

    def loss_fn(values: torch.Tensor) -> torch.Tensor:
        return dimer_loss(values, target)

    first = _value_and_grad_once(loss_fn, params, device=device)
    train = _train(loss_fn, params, steps=args.steps, lr=args.lr, device=device)
    timed_params = train.get("parameters_after_training")
    if not isinstance(timed_params, torch.Tensor):
        timed_params = params
    steady = time_value_and_grad(loss_fn, timed_params, iters=args.iters, warmup=args.warmup, device=device)
    return {
        "case": f"dimer:{int(n_wires)}",
        "kind": "structured_dimer_mps",
        "n_wires": int(n_wires),
        "n_pairs": int(n_wires) // 2,
        "mps_bond": 2,
        "parameters": int(params.numel()),
        "first_loss_grad": first,
        "training": {key: value for key, value in train.items() if key != "parameters_after_training"},
        "steady_loss_grad": steady,
        "claim_boundary": "structured low-bond MPS; not arbitrary 1000-qubit circuit training",
    }


def _run_hardware_case(n_wires: int, args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    torch.manual_seed(19)
    n_params = int(n_wires) * int(args.layers) * 2
    params = (0.15 * torch.randn(n_params, device=device)).requires_grad_(False)
    hamiltonian = fq.zz_chain_hamiltonian(int(n_wires), coupling=-1.0, field=0.1)

    def build(values: torch.Tensor) -> fq.Circuit:
        return _build_hardware_ansatz(values, n_wires=int(n_wires), layers=args.layers, device=device)

    kernel_start = time.perf_counter()
    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=int(n_wires),
        hamiltonian=hamiltonian,
        max_bond=args.max_bond,
        jit=not args.no_jit,
    )
    kernel_setup_seconds = time.perf_counter() - kernel_start

    def loss_fn(values: torch.Tensor) -> torch.Tensor:
        return kernel(values).sum()

    first = _value_and_grad_once(loss_fn, params, device=device)
    train = _train(loss_fn, params, steps=args.steps, lr=args.lr, device=device)
    timed_params = train.get("parameters_after_training")
    if not isinstance(timed_params, torch.Tensor):
        timed_params = params
    steady = time_value_and_grad(loss_fn, timed_params, iters=args.iters, warmup=args.warmup, device=device)
    return {
        "case": f"hardware:{int(n_wires)}",
        "kind": "nearest_neighbor_hardware_efficient_mps",
        "n_wires": int(n_wires),
        "layers": int(args.layers),
        "max_bond": int(args.max_bond),
        "parameters": int(params.numel()),
        "kernel_setup_seconds": kernel_setup_seconds,
        "kernel_summary": kernel.summary(),
        "first_loss_grad": first,
        "training": {key: value for key, value in train.items() if key != "parameters_after_training"},
        "steady_loss_grad": steady,
        "claim_boundary": "low-bond nearest-neighbor MPS; not high-entanglement arbitrary circuit training",
    }


def _parse_cases(text: str) -> list[tuple[str, int]]:
    cases = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Cases must use kind:n_wires, for example dimer:1000 or hardware:60.")
        kind, wires = item.split(":", 1)
        kind = kind.strip().lower()
        if kind not in {"dimer", "hardware"}:
            raise ValueError("Case kind must be 'dimer' or 'hardware'.")
        cases.append((kind, int(wires)))
    if not cases:
        raise ValueError("At least one case is required.")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="dimer:1000,hardware:60")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-bond", type=int, default=4)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--jax-cache-dir", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    cache = configure_jax_compilation_cache(args.jax_cache_dir)
    has_jax, jax_error = jax_available()
    if not has_jax:
        raise SystemExit(f"JAX backend unavailable: {jax_error}")
    device = _device(args.device)

    results = []
    for kind, n_wires in _parse_cases(args.cases):
        if kind == "dimer":
            results.append(_run_dimer_case(n_wires, args, device=device))
        else:
            results.append(_run_hardware_case(n_wires, args, device=device))

    payload = {
        "benchmark": "flagship_mps_training",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "jax_cache": cache,
        "configuration": {
            "cases": args.cases,
            "layers": args.layers,
            "max_bond": args.max_bond,
            "steps": args.steps,
            "iters": args.iters,
            "warmup": args.warmup,
            "jit": not args.no_jit,
        },
        "results": results,
        "recommended_claim": (
            "FlagQuantum provides a PyTorch-facing JAX-MPS single-device fast path for structured "
            "low-bond quantum AI training, with auditable loss+gradient timing and explicit claim boundaries."
        ),
        "do_not_claim": (
            "This benchmark does not demonstrate arbitrary high-entanglement 1000-qubit circuit training "
            "or distributed sharded scalability."
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json(args.json_output, payload)


if __name__ == "__main__":
    main()
