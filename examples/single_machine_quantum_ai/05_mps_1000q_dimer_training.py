"""Structured 1000-qubit MPS quantum AI training example.

This example uses a dimerized low-bond MPS task: every neighboring pair is a
small entangled quantum unit, and the global 1000-qubit state is their tensor
product. Dense statevector simulators must represent 2**1000 amplitudes, while
the MPS/JAX kernel evaluates the local training objective in O(n) time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
from common import (  # noqa: E402
    configure_jax_compilation_cache,
    jax_available,
    print_kv,
    print_section,
    print_table,
    sync_if_needed,
    time_value_and_grad,
)


def _init_teacher(n_pairs: int, *, device: str) -> torch.Tensor:
    base = torch.linspace(-0.7, 0.7, steps=int(n_pairs), dtype=torch.float32, device=device)
    return torch.stack(
        (
            0.35 * torch.sin(1.7 * base) + 0.15,
            -0.30 * torch.cos(1.3 * base) - 0.05,
            0.45 * torch.sin(0.9 * base + 0.2),
        ),
        dim=-1,
    )


def _init_trainable(teacher: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor([0.45, -0.35, 0.25], dtype=teacher.dtype, device=teacher.device)
    return (teacher.detach() + offset).clone().requires_grad_(True)


def _torch_to_jax(tensor: torch.Tensor) -> Any:
    import jax.dlpack

    return jax.dlpack.from_dlpack(tensor.detach().contiguous())


def _jax_to_torch(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(value).to(device=like.device, dtype=like.dtype)


def _dimer_observables_jax(params: Any) -> Any:
    import jax.numpy as jnp

    alpha = params[:, 0]
    beta = params[:, 1]
    gamma = params[:, 2]
    ca = jnp.cos(alpha / 2)
    sa = jnp.sin(alpha / 2)
    cb = jnp.cos(beta / 2)
    sb = jnp.sin(beta / 2)
    cg = jnp.cos(gamma / 2)
    sg = jnp.sin(gamma / 2)

    amp00 = cg * ca * cb - 1j * sg * sa * sb
    amp01 = cg * ca * sb - 1j * sg * sa * cb
    amp10 = cg * sa * cb - 1j * sg * ca * sb
    amp11 = cg * sa * sb - 1j * sg * ca * cb
    p00 = jnp.real(jnp.conj(amp00) * amp00)
    p01 = jnp.real(jnp.conj(amp01) * amp01)
    p10 = jnp.real(jnp.conj(amp10) * amp10)
    p11 = jnp.real(jnp.conj(amp11) * amp11)

    z_left = p00 + p01 - p10 - p11
    z_right = p00 - p01 + p10 - p11
    zz = p00 - p01 - p10 + p11
    xx = 2.0 * jnp.real(jnp.conj(amp00) * amp11 + jnp.conj(amp01) * amp10)
    return jnp.stack((z_left, z_right, zz, xx), axis=-1)


def _structured_loss_jax(params: Any, target: Any) -> Any:
    import jax.numpy as jnp

    pred = _dimer_observables_jax(params)
    diff = pred - target
    return jnp.mean(jnp.sum(diff * diff, axis=-1))


_JIT_DIMER_VALUE_AND_GRAD: Any | None = None


def _dimer_value_and_grad() -> Any:
    global _JIT_DIMER_VALUE_AND_GRAD
    if _JIT_DIMER_VALUE_AND_GRAD is None:
        import jax

        _JIT_DIMER_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(_structured_loss_jax))
    return _JIT_DIMER_VALUE_AND_GRAD


class _DimerMPSLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        jax_params = _torch_to_jax(params)
        jax_target = _torch_to_jax(target)
        value, grad = _dimer_value_and_grad()(jax_params, jax_target)
        torch_grad = _jax_to_torch(grad, like=params)
        ctx.save_for_backward(torch_grad)
        return _jax_to_torch(value, like=params)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (grad,) = ctx.saved_tensors
        return grad_output.reshape(()) * grad, None


def structured_dimer_mps_loss(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _DimerMPSLoss.apply(params, target)


def build_small_flagquantum_circuit(params: torch.Tensor, *, n_wires: int, device: str) -> fq.Circuit:
    circuit = fq.Circuit(int(n_wires), device=device)
    pairs = int(n_wires) // 2
    for pair in range(pairs):
        left = 2 * pair
        right = left + 1
        circuit.ry(left, theta=params[pair, 0])
        circuit.ry(right, theta=params[pair, 1])
        circuit.rxx(left, right, theta=params[pair, 2])
    return circuit


def run_training(args: argparse.Namespace) -> None:
    cache_report = configure_jax_compilation_cache(args.jax_cache_dir)
    has_jax, jax_error = jax_available()
    if not has_jax:
        raise SystemExit(f"JAX backend requested but unavailable: {jax_error}.")

    if args.n_qubits % 2:
        raise SystemExit("--n-qubits must be even for the dimerized MPS example.")
    n_pairs = args.n_qubits // 2
    torch.manual_seed(23)
    teacher = _init_teacher(n_pairs, device=args.device)
    with torch.no_grad():
        target = torch.utils.dlpack.from_dlpack(_dimer_observables_jax(_torch_to_jax(teacher))).to(
            device=teacher.device,
            dtype=teacher.dtype,
        )
    params = _init_trainable(teacher)
    optimizer = torch.optim.Adam([params], lr=args.lr)

    first_start = time.perf_counter()
    first_loss = structured_dimer_mps_loss(params, target)
    first_loss.backward()
    sync_if_needed(args.device)
    first_seconds = time.perf_counter() - first_start
    optimizer.step()
    optimizer.zero_grad()

    initial = float(first_loss.detach())
    for step in range(1, int(args.steps)):
        loss = structured_dimer_mps_loss(params, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step + 1 == args.steps or (step + 1) % max(1, args.steps // 5) == 0:
            print({"step": step + 1, "loss": float(loss.detach())})

    final_loss = float(structured_dimer_mps_loss(params, target).detach())
    speed = time_value_and_grad(
        lambda theta: structured_dimer_mps_loss(theta, target),
        params.detach(),
        iters=args.bench_iters,
        warmup=args.bench_warmup,
        device=args.device,
    )

    print_section("Single-Machine Quantum AI (1000q Structured MPS)")
    print_kv(
        {
            "example": "single_machine_1000q_dimer_mps_training",
            "n_qubits": args.n_qubits,
            "n_pairs": n_pairs,
            "mps_bond": 2,
            "parameters": int(params.numel()),
            "training_backend": "jax_kernel_torch_autograd",
            "distribution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "initial_loss": initial,
            "final_loss": final_loss,
            "first_loss_grad_s": first_seconds,
            "steady_loss_grad_s": speed["avg_seconds"],
            "steady_grad_norm": speed["grad_norm"],
            "jax_cache_enabled": cache_report.get("enabled"),
            "jax_cache_dir": cache_report.get("cache_dir"),
        }
    )
    print_section("Why This Is A FlagQuantum-Favorable Benchmark")
    print_table(
        [
            {"path": "FlagQuantum structured JAX MPS", "memory": "O(n)", "trainable": "yes", "1000q": "yes"},
            {"path": "dense statevector/default.qubit", "memory": "O(2^n)", "trainable": "no", "1000q": "no"},
            {"path": "generic MPS without structure", "memory": "O(n chi^2)", "trainable": "maybe", "1000q": "compile-heavy"},
        ],
        columns=["path", "memory", "trainable", "1000q"],
    )

    if args.show_small_ir:
        small_wires = min(args.n_qubits, 8)
        small_circuit = build_small_flagquantum_circuit(params.detach()[: small_wires // 2], n_wires=small_wires, device=args.device)
        print_section("Small FlagQuantum IR Preview")
        print_kv({"n_wires": small_wires, "ir_instructions": len(small_circuit.to_ir())})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--bench-warmup", type=int, default=3)
    parser.add_argument("--jax-cache-dir", default=None)
    parser.add_argument("--show-small-ir", action="store_true")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
