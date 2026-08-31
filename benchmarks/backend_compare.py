"""Compare FlagQuantum PyTorch and JAX quantum-kernel backends.

Examples
--------
Single-sample Z-sum benchmark:

    python benchmarks/backend_compare.py --n-wires 8 --layers 2 --iters 50 --warmup 10 \
          --json-output backend_compare_8q_batch1_z_sum.json

Batched Hamiltonian benchmark:

    python benchmarks/backend_compare.py --n-wires 8 --layers 2 --batch-size 16 \
        --observable ising --iters 50 --warmup 10 --json-output backend_compare_8q_batch16_ising.json

JAX MPS vs PyTorch MPS：
    python benchmarks/backend_compare.py --device cpu --mode mps --max-bond 32 --n-wires 8 \
        --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_jax_mps_compare_8q_b16_ising.json



The PyTorch path is the native FlagQuantum statevector/autograd path. The JAX
path is FlagQuantum's JAX quantum kernel exposed through the PyTorch interface.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPRO_COMMANDS = """
CPU statevector:
  python benchmarks/backend_compare.py --device cpu --mode statevector --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_jax_statevector_compare_8q_b16_ising.json

CPU MPS:
  python benchmarks/backend_compare.py --device cpu --mode mps --max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_jax_mps_compare_8q_b16_ising.json

CPU MPS high precision:
  python benchmarks/backend_compare.py --device cpu --mode mps --max-bond 32 --jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_jax_mps_compare_8q_b16_ising_fp64.json

CPU tensor network:
  python benchmarks/backend_compare.py --device cpu --mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_jax_tn_compare_8q_b16_ising.json

CPU tensor network high precision:
  python benchmarks/backend_compare.py --device cpu --mode tensor_network --jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_jax_tn_compare_8q_b16_ising_fp64.json

Quick smoke:
  python benchmarks/backend_compare.py --device cpu --mode mps --max-bond 8 --n-wires 4 --layers 1 --batch-size 2 --observable z_sum --iters 2 --warmup 1
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
import flagquantum.backends as fqb  # noqa: E402


def _configure_torch_precision(mode: str) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(mode)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = mode != "highest"
        torch.backends.cudnn.allow_tf32 = mode != "highest"


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _init_params(
    n_wires: int,
    layers: int,
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    total = int(layers) * int(n_wires) * 3
    base = torch.linspace(-0.37, 0.41, steps=total, dtype=torch.float32, device=device).reshape(
        int(layers),
        int(n_wires),
        3,
    )
    if int(batch_size) <= 1:
        return base
    offsets = torch.linspace(-0.09, 0.09, steps=int(batch_size), dtype=torch.float32, device=device)
    return base.unsqueeze(0) + offsets.reshape(int(batch_size), 1, 1, 1)


def _build_circuit(params: torch.Tensor, *, device: str) -> fq.Circuit:
    layers, n_wires, _ = params.shape
    circuit = fq.Circuit(int(n_wires), device=device)
    for layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.rx(wire, theta=params[layer, wire, 0])
            circuit.ry(wire, theta=params[layer, wire, 1])
            circuit.rz(wire, theta=params[layer, wire, 2])
        for wire in range(int(n_wires) - 1):
            circuit.cx(wire, wire + 1)
        if int(n_wires) > 2:
            circuit.rxx(0, int(n_wires) - 1, theta=params[layer, 0, 0] * 0.25)
    return circuit


def _ising_hamiltonian(n_wires: int) -> fq.Hamiltonian:
    terms = []
    for wire in range(int(n_wires) - 1):
        terms.append(fq.pauli_term(0.7, "ZZ", (wire, wire + 1)))
    for wire in range(int(n_wires)):
        terms.append(fq.pauli_term(-0.2, "X", (wire,)))
        terms.append(fq.pauli_term(0.05, "Z", (wire,)))
    return fq.Hamiltonian(terms)


def _pauli_dense_matrix(
    ops: tuple[tuple[int, str], ...],
    n_wires: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    op_map = {int(wire): str(name).lower() for wire, name in ops}
    complex_dtype = dtype if dtype.is_complex else torch.complex64
    identity = torch.eye(2, dtype=complex_dtype, device=device)
    matrices = {
        "i": identity,
        "x": torch.tensor([[0, 1], [1, 0]], dtype=complex_dtype, device=device),
        "y": torch.tensor([[0, -1j], [1j, 0]], dtype=complex_dtype, device=device),
        "z": torch.tensor([[1, 0], [0, -1]], dtype=complex_dtype, device=device),
    }
    out = matrices[op_map.get(0, "i")]
    for wire in range(1, int(n_wires)):
        out = torch.kron(out, matrices[op_map.get(wire, "i")])
    return out


def _dense_hamiltonian_loss(state_or_circuit: Any, hamiltonian: fq.Hamiltonian) -> torch.Tensor:
    state = state_or_circuit.state()
    if state.ndim == 1:
        state = state.reshape(1, -1)
    n_wires = int(round(torch.log2(torch.tensor(state.shape[-1], dtype=torch.float64)).item()))
    total = None
    for term in hamiltonian.terms:
        ops = tuple((int(wire), str(name).lower()) for wire, name in term.ops if str(name).lower() != "i")
        if not ops:
            value = torch.ones(state.shape[0], dtype=torch.float32, device=state.device)
        else:
            matrix = _pauli_dense_matrix(ops, n_wires, device=state.device, dtype=state.dtype)
            transformed = state @ matrix.transpose(-1, -2)
            value = torch.real(torch.sum(torch.conj(state) * transformed, dim=-1))
        contribution = torch.real(term.coefficient * value)
        total = contribution if total is None else total + contribution
    assert total is not None
    return total.sum()


def _loss_from_state(
    state_or_circuit: Any,
    *,
    observable: str,
    hamiltonian: fq.Hamiltonian | None,
) -> torch.Tensor:
    if observable == "z_sum":
        if hasattr(state_or_circuit, "expectation_z_sum"):
            return state_or_circuit.expectation_z_sum().sum()
        return state_or_circuit.expectation_z().sum()
    if observable == "ising":
        assert hamiltonian is not None
        return _dense_hamiltonian_loss(state_or_circuit, hamiltonian)
    raise ValueError(f"Unsupported observable {observable!r}.")


def _pytorch_value_and_grad(
    params_seed: torch.Tensor,
    *,
    device: str,
    mode: str,
    max_bond: int | None,
    observable: str,
    hamiltonian: fq.Hamiltonian | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    def loss_for(row: torch.Tensor) -> torch.Tensor:
        circuit = _build_circuit(row, device=device)
        if mode == "statevector":
            return _loss_from_state(circuit, observable=observable, hamiltonian=hamiltonian)
        if mode == "mps":
            return _loss_from_state(
                fqb.run_mps(circuit, max_bond=max_bond),
                observable=observable,
                hamiltonian=hamiltonian,
            )
        if mode in {"tensor_network", "tn"}:
            return _loss_from_state(
                fqb.run_tensor_network(circuit),
                observable=observable,
                hamiltonian=hamiltonian,
            )
        raise ValueError("mode must be 'statevector', 'mps', or 'tensor_network'.")

    if params.ndim == 3:
        loss = loss_for(params)
    else:
        losses = [loss_for(row) for row in params]
        loss = torch.stack(losses).sum()
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _jax_value_and_grad(
    kernel: Any,
    params_seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    values = kernel(params)
    loss = values.sum()
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_value_and_grad(fn: Any, *, warmup: int, iters: int, device: str) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
    _sync(device)
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = fn()
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())


def _conclusion(
    *,
    status: str,
    speedup: float | None,
    loss_abs_error: float,
    grad_max_abs_error: float,
    observable: str,
    batch_size: int,
    n_wires: int,
    layers: int,
    device: str,
) -> dict[str, Any]:
    headline = (
        "FlagQuantum JAX quantum-kernel backend matches PyTorch precision and is faster."
        if status == "ok"
        else "FlagQuantum JAX quantum-kernel backend did not meet the requested precision thresholds."
    )
    return {
        "headline": headline,
        "precision_status": status,
        "recommended_claim": (
            f"On {device}, {int(n_wires)} wires, {int(layers)} layers, batch_size={int(batch_size)}, "
            f"observable={observable}, FlagQuantum JAX is {float(speedup):.2f}x faster than "
            f"FlagQuantum PyTorch with loss_abs_error={loss_abs_error:.3e} and "
            f"grad_max_abs_error={grad_max_abs_error:.3e}."
            if speedup is not None
            else None
        ),
    }


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("statevector", "mps", "tensor_network", "tn"), default="statevector")
    parser.add_argument("--observable", choices=("z_sum", "ising"), default="z_sum")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-jax-jit", action="store_true")
    parser.add_argument("--jax-matmul-precision", default="highest")
    parser.add_argument("--jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path; stdout is used when omitted.",
    )
    args = parser.parse_args()
    _configure_torch_precision(args.torch_matmul_precision)

    device = _device(args.device)
    params_seed = _init_params(
        args.n_wires,
        args.layers,
        batch_size=args.batch_size,
        device=device,
    )
    example_params = params_seed[0].detach() if params_seed.ndim == 4 else params_seed.detach()
    hamiltonian = _ising_hamiltonian(args.n_wires) if args.observable == "ising" else None

    pytorch_result = _time_value_and_grad(
        lambda: _pytorch_value_and_grad(
            params_seed,
            device=device,
            mode=args.mode,
            max_bond=args.max_bond,
            observable=args.observable,
            hamiltonian=hamiltonian,
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    payload: dict[str, Any] = {
        "benchmark": "backend_compare",
        "n_wires": args.n_wires,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "observable": args.observable,
        "mode": args.mode,
        "device": device,
        "iters": args.iters,
        "warmup": args.warmup,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
        },
        "flagquantum_pytorch": {
            "backend": "pytorch",
            "mode": args.mode,
            "avg_seconds": pytorch_result["avg_seconds"],
            "loss": pytorch_result["loss"],
        },
    }

    if importlib.util.find_spec("jax") is None:
        payload["flagquantum_jax"] = {
            "backend": "jax",
            "status": "unavailable",
            "reason": "jax is not installed",
        }
        payload["comparison"] = {"status": "unavailable", "reason": "jax is not installed"}
    else:
        kernel = fq.compile_quantum_kernel(
            lambda values: _build_circuit(values, device=device),
            example_params,
            backend="jax",
            interface="torch",
            mode="tensor_network" if args.mode == "tn" else args.mode,
            n_wires=args.n_wires,
            observable="z_sum" if args.observable == "z_sum" else "hamiltonian",
            hamiltonian=hamiltonian,
            jit=not args.no_jax_jit,
            matmul_precision=None if args.jax_matmul_precision == "default" else args.jax_matmul_precision,
            compute_dtype=args.jax_compute_dtype,
            max_bond=args.max_bond,
        )
        jax_result = _time_value_and_grad(
            lambda: _jax_value_and_grad(kernel, params_seed),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        loss_abs_error = abs(float(pytorch_result["loss"]) - float(jax_result["loss"]))
        grad_max_abs_error = _max_abs_diff(pytorch_result["grad"], jax_result["grad"])
        status = (
            "ok"
            if loss_abs_error <= float(args.loss_atol) and grad_max_abs_error <= float(args.grad_atol)
            else "precision_mismatch"
        )
        payload["flagquantum_jax"] = {
            "backend": "jax",
            "interface": "torch",
            "mode": args.mode,
            "jit": not args.no_jax_jit,
            "status": status,
            "avg_seconds": jax_result["avg_seconds"],
            "loss": jax_result["loss"],
            "kernel": kernel.summary(),
        }
        payload["comparison"] = {
            "status": status,
            "loss_abs_error": loss_abs_error,
            "grad_max_abs_error": grad_max_abs_error,
            "loss_atol": float(args.loss_atol),
            "grad_atol": float(args.grad_atol),
            "speedup_jax_over_pytorch": pytorch_result["avg_seconds"] / jax_result["avg_seconds"]
            if jax_result["avg_seconds"] > 0
            else None,
            "slowdown_jax_vs_pytorch": jax_result["avg_seconds"] / pytorch_result["avg_seconds"]
            if pytorch_result["avg_seconds"] > 0
            else None,
        }
        payload["conclusion"] = _conclusion(
            status=status,
            speedup=payload["comparison"]["speedup_jax_over_pytorch"],
            loss_abs_error=loss_abs_error,
            grad_max_abs_error=grad_max_abs_error,
            observable=args.observable,
            batch_size=args.batch_size,
            n_wires=args.n_wires,
            layers=args.layers,
            device=device,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json_output(args.json_output, text)


if __name__ == "__main__":
    main()
