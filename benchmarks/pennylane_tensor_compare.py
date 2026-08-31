"""Focused MPS/TN comparison between FlagQuantum and PennyLane tensor devices.

This benchmark is intentionally separate from ``pennylane_backend_compare.py``.
The general benchmark uses a batched Ising Hamiltonian workload, which is useful
as a stress test but can make PennyLane tensor devices extremely slow when the
gradient method is parameter-shift.

Recommended quick commands
--------------------------
CPU MPS quick gradient:
  python benchmarks/pennylane_tensor_compare.py --device cpu --pennylane-device default.tensor --method mps --n-wires 4 --layers 1 --observable z_sum --iters 3 --warmup 1 --json-output cpu_pl_tensor_mps_quick.json

CPU MPS quick gradient, PennyLane JAX interface:
  python benchmarks/pennylane_tensor_compare.py --device cpu --pennylane-device default.tensor --pennylane-interface jax --method mps --n-wires 4 --layers 1 --observable z_sum --iters 3 --warmup 1 --pennylane-run-timeout-seconds 30 --json-output cpu_pl_tensor_mps_jax_quick.json

CPU MPS quick timeout/lower-bound:
  python benchmarks/pennylane_tensor_compare.py --device cpu --pennylane-device default.tensor --method mps --n-wires 4 --layers 1 --observable z_sum --iters 3 --warmup 0 --pennylane-run-timeout-seconds 10 --json-output cpu_pl_tensor_mps_timeout.json

CPU TN quick gradient:
  python benchmarks/pennylane_tensor_compare.py --device cpu --pennylane-device default.tensor --method tn --n-wires 4 --layers 1 --observable z_sum --iters 3 --warmup 1 --json-output cpu_pl_tensor_tn_quick.json

CPU TN quick timeout/lower-bound:
  python benchmarks/pennylane_tensor_compare.py --device cpu --pennylane-device default.tensor --method tn --n-wires 4 --layers 1 --observable z_sum --iters 3 --warmup 0 --pennylane-run-timeout-seconds 10 --json-output cpu_pl_tensor_tn_timeout.json

GPU lightning.tensor MPS quick gradient, when installed:
  python benchmarks/pennylane_tensor_compare.py --device cuda --pennylane-device lightning.tensor --method mps --max-bond 32 --n-wires 6 --layers 1 --observable z_sum --iters 5 --warmup 1 --json-output gpu_pl_lightning_tensor_mps_quick.json

GPU lightning.tensor MPS quick timeout/lower-bound, when installed:
  python benchmarks/pennylane_tensor_compare.py --device cuda --pennylane-device lightning.tensor --method mps --max-bond 32 --n-wires 6 --layers 1 --observable z_sum --iters 5 --warmup 0 --pennylane-run-timeout-seconds 30 --json-output gpu_pl_lightning_tensor_mps_timeout.json

GPU lightning.tensor TN quick gradient, when installed:
  python benchmarks/pennylane_tensor_compare.py --device cuda --pennylane-device lightning.tensor --method tn --n-wires 6 --layers 1 --observable z_sum --iters 5 --warmup 1 --json-output gpu_pl_lightning_tensor_tn_quick.json

Forward-only sanity check:
  python benchmarks/pennylane_tensor_compare.py --device cpu --pennylane-device default.tensor --method mps --forward-only --n-wires 8 --layers 2 --observable z_sum --iters 10 --warmup 2 --json-output cpu_pl_tensor_mps_forward.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
import flagquantum.backends as fqb  # noqa: E402


def _torch_to_jax(value: torch.Tensor) -> Any:
    import jax.dlpack

    return jax.dlpack.from_dlpack(value.detach().contiguous())


def _jax_to_torch(value: Any) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(value).detach().cpu()


def _block_jax(value: Any) -> None:
    import jax

    jax.block_until_ready(value)


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
        torch.cuda.synchronize(torch.device(device))


def _init_params(n_wires: int, layers: int, *, device: str) -> torch.Tensor:
    total = int(layers) * int(n_wires) * 2
    return torch.linspace(-0.31, 0.37, steps=total, dtype=torch.float32, device=device).reshape(
        int(layers),
        int(n_wires),
        2,
    )


def _build_circuit(params: torch.Tensor, *, device: str) -> fq.Circuit:
    layers, n_wires, _ = params.shape
    circuit = fq.Circuit(int(n_wires), device=device)
    for layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.ry(wire, theta=params[layer, wire, 0])
            circuit.rz(wire, theta=params[layer, wire, 1])
        for wire in range(int(n_wires) - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def _loss_from_flagquantum_state(state_or_circuit: Any, observable: str) -> torch.Tensor:
    if observable == "z0":
        return state_or_circuit.expectation_z(0).sum()
    if observable == "z_sum":
        if hasattr(state_or_circuit, "expectation_z_sum"):
            return state_or_circuit.expectation_z_sum().sum()
        return state_or_circuit.expectation_z().sum()
    if observable == "edge_zz":
        if hasattr(state_or_circuit, "expectation_ps"):
            return state_or_circuit.expectation_ps(z=[0, state_or_circuit.n_wires - 1]).sum()
        n_wires = int(state_or_circuit.n_wires)
        return state_or_circuit.expectation_ps(z=[0, n_wires - 1]).sum()
    raise ValueError(f"Unsupported observable {observable!r}.")


def _flagquantum_loss(
    params: torch.Tensor,
    *,
    mode: str,
    max_bond: int | None,
    observable: str,
    device: str,
) -> torch.Tensor:
    circuit = _build_circuit(params, device=device)
    if mode == "mps":
        return _loss_from_flagquantum_state(
            fqb.run_mps(circuit, max_bond=max_bond), observable
        )
    if mode in {"tn", "tensor_network"}:
        return _loss_from_flagquantum_state(
            fqb.run_tensor_network(circuit), observable
        )
    raise ValueError("mode must be 'mps' or 'tn'.")


def _flagquantum_jax_loss_fn(
    *,
    example_params: torch.Tensor,
    mode: str,
    max_bond: int | None,
    observable: str,
    n_wires: int,
    device: str,
    jit: bool,
    compute_dtype: str,
    matmul_precision: str | None,
) -> Any:
    kernel = fq.compile_quantum_kernel(
        lambda values: _build_circuit(values, device=device),
        example_params,
        backend="jax",
        interface="torch",
        mode="tensor_network" if mode == "tn" else mode,
        n_wires=int(n_wires),
        observable="z_sum" if observable == "z_sum" else "z",
        observable_wires=(0,) if observable in {"z0", "edge_zz"} else None,
        jit=jit,
        max_bond=max_bond,
        compute_dtype=compute_dtype,
        matmul_precision=matmul_precision,
    )

    def loss(values: torch.Tensor) -> torch.Tensor:
        out = kernel(values)
        if observable == "edge_zz":
            return _flagquantum_loss(values, mode=mode, max_bond=max_bond, observable=observable, device=device)
        return out.sum()

    return loss, kernel


def _build_pennylane_loss(
    *,
    n_wires: int,
    observable: str,
    device_name: str,
    method: str,
    max_bond: int | None,
    cutoff: float | None,
    diff_method: str,
) -> Any:
    import pennylane as qml

    kwargs: dict[str, Any] = {"method": "tn" if method == "tn" else "mps"}
    if max_bond is not None:
        kwargs["max_bond_dim"] = int(max_bond)
    if cutoff is not None:
        kwargs["cutoff"] = float(cutoff)
    qdev = qml.device(device_name, wires=int(n_wires), **kwargs)

    @qml.qnode(qdev, interface="torch", diff_method=diff_method)
    def qnode(values: torch.Tensor) -> Any:
        for layer in range(int(values.shape[0])):
            for wire in range(int(n_wires)):
                qml.RY(values[layer, wire, 0], wires=wire)
                qml.RZ(values[layer, wire, 1], wires=wire)
            for wire in range(int(n_wires) - 1):
                qml.CNOT(wires=(wire, wire + 1))
        if observable == "z0":
            return qml.expval(qml.PauliZ(0))
        if observable == "edge_zz":
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(int(n_wires) - 1))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(int(n_wires))]

    def loss(values: torch.Tensor) -> torch.Tensor:
        out = qnode(values)
        if observable == "z_sum":
            return torch.stack(tuple(out)).sum()
        return out

    return loss


def _build_pennylane_jax_value_and_grad(
    *,
    n_wires: int,
    observable: str,
    device_name: str,
    method: str,
    max_bond: int | None,
    cutoff: float | None,
    diff_method: str,
    jit: bool,
) -> Any:
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    kwargs: dict[str, Any] = {"method": "tn" if method == "tn" else "mps"}
    if max_bond is not None:
        kwargs["max_bond_dim"] = int(max_bond)
    if cutoff is not None:
        kwargs["cutoff"] = float(cutoff)
    qdev = qml.device(device_name, wires=int(n_wires), **kwargs)

    @qml.qnode(qdev, interface="jax", diff_method=diff_method)
    def qnode(values: Any) -> Any:
        for layer in range(int(values.shape[0])):
            for wire in range(int(n_wires)):
                qml.RY(values[layer, wire, 0], wires=wire)
                qml.RZ(values[layer, wire, 1], wires=wire)
            for wire in range(int(n_wires) - 1):
                qml.CNOT(wires=(wire, wire + 1))
        if observable == "z0":
            return qml.expval(qml.PauliZ(0))
        if observable == "edge_zz":
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(int(n_wires) - 1))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(int(n_wires))]

    def loss(values: Any) -> Any:
        out = qnode(values)
        if observable == "z_sum":
            return jnp.sum(jnp.stack(out))
        return out

    value_and_grad = jax.value_and_grad(loss)
    return jax.jit(value_and_grad) if jit else value_and_grad


def _pennylane_jax_value_and_grad(value_and_grad: Any, params_seed: torch.Tensor, *, forward_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
    import jax

    params = _torch_to_jax(params_seed)
    loss, grad = value_and_grad(params)
    if forward_only:
        _block_jax(loss)
        return _jax_to_torch(loss), torch.zeros_like(params_seed)
    jax.block_until_ready((loss, grad))
    return _jax_to_torch(loss), _jax_to_torch(grad)


def _value_and_grad(loss_fn: Any, params_seed: torch.Tensor, *, forward_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(not forward_only)
    loss = loss_fn(params)
    if forward_only:
        grad = torch.zeros_like(params)
    else:
        loss.backward()
        grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_value_and_grad(
    fn: Any,
    *,
    warmup: int,
    iters: int,
    device: str,
    reference_avg_seconds: float | None = None,
    early_stop_speedup: float | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
    _sync(device)
    start = time.perf_counter()
    loss = None
    grad = None
    completed = 0
    early_stop_reason = None
    for _ in range(int(iters)):
        iter_start = time.perf_counter()
        loss, grad = fn()
        _sync(device)
        completed += 1
        iter_seconds = time.perf_counter() - iter_start
        elapsed_so_far = time.perf_counter() - start
        if (
            reference_avg_seconds is not None
            and early_stop_speedup is not None
            and reference_avg_seconds > 0
            and iter_seconds / reference_avg_seconds >= float(early_stop_speedup)
        ):
            early_stop_reason = "speedup_threshold_reached"
            break
        if max_seconds is not None and elapsed_so_far >= float(max_seconds):
            early_stop_reason = "max_seconds_reached"
            break
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    avg_seconds = elapsed / max(1, int(completed))
    return {
        "avg_seconds": avg_seconds,
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
        "completed_iters": int(completed),
        "requested_iters": int(iters),
        "early_stopped": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "speedup_lower_bound_over_reference": (
            avg_seconds / reference_avg_seconds
            if reference_avg_seconds is not None and reference_avg_seconds > 0
            else None
        ),
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.write_text(text + "\n", encoding="utf-8")


def _pennylane_worker(config: dict[str, Any], queue: Any) -> None:
    try:
        _configure_torch_precision(str(config["torch_matmul_precision"]))
        device = _device(str(config["device"]))
        params_seed = _init_params(int(config["n_wires"]), int(config["layers"]), device=device)
        if str(config["pennylane_interface"]) == "jax":
            if device.startswith("cuda"):
                params_seed = params_seed.cpu()
            pl_value_and_grad = _build_pennylane_jax_value_and_grad(
                n_wires=int(config["n_wires"]),
                observable=str(config["observable"]),
                device_name=str(config["pennylane_device"]),
                method=str(config["method"]),
                max_bond=config["max_bond"],
                cutoff=config["cutoff"],
                diff_method=str(config["pennylane_diff_method"]),
                jit=not bool(config["no_pennylane_jit"]),
            )
            fn = lambda: _pennylane_jax_value_and_grad(
                pl_value_and_grad,
                params_seed,
                forward_only=bool(config["forward_only"]),
            )
        else:
            pl_loss = _build_pennylane_loss(
                n_wires=int(config["n_wires"]),
                observable=str(config["observable"]),
                device_name=str(config["pennylane_device"]),
                method=str(config["method"]),
                max_bond=config["max_bond"],
                cutoff=config["cutoff"],
                diff_method=str(config["pennylane_diff_method"]),
            )
            fn = lambda: _value_and_grad(pl_loss, params_seed, forward_only=bool(config["forward_only"]))
        result = _time_value_and_grad(
            fn,
            warmup=int(config["warmup"]),
            iters=int(config["iters"]),
            device=device,
            reference_avg_seconds=float(config["reference_avg_seconds"]),
            early_stop_speedup=config["pennylane_early_stop_speedup"],
            max_seconds=config["pennylane_max_seconds"],
        )
        result["grad_shape"] = list(result["grad"].shape)
        result["grad"] = result["grad"].detach().cpu().reshape(-1).tolist()
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})


def _time_pennylane_in_subprocess(config: dict[str, Any], *, timeout_seconds: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    process = context.Process(target=_pennylane_worker, args=(config, queue))
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, {
            "type": "TimeoutError",
            "message": f"PennyLane subprocess exceeded {float(timeout_seconds):.3f} seconds.",
            "timeout_seconds": float(timeout_seconds),
        }
    if queue.empty():
        return None, {
            "type": "RuntimeError",
            "message": f"PennyLane subprocess exited with code {process.exitcode} without returning a result.",
            "exitcode": process.exitcode,
        }
    payload = queue.get()
    if not payload.get("ok"):
        return None, payload.get("error")
    result = payload["result"]
    result["grad"] = torch.tensor(result["grad"], dtype=torch.float32).reshape(tuple(result["grad_shape"]))
    del result["grad_shape"]
    return result, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--observable", choices=("z0", "z_sum", "edge_zz"), default="z_sum")
    parser.add_argument("--method", choices=("mps", "tn"), default="mps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pennylane-device", default="default.tensor")
    parser.add_argument("--pennylane-interface", choices=("torch", "jax"), default="torch")
    parser.add_argument("--pennylane-diff-method", default="parameter-shift")
    parser.add_argument("--no-pennylane-jit", action="store_true")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--use-flagquantum-jax", action="store_true")
    parser.add_argument("--no-flagquantum-jit", action="store_true")
    parser.add_argument("--flagquantum-jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--flagquantum-jax-matmul-precision", default="highest")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--pennylane-max-seconds", type=float, default=None)
    parser.add_argument("--pennylane-run-timeout-seconds", type=float, default=None)
    parser.add_argument("--pennylane-early-stop-speedup", type=float, default=None)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    _configure_torch_precision(args.torch_matmul_precision)
    device = _device(args.device)
    params_seed = _init_params(args.n_wires, args.layers, device=device)

    if args.use_flagquantum_jax:
        fq_loss, fq_kernel = _flagquantum_jax_loss_fn(
            example_params=params_seed.detach(),
            mode=args.method,
            max_bond=args.max_bond,
            observable=args.observable,
            n_wires=args.n_wires,
            device=device,
            jit=not args.no_flagquantum_jit,
            compute_dtype=args.flagquantum_jax_compute_dtype,
            matmul_precision=None
            if args.flagquantum_jax_matmul_precision == "default"
            else args.flagquantum_jax_matmul_precision,
        )
        flagquantum_backend = "jax"
    else:
        fq_loss = lambda values: _flagquantum_loss(
            values,
            mode=args.method,
            max_bond=args.max_bond,
            observable=args.observable,
            device=device,
        )
        fq_kernel = None
        flagquantum_backend = "pytorch"

    flagquantum_result = _time_value_and_grad(
        lambda: _value_and_grad(fq_loss, params_seed, forward_only=args.forward_only),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    pennylane_result = None
    pennylane_error = None
    if importlib.util.find_spec("pennylane") is None:
        pennylane_error = {"type": "ModuleNotFoundError", "message": "pennylane is not installed"}
    else:
        try:
            if args.pennylane_run_timeout_seconds is not None:
                pennylane_result, pennylane_error = _time_pennylane_in_subprocess(
                    {
                        "n_wires": args.n_wires,
                        "layers": args.layers,
                        "observable": args.observable,
                        "method": args.method,
                        "device": device,
                        "pennylane_device": args.pennylane_device,
                        "pennylane_interface": args.pennylane_interface,
                        "pennylane_diff_method": args.pennylane_diff_method,
                        "no_pennylane_jit": args.no_pennylane_jit,
                        "max_bond": args.max_bond,
                        "cutoff": args.cutoff,
                        "forward_only": args.forward_only,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "torch_matmul_precision": args.torch_matmul_precision,
                        "reference_avg_seconds": flagquantum_result["avg_seconds"],
                        "pennylane_early_stop_speedup": args.pennylane_early_stop_speedup,
                        "pennylane_max_seconds": args.pennylane_max_seconds,
                    },
                    timeout_seconds=float(args.pennylane_run_timeout_seconds),
                )
            else:
                if args.pennylane_interface == "jax":
                    pl_value_and_grad = _build_pennylane_jax_value_and_grad(
                        n_wires=args.n_wires,
                        observable=args.observable,
                        device_name=args.pennylane_device,
                        method=args.method,
                        max_bond=args.max_bond,
                        cutoff=args.cutoff,
                        diff_method=args.pennylane_diff_method,
                        jit=not args.no_pennylane_jit,
                    )
                    pl_fn = lambda: _pennylane_jax_value_and_grad(
                        pl_value_and_grad,
                        params_seed.cpu() if device.startswith("cuda") else params_seed,
                        forward_only=args.forward_only,
                    )
                else:
                    pl_loss = _build_pennylane_loss(
                        n_wires=args.n_wires,
                        observable=args.observable,
                        device_name=args.pennylane_device,
                        method=args.method,
                        max_bond=args.max_bond,
                        cutoff=args.cutoff,
                        diff_method=args.pennylane_diff_method,
                    )
                    pl_fn = lambda: _value_and_grad(pl_loss, params_seed, forward_only=args.forward_only)
                pennylane_result = _time_value_and_grad(
                    pl_fn,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                    reference_avg_seconds=flagquantum_result["avg_seconds"],
                    early_stop_speedup=args.pennylane_early_stop_speedup,
                    max_seconds=args.pennylane_max_seconds,
                )
        except Exception as exc:
            pennylane_error = {"type": type(exc).__name__, "message": str(exc)}

    comparison: dict[str, Any]
    if pennylane_result is None:
        lower_bound = None
        if pennylane_error and pennylane_error.get("type") == "TimeoutError" and flagquantum_result["avg_seconds"] > 0:
            lower_bound = float(pennylane_error["timeout_seconds"]) / float(flagquantum_result["avg_seconds"])
        comparison = {
            "status": "timeout" if lower_bound is not None else "unavailable",
            "reason": pennylane_error,
            "speedup_lower_bound_flagquantum_over_pennylane": lower_bound,
        }
    else:
        loss_abs_error = abs(float(flagquantum_result["loss"]) - float(pennylane_result["loss"]))
        grad_max_abs_error = 0.0 if args.forward_only else _max_abs_diff(flagquantum_result["grad"], pennylane_result["grad"])
        precision_ok = loss_abs_error <= float(args.loss_atol) and (
            args.forward_only or grad_max_abs_error <= float(args.grad_atol)
        )
        comparison = {
            "status": "ok" if precision_ok else "precision_mismatch",
            "loss_abs_error": loss_abs_error,
            "grad_max_abs_error": grad_max_abs_error,
            "loss_atol": float(args.loss_atol),
            "grad_atol": float(args.grad_atol),
            "speedup_flagquantum_over_pennylane": (
                pennylane_result["avg_seconds"] / flagquantum_result["avg_seconds"]
                if flagquantum_result["avg_seconds"] > 0
                else None
            ),
            "pennylane_early_stopped": pennylane_result["early_stopped"],
            "pennylane_early_stop_reason": pennylane_result["early_stop_reason"],
            "speedup_lower_bound_flagquantum_over_pennylane": pennylane_result[
                "speedup_lower_bound_over_reference"
            ],
        }

    payload = {
        "benchmark": "pennylane_tensor_compare",
        "n_wires": args.n_wires,
        "layers": args.layers,
        "observable": args.observable,
        "method": args.method,
        "device": device,
        "forward_only": bool(args.forward_only),
        "iters": args.iters,
        "warmup": args.warmup,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pennylane": None if importlib.util.find_spec("pennylane") is None else __import__("pennylane").__version__,
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
            "torch_float32_matmul_precision": args.torch_matmul_precision,
        },
        "flagquantum": {
            "backend": flagquantum_backend,
            "mode": args.method,
            "max_bond": args.max_bond,
            "avg_seconds": flagquantum_result["avg_seconds"],
            "loss": flagquantum_result["loss"],
            "kernel": None if fq_kernel is None else fq_kernel.summary(),
        },
        "pennylane": {
            "backend": args.pennylane_device,
            "interface": args.pennylane_interface,
            "method": args.method,
            "diff_method": args.pennylane_diff_method,
            "jit": (not args.no_pennylane_jit) if args.pennylane_interface == "jax" else False,
            "max_bond_dim": args.max_bond,
            "cutoff": args.cutoff,
            "avg_seconds": None if pennylane_result is None else pennylane_result["avg_seconds"],
            "loss": None if pennylane_result is None else pennylane_result["loss"],
            "completed_iters": None if pennylane_result is None else pennylane_result["completed_iters"],
            "requested_iters": None if pennylane_result is None else pennylane_result["requested_iters"],
            "early_stopped": None if pennylane_result is None else pennylane_result["early_stopped"],
            "early_stop_reason": None if pennylane_result is None else pennylane_result["early_stop_reason"],
            "error": pennylane_error,
        },
        "comparison": comparison,
    }
    if comparison["status"] in {"ok", "precision_mismatch"}:
        speedup = comparison.get("speedup_flagquantum_over_pennylane")
        payload["conclusion"] = {
            "headline": (
                "FlagQuantum matches PennyLane tensor-device precision for this workload."
                if comparison["status"] == "ok"
                else "FlagQuantum and PennyLane tensor-device precision differed for this workload."
            ),
            "recommended_claim": (
                f"On {device}, method={args.method}, {args.n_wires} wires, {args.layers} layers, "
                f"observable={args.observable}, FlagQuantum {flagquantum_backend} is {float(speedup):.2f}x "
                f"faster than PennyLane {args.pennylane_device}."
                if speedup is not None
                else None
            ),
        }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json_output(args.json_output, text)


if __name__ == "__main__":
    main()
