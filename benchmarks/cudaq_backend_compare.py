"""Compare FlagQuantum JAX kernels with CUDA-Q training APIs.

This benchmark replaces the cuTensorNet benchmark for training claims. CUDA-Q is
a quantum programming stack with official gradient utilities, while cuTensorNet
is a lower-level tensor contraction engine. The workload here keeps the circuit,
parameters, observable, warmup, and iteration counts aligned, and runs CUDA-Q in
a subprocess so a slow backend cannot hang the benchmark driver.

Recommended commands
--------------------
CPU/GPU quick availability smoke:
  python benchmarks/cudaq_backend_compare.py --device cpu --n-wires 4 --layers 1 --batch-size 1 --observable z_sum --iters 2 --warmup 0 --cudaq-run-timeout-seconds 30 --json-output cudaq_smoke.json

CPU CUDA-Q ParameterShift vs FlagQuantum JAX:
  python benchmarks/cudaq_backend_compare.py --device cpu --n-wires 6 --layers 1 --batch-size 1 --observable ising --iters 5 --warmup 1 --cudaq-gradient-method parameter-shift --cudaq-run-timeout-seconds 120 --json-output cpu_cudaq_parameter_shift_vs_fq_jax.json

GPU CUDA-Q ParameterShift vs FlagQuantum JAX:
  python benchmarks/cudaq_backend_compare.py --device cuda --n-wires 8 --layers 2 --batch-size 1 --observable ising --iters 10 --warmup 2 --cudaq-gradient-method parameter-shift --cudaq-run-timeout-seconds 300 --json-output gpu_cudaq_parameter_shift_vs_fq_jax.json

Forward-only CUDA-Q vs FlagQuantum JAX:
  python benchmarks/cudaq_backend_compare.py --device cuda --n-wires 8 --layers 2 --batch-size 16 --observable ising --forward-only --iters 50 --warmup 10 --cudaq-run-timeout-seconds 120 --json-output gpu_cudaq_forward_vs_fq_jax.json

Notes
-----
CUDA-Q APIs have changed across releases. This benchmark uses conservative
runtime introspection:

* The circuit uses RX/RY/RZ plus nearest-neighbor controlled-X gates.
* The observable uses ``cudaq.spin`` terms when available.
* Gradients first try ``cudaq.gradients.ParameterShift`` / finite-difference
  classes. If the installed CUDA-Q build exposes a different signature, the
  JSON result reports the exact error instead of hanging.
"""

from __future__ import annotations

import argparse
import importlib
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


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.write_text(text + "\n", encoding="utf-8")


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
    return circuit


def _ising_hamiltonian(n_wires: int) -> fq.Hamiltonian:
    terms = []
    for wire in range(int(n_wires) - 1):
        terms.append(fq.pauli_term(0.7, "ZZ", (wire, wire + 1)))
    for wire in range(int(n_wires)):
        terms.append(fq.pauli_term(-0.2, "X", (wire,)))
        terms.append(fq.pauli_term(0.05, "Z", (wire,)))
    return fq.Hamiltonian(terms)


def _fq_value_and_grad(kernel: Any, params_seed: torch.Tensor, *, forward_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(not forward_only)
    loss = kernel(params).sum()
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


def _cudaq_set_target(cudaq: Any, target: str | None) -> None:
    if not target:
        return
    if hasattr(cudaq, "set_target"):
        cudaq.set_target(str(target))


def _cudaq_flat_params(params: torch.Tensor) -> list[float]:
    return [float(x) for x in params.detach().cpu().reshape(-1).tolist()]


def _build_cudaq_kernel(cudaq: Any, n_wires: int, layers: int) -> Any:
    for gate_name in ("rx", "ry", "rz", "x"):
        if hasattr(cudaq, gate_name):
            globals()[gate_name] = getattr(cudaq, gate_name)

    @cudaq.kernel
    def ansatz(theta: list[float]):  # type: ignore[valid-type]
        q = cudaq.qvector(n_wires)
        idx = 0
        for _layer in range(layers):
            for wire in range(n_wires):
                rx(theta[idx], q[wire])
                idx += 1
                ry(theta[idx], q[wire])
                idx += 1
                rz(theta[idx], q[wire])
                idx += 1
            for wire in range(n_wires - 1):
                x.ctrl(q[wire], q[wire + 1])

    return ansatz


def _cudaq_spin_hamiltonian(cudaq: Any, n_wires: int, observable: str) -> Any:
    spin = cudaq.spin
    if observable == "z_sum":
        ham = 0.0 * spin.z(0)
        for wire in range(int(n_wires)):
            ham = ham + spin.z(wire)
        return ham
    if observable == "ising":
        ham = 0.0 * spin.z(0)
        for wire in range(int(n_wires) - 1):
            ham = ham + 0.7 * spin.z(wire) * spin.z(wire + 1)
        for wire in range(int(n_wires)):
            ham = ham - 0.2 * spin.x(wire)
            ham = ham + 0.05 * spin.z(wire)
        return ham
    raise ValueError(f"Unsupported observable {observable!r}.")


def _cudaq_observe_loss(cudaq: Any, kernel: Any, hamiltonian: Any, values: list[float]) -> float:
    result = cudaq.observe(kernel, hamiltonian, values)
    if hasattr(result, "expectation"):
        return float(result.expectation())
    return float(result)


def _cudaq_gradient_instance(cudaq: Any, method: str) -> Any:
    gradients = getattr(cudaq, "gradients", None)
    if gradients is None:
        raise AttributeError("cudaq.gradients is not available in this CUDA-Q installation.")
    names = {
        "parameter-shift": ("ParameterShift", "ParameterShiftGradient"),
        "central-difference": ("CentralDifference", "CentralDifferenceGradient"),
        "forward-difference": ("ForwardDifference", "ForwardDifferenceGradient"),
    }[method]
    for name in names:
        if hasattr(gradients, name):
            return getattr(gradients, name)()
    raise AttributeError(f"CUDA-Q gradients module does not expose any of {names}.")


def _cudaq_gradient_compute(gradient: Any, loss_fn: Any, values: list[float]) -> list[float]:
    attempts = []
    call_patterns = (
        lambda: gradient.compute(values, loss_fn),
        lambda: gradient.compute(loss_fn, values),
        lambda: gradient.compute(values),
    )
    for pattern in call_patterns:
        try:
            out = pattern()
            if isinstance(out, tuple):
                out = out[-1]
            return [float(x) for x in list(out)]
        except Exception as exc:
            attempts.append({"type": type(exc).__name__, "message": str(exc)})
    raise RuntimeError(f"CUDA-Q gradient compute failed for known signatures: {attempts}")


def _cudaq_finite_difference(loss_fn: Any, values: list[float], eps: float = 1e-3) -> list[float]:
    grad = []
    plus = list(values)
    minus = list(values)
    for index, value in enumerate(values):
        plus[index] = value + eps
        minus[index] = value - eps
        grad.append((float(loss_fn(plus)) - float(loss_fn(minus))) / (2.0 * eps))
        plus[index] = value
        minus[index] = value
    return grad


def _cudaq_value_and_grad(
    *,
    cudaq: Any,
    kernel: Any,
    hamiltonian: Any,
    params_seed: torch.Tensor,
    method: str,
    forward_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = params_seed.detach().cpu()
    batched = rows.ndim == 4
    if not batched:
        rows = rows.reshape(1, *rows.shape)
    losses = []
    grads = []
    gradient = None if forward_only or method == "finite-difference" else _cudaq_gradient_instance(cudaq, method)
    for row in rows:
        values = _cudaq_flat_params(row)
        loss_fn = lambda vector: _cudaq_observe_loss(cudaq, kernel, hamiltonian, list(vector))
        losses.append(loss_fn(values))
        if forward_only:
            grads.append(torch.zeros_like(row))
        elif method == "finite-difference":
            grads.append(torch.tensor(_cudaq_finite_difference(loss_fn, values), dtype=torch.float32).reshape_as(row))
        else:
            grads.append(torch.tensor(_cudaq_gradient_compute(gradient, loss_fn, values), dtype=torch.float32).reshape_as(row))
    loss = torch.tensor(float(sum(losses)), dtype=torch.float32)
    grad = grads[0] if not batched else torch.stack(grads, dim=0)
    return loss, grad


def _cudaq_worker(config: dict[str, Any], queue: Any) -> None:
    try:
        cudaq = importlib.import_module("cudaq")
        _cudaq_set_target(cudaq, config.get("cudaq_target"))
        params_seed = _init_params(
            int(config["n_wires"]),
            int(config["layers"]),
            batch_size=int(config["batch_size"]),
            device="cpu",
        )
        kernel = _build_cudaq_kernel(cudaq, int(config["n_wires"]), int(config["layers"]))
        hamiltonian = _cudaq_spin_hamiltonian(cudaq, int(config["n_wires"]), str(config["observable"]))
        result = _time_value_and_grad(
            lambda: _cudaq_value_and_grad(
                cudaq=cudaq,
                kernel=kernel,
                hamiltonian=hamiltonian,
                params_seed=params_seed,
                method=str(config["gradient_method"]),
                forward_only=bool(config["forward_only"]),
            ),
            warmup=int(config["warmup"]),
            iters=int(config["iters"]),
            device="cpu",
            reference_avg_seconds=float(config["reference_avg_seconds"]),
            early_stop_speedup=config["early_stop_speedup"],
            max_seconds=config["max_seconds"],
        )
        result["grad_shape"] = list(result["grad"].shape)
        result["grad"] = result["grad"].detach().cpu().reshape(-1).tolist()
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})


def _time_cudaq_in_subprocess(
    config: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    process = context.Process(target=_cudaq_worker, args=(config, queue))
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, {
            "type": "TimeoutError",
            "message": f"CUDA-Q subprocess exceeded {float(timeout_seconds):.3f} seconds.",
            "timeout_seconds": float(timeout_seconds),
        }
    if queue.empty():
        return None, {
            "type": "RuntimeError",
            "message": f"CUDA-Q subprocess exited with code {process.exitcode} without returning a result.",
            "exitcode": process.exitcode,
        }
    payload = queue.get()
    if not payload.get("ok"):
        return None, payload.get("error")
    result = payload["result"]
    result["grad"] = torch.tensor(result["grad"], dtype=torch.float32).reshape(tuple(result["grad_shape"]))
    del result["grad_shape"]
    return result, None


def _comparison_payload(
    *,
    flagquantum_result: dict[str, Any],
    cudaq_result: dict[str, Any] | None,
    cudaq_error: dict[str, Any] | None,
    loss_atol: float,
    grad_atol: float,
    forward_only: bool,
) -> dict[str, Any]:
    if cudaq_error is not None:
        if cudaq_error.get("type") == "TimeoutError":
            lower_bound = float(cudaq_error["timeout_seconds"]) / float(flagquantum_result["avg_seconds"])
            return {
                "status": "timeout",
                "reason": cudaq_error,
                "speedup_lower_bound_flagquantum_over_cudaq": lower_bound,
            }
        return {"status": "unavailable", "reason": cudaq_error}
    assert cudaq_result is not None
    loss_abs_error = abs(float(flagquantum_result["loss"]) - float(cudaq_result["loss"]))
    grad_max_abs_error = 0.0 if forward_only else _max_abs_diff(flagquantum_result["grad"], cudaq_result["grad"])
    status = (
        "ok"
        if loss_abs_error <= float(loss_atol) and (forward_only or grad_max_abs_error <= float(grad_atol))
        else "precision_mismatch"
    )
    return {
        "status": status,
        "loss_abs_error": loss_abs_error,
        "grad_max_abs_error": None if forward_only else grad_max_abs_error,
        "loss_atol": float(loss_atol),
        "grad_atol": None if forward_only else float(grad_atol),
        "speedup_flagquantum_over_cudaq": cudaq_result["avg_seconds"] / flagquantum_result["avg_seconds"]
        if flagquantum_result["avg_seconds"] > 0
        else None,
        "slowdown_flagquantum_vs_cudaq": flagquantum_result["avg_seconds"] / cudaq_result["avg_seconds"]
        if cudaq_result["avg_seconds"] > 0
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=6)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--observable", choices=("z_sum", "ising"), default="ising")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("statevector", "mps", "tensor_network", "tn"), default="statevector")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--cudaq-target", default=None)
    parser.add_argument(
        "--cudaq-gradient-method",
        choices=("parameter-shift", "central-difference", "forward-difference", "finite-difference"),
        default="parameter-shift",
    )
    parser.add_argument("--cudaq-run-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cudaq-max-seconds", type=float, default=None)
    parser.add_argument("--cudaq-early-stop-speedup", type=float, default=30000.0)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-jax-jit", action="store_true")
    parser.add_argument("--jax-matmul-precision", default="highest")
    parser.add_argument("--jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    _configure_torch_precision(args.torch_matmul_precision)
    device = _device(args.device)
    fq_mode = "tensor_network" if args.mode == "tn" else args.mode
    params_seed = _init_params(args.n_wires, args.layers, batch_size=args.batch_size, device=device)
    example_params = params_seed[0].detach() if params_seed.ndim == 4 else params_seed.detach()
    hamiltonian = _ising_hamiltonian(args.n_wires) if args.observable == "ising" else None

    payload: dict[str, Any] = {
        "benchmark": "cudaq_backend_compare",
        "n_wires": int(args.n_wires),
        "layers": int(args.layers),
        "batch_size": int(args.batch_size),
        "observable": args.observable,
        "device": device,
        "mode": fq_mode,
        "forward_only": bool(args.forward_only),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": platform.platform(),
            "cudaq_installed": importlib.util.find_spec("cudaq") is not None,
        },
    }

    if importlib.util.find_spec("jax") is None:
        payload["flagquantum_jax"] = {"status": "unavailable", "reason": "jax is not installed"}
        payload["comparison"] = {"status": "unavailable", "reason": "jax is not installed"}
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)
        return

    kernel = fq.compile_quantum_kernel(
        lambda values: _build_circuit(values, device=device),
        example_params,
        backend="jax",
        interface="torch",
        mode=fq_mode,
        n_wires=args.n_wires,
        observable="z_sum" if args.observable == "z_sum" else "hamiltonian",
        hamiltonian=hamiltonian,
        jit=not args.no_jax_jit,
        matmul_precision=None if args.jax_matmul_precision == "default" else args.jax_matmul_precision,
        compute_dtype=args.jax_compute_dtype,
        max_bond=args.max_bond,
    )
    flagquantum_result = _time_value_and_grad(
        lambda: _fq_value_and_grad(kernel, params_seed, forward_only=bool(args.forward_only)),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    payload["flagquantum_jax"] = {
        "backend": "jax",
        "interface": "torch",
        "mode": fq_mode,
        "jit": not args.no_jax_jit,
        "avg_seconds": flagquantum_result["avg_seconds"],
        "loss": flagquantum_result["loss"],
        "kernel": kernel.summary(),
    }

    cudaq_result = None
    cudaq_error = None
    if importlib.util.find_spec("cudaq") is None:
        cudaq_error = {"type": "ModuleNotFoundError", "message": "cudaq is not installed"}
    else:
        cudaq_result, cudaq_error = _time_cudaq_in_subprocess(
            {
                "n_wires": int(args.n_wires),
                "layers": int(args.layers),
                "batch_size": int(args.batch_size),
                "observable": args.observable,
                "gradient_method": args.cudaq_gradient_method,
                "forward_only": bool(args.forward_only),
                "cudaq_target": args.cudaq_target,
                "warmup": int(args.warmup),
                "iters": int(args.iters),
                "reference_avg_seconds": float(flagquantum_result["avg_seconds"]),
                "early_stop_speedup": args.cudaq_early_stop_speedup,
                "max_seconds": args.cudaq_max_seconds,
            },
            timeout_seconds=float(args.cudaq_run_timeout_seconds),
        )

    if cudaq_result is None:
        payload["cudaq"] = {
            "backend": "cudaq",
            "target": args.cudaq_target,
            "gradient_method": args.cudaq_gradient_method,
            "status": "timeout" if cudaq_error and cudaq_error.get("type") == "TimeoutError" else "unavailable",
            "error": cudaq_error,
        }
    else:
        payload["cudaq"] = {
            "backend": "cudaq",
            "target": args.cudaq_target,
            "gradient_method": args.cudaq_gradient_method,
            "status": "ok",
            "avg_seconds": cudaq_result["avg_seconds"],
            "loss": cudaq_result["loss"],
            "completed_iters": cudaq_result["completed_iters"],
            "requested_iters": cudaq_result["requested_iters"],
            "early_stopped": cudaq_result["early_stopped"],
            "early_stop_reason": cudaq_result["early_stop_reason"],
        }

    payload["comparison"] = _comparison_payload(
        flagquantum_result=flagquantum_result,
        cudaq_result=cudaq_result,
        cudaq_error=cudaq_error,
        loss_atol=float(args.loss_atol),
        grad_atol=float(args.grad_atol),
        forward_only=bool(args.forward_only),
    )
    payload["conclusion"] = {
        "headline": (
            "FlagQuantum JAX matches CUDA-Q precision for this workload."
            if payload["comparison"]["status"] == "ok"
            else "CUDA-Q comparison did not produce a precision-aligned result for this workload."
        ),
        "recommended_claim": (
            f"FlagQuantum is {payload['comparison'].get('speedup_flagquantum_over_cudaq'):.2f}x faster than "
            f"CUDA-Q with loss_abs_error={payload['comparison'].get('loss_abs_error'):.3e}."
            if payload["comparison"].get("speedup_flagquantum_over_cudaq") is not None
            else None
        ),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json_output(args.json_output, text)


if __name__ == "__main__":
    main()
