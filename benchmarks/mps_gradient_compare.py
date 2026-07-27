"""Compare FlagQuantum distributed MPS gradients with tensorcircuit-ng MPS.

Run on a GPU node, for example:

    torchrun --standalone --nproc_per_node=2 benchmarks/mps_gradient_compare.py \
        --n-wires 8 --layers 2 --max-bond 32 --iters 20 --warmup 3 --tc-backend jax

    torchrun --standalone --nproc_per_node=2 benchmarks/mps_gradient_compare.py \
        --n-wires 8 \
        --layers 2 \
        --max-bond 32 \
        --iters 20 \
        --warmup 3 \
        --device cuda \
        --tc-backend pytorch \
        --tc-gradient-method auto \
        --no-tc-jit \
        --include-tc-jax-baseline

The script prints a JSON payload on rank 0 with loss/gradient precision and
timing data. FlagQuantum runs on every torch.distributed rank. The
tensorcircuit-ng reference runs on rank 0 because it is not a torch.distributed
MPS executor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _init_params(n_wires: int, layers: int, *, device: str) -> torch.Tensor:
    values = torch.linspace(
        -0.37,
        0.41,
        steps=int(n_wires) * int(layers) * 3,
        dtype=torch.float32,
        device=device,
    )
    return values.reshape(int(layers), int(n_wires), 3)


def _observable_wires(n_wires: int) -> tuple[int, ...]:
    return tuple(range(int(n_wires)))


def _z_training_loss(result: Any, n_wires: int) -> torch.Tensor:
    wires = _observable_wires(n_wires)
    if hasattr(result, "expectation_z_sum"):
        return result.expectation_z_sum(wires).sum()
    return result.expectation_z(wires).sum()


def _build_flagquantum_circuit(params: torch.Tensor, *, device: str) -> fq.Circuit:
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


def _flagquantum_value_and_grad(
    params_seed: torch.Tensor,
    *,
    device: str,
    max_bond: int | None,
    boundary_transport: str,
    gradient_fastpath: bool,
    fuse_single_qubit: bool,
    dense_observable_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    circuit = _build_flagquantum_circuit(params, device=device)
    result = circuit.run(
        mode="distributed_mps",
        world_size=_world_size(),
        distributed_executor="torch" if _world_size() > 1 else "auto",
        device=device,
        max_bond=max_bond,
        boundary_transport=boundary_transport,
        gradient_fastpath=gradient_fastpath,
        fuse_single_qubit=fuse_single_qubit,
        dense_observable_wires=dense_observable_wires,
    )
    loss = _z_training_loss(result, int(params.shape[1]))
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _flagquantum_local_mps_value_and_grad(
    params_seed: torch.Tensor,
    *,
    device: str,
    max_bond: int | None,
    fuse_single_qubit: bool,
    dense_observable_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    circuit = _build_flagquantum_circuit(params, device=device)
    result = fq.run_mps(
        circuit,
        max_bond=max_bond,
        fuse_single_qubit=fuse_single_qubit,
        dense_observable_wires=dense_observable_wires,
    )
    loss = _z_training_loss(result, int(params.shape[1]))
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_flagquantum(
    params_seed: torch.Tensor,
    *,
    device: str,
    max_bond: int | None,
    boundary_transport: str,
    gradient_fastpath: bool,
    fuse_single_qubit: bool,
    dense_observable_wires: int,
    compile_step: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    training_step = None
    if compile_step:
        training_step = fq.compile_mps_training_step(
            lambda values: _build_flagquantum_circuit(values, device=device),
            params_seed,
            compile=True,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            fallback=True,
            max_bond=max_bond,
            fuse_single_qubit=fuse_single_qubit,
            dense_observable_wires=dense_observable_wires,
        )
    for _ in range(int(warmup)):
        if training_step is None:
            _flagquantum_value_and_grad(
                params_seed,
                device=device,
                max_bond=max_bond,
                boundary_transport=boundary_transport,
                gradient_fastpath=gradient_fastpath,
                fuse_single_qubit=fuse_single_qubit,
                dense_observable_wires=dense_observable_wires,
            )
        else:
            training_step(params_seed)
    _sync(device)
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        if training_step is None:
            loss, grad = _flagquantum_value_and_grad(
                params_seed,
                device=device,
                max_bond=max_bond,
                boundary_transport=boundary_transport,
                gradient_fastpath=gradient_fastpath,
                fuse_single_qubit=fuse_single_qubit,
                dense_observable_wires=dense_observable_wires,
            )
        else:
            loss, grad = training_step(params_seed)
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    summary = {
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
        "avg_seconds": elapsed / max(1, int(iters)),
    }
    if training_step is not None:
        summary["compiled_training_step"] = training_step.summary()
    return summary


def _time_flagquantum_local_mps(
    params_seed: torch.Tensor,
    *,
    device: str,
    max_bond: int | None,
    fuse_single_qubit: bool,
    dense_observable_wires: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        _flagquantum_local_mps_value_and_grad(
            params_seed,
            device=device,
            max_bond=max_bond,
            fuse_single_qubit=fuse_single_qubit,
            dense_observable_wires=dense_observable_wires,
        )
    _sync(device)
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = _flagquantum_local_mps_value_and_grad(
            params_seed,
            device=device,
            max_bond=max_bond,
            fuse_single_qubit=fuse_single_qubit,
            dense_observable_wires=dense_observable_wires,
        )
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
        "avg_seconds": elapsed / max(1, int(iters)),
    }


def _time_flagquantum_jax_torch_bridge(
    params_seed: torch.Tensor,
    *,
    device: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    kernel = fq.compile_quantum_kernel(
        lambda values: _build_flagquantum_circuit(values, device=device),
        params_seed,
        backend="jax",
        interface="torch",
        mode="statevector",
        n_wires=int(params_seed.shape[1]),
        observable="z_sum",
        jit=True,
    )
    for _ in range(int(warmup)):
        params = params_seed.detach().clone().requires_grad_(True)
        loss = kernel(params)
        loss.backward()
    _sync(device)
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        params = params_seed.detach().clone().requires_grad_(True)
        loss = kernel(params)
        loss.backward()
        grad = params.grad if params.grad is not None else torch.zeros_like(params)
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
        "avg_seconds": elapsed / max(1, int(iters)),
        "summary": kernel.summary(),
    }


def _tensorcircuit_value_and_grad(
    params_np: Any,
    *,
    n_wires: int,
    layers: int,
    max_bond: int | None,
    backend: str,
    jit: bool,
    gradient_method: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    import numpy as np
    import tensorcircuit as tc

    with tc.runtime_backend(backend):

        def loss_fn(params: Any) -> Any:
            circuit = tc.MPSCircuit(int(n_wires))
            if max_bond is not None:
                circuit.set_split_rules({"max_singular_values": int(max_bond)})
            for layer in range(int(layers)):
                for wire in range(int(n_wires)):
                    circuit.rx(wire, theta=params[layer, wire, 0])
                    circuit.ry(wire, theta=params[layer, wire, 1])
                    circuit.rz(wire, theta=params[layer, wire, 2])
                for wire in range(int(n_wires) - 1):
                    circuit.cx(wire, wire + 1)
            value = 0.0
            for wire in _observable_wires(n_wires):
                value = value + circuit.expectation_ps(z=[wire])
            return tc.backend.real(value)

        def loss_numpy(params_value: Any) -> float:
            params_tensor = tc.backend.convert_to_tensor(np.asarray(params_value, dtype=np.float32))
            return float(tc.backend.numpy(loss_fn(params_tensor)))

        if gradient_method == "parameter-shift":
            params = np.asarray(params_np, dtype=np.float32)

            def parameter_shift_eval(params_value: Any) -> tuple[float, Any]:
                loss_value = loss_numpy(params_value)
                flat = np.asarray(params_value, dtype=np.float32).reshape(-1)
                grad_flat = np.zeros_like(flat)
                shift = np.float32(np.pi / 2)
                for index in range(flat.size):
                    plus = flat.copy()
                    minus = flat.copy()
                    plus[index] += shift
                    minus[index] -= shift
                    plus_loss = loss_numpy(plus.reshape(params.shape))
                    minus_loss = loss_numpy(minus.reshape(params.shape))
                    grad_flat[index] = 0.5 * (plus_loss - minus_loss)
                return loss_value, grad_flat.reshape(params.shape)

            for _ in range(int(warmup)):
                loss, grad = parameter_shift_eval(params)
            start = time.perf_counter()
            loss = None
            grad = None
            for _ in range(int(iters)):
                loss, grad = parameter_shift_eval(params)
            elapsed = time.perf_counter() - start
            assert loss is not None and grad is not None
            return {
                "loss": float(loss),
                "grad": np.asarray(grad, dtype=np.float32),
                "avg_seconds": elapsed / max(1, int(iters)),
                "gradient_method": "parameter-shift",
            }

        value_and_grad = tc.backend.value_and_grad(loss_fn)
        if jit:
            value_and_grad = tc.backend.jit(value_and_grad)
        params = tc.backend.convert_to_tensor(np.asarray(params_np, dtype=np.float32))
        for _ in range(int(warmup)):
            loss, grad = value_and_grad(params)
            tc.backend.numpy(loss)
            tc.backend.numpy(grad)
        start = time.perf_counter()
        loss = None
        grad = None
        for _ in range(int(iters)):
            loss, grad = value_and_grad(params)
            tc.backend.numpy(loss)
            tc.backend.numpy(grad)
        elapsed = time.perf_counter() - start
        assert loss is not None and grad is not None
        return {
            "loss": float(tc.backend.numpy(loss)),
            "grad": np.asarray(tc.backend.numpy(grad), dtype=np.float32),
            "avg_seconds": elapsed / max(1, int(iters)),
            "gradient_method": "autograd",
        }


def _tc_backend_available(backend: str) -> tuple[bool, str]:
    if backend == "jax":
        return importlib.util.find_spec("jax") is not None, "jax"
    if backend == "tensorflow":
        return importlib.util.find_spec("tensorflow") is not None, "tensorflow"
    if backend == "pytorch":
        return importlib.util.find_spec("torch") is not None, "torch"
    if backend == "numpy":
        return True, "numpy"
    return False, backend


def _resolve_tc_backend(requested: str) -> tuple[str, str | None]:
    if requested != "auto":
        available, dependency = _tc_backend_available(requested)
        if available:
            return requested, None
        raise RuntimeError(
            f"tensorcircuit-ng backend {requested!r} requires {dependency!r}, but it is not installed. "
            "Install the missing dependency or rerun with --tc-backend auto/pytorch/numpy."
        )
    for candidate in ("jax", "tensorflow", "pytorch", "numpy"):
        available, _ = _tc_backend_available(candidate)
        if available:
            fallback = None if candidate == "jax" else "jax backend unavailable; selected first available backend"
            return candidate, fallback
    raise RuntimeError("No tensorcircuit-ng backend is available.")


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())


def _comparison_payload(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_name: str,
    candidate_name: str,
    loss_atol: float,
    grad_atol: float,
) -> dict[str, Any]:
    reference_grad = reference["grad"]
    candidate_grad = torch.as_tensor(candidate["grad"], dtype=reference_grad.dtype)
    loss_abs_error = abs(float(reference["loss"]) - float(candidate["loss"]))
    grad_max_abs_error = _max_abs_diff(reference_grad, candidate_grad)
    status = (
        "ok"
        if loss_abs_error <= float(loss_atol) and grad_max_abs_error <= float(grad_atol)
        else "precision_mismatch"
    )
    return {
        "status": status,
        "loss_abs_error": loss_abs_error,
        "grad_max_abs_error": grad_max_abs_error,
        "loss_atol": float(loss_atol),
        "grad_atol": float(grad_atol),
        f"speedup_{reference_name}_over_{candidate_name}": candidate["avg_seconds"] / reference["avg_seconds"]
        if reference["avg_seconds"] > 0
        else None,
        f"slowdown_{reference_name}_vs_{candidate_name}": reference["avg_seconds"] / candidate["avg_seconds"]
        if candidate["avg_seconds"] > 0
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--boundary-transport", default="auto")
    parser.add_argument("--no-fq-gradient-fastpath", action="store_true")
    parser.add_argument("--no-fq-single-qubit-fusion", action="store_true")
    parser.add_argument("--fq-dense-observable-wires", type=int, default=12)
    parser.add_argument("--fq-compile", action="store_true")
    parser.add_argument("--fq-compile-backend", default="")
    parser.add_argument("--fq-compile-mode", default="reduce-overhead")
    parser.add_argument("--tc-backend", default="auto", choices=("auto", "jax", "tensorflow", "pytorch", "numpy"))
    parser.add_argument(
        "--tc-gradient-method",
        default="auto",
        choices=("auto", "autograd", "parameter-shift"),
        help="Gradient method for tensorcircuit-ng. auto tries autograd first, then parameter-shift.",
    )
    parser.add_argument("--no-tc-jit", action="store_true")
    parser.add_argument(
        "--include-tc-jax-baseline",
        action="store_true",
        help="Also run tensorcircuit-ng on its JAX/JIT path to show best-path precision alignment.",
    )
    parser.add_argument(
        "--include-fq-jax-torch-bridge",
        action="store_true",
        help="Also run FlagQuantum's JAX quantum kernel through the PyTorch autograd bridge.",
    )
    parser.add_argument("--skip-fq-local-mps", action="store_true")
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument("--json-output", default="mps_benchmark")
    args = parser.parse_args()

    device = _device(args.device)
    tc_backend = None
    tc_backend_note = None
    if _rank() == 0:
        tc_backend, tc_backend_note = _resolve_tc_backend(args.tc_backend)
    params_seed = _init_params(args.n_wires, args.layers, device=device)
    fq_result = _time_flagquantum(
        params_seed,
        device=device,
        max_bond=args.max_bond,
        boundary_transport=args.boundary_transport,
        gradient_fastpath=not args.no_fq_gradient_fastpath,
        fuse_single_qubit=not args.no_fq_single_qubit_fusion,
        dense_observable_wires=args.fq_dense_observable_wires,
        compile_step=args.fq_compile,
        
        compile_backend=args.fq_compile_backend or None,
        compile_mode=args.fq_compile_mode or None,
        warmup=args.warmup,
        iters=args.iters,
    )
    compiled_training_step = fq_result.get("compiled_training_step")

    payload: dict[str, Any] = {
        "benchmark": "mps_gradient_compare",
        "rank": _rank(),
        "world_size": _world_size(),
        "n_wires": args.n_wires,
        "layers": args.layers,
        "max_bond": args.max_bond,
        "iters": args.iters,
        "warmup": args.warmup,
        "flagquantum": {
            "avg_seconds": fq_result["avg_seconds"],
            "gradient_fastpath": not args.no_fq_gradient_fastpath,
            "single_qubit_fusion": not args.no_fq_single_qubit_fusion,
            "dense_observable_wires": args.fq_dense_observable_wires,
            "compiled_training_step": compiled_training_step,
            "loss": fq_result["loss"],
        },
        "flagquantum_distributed_mps": {
            "avg_seconds": fq_result["avg_seconds"],
            "gradient_fastpath": not args.no_fq_gradient_fastpath,
            "single_qubit_fusion": not args.no_fq_single_qubit_fusion,
            "dense_observable_wires": args.fq_dense_observable_wires,
            "compiled_training_step": compiled_training_step,
            "loss": fq_result["loss"],
        },
    }

    if _rank() == 0:
        fq_local_result = None
        if not args.skip_fq_local_mps:
            fq_local_result = _time_flagquantum_local_mps(
                params_seed,
                device=device,
                max_bond=args.max_bond,
                fuse_single_qubit=not args.no_fq_single_qubit_fusion,
                dense_observable_wires=args.fq_dense_observable_wires,
                warmup=args.warmup,
                iters=args.iters,
            )
            payload["flagquantum_local_mps"] = {
                "avg_seconds": fq_local_result["avg_seconds"],
                "single_qubit_fusion": not args.no_fq_single_qubit_fusion,
                "dense_observable_wires": args.fq_dense_observable_wires,
                "loss": fq_local_result["loss"],
            }
            payload["local_vs_distributed_mps_comparison"] = _comparison_payload(
                fq_result,
                fq_local_result,
                reference_name="flagquantum_distributed_mps",
                candidate_name="flagquantum_local_mps",
                loss_atol=args.loss_atol,
                grad_atol=args.grad_atol,
            )
        if args.include_fq_jax_torch_bridge:
            try:
                fq_bridge_result = _time_flagquantum_jax_torch_bridge(
                    params_seed,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except Exception as exc:  # noqa: BLE001 - optional bridge benchmark should preserve main result.
                payload["flagquantum_jax_torch_bridge"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            else:
                bridge_comparison = _comparison_payload(
                    fq_result,
                    fq_bridge_result,
                    reference_name="flagquantum",
                    candidate_name="flagquantum_jax_torch_bridge",
                    loss_atol=args.loss_atol,
                    grad_atol=args.grad_atol,
                )
                payload["flagquantum_jax_torch_bridge"] = {
                    "status": bridge_comparison["status"],
                    "avg_seconds": fq_bridge_result["avg_seconds"],
                    "loss": fq_bridge_result["loss"],
                    "summary": fq_bridge_result["summary"],
                }
                payload["jax_torch_bridge_comparison"] = bridge_comparison
        assert tc_backend is not None
        tc_error = None
        tc_gradient_method = args.tc_gradient_method
        if tc_gradient_method == "auto":
            tc_gradient_method = "autograd"
        try:
            tc_result = _tensorcircuit_value_and_grad(
                params_seed.detach().cpu().numpy(),
                n_wires=args.n_wires,
                layers=args.layers,
                max_bond=args.max_bond,
                backend=tc_backend,
                jit=not args.no_tc_jit,
                gradient_method=tc_gradient_method,
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should preserve partial results.
            tc_error = exc
            if args.tc_gradient_method == "auto":
                try:
                    tc_result = _tensorcircuit_value_and_grad(
                        params_seed.detach().cpu().numpy(),
                        n_wires=args.n_wires,
                        layers=args.layers,
                        max_bond=args.max_bond,
                        backend=tc_backend,
                        jit=False,
                        gradient_method="parameter-shift",
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                except Exception as fallback_exc:  # noqa: BLE001 - benchmark should preserve partial results.
                    payload["tensorcircuit_ng"] = {
                        "backend": tc_backend,
                        "requested_backend": args.tc_backend,
                        "backend_note": tc_backend_note,
                        "jit": not args.no_tc_jit,
                        "status": "failed",
                        "gradient_method": args.tc_gradient_method,
                        "error_type": type(fallback_exc).__name__,
                        "error": str(fallback_exc),
                        "autograd_error_type": type(tc_error).__name__,
                        "autograd_error": str(tc_error),
                    }
                    payload["comparison"] = {
                        "status": "unavailable",
                        "reason": "tensorcircuit-ng backend failed before producing gradients",
                    }
                else:
                    tc_result["autograd_fallback_error_type"] = type(tc_error).__name__
                    tc_result["autograd_fallback_error"] = str(tc_error)
            else:
                payload["tensorcircuit_ng"] = {
                    "backend": tc_backend,
                    "requested_backend": args.tc_backend,
                    "backend_note": tc_backend_note,
                    "jit": not args.no_tc_jit,
                    "status": "failed",
                    "gradient_method": tc_gradient_method,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                payload["comparison"] = {
                    "status": "unavailable",
                    "reason": "tensorcircuit-ng backend failed before producing gradients",
                }
        else:
            tc_result["autograd_fallback_error_type"] = None
            tc_result["autograd_fallback_error"] = None

        if "comparison" not in payload:
            comparison = _comparison_payload(
                fq_result,
                tc_result,
                reference_name="flagquantum",
                candidate_name="tensorcircuit_ng",
                loss_atol=args.loss_atol,
                grad_atol=args.grad_atol,
            )
            payload["tensorcircuit_ng"] = {
                "backend": tc_backend,
                "requested_backend": args.tc_backend,
                "backend_note": tc_backend_note,
                "jit": (not args.no_tc_jit) if tc_result["gradient_method"] == "autograd" else False,
                "status": comparison["status"],
                "gradient_method": tc_result["gradient_method"],
                "avg_seconds": tc_result["avg_seconds"],
                "loss": tc_result["loss"],
                "autograd_fallback_error_type": tc_result["autograd_fallback_error_type"],
                "autograd_fallback_error": tc_result["autograd_fallback_error"],
            }
            payload["comparison"] = comparison
            if fq_local_result is not None:
                payload["local_mps_comparison"] = _comparison_payload(
                    fq_local_result,
                    tc_result,
                    reference_name="flagquantum_local_mps",
                    candidate_name="tensorcircuit_ng",
                    loss_atol=args.loss_atol,
                    grad_atol=args.grad_atol,
                )
        if args.include_tc_jax_baseline and tc_backend != "jax":
            available, dependency = _tc_backend_available("jax")
            if not available:
                payload["tensorcircuit_ng_jax_baseline"] = {
                    "status": "unavailable",
                    "reason": f"jax backend requires {dependency!r}, but it is not installed",
                }
            else:
                try:
                    jax_result = _tensorcircuit_value_and_grad(
                        params_seed.detach().cpu().numpy(),
                        n_wires=args.n_wires,
                        layers=args.layers,
                        max_bond=args.max_bond,
                        backend="jax",
                        jit=True,
                        gradient_method="autograd",
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark should preserve partial results.
                    payload["tensorcircuit_ng_jax_baseline"] = {
                        "status": "failed",
                        "backend": "jax",
                        "jit": True,
                        "gradient_method": "autograd",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                else:
                    jax_comparison = _comparison_payload(
                        fq_result,
                        jax_result,
                        reference_name="flagquantum",
                        candidate_name="tensorcircuit_ng_jax",
                        loss_atol=args.loss_atol,
                        grad_atol=args.grad_atol,
                    )
                    payload["tensorcircuit_ng_jax_baseline"] = {
                        "status": jax_comparison["status"],
                        "backend": "jax",
                        "jit": True,
                        "gradient_method": "autograd",
                        "avg_seconds": jax_result["avg_seconds"],
                        "loss": jax_result["loss"],
                    }
                    payload["jax_baseline_comparison"] = jax_comparison
                    if fq_local_result is not None:
                        payload["local_mps_jax_baseline_comparison"] = _comparison_payload(
                            fq_local_result,
                            jax_result,
                            reference_name="flagquantum_local_mps",
                            candidate_name="tensorcircuit_ng_jax",
                            loss_atol=args.loss_atol,
                            grad_atol=args.grad_atol,
                        )
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            Path(args.json_output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
