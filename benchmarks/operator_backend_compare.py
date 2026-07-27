"""Compare native PyTorch execution with optional FlagGems ATen replacement.

Repro commands
--------------
CPU smoke, statevector:

    python benchmarks/operator_backend_compare.py --device cpu --mode statevector --n-wires 6 --layers 2 --batch-size 2 --observable z_sum --iters 10 --warmup 3 --json-output operator_backend_statevector_cpu.json

CPU smoke, MPS:

    python benchmarks/operator_backend_compare.py --device cpu --mode mps --max-bond 16 --n-wires 8 --layers 2 --batch-size 2 --observable ising --iters 10 --warmup 3 --json-output operator_backend_mps_cpu.json

GPU / vendor accelerator run with strict FlagGems activation:

    python benchmarks/operator_backend_compare.py --device cuda --mode mps --max-bond 32 --n-wires 12 --layers 3 --batch-size 8 --observable ising --iters 50 --warmup 10 --flaggems-strict --json-output operator_backend_mps_cuda_flaggems.json

GPU run without automatic complex64 op filtering:

    python benchmarks/operator_backend_compare.py --device cuda --mode mps --max-bond 32 --n-wires 12 --layers 3 --batch-size 8 --observable ising --iters 50 --warmup 10 --flaggems-strict --no-flaggems-op-validation

Enable experimental ops after target-hardware preflight:

    python benchmarks/operator_backend_compare.py --device cuda --mode tensor_network --n-wires 8 --layers 2 --batch-size 4 --observable z_sum --iters 20 --warmup 5 --flaggems-include-experimental --json-output operator_backend_tn_cuda_experimental.json

Run against a local FlagGems source checkout:

    python benchmarks/operator_backend_compare.py --device cuda --mode mps --n-wires 8 --layers 2 --batch-size 4 --observable ising --flaggems-src ../FlagGems-master/src --flaggems-strict
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _init_params(
    n_wires: int,
    layers: int,
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    total = int(layers) * int(n_wires) * 3
    base = torch.linspace(-0.31, 0.43, steps=total, dtype=torch.float32, device=device).reshape(
        int(layers),
        int(n_wires),
        3,
    )
    if int(batch_size) <= 1:
        return base
    offsets = torch.linspace(-0.05, 0.05, steps=int(batch_size), dtype=torch.float32, device=device)
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


def _ising_terms(n_wires: int) -> tuple[tuple[float, tuple[tuple[int, str], ...]], ...]:
    terms: list[tuple[float, tuple[tuple[int, str], ...]]] = []
    for wire in range(int(n_wires) - 1):
        terms.append((0.7, ((wire, "z"), (wire + 1, "z"))))
    for wire in range(int(n_wires)):
        terms.append((-0.2, ((wire, "x"),)))
        terms.append((0.05, ((wire, "z"),)))
    return tuple(terms)


def _statevector_expectation_ps(
    state: torch.Tensor,
    n_wires: int,
    *,
    x: Sequence[int] = (),
    y: Sequence[int] = (),
    z: Sequence[int] = (),
) -> torch.Tensor:
    target = state
    circuit = fq.Circuit(int(n_wires), device=state.device, inputs=target)
    for wire in x:
        circuit.x(int(wire))
    for wire in y:
        circuit.y(int(wire))
    for wire in z:
        circuit.z(int(wire))
    transformed = circuit.state(refresh=True)
    return torch.real(torch.sum(torch.conj(state) * transformed, dim=-1))


def _expectation_ps(target: Any, n_wires: int, ops: tuple[tuple[int, str], ...]) -> torch.Tensor:
    x = tuple(wire for wire, name in ops if name == "x")
    y = tuple(wire for wire, name in ops if name == "y")
    z = tuple(wire for wire, name in ops if name == "z")
    if hasattr(target, "expectation_ps"):
        return target.expectation_ps(x=x, y=y, z=z)
    return _statevector_expectation_ps(target, n_wires, x=x, y=y, z=z)


def _target_from_circuit(
    circuit: fq.Circuit,
    *,
    mode: str,
    max_bond: int | None,
) -> Any:
    if mode == "statevector":
        return circuit
    if mode == "mps":
        return fq.run_mps(circuit, max_bond=max_bond)
    if mode == "tensor_network":
        return fq.run_tensor_network(circuit)
    raise ValueError("mode must be statevector, mps, or tensor_network.")


def _loss_single(
    params: torch.Tensor,
    *,
    mode: str,
    observable: str,
    device: str,
    max_bond: int | None,
) -> torch.Tensor:
    n_wires = int(params.shape[1])
    circuit = _build_circuit(params, device=device)
    target = _target_from_circuit(circuit, mode=mode, max_bond=max_bond)
    if observable == "z_sum":
        if hasattr(target, "expectation_z_sum"):
            return target.expectation_z_sum().sum()
        return target.expectation_z().sum()
    if observable == "ising":
        total = None
        for coeff, ops in _ising_terms(n_wires):
            value = coeff * _expectation_ps(target, n_wires, ops)
            total = value if total is None else total + value
        assert total is not None
        return total.sum()
    raise ValueError("observable must be z_sum or ising.")


def _loss(
    params: torch.Tensor,
    *,
    mode: str,
    observable: str,
    device: str,
    max_bond: int | None,
) -> torch.Tensor:
    if params.ndim == 3:
        return _loss_single(
            params,
            mode=mode,
            observable=observable,
            device=device,
            max_bond=max_bond,
        )
    losses = [
        _loss_single(row, mode=mode, observable=observable, device=device, max_bond=max_bond)
        for row in params
    ]
    return torch.stack(losses).sum()


def _value_and_grad(
    params_template: torch.Tensor,
    *,
    mode: str,
    observable: str,
    device: str,
    max_bond: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_template.detach().clone().requires_grad_(True)
    loss = _loss(
        params,
        mode=mode,
        observable=observable,
        device=device,
        max_bond=max_bond,
    )
    loss.backward()
    grad = params.grad.detach().clone()
    return loss.detach(), grad


def _time_value_and_grad(
    params_template: torch.Tensor,
    *,
    mode: str,
    observable: str,
    device: str,
    max_bond: int | None,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    loss = None
    grad = None
    for _ in range(int(warmup)):
        loss, grad = _value_and_grad(
            params_template,
            mode=mode,
            observable=observable,
            device=device,
            max_bond=max_bond,
        )
    _sync(device)
    start = time.perf_counter()
    for _ in range(int(iters)):
        loss, grad = _value_and_grad(
            params_template,
            mode=mode,
            observable=observable,
            device=device,
            max_bond=max_bond,
        )
    _sync(device)
    seconds = (time.perf_counter() - start) / max(1, int(iters))
    assert loss is not None and grad is not None
    return {
        "avg_seconds": seconds,
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad.detach()).cpu()),
        "grad": grad.detach().cpu(),
    }


def _comparison(reference: dict[str, Any], candidate: dict[str, Any], *, atol: float) -> dict[str, Any]:
    grad_error = float(torch.max(torch.abs(reference["grad"] - candidate["grad"])).item())
    loss_error = abs(float(reference["loss"]) - float(candidate["loss"]))
    speedup = float(reference["avg_seconds"]) / float(candidate["avg_seconds"])
    status = "ok" if grad_error <= float(atol) and loss_error <= float(atol) else "precision_mismatch"
    return {
        "status": status,
        "loss_abs_error": loss_error,
        "grad_max_abs_error": grad_error,
        "atol": float(atol),
        "speedup_flaggems_over_pytorch": speedup,
        "slowdown_flaggems_vs_pytorch": 1.0 / speedup if speedup else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="statevector", choices=["statevector", "mps", "tensor_network"])
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--observable", default="z_sum", choices=["z_sum", "ising"])
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--flaggems-include", default=None)
    parser.add_argument("--flaggems-include-experimental", action="store_true")
    parser.add_argument("--flaggems-src", default=None, help="Optional path to FlagGems src for source-tree benchmarking.")
    parser.add_argument("--flaggems-strict", action="store_true")
    parser.add_argument(
        "--no-flaggems-op-validation",
        action="store_true",
        help="Disable per-op forward/backward validation before enabling FlagGems.",
    )
    parser.add_argument(
        "--flaggems-validation-dtype",
        default="complex64",
        help="Dtype used by the per-op FlagGems smoke tests.",
    )
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    if args.flaggems_src:
        src = Path(args.flaggems_src).resolve()
        sys.path.insert(0, str(src))
        existing_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = (
            str(src) if not existing_pythonpath else str(src) + os.pathsep + existing_pythonpath
        )

    device = _device(args.device)
    params = _init_params(
        args.n_wires,
        args.layers,
        batch_size=args.batch_size,
        device=device,
    )
    include = args.flaggems_include
    preflight = fq.flaggems_preflight(
        requested_ops=include.split(",") if include else None,
        include_experimental=args.flaggems_include_experimental,
    )
    planned_flaggems_ops = tuple(preflight["replacement_plan"]["enabled_ops"])
    flaggems_validation = None
    flaggems_include_for_run = include
    if not args.no_flaggems_op_validation and preflight["availability"]["available"]:
        validation = fq.validate_flaggems_ops(
            planned_flaggems_ops,
            device=device,
            dtype=args.flaggems_validation_dtype,
            include_experimental=args.flaggems_include_experimental,
        )
        flaggems_validation = validation.summary()
        flaggems_include_for_run = list(validation.usable_ops)

    pytorch_result = _time_value_and_grad(
        params,
        mode=args.mode,
        observable=args.observable,
        device=device,
        max_bond=args.max_bond,
        iters=args.iters,
        warmup=args.warmup,
    )
    pytorch_public = {key: value for key, value in pytorch_result.items() if key != "grad"}

    flaggems_result = None
    flaggems_error = None
    flaggems_session = None
    if flaggems_validation is not None and not flaggems_include_for_run:
        flaggems_error = {
            "type": "OperatorValidationError",
            "message": "No FlagGems operator passed the target device/dtype validation.",
        }
    else:
        try:
            with fq.operator_backend(
                "flaggems",
                include=flaggems_include_for_run,
                include_experimental=args.flaggems_include_experimental,
                strict=args.flaggems_strict,
            ) as session:
                flaggems_session = session.summary()
                if session.enabled:
                    flaggems_result = _time_value_and_grad(
                        params,
                        mode=args.mode,
                        observable=args.observable,
                        device=device,
                        max_bond=args.max_bond,
                        iters=args.iters,
                        warmup=args.warmup,
                    )
        except BaseException as exc:  # pragma: no cover - target hardware dependent.
            flaggems_error = {"type": type(exc).__name__, "message": str(exc)}

    if flaggems_result is None:
        comparison = {
            "status": "unavailable",
            "reason": flaggems_error
            or (
                flaggems_session["availability"]["reason"]
                if flaggems_session is not None
                else preflight["availability"]["reason"]
            ),
        }
        flaggems_public = {
            "status": "unavailable",
            "error": flaggems_error,
            "session": flaggems_session,
        }
    else:
        comparison = _comparison(pytorch_result, flaggems_result, atol=args.atol)
        flaggems_public = {key: value for key, value in flaggems_result.items() if key != "grad"}
        flaggems_public["status"] = comparison["status"]
        flaggems_public["session"] = flaggems_session

    result = {
        "benchmark": "operator_backend_compare",
        "purpose": "Native PyTorch ATen operator replacement only; this is not a distributed scalability claim.",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "configuration": {
            "device": device,
            "mode": args.mode,
            "n_wires": args.n_wires,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "observable": args.observable,
            "max_bond": args.max_bond if args.mode == "mps" else None,
            "iters": args.iters,
            "warmup": args.warmup,
            "flaggems_op_validation": not args.no_flaggems_op_validation,
            "flaggems_validation_dtype": args.flaggems_validation_dtype,
        },
        "operator_audit": preflight,
        "operator_validation": flaggems_validation,
        "flagquantum_pytorch": pytorch_public,
        "flagquantum_pytorch_flaggems": flaggems_public,
        "comparison": comparison,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
