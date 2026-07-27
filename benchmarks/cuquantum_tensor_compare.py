"""Direct tensor-network comparison between FlagQuantum and NVIDIA cuQuantum.

This benchmark intentionally bypasses PennyLane and calls cuQuantum directly.
It is meant to answer a narrower and cleaner question than the PennyLane
benchmarks: how does FlagQuantum's JAX quantum kernel compare with a direct
cuTensorNet-backed contraction primitive on the same circuit and observable?

Recommended commands
--------------------
GPU direct cuTensorNet TN forward comparison:
  python benchmarks/cuquantum_tensor_compare.py --device cuda --cuquantum-api cutensornet --flagquantum-mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output gpu_cutensornet_tn_vs_fq_jax_tn.json

GPU direct cuTensorNet TN vs FlagQuantum MPS:
  python benchmarks/cuquantum_tensor_compare.py --device cuda --cuquantum-api cutensornet --flagquantum-mode mps --max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output gpu_cutensornet_tn_vs_fq_jax_mps.json

GPU direct cuTensorNet TN with optional finite-difference gradient:
  python benchmarks/cuquantum_tensor_compare.py --device cuda --cuquantum-api cutensornet --flagquantum-mode tensor_network --n-wires 6 --layers 1 --batch-size 1 --observable z_sum --iters 10 --warmup 2 --include-cuquantum-gradient --json-output gpu_cutensornet_tn_grad_vs_fq_jax_tn.json

CPU/package availability smoke:
  python benchmarks/cuquantum_tensor_compare.py --device cuda --cuquantum-api cutensornet --n-wires 4 --layers 1 --batch-size 1 --observable z_sum --iters 1 --warmup 0

Notes
-----
This file explicitly prefers cuTensorNet's high-level Network API for the
NVIDIA path. cuQuantum 26.6+ exposes this as ``cuquantum.tensornet.Network``;
older builds exposed it as ``cuquantum.cutensornet.experimental.Network``.
``cuquantum.contract`` remains available through ``--cuquantum-api contract`` as
a compatibility fallback. If ``cuquantum`` or ``cupy`` is not installed, the
benchmark emits a JSON payload with ``status=unavailable`` instead of failing.
cuTensorNet does not provide the same PyTorch autograd interface as FlagQuantum
here; optional gradient comparison therefore uses central finite differences
and should be read as a reference, not as a native autograd performance result.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import string
import sys
import time
from pathlib import Path
from typing import Any

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


def _contract_with_cutensornet(network_cls: Any) -> Any:
    def contract(equation: str, *operands: Any) -> Any:
        network = network_cls(equation, *operands)
        try:
            if hasattr(network, "contract_path"):
                network.contract_path()
            result = network.contract()
            if isinstance(result, tuple):
                result = result[0]
            return result
        finally:
            if hasattr(network, "free"):
                network.free()
            elif hasattr(network, "close"):
                network.close()

    return contract


def _load_cutensornet_network() -> tuple[Any | None, str | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    candidates = (
        ("cuquantum.tensornet", "Network", "cuquantum.tensornet.Network"),
        (
            "cuquantum.cutensornet.experimental",
            "Network",
            "cuquantum.cutensornet.experimental.Network",
        ),
    )
    for module_name, class_name, backend_name in candidates:
        try:
            module = importlib.import_module(module_name)
            network_cls = getattr(module, class_name)
            return network_cls, backend_name, errors
        except Exception as exc:
            errors.append(
                {
                    "module": module_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    try:
        importlib.import_module("cuquantum.bindings.cutensornet")
        errors.append(
            {
                "module": "cuquantum.bindings.cutensornet",
                "type": "LowLevelBindingOnly",
                "message": (
                    "Low-level cuTensorNet bindings are installed, but this benchmark "
                    "uses the high-level Network API to keep the comparison concise."
                ),
            }
        )
    except Exception as exc:
        errors.append(
            {
                "module": "cuquantum.bindings.cutensornet",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
    return None, None, errors


def _load_contract_fallback(cuquantum: Any) -> tuple[Any | None, str | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if hasattr(cuquantum, "contract"):
        return cuquantum.contract, "cuquantum.contract", errors
    try:
        tensornet = importlib.import_module("cuquantum.tensornet")
        if hasattr(tensornet, "contract"):
            return tensornet.contract, "cuquantum.tensornet.contract", errors
        errors.append(
            {
                "module": "cuquantum.tensornet",
                "type": "AttributeError",
                "message": "cuquantum.tensornet.contract is not available",
            }
        )
    except Exception as exc:
        errors.append(
            {
                "module": "cuquantum.tensornet",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
    errors.append(
        {
            "module": "cuquantum",
            "type": "AttributeError",
            "message": "cuquantum.contract is not available",
        }
    )
    return None, None, errors


def _load_cuquantum(api: str) -> tuple[Any | None, Any | None, str | None, dict[str, Any] | None]:
    if importlib.util.find_spec("cupy") is None:
        return None, None, None, {"type": "ModuleNotFoundError", "message": "cupy is not installed"}
    if importlib.util.find_spec("cuquantum") is None:
        return None, None, None, {"type": "ModuleNotFoundError", "message": "cuquantum is not installed"}
    cupy = importlib.import_module("cupy")
    cuquantum = importlib.import_module("cuquantum")
    if api in {"cutensornet", "auto"}:
        network_cls, backend_name, network_errors = _load_cutensornet_network()
        if network_cls is not None and backend_name is not None:
            return cupy, _contract_with_cutensornet(network_cls), backend_name, None
        if api == "cutensornet":
            return cupy, None, None, {
                "type": "ModuleNotFoundError",
                "message": (
                    "cuTensorNet high-level Network API was not found. For cuQuantum 26.6+ "
                    "the expected path is cuquantum.tensornet.Network; for older builds it "
                    "is cuquantum.cutensornet.experimental.Network. Pass --cuquantum-api "
                    "contract to use the generic contract fallback."
                ),
                "attempts": network_errors,
            }
    contract, backend_name, contract_errors = _load_contract_fallback(cuquantum)
    if contract is None or backend_name is None:
        return cupy, None, None, {
            "type": "AttributeError",
            "message": "No cuQuantum contract fallback API is available.",
            "attempts": contract_errors,
        }
    return cupy, contract, backend_name, None


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


def _fq_forward(kernel: Any, params_seed: torch.Tensor) -> torch.Tensor:
    return kernel(params_seed.detach()).sum().detach()


def _fq_value_and_grad(kernel: Any, params_seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    loss = kernel(params).sum()
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_forward(fn: Any, *, warmup: int, iters: int, device: str) -> dict[str, Any]:
    for _ in range(int(warmup)):
        value = fn()
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
    _sync(device)
    start = time.perf_counter()
    value = None
    for _ in range(int(iters)):
        value = fn()
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
    _sync(device)
    elapsed = time.perf_counter() - start
    assert value is not None
    return {"avg_seconds": elapsed / max(1, int(iters)), "loss": float(value.detach().cpu())}


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


def _label_pool(n_wires: int) -> list[str]:
    if int(n_wires) > 48:
        raise ValueError("The direct cuQuantum einsum benchmark supports at most 48 wires.")
    return list(string.ascii_letters)


def _fresh_labels(current: list[str], count: int) -> list[str]:
    labels = [label for label in string.ascii_letters if label not in set(current)]
    if len(labels) < int(count):
        raise ValueError("Not enough einsum labels for this circuit width.")
    return labels[: int(count)]


def _apply_one_qubit_gate(
    state: Any,
    labels: list[str],
    gate: Any,
    wire: int,
    *,
    contract: Any,
) -> tuple[Any, list[str]]:
    new_label = _fresh_labels(labels, 1)[0]
    old_label = labels[int(wire)]
    out_labels = list(labels)
    out_labels[int(wire)] = new_label
    expr = f"{new_label}{old_label},{''.join(labels)}->{''.join(out_labels)}"
    return contract(expr, gate, state), out_labels


def _apply_two_qubit_gate(
    state: Any,
    labels: list[str],
    gate: Any,
    wire0: int,
    wire1: int,
    *,
    contract: Any,
) -> tuple[Any, list[str]]:
    new0, new1 = _fresh_labels(labels, 2)
    old0 = labels[int(wire0)]
    old1 = labels[int(wire1)]
    out_labels = list(labels)
    out_labels[int(wire0)] = new0
    out_labels[int(wire1)] = new1
    expr = f"{new0}{new1}{old0}{old1},{''.join(labels)}->{''.join(out_labels)}"
    return contract(expr, gate, state), out_labels


def _cu_rx(cp: Any, theta: Any, dtype: Any) -> Any:
    half = theta / 2.0
    c = cp.cos(half)
    s = cp.sin(half)
    gate = cp.empty((2, 2), dtype=dtype)
    gate[0, 0] = c
    gate[0, 1] = -1j * s
    gate[1, 0] = -1j * s
    gate[1, 1] = c
    return gate


def _cu_ry(cp: Any, theta: Any, dtype: Any) -> Any:
    half = theta / 2.0
    c = cp.cos(half)
    s = cp.sin(half)
    gate = cp.empty((2, 2), dtype=dtype)
    gate[0, 0] = c
    gate[0, 1] = -s
    gate[1, 0] = s
    gate[1, 1] = c
    return gate


def _cu_rz(cp: Any, theta: Any, dtype: Any) -> Any:
    half = theta / 2.0
    gate = cp.zeros((2, 2), dtype=dtype)
    gate[0, 0] = cp.exp(-1j * half)
    gate[1, 1] = cp.exp(1j * half)
    return gate


def _cu_cx(cp: Any, dtype: Any) -> Any:
    matrix = cp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=dtype,
    )
    return matrix.reshape(2, 2, 2, 2)


def _cu_rxx(cp: Any, theta: Any, dtype: Any) -> Any:
    half = theta / 2.0
    c = cp.cos(half)
    s = cp.sin(half)
    matrix = cp.zeros((4, 4), dtype=dtype)
    matrix[0, 0] = c
    matrix[0, 3] = -1j * s
    matrix[1, 1] = c
    matrix[1, 2] = -1j * s
    matrix[2, 1] = -1j * s
    matrix[2, 2] = c
    matrix[3, 0] = -1j * s
    matrix[3, 3] = c
    return matrix.reshape(2, 2, 2, 2)


def _cu_initial_state(cp: Any, n_wires: int, dtype: Any) -> Any:
    state = cp.zeros((2,) * int(n_wires), dtype=dtype)
    state[(0,) * int(n_wires)] = 1.0
    return state


def _cu_state_for_params(
    params: Any,
    *,
    n_wires: int,
    cp: Any,
    contract: Any,
    dtype: Any,
) -> Any:
    _label_pool(n_wires)
    labels = list(string.ascii_letters[: int(n_wires)])
    state = _cu_initial_state(cp, int(n_wires), dtype)
    layers = int(params.shape[0])
    for layer in range(layers):
        for wire in range(int(n_wires)):
            state, labels = _apply_one_qubit_gate(
                state,
                labels,
                _cu_rx(cp, params[layer, wire, 0], dtype),
                wire,
                contract=contract,
            )
            state, labels = _apply_one_qubit_gate(
                state,
                labels,
                _cu_ry(cp, params[layer, wire, 1], dtype),
                wire,
                contract=contract,
            )
            state, labels = _apply_one_qubit_gate(
                state,
                labels,
                _cu_rz(cp, params[layer, wire, 2], dtype),
                wire,
                contract=contract,
            )
        cx = _cu_cx(cp, dtype)
        for wire in range(int(n_wires) - 1):
            state, labels = _apply_two_qubit_gate(state, labels, cx, wire, wire + 1, contract=contract)
        if int(n_wires) > 2:
            state, labels = _apply_two_qubit_gate(
                state,
                labels,
                _cu_rxx(cp, params[layer, 0, 0] * 0.25, dtype),
                0,
                int(n_wires) - 1,
                contract=contract,
            )
    if labels != list(string.ascii_letters[: int(n_wires)]):
        expr = f"{''.join(labels)}->{''.join(string.ascii_letters[: int(n_wires)])}"
        state = contract(expr, state)
    return state.reshape(-1)


def _cu_z_expectation(cp: Any, state: Any, wire: int, n_wires: int) -> Any:
    indices = cp.arange(2**int(n_wires), dtype=cp.int64)
    bit = (indices >> (int(n_wires) - 1 - int(wire))) & 1
    signs = 1.0 - 2.0 * bit.astype(cp.float32)
    probs = cp.real(cp.conj(state) * state)
    return cp.sum(probs * signs)


def _cu_zz_expectation(cp: Any, state: Any, wire0: int, wire1: int, n_wires: int) -> Any:
    indices = cp.arange(2**int(n_wires), dtype=cp.int64)
    bit0 = (indices >> (int(n_wires) - 1 - int(wire0))) & 1
    bit1 = (indices >> (int(n_wires) - 1 - int(wire1))) & 1
    signs = 1.0 - 2.0 * (bit0 ^ bit1).astype(cp.float32)
    probs = cp.real(cp.conj(state) * state)
    return cp.sum(probs * signs)


def _cu_x_expectation(cp: Any, state: Any, wire: int, n_wires: int) -> Any:
    indices = cp.arange(2**int(n_wires), dtype=cp.int64)
    flipped = indices ^ (1 << (int(n_wires) - 1 - int(wire)))
    return cp.real(cp.sum(cp.conj(state) * state[flipped]))


def _cu_loss_for_row(
    row: Any,
    *,
    n_wires: int,
    observable: str,
    cp: Any,
    contract: Any,
    dtype: Any,
) -> Any:
    state = _cu_state_for_params(row, n_wires=n_wires, cp=cp, contract=contract, dtype=dtype)
    if observable == "z_sum":
        total = cp.asarray(0.0, dtype=cp.float32)
        for wire in range(int(n_wires)):
            total = total + _cu_z_expectation(cp, state, wire, int(n_wires))
        return total
    if observable == "ising":
        total = cp.asarray(0.0, dtype=cp.float32)
        for wire in range(int(n_wires) - 1):
            total = total + 0.7 * _cu_zz_expectation(cp, state, wire, wire + 1, int(n_wires))
        for wire in range(int(n_wires)):
            total = total - 0.2 * _cu_x_expectation(cp, state, wire, int(n_wires))
            total = total + 0.05 * _cu_z_expectation(cp, state, wire, int(n_wires))
        return total
    raise ValueError(f"Unsupported observable {observable!r}.")


def _torch_params_to_cupy(cp: Any, params: torch.Tensor, dtype: Any) -> Any:
    array = params.detach().cpu().numpy()
    real_dtype = cp.float64 if dtype == cp.complex128 else cp.float32
    return cp.asarray(array, dtype=real_dtype)


def _cu_forward_loss(
    params: torch.Tensor,
    *,
    n_wires: int,
    observable: str,
    cp: Any,
    contract: Any,
    dtype: Any,
) -> Any:
    values = _torch_params_to_cupy(cp, params, dtype)
    if values.ndim == 3:
        return _cu_loss_for_row(values, n_wires=n_wires, observable=observable, cp=cp, contract=contract, dtype=dtype)
    total = cp.asarray(0.0, dtype=cp.float32)
    for batch in range(int(values.shape[0])):
        total = total + _cu_loss_for_row(
            values[batch],
            n_wires=n_wires,
            observable=observable,
            cp=cp,
            contract=contract,
            dtype=dtype,
        )
    return total


def _time_cu_forward(
    params: torch.Tensor,
    *,
    n_wires: int,
    observable: str,
    cp: Any,
    contract: Any,
    dtype: Any,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        value = _cu_forward_loss(params, n_wires=n_wires, observable=observable, cp=cp, contract=contract, dtype=dtype)
        cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    value = None
    for _ in range(int(iters)):
        value = _cu_forward_loss(params, n_wires=n_wires, observable=observable, cp=cp, contract=contract, dtype=dtype)
        cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    assert value is not None
    return {"avg_seconds": elapsed / max(1, int(iters)), "loss": float(cp.asnumpy(value))}


def _cu_finite_difference_grad(
    params: torch.Tensor,
    *,
    n_wires: int,
    observable: str,
    cp: Any,
    contract: Any,
    dtype: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = torch.zeros_like(params.detach().cpu())
    flat = params.detach().cpu().reshape(-1)
    grad_flat = grad.reshape(-1)
    plus = flat.clone()
    minus = flat.clone()
    eps = 1e-3 if dtype == cp.complex64 else 1e-5
    for index in range(flat.numel()):
        plus[index] = flat[index] + eps
        minus[index] = flat[index] - eps
        plus_loss = _cu_forward_loss(
            plus.reshape_as(params).to(params.device),
            n_wires=n_wires,
            observable=observable,
            cp=cp,
            contract=contract,
            dtype=dtype,
        )
        minus_loss = _cu_forward_loss(
            minus.reshape_as(params).to(params.device),
            n_wires=n_wires,
            observable=observable,
            cp=cp,
            contract=contract,
            dtype=dtype,
        )
        grad_flat[index] = float(cp.asnumpy((plus_loss - minus_loss) / (2.0 * eps)))
        plus[index] = flat[index]
        minus[index] = flat[index]
    loss = _cu_forward_loss(params, n_wires=n_wires, observable=observable, cp=cp, contract=contract, dtype=dtype)
    return torch.tensor(float(cp.asnumpy(loss))), grad


def _time_cu_finite_difference(
    params: torch.Tensor,
    *,
    n_wires: int,
    observable: str,
    cp: Any,
    contract: Any,
    dtype: Any,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        _cu_finite_difference_grad(
            params,
            n_wires=n_wires,
            observable=observable,
            cp=cp,
            contract=contract,
            dtype=dtype,
        )
        cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = _cu_finite_difference_grad(
            params,
            n_wires=n_wires,
            observable=observable,
            cp=cp,
            contract=contract,
            dtype=dtype,
        )
        cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())


def _comparison_status(loss_abs_error: float, *, loss_atol: float) -> str:
    return "ok" if float(loss_abs_error) <= float(loss_atol) else "precision_mismatch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--observable", choices=("z_sum", "ising"), default="ising")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flagquantum-mode", choices=("mps", "tensor_network", "tn"), default="tensor_network")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-jax-jit", action="store_true")
    parser.add_argument("--jax-matmul-precision", default="highest")
    parser.add_argument("--jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--cuquantum-api", choices=("cutensornet", "contract", "auto"), default="cutensornet")
    parser.add_argument("--cuquantum-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument("--include-cuquantum-gradient", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    _configure_torch_precision(args.torch_matmul_precision)
    device = _device(args.device)
    fq_mode = "tensor_network" if args.flagquantum_mode == "tn" else args.flagquantum_mode
    payload: dict[str, Any] = {
        "benchmark": "cuquantum_tensor_compare",
        "n_wires": int(args.n_wires),
        "layers": int(args.layers),
        "batch_size": int(args.batch_size),
        "observable": args.observable,
        "device": device,
        "flagquantum_mode": fq_mode,
        "cuquantum_mode": "direct_exact_tensor_network",
        "cuquantum_requested_api": args.cuquantum_api,
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": platform.platform(),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None,
            "torch_cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32
            if torch.cuda.is_available()
            else None,
            "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
        },
    }

    if not device.startswith("cuda"):
        payload["comparison"] = {
            "status": "unavailable",
            "reason": {"type": "DeviceError", "message": "cuQuantum direct comparison requires --device cuda."},
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)
        return
    if not torch.cuda.is_available():
        payload["comparison"] = {
            "status": "unavailable",
            "reason": {
                "type": "DeviceError",
                "message": "Requested --device cuda, but this PyTorch installation does not have CUDA available.",
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)
        return

    cp, contract, cuquantum_backend, import_error = _load_cuquantum(args.cuquantum_api)
    if import_error is not None:
        payload["cuquantum"] = {
            "status": "unavailable",
            "requested_api": args.cuquantum_api,
            "error": import_error,
        }
        payload["comparison"] = {"status": "unavailable", "reason": import_error}
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)
        return
    assert cp is not None and contract is not None and cuquantum_backend is not None
    cu_dtype = cp.complex128 if args.cuquantum_compute_dtype == "complex128" else cp.complex64

    params_seed = _init_params(args.n_wires, args.layers, batch_size=args.batch_size, device=device)
    example_params = params_seed[0].detach() if params_seed.ndim == 4 else params_seed.detach()
    hamiltonian = _ising_hamiltonian(args.n_wires) if args.observable == "ising" else None

    if importlib.util.find_spec("jax") is None:
        payload["flagquantum_jax"] = {
            "backend": "jax",
            "status": "unavailable",
            "reason": "jax is not installed",
        }
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

    fq_forward = _time_forward(
        lambda: _fq_forward(kernel, params_seed),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    fq_value_grad = _time_value_and_grad(
        lambda: _fq_value_and_grad(kernel, params_seed),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    try:
        cu_forward = _time_cu_forward(
            params_seed,
            n_wires=args.n_wires,
            observable=args.observable,
            cp=cp,
            contract=contract,
            dtype=cu_dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    except Exception as exc:  # pragma: no cover - depends on external cuQuantum runtime
        payload["flagquantum_jax"] = {
            "backend": "jax",
            "interface": "torch",
            "mode": fq_mode,
            "jit": not args.no_jax_jit,
            "forward_avg_seconds": fq_forward["avg_seconds"],
            "value_grad_avg_seconds": fq_value_grad["avg_seconds"],
            "loss": fq_forward["loss"],
            "kernel": kernel.summary(),
        }
        payload["cuquantum"] = {
            "backend": cuquantum_backend,
            "mode": "direct_exact_tensor_network",
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        payload["comparison"] = {"status": "error", "reason": payload["cuquantum"]["error"]}
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)
        return

    loss_abs_error = abs(float(fq_forward["loss"]) - float(cu_forward["loss"]))
    status = _comparison_status(loss_abs_error, loss_atol=args.loss_atol)
    payload["flagquantum_jax"] = {
        "backend": "jax",
        "interface": "torch",
        "mode": fq_mode,
        "jit": not args.no_jax_jit,
        "forward_avg_seconds": fq_forward["avg_seconds"],
        "value_grad_avg_seconds": fq_value_grad["avg_seconds"],
        "loss": fq_forward["loss"],
        "kernel": kernel.summary(),
    }
    payload["cuquantum"] = {
        "backend": cuquantum_backend,
        "requested_api": args.cuquantum_api,
        "mode": "direct_exact_tensor_network",
        "compute_dtype": args.cuquantum_compute_dtype,
        "status": status,
        "forward_avg_seconds": cu_forward["avg_seconds"],
        "loss": cu_forward["loss"],
        "native_autograd": False,
    }
    payload["comparison"] = {
        "status": status,
        "loss_abs_error": loss_abs_error,
        "loss_atol": float(args.loss_atol),
        "speedup_flagquantum_forward_over_cuquantum": cu_forward["avg_seconds"] / fq_forward["avg_seconds"]
        if fq_forward["avg_seconds"] > 0
        else None,
        "speedup_flagquantum_value_grad_over_cuquantum_forward": cu_forward["avg_seconds"]
        / fq_value_grad["avg_seconds"]
        if fq_value_grad["avg_seconds"] > 0
        else None,
    }
    payload["conclusion"] = {
        "headline": (
            "FlagQuantum JAX matches direct cuQuantum tensor contraction forward precision."
            if status == "ok"
            else "FlagQuantum JAX and direct cuQuantum tensor contraction did not meet precision thresholds."
        ),
        "recommended_claim": (
            f"On {device}, {int(args.n_wires)} wires, {int(args.layers)} layers, "
            f"batch_size={int(args.batch_size)}, observable={args.observable}, "
            f"FlagQuantum {fq_mode} forward is "
            f"{payload['comparison']['speedup_flagquantum_forward_over_cuquantum']:.2f}x faster than "
            f"direct cuQuantum contraction with loss_abs_error={loss_abs_error:.3e}."
            if payload["comparison"]["speedup_flagquantum_forward_over_cuquantum"] is not None
            else None
        ),
        "gradient_note": "cuQuantum path is a direct contraction primitive here; native PyTorch autograd is provided by FlagQuantum, not by this cuQuantum path.",
    }

    if args.include_cuquantum_gradient:
        cu_grad = _time_cu_finite_difference(
            params_seed,
            n_wires=args.n_wires,
            observable=args.observable,
            cp=cp,
            contract=contract,
            dtype=cu_dtype,
            warmup=0,
            iters=max(1, min(1, int(args.iters))),
        )
        grad_max_abs_error = _max_abs_diff(fq_value_grad["grad"], cu_grad["grad"])
        grad_status = "ok" if grad_max_abs_error <= float(args.grad_atol) else "precision_mismatch"
        payload["cuquantum"]["finite_difference_gradient"] = {
            "status": grad_status,
            "avg_seconds": cu_grad["avg_seconds"],
            "loss": cu_grad["loss"],
            "grad_max_abs_error_vs_flagquantum_jax": grad_max_abs_error,
            "grad_atol": float(args.grad_atol),
            "note": "Central finite differences are used because this direct cuQuantum path is not a native PyTorch autograd backend.",
        }
        payload["comparison"]["grad_max_abs_error"] = grad_max_abs_error
        payload["comparison"]["grad_atol"] = float(args.grad_atol)
        payload["comparison"]["gradient_status"] = grad_status
        payload["comparison"]["speedup_flagquantum_value_grad_over_cuquantum_finite_difference"] = cu_grad[
            "avg_seconds"
        ] / fq_value_grad["avg_seconds"] if fq_value_grad["avg_seconds"] > 0 else None

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json_output(args.json_output, text)


if __name__ == "__main__":
    main()
