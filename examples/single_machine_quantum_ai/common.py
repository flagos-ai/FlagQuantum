"""Shared helpers for single-machine quantum AI examples."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import torch


def jax_available() -> tuple[bool, str | None]:
    try:
        import jax  # noqa: F401

        return True, None
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, str(exc)


def configure_jax_compilation_cache(cache_dir: str | None) -> dict[str, Any]:
    if not cache_dir:
        return {"enabled": False, "cache_dir": None}

    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(path))
    try:
        import jax

        jax.config.update("jax_compilation_cache_dir", str(path))
        for key, value in (
            ("jax_persistent_cache_min_compile_time_secs", 0),
            ("jax_persistent_cache_min_entry_size_bytes", -1),
        ):
            try:
                jax.config.update(key, value)
            except Exception:
                pass
        return {"enabled": True, "cache_dir": str(path)}
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"enabled": False, "cache_dir": str(path), "reason": str(exc)}


def sync_if_needed(device: str | torch.device) -> None:
    text = str(device)
    if text.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_value_and_grad(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    iters: int = 5,
    warmup: int = 2,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Measure scalar loss + gradient time for a PyTorch-facing function."""

    params = parameters.detach().clone().requires_grad_(True)
    for _ in range(max(0, int(warmup))):
        loss = loss_fn(params)
        grad = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)[0]
        del grad
    sync_if_needed(device)
    start = time.perf_counter()
    last_loss = None
    last_grad = None
    for _ in range(max(1, int(iters))):
        loss = loss_fn(params)
        grad = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)[0]
        last_loss = loss.detach()
        last_grad = grad.detach()
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "loss": float(last_loss.reshape(())) if last_loss is not None else None,
        "grad_norm": float(last_grad.norm()) if last_grad is not None else None,
    }


def time_forward(
    fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    iters: int = 5,
    warmup: int = 2,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Measure forward-only time for an inference-style function."""

    params = parameters.detach().clone()
    for _ in range(max(0, int(warmup))):
        fn(params)
    sync_if_needed(device)
    start = time.perf_counter()
    value = None
    for _ in range(max(1, int(iters))):
        value = fn(params)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "value_mean": float(value.detach().float().mean()) if value is not None else None,
    }


def dense_hamiltonian_matrix(hamiltonian: Any, n_wires: int) -> torch.Tensor:
    """Build a small dense Hamiltonian matrix for exact local references."""

    matrices = {
        "i": torch.eye(2, dtype=torch.complex64),
        "x": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64),
        "y": torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex64),
        "z": torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64),
    }
    total = torch.zeros((2**n_wires, 2**n_wires), dtype=torch.complex64)
    for term in hamiltonian.terms:
        ops = {wire: name for wire, name in term.ops}
        op = torch.ones((1, 1), dtype=torch.complex64)
        for wire in range(n_wires):
            op = torch.kron(op, matrices[ops.get(wire, "i")])
        total = total + torch.as_tensor(term.coefficient, dtype=torch.complex64) * op
    return total


def exact_ground_energy(hamiltonian: Any, n_wires: int) -> float:
    matrix = dense_hamiltonian_matrix(hamiltonian, n_wires)
    values = torch.linalg.eigvalsh(matrix)
    return float(values[0].real)


def speedup(numerator_seconds: float | None, denominator_seconds: float | None) -> float | None:
    if numerator_seconds is None or denominator_seconds is None or denominator_seconds <= 0:
        return None
    return float(numerator_seconds / denominator_seconds)


def fmt(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_kv(items: dict[str, Any], *, indent: int = 2) -> None:
    pad = " " * indent
    width = max((len(str(key)) for key in items), default=0)
    for key, value in items.items():
        print(f"{pad}{key:<{width}} : {fmt(value)}")


def print_table(rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    if not rows:
        print("  n/a")
        return
    widths = {
        column: max(len(column), *(len(fmt(row.get(column))) for row in rows))
        for column in columns
    }
    header = "  " + "  ".join(f"{column:<{widths[column]}}" for column in columns)
    print(header)
    print("  " + "  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  " + "  ".join(f"{fmt(row.get(column)):<{widths[column]}}" for column in columns))


def print_speed_compare(speed_compare: dict[str, Any]) -> None:
    print_section("Backend Speed Comparison")
    for name, result in speed_compare.items():
        if name.startswith("speedup"):
            continue
        if not isinstance(result, dict):
            continue
        print(f"  {name}")
        if result.get("status") == "unavailable":
            print(f"    status : unavailable")
            print(f"    reason : {result.get('reason')}")
            continue
        for key in ("avg_seconds", "loss", "grad_norm", "value_mean"):
            if key in result:
                print(f"    {key:<12}: {fmt(result[key])}")
    speed = speed_compare.get("speedup_jax_over_pytorch")
    if speed is not None:
        print(f"  {'speedup_jax_over_pytorch':<26}: {fmt(speed)}x")


def print_training_summary(
    *,
    title: str,
    example: str,
    metrics: dict[str, Any],
    speed_compare: dict[str, Any] | None = None,
    model_summary: dict[str, Any] | None = None,
) -> None:
    print_section(title)
    print_kv({"example": example, **metrics})
    if speed_compare is not None:
        print_speed_compare(speed_compare)
    if model_summary:
        print_section("Model Summary")
        compact = {
            key: model_summary[key]
            for key in (
                "state_mode",
                "n_wires",
                "batch_size",
                "max_bond",
                "mean_bond",
                "parameter_count",
                "truncation_error",
                "dtype",
                "device",
            )
            if key in model_summary
        }
        print_kv(compact)
