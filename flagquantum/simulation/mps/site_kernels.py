"""Audited eager and compiled kernels for MPS site computation."""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch.profiler import record_function


@dataclass
class SiteKernelStats:
    ry_bucket_calls: int = 0
    rxx_bucket_calls: int = 0
    transfer_calls: int = 0
    compiled_calls: int = 0
    compile_seconds: float = 0.0
    dynamo_graphs: int = 0
    triton_kernels: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    eager_fallbacks: int = 0
    prewarm_compiles: int = 0
    warm_execution_seconds: float = 0.0
    nonfinite_compiled_outputs: int = 0
    nonfinite_eager_recoveries: int = 0


@dataclass(frozen=True)
class SiteKernelCachePolicy:
    """Bounded compile policy for recurring MPS bond shapes."""

    bond_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    max_entries: int = 32
    max_accounted_input_bytes: int = 512 * 1024 * 1024
    backend: str | None = None
    mode: str | None = "default"
    unsupported_shape: str = "eager"
    policy_id: str = "mps_dynamic_bond_v1"

    def __post_init__(self) -> None:
        if not self.bond_buckets or any(value <= 0 for value in self.bond_buckets):
            raise ValueError("bond_buckets must contain positive dimensions")
        if tuple(sorted(set(self.bond_buckets))) != self.bond_buckets:
            raise ValueError("bond_buckets must be sorted and unique")
        if self.max_entries <= 0 or self.max_accounted_input_bytes <= 0:
            raise ValueError("site-kernel cache limits must be positive")
        if self.unsupported_shape not in {"eager", "error"}:
            raise ValueError("unsupported_shape must be eager or error")


@dataclass(frozen=True)
class SiteKernelBucket:
    """Synthetic hot bucket declaration used by :func:`prewarm_site_kernel_buckets`."""

    kind: str
    bucket_size: int = 1
    batch_size: int = 1
    left_bond: int = 1
    middle_bond: int = 1
    right_bond: int = 1
    channels: int = 1
    z: bool = False


_STATS = SiteKernelStats()
_POLICY = SiteKernelCachePolicy()
_CACHE: OrderedDict[tuple[Any, ...], tuple[Callable[..., torch.Tensor], int]] = (
    OrderedDict()
)
_CACHE_ACCOUNTED_BYTES = 0
_EVENTS: list[dict[str, Any]] = []
_MAX_EVENTS = 4096


def reset_site_kernel_stats(*, clear_cache: bool = False) -> None:
    global _STATS
    _STATS = SiteKernelStats()
    _EVENTS.clear()
    if clear_cache:
        clear_site_kernel_cache()


def site_kernel_stats() -> dict[str, int | float]:
    return {
        **asdict(_STATS),
        "cache_entries": len(_CACHE),
        "cache_accounted_input_bytes": _CACHE_ACCOUNTED_BYTES,
        "cache_max_entries": _POLICY.max_entries,
        "cache_max_accounted_input_bytes": _POLICY.max_accounted_input_bytes,
    }


def site_kernel_cache_events() -> tuple[Mapping[str, Any], ...]:
    """Return JSON-safe cold/warm/cache/fallback events in execution order."""

    return tuple(dict(event) for event in _EVENTS)


def site_kernel_cache_policy() -> SiteKernelCachePolicy:
    return _POLICY


def configure_site_kernel_cache(policy: SiteKernelCachePolicy) -> None:
    """Install a policy, explicitly invalidating graphs from the old policy."""

    global _POLICY
    if not isinstance(policy, SiteKernelCachePolicy):
        raise TypeError("policy must be SiteKernelCachePolicy")
    if policy != _POLICY:
        clear_site_kernel_cache()
        _POLICY = policy


def clear_site_kernel_cache() -> None:
    global _CACHE_ACCOUNTED_BYTES
    _CACHE.clear()
    _CACHE_ACCOUNTED_BYTES = 0


def _log_event(
    event: str, *, kind: str, key: tuple[Any, ...] | None = None, **fields: Any
) -> None:
    item = {"event": event, "kind": kind, "time_ns": time.time_ns(), **fields}
    if key is not None:
        item["key"] = repr(key)
    _EVENTS.append(item)
    if len(_EVENTS) > _MAX_EVENTS:
        del _EVENTS[: len(_EVENTS) - _MAX_EVENTS]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def site_kernel_bucket_capacity(*sample_items: torch.Tensor) -> int:
    """Maximum equal-shape items whose packed inputs fit the cache byte bound."""

    if not sample_items:
        raise ValueError("site-kernel bucket capacity requires sample tensors")
    item_bytes = sum(_tensor_bytes(tensor) for tensor in sample_items)
    if item_bytes <= 0:
        raise ValueError("site-kernel bucket sample tensors must be non-empty")
    return max(1, _POLICY.max_accounted_input_bytes // item_bytes)


def _cache_key(kind: str, args: Sequence[torch.Tensor]) -> tuple[Any, ...]:
    devices = tuple((arg.device.type, arg.device.index) for arg in args)
    return (
        kind,
        tuple(tuple(int(value) for value in arg.shape) for arg in args),
        tuple(str(arg.dtype) for arg in args),
        devices,
        _POLICY,
    )


def _supported_shape(kind: str, args: Sequence[torch.Tensor]) -> bool:
    if sum(_tensor_bytes(arg) for arg in args) > _POLICY.max_accounted_input_bytes:
        return False
    allowed = set(_POLICY.bond_buckets)
    if kind == "ry":
        return int(args[0].shape[2]) in allowed and int(args[0].shape[4]) in allowed
    if kind == "rxx_contraction":
        return all(
            int(value) in allowed
            for value in (args[0].shape[2], args[0].shape[4], args[1].shape[4])
        )
    if kind.startswith("transfer_") and kind != "transfer_observable_channels":
        return int(args[1].shape[1]) in allowed and int(args[1].shape[3]) in allowed
    if kind == "transfer_observable_channels":
        return int(args[1].shape[1]) in allowed and int(args[1].shape[3]) in allowed
    return False


def _compile(eager: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    if _POLICY.backend is not None:
        kwargs["backend"] = _POLICY.backend
    if _POLICY.mode is not None and _POLICY.backend != "eager":
        kwargs["mode"] = _POLICY.mode
    return torch.compile(eager, **kwargs)


def _recompile_limit_context() -> AbstractContextManager[object]:
    """Keep Dynamo's code-object limit consistent with the bounded shape cache."""

    try:
        from torch._dynamo import config

        # One logical LRU entry can create distinct code-object specializations
        # for a microbatch leading dimension and for real/complex views.  Keep
        # Dynamo bounded, but provision those variants instead of failing at
        # the 33rd valid ISSUE-108 RXX shape while our own 32-entry LRU remains
        # within policy.
        specialization_limit = 4 * _POLICY.max_entries
        context = config.patch(
            recompile_limit=max(int(config.recompile_limit), specialization_limit),
            accumulated_recompile_limit=max(
                int(config.accumulated_recompile_limit), specialization_limit
            ),
        )
        if not isinstance(context, AbstractContextManager):
            raise TypeError("Dynamo configuration patch must be a context manager.")
        return context
    except (ImportError, AttributeError):
        return nullcontext()


def _insert_cache(
    key: tuple[Any, ...],
    compiled: Callable[..., torch.Tensor],
    accounted: int,
    kind: str,
) -> None:
    global _CACHE_ACCOUNTED_BYTES
    while _CACHE and (
        len(_CACHE) >= _POLICY.max_entries
        or _CACHE_ACCOUNTED_BYTES + accounted > _POLICY.max_accounted_input_bytes
    ):
        evicted_key, (_, evicted_bytes) = _CACHE.popitem(last=False)
        _CACHE_ACCOUNTED_BYTES -= evicted_bytes
        _STATS.cache_evictions += 1
        _log_event(
            "eviction",
            kind=str(evicted_key[0]),
            key=evicted_key,
            accounted_input_bytes=evicted_bytes,
        )
    _CACHE[key] = (compiled, accounted)
    _CACHE_ACCOUNTED_BYTES += accounted


def _metrics_count() -> int:
    try:
        from torch._inductor import metrics

        return int(metrics.generated_kernel_count)
    except (ImportError, AttributeError):
        return 0


def _run(
    eager: Callable[..., torch.Tensor],
    args: Sequence[torch.Tensor],
    *,
    kind: str,
    compiled: bool,
    prewarm: bool = False,
) -> torch.Tensor:
    if not compiled:
        return eager(*args)
    key = _cache_key(kind, args)
    if not _supported_shape(kind, args):
        _STATS.eager_fallbacks += 1
        _log_event(
            "unsupported_shape_fallback",
            kind=kind,
            key=key,
            policy=_POLICY.unsupported_shape,
        )
        if _POLICY.unsupported_shape == "error":
            raise RuntimeError(
                f"unsupported compiled site-kernel shape for {kind}: {key[1]}"
            )
        return eager(*args)
    cached = _CACHE.get(key)
    cold = cached is None
    if cached is None:
        _STATS.cache_misses += 1
        _log_event("miss", kind=kind, key=key)
        kernel = _compile(eager)
    else:
        _STATS.cache_hits += 1
        _CACHE.move_to_end(key)
        kernel = cached[0]
        _log_event("hit", kind=kind, key=key)
    before = _metrics_count()
    started = time.perf_counter()
    try:
        # torch.compile tracks recompiles per Python code object, while this
        # module intentionally owns several exact-shape callables for that code
        # object. Keep both bounded policies aligned so 9+ valid bond buckets do
        # not fail before the LRU cache reaches its declared maximum.
        with _recompile_limit_context():
            result = kernel(*args)
        if args[0].is_cuda:
            torch.accelerator.synchronize(args[0].device)
    except Exception as error:
        raise RuntimeError(
            f"compiled site-sharded {kind} kernel failed closed: {error}"
        ) from error
    elapsed = time.perf_counter() - started
    after = _metrics_count()
    _STATS.compiled_calls += 1
    if cold:
        _STATS.compile_seconds += elapsed
        _STATS.dynamo_graphs += 1
        _STATS.triton_kernels += max(0, after - before)
        accounted = sum(_tensor_bytes(arg) for arg in args)
        _insert_cache(key, kernel, accounted, kind)
        if prewarm:
            _STATS.prewarm_compiles += 1
        _log_event(
            "compile", kind=kind, key=key, setup_seconds=elapsed, prewarm=prewarm
        )
    else:
        _STATS.warm_execution_seconds += elapsed
        _log_event(
            "warm_execution",
            kind=kind,
            key=key,
            execution_seconds=elapsed,
            prewarm=prewarm,
        )
    return result


def _ry_bucket_real(tensors: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    # tensors [sites,batch,left,2,right,real_imag]
    # matrices [sites,batch,2,2,real_imag]
    tr, ti = tensors[..., 0], tensors[..., 1]
    mr, mi = matrices[..., 0], matrices[..., 1]
    real = torch.einsum("kbpq,kblqr->kblpr", mr, tr) - torch.einsum(
        "kbpq,kblqr->kblpr", mi, ti
    )
    imag = torch.einsum("kbpq,kblqr->kblpr", mr, ti) + torch.einsum(
        "kbpq,kblqr->kblpr", mi, tr
    )
    return torch.stack((real, imag), dim=-1)


def _recover_nonfinite_compiled_output(
    output: torch.Tensor,
    *,
    eager: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    kind: str,
) -> torch.Tensor:
    if bool(torch.isfinite(output).all()):
        return output
    _STATS.nonfinite_compiled_outputs += 1
    recovered = eager(*inputs)
    if not bool(torch.isfinite(recovered).all()):
        finite_inputs = tuple(bool(torch.isfinite(item).all()) for item in inputs)
        raise RuntimeError(
            f"MPS {kind} produced non-finite values in both compiled and eager "
            f"kernels; finite_inputs={finite_inputs}"
        )
    _STATS.nonfinite_eager_recoveries += 1
    return recovered


def apply_ry_bucket(
    tensors: torch.Tensor, matrices: torch.Tensor, *, compiled: bool
) -> torch.Tensor:
    """Apply equal-shape rank-local RY matrices without changing MPS gauge."""

    _STATS.ry_bucket_calls += 1
    if not tensors.is_complex() or not matrices.is_complex():
        raise TypeError("site-sharded RY bucket requires complex tensors and matrices")
    if matrices.ndim == 3:
        matrices = matrices.unsqueeze(1).expand(-1, tensors.shape[1], -1, -1)
    real_tensors = torch.view_as_real(tensors)
    real_matrices = torch.view_as_real(matrices)
    real_inputs = (real_tensors, real_matrices)
    output = _run(
        _ry_bucket_real,
        real_inputs,
        kind="ry",
        compiled=compiled,
    )
    if compiled:
        input_norm = torch.linalg.vector_norm(real_tensors, dim=(-4, -3, -2, -1))
        output_norm = torch.linalg.vector_norm(output, dim=(-4, -3, -2, -1))
        if not bool(torch.allclose(input_norm, output_norm, rtol=2e-5, atol=2e-6)):
            output = torch.full_like(output, float("nan"))
        output = _recover_nonfinite_compiled_output(
            output,
            eager=_ry_bucket_real,
            inputs=real_inputs,
            kind="single-site contraction",
        )
    return torch.view_as_complex(output.contiguous())


def _two_site_gate_contraction_real(
    left: torch.Tensor, right: torch.Tensor, matrices: torch.Tensor
) -> torch.Tensor:
    lr, li = left[..., 0], left[..., 1]
    rr, ri = right[..., 0], right[..., 1]
    real = torch.einsum("kblsm,kbmtr->kblstr", lr, rr) - torch.einsum(
        "kblsm,kbmtr->kblstr", li, ri
    )
    imag = torch.einsum("kblsm,kbmtr->kblstr", lr, ri) + torch.einsum(
        "kblsm,kbmtr->kblstr", li, rr
    )
    theta = torch.stack((real, imag), dim=-1)
    bonds, batch, left_dim, _, _, right_dim, _ = theta.shape
    flat = theta.reshape(bonds, batch, left_dim, 4, right_dim, 2)
    tr, ti = flat[..., 0], flat[..., 1]
    mr, mi = matrices[..., 0], matrices[..., 1]
    out_r = torch.einsum("kbij,kbljr->kblir", mr, tr) - torch.einsum(
        "kbij,kbljr->kblir", mi, ti
    )
    out_i = torch.einsum("kbij,kbljr->kblir", mr, ti) + torch.einsum(
        "kbij,kbljr->kblir", mi, tr
    )
    return torch.stack((out_r, out_i), dim=-1).reshape(
        bonds, batch, left_dim * 2, 2 * right_dim, 2
    )


def apply_two_site_gate_contraction_bucket(
    left: torch.Tensor, right: torch.Tensor, matrices: torch.Tensor, *, compiled: bool
) -> torch.Tensor:
    """Contract equal-shape local two-site gates."""

    _STATS.rxx_bucket_calls += 1
    if matrices.ndim == 3:
        matrices = matrices.unsqueeze(1).expand(-1, left.shape[1], -1, -1)
    real_inputs = (
        torch.view_as_real(left),
        torch.view_as_real(right),
        torch.view_as_real(matrices),
    )
    output = _run(
        _two_site_gate_contraction_real,
        real_inputs,
        kind="rxx_contraction",
        compiled=compiled,
    )
    if compiled:
        left_norm = torch.linalg.vector_norm(real_inputs[0], dim=(-4, -3, -2, -1))
        right_norm = torch.linalg.vector_norm(real_inputs[1], dim=(-4, -3, -2, -1))
        output_norm = torch.linalg.vector_norm(output, dim=(-3, -2, -1))
        norm_bound = left_norm * right_norm
        if not bool(torch.all(output_norm <= norm_bound * 1.00002 + 2e-6)):
            output = torch.full_like(output, float("nan"))
        output = _recover_nonfinite_compiled_output(
            output,
            eager=_two_site_gate_contraction_real,
            inputs=real_inputs,
            kind="two-site contraction",
        )
    return torch.view_as_complex(output.contiguous())


def _transfer_real(
    env: torch.Tensor, tensor: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    er, ei = env[..., 0], env[..., 1]
    ar, ai = tensor[..., 0], tensor[..., 1]
    kr = ar * signs.reshape(1, 1, 2, 1)
    ki = ai * signs.reshape(1, 1, 2, 1)
    cr = torch.einsum("bij,bipr->bjpr", er, ar) + torch.einsum("bij,bipr->bjpr", ei, ai)
    ci = -torch.einsum("bij,bipr->bjpr", er, ai) + torch.einsum(
        "bij,bipr->bjpr", ei, ar
    )
    return torch.stack(
        (
            torch.einsum("bjpr,bjps->brs", cr, kr)
            - torch.einsum("bjpr,bjps->brs", ci, ki),
            torch.einsum("bjpr,bjps->brs", cr, ki)
            + torch.einsum("bjpr,bjps->brs", ci, kr),
        ),
        dim=-1,
    )


def environment_transfer(
    env: torch.Tensor, tensor: torch.Tensor, *, z: bool, compiled: bool
) -> torch.Tensor:
    """Real-channel identity/Z transfer used inside one rank's environment scan."""

    _STATS.transfer_calls += 1
    signs = tensor.real.new_tensor((1.0, -1.0) if z else (1.0, 1.0))
    with record_function("flagquantum::mps::environment_transfer"):
        output = _run(
            _transfer_real,
            (torch.view_as_real(env), torch.view_as_real(tensor), signs),
            kind="transfer_z" if z else "transfer_i",
            compiled=compiled,
        )
    return torch.view_as_complex(output.contiguous())


def environment_transfer_channels(
    channels: torch.Tensor, tensor: torch.Tensor, *, compiled: bool
) -> torch.Tensor:
    """Propagate many observable channels through one site in one launch."""

    _STATS.transfer_calls += 1
    with record_function("flagquantum::mps::environment_transfer_channels"):
        return _run(
            _transfer_channels_complex,
            (channels, tensor),
            kind="transfer_observable_channels",
            compiled=compiled,
        )


def _transfer_channels_complex(
    channels: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("tbij,bipr,bjps->tbrs", channels, tensor.conj(), tensor)


def prewarm_site_kernel_buckets(
    buckets: Iterable[SiteKernelBucket],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
) -> tuple[Mapping[str, Any], ...]:
    """Compile declared hot buckets before a timed training interval."""

    resolved = torch.device(device)
    if dtype not in {torch.complex64, torch.complex128}:
        raise TypeError("site-kernel prewarm dtype must be complex64 or complex128")
    event_start = len(_EVENTS)
    for bucket in buckets:
        if (
            min(
                bucket.bucket_size,
                bucket.batch_size,
                bucket.left_bond,
                bucket.middle_bond,
                bucket.right_bond,
                bucket.channels,
            )
            <= 0
        ):
            raise ValueError("prewarm bucket dimensions must be positive")
        if bucket.kind == "ry":
            tensors = torch.zeros(
                bucket.bucket_size,
                bucket.batch_size,
                bucket.left_bond,
                2,
                bucket.right_bond,
                dtype=dtype,
                device=resolved,
            )
            matrices = torch.eye(2, dtype=dtype, device=resolved).expand(
                bucket.bucket_size, bucket.batch_size, 2, 2
            )
            real_args = (torch.view_as_real(tensors), torch.view_as_real(matrices))
            _run(_ry_bucket_real, real_args, kind="ry", compiled=True, prewarm=True)
        elif bucket.kind == "rxx_contraction":
            left = torch.zeros(
                bucket.bucket_size,
                bucket.batch_size,
                bucket.left_bond,
                2,
                bucket.middle_bond,
                dtype=dtype,
                device=resolved,
            )
            right = torch.zeros(
                bucket.bucket_size,
                bucket.batch_size,
                bucket.middle_bond,
                2,
                bucket.right_bond,
                dtype=dtype,
                device=resolved,
            )
            matrices = torch.eye(4, dtype=dtype, device=resolved).expand(
                bucket.bucket_size, bucket.batch_size, 4, 4
            )
            _run(
                _two_site_gate_contraction_real,
                tuple(torch.view_as_real(item) for item in (left, right, matrices)),
                kind="rxx_contraction",
                compiled=True,
                prewarm=True,
            )
        elif bucket.kind in {"transfer_i", "transfer_z"}:
            env = torch.zeros(
                bucket.batch_size,
                bucket.left_bond,
                bucket.left_bond,
                dtype=dtype,
                device=resolved,
            )
            tensor = torch.zeros(
                bucket.batch_size,
                bucket.left_bond,
                2,
                bucket.right_bond,
                dtype=dtype,
                device=resolved,
            )
            signs = tensor.real.new_tensor(
                (1.0, -1.0) if bucket.kind == "transfer_z" else (1.0, 1.0)
            )
            _run(
                _transfer_real,
                (torch.view_as_real(env), torch.view_as_real(tensor), signs),
                kind=bucket.kind,
                compiled=True,
                prewarm=True,
            )
        elif bucket.kind == "transfer_observable_channels":
            channels = torch.zeros(
                bucket.channels,
                bucket.batch_size,
                bucket.left_bond,
                bucket.left_bond,
                dtype=dtype,
                device=resolved,
            )
            tensor = torch.zeros(
                bucket.batch_size,
                bucket.left_bond,
                2,
                bucket.right_bond,
                dtype=dtype,
                device=resolved,
            )
            _run(
                _transfer_channels_complex,
                (channels, tensor),
                kind=bucket.kind,
                compiled=True,
                prewarm=True,
            )
        else:
            raise ValueError(f"unsupported prewarm site-kernel kind: {bucket.kind}")
    return tuple(dict(event) for event in _EVENTS[event_start:])


def require_warm_step_regression(
    compiled_seconds: Sequence[float],
    eager_seconds: Sequence[float],
    *,
    maximum_ratio: float = 1.05,
) -> Mapping[str, float | bool]:
    """Reject a compiled warm path that regresses beyond the declared budget."""

    if not compiled_seconds or not eager_seconds:
        raise ValueError("warm regression gate requires compiled and eager samples")
    if maximum_ratio <= 0:
        raise ValueError("maximum_ratio must be positive")
    compiled_mean = sum(float(value) for value in compiled_seconds) / len(
        compiled_seconds
    )
    eager_mean = sum(float(value) for value in eager_seconds) / len(eager_seconds)
    if eager_mean <= 0:
        raise ValueError("eager warm samples must have a positive mean")
    ratio = compiled_mean / eager_mean
    if ratio > maximum_ratio:
        raise RuntimeError(
            f"compiled warm-step regression {ratio:.6f} exceeds limit {maximum_ratio:.6f}"
        )
    return {
        "passed": True,
        "compiled_mean_seconds": compiled_mean,
        "eager_mean_seconds": eager_mean,
        "compiled_to_eager_ratio": ratio,
        "maximum_ratio": maximum_ratio,
    }


__all__ = (
    "SiteKernelBucket",
    "SiteKernelCachePolicy",
    "apply_ry_bucket",
    "apply_two_site_gate_contraction_bucket",
    "clear_site_kernel_cache",
    "configure_site_kernel_cache",
    "environment_transfer",
    "environment_transfer_channels",
    "prewarm_site_kernel_buckets",
    "require_warm_step_regression",
    "reset_site_kernel_stats",
    "site_kernel_bucket_capacity",
    "site_kernel_cache_events",
    "site_kernel_cache_policy",
    "site_kernel_stats",
)
