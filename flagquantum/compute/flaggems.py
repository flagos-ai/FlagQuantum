"""FlagGems operators for directly controlled PyTorch devices.

FlagQuantum owns the quantum execution semantics.  Operator backends such as
FlagGems sit below the native PyTorch path and may replace selected ATen
operators without changing the Circuit/MPS/TN API.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

from .registry import get_platform_runtime

FLAGGEMS_SAFE_OPS: tuple[str, ...] = (
    "abs",
    "add",
    "addmm",
    "bmm",
    "dot",
    "gather",
    "index_select",
    "kron",
    "mm",
    "mul",
    "outer",
    "sum",
    "vdot",
    "where_self",
    "zeros",
    "zeros_like",
)

FLAGGEMS_EXPERIMENTAL_OPS: tuple[str, ...] = (
    "einsum",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "svd",
)

FLAGGEMS_NON_OPERATOR_REQUIREMENTS: tuple[str, ...] = (
    "jax.pmap",
    "jax.shard_map",
    "torch.distributed.all_to_all",
    "torch.distributed.batch_isend_irecv",
    "torch.distributed.collectives",
    "mps_site_sharding",
    "tn_slice_reduction",
    "statevector_rank_transport",
)

FLAGGEMS_FEATURE_REQUESTS: Mapping[str, str] = {
    "einsum": "Need target-hardware validation for complex high-rank TN/MPS contractions and backward stability.",
    "svd": "Need stable complex SVD/truncation backward for MPS before default enablement.",
    "scatter": "Need complex dtype and autograd validation for sharded-state index transport patterns.",
    "scatter_": "Need in-place scatter safety validation for autograd and aliasing.",
    "scatter_add_": "Need reduction determinism and complex gradient validation.",
    "scatter_reduce": "Need deterministic reduction semantics for quantum gradient checks.",
    "scatter_reduce_": "Need in-place reduction safety validation for autograd and aliasing.",
    "jax.pmap": "Out of scope for PyTorch ATen replacement; keep in FlagQuantum JAX runtime.",
    "jax.shard_map": "Out of scope for PyTorch ATen replacement; keep in FlagQuantum JAX runtime.",
    "torch.distributed.all_to_all": "Out of scope for local ATen kernels; keep in FlagQuantum distributed runtime/NCCL.",
    "torch.distributed.batch_isend_irecv": "Out of scope for local ATen kernels; keep in FlagQuantum distributed runtime/NCCL.",
    "torch.distributed.collectives": "Out of scope for local ATen kernels; keep in FlagQuantum distributed runtime/NCCL.",
    "mps_site_sharding": "Requires FlagQuantum distributed MPS ownership and boundary-gradient protocol.",
    "tn_slice_reduction": "Requires FlagQuantum distributed TN slicing/reduction protocol.",
    "statevector_rank_transport": "Requires FlagQuantum distributed statevector transport protocol.",
}

FLAGGEMS_OP_ALIASES: Mapping[str, str] = {
    "where.self": "where_self",
    "where.self_out": "where_self_out",
    "scatter.src": "scatter",
    "scatter_.src": "scatter_",
    "scatter.reduce": "scatter_reduce",
    "scatter_.reduce": "scatter_reduce_",
}

FLAGGEMS_NATIVE_PYTORCH_HOTSPOTS: Mapping[str, tuple[str, ...]] = {
    "statevector": (
        "bmm",
        "mm",
        "sum",
        "abs",
        "mul",
        "gather",
        "index_select",
    ),
    "mps": (
        "einsum",
        "bmm",
        "mm",
        "sum",
        "abs",
        "mul",
        "svd",
    ),
    "tensor_network": (
        "einsum",
        "sum",
        "mul",
        "add",
        "scatter",
    ),
    "density_matrix": (
        "bmm",
        "mm",
        "sum",
        "abs",
        "mul",
        "kron",
    ),
}


@dataclass(frozen=True)
class OperatorBackendAvailability:
    """Runtime availability of an optional operator backend."""

    name: str
    available: bool
    reason: str | None = None
    module_version: str | None = None
    device: str | None = None
    vendor: str | None = None
    registered_op_count: int | None = None
    registered_keys: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "module_version": self.module_version,
            "device": self.device,
            "vendor": self.vendor,
            "registered_op_count": self.registered_op_count,
            "registered_keys_sample": self.registered_keys[:32],
        }


@dataclass(frozen=True)
class OperatorReplacementPlan:
    """Classification of FlagQuantum PyTorch ops for an operator backend."""

    backend: str
    requested_ops: tuple[str, ...]
    runtime_replaceable_ops: tuple[str, ...]
    catalog_safe_ops: tuple[str, ...]
    experimental_ops: tuple[str, ...]
    unavailable_ops: tuple[str, ...]
    non_operator_requirements: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def enabled_ops(self) -> tuple[str, ...]:
        return self.runtime_replaceable_ops or self.catalog_safe_ops

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_ops": self.requested_ops,
            "runtime_replaceable_ops": self.runtime_replaceable_ops,
            "catalog_safe_ops": self.catalog_safe_ops,
            "experimental_ops": self.experimental_ops,
            "unavailable_ops": self.unavailable_ops,
            "non_operator_requirements": self.non_operator_requirements,
            "notes": self.notes,
            "enabled_ops": self.enabled_ops,
        }


@dataclass(frozen=True)
class OperatorBackendSession:
    """Information yielded by ``operator_backend``."""

    name: str
    enabled: bool
    availability: OperatorBackendAvailability
    plan: OperatorReplacementPlan
    active_registered_keys: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "availability": self.availability.summary(),
            "replacement_plan": self.plan.summary(),
            "active_registered_keys": self.active_registered_keys,
        }


@dataclass(frozen=True)
class OperatorValidationResult:
    """Forward/backward smoke validation for a concrete operator backend."""

    backend: str
    device: str
    dtype: str
    requested_ops: tuple[str, ...]
    passed_ops: tuple[str, ...]
    failed_ops: Mapping[str, str]

    @property
    def usable_ops(self) -> tuple[str, ...]:
        return self.passed_ops

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "requested_ops": self.requested_ops,
            "passed_ops": self.passed_ops,
            "failed_ops": dict(self.failed_ops),
            "usable_ops": self.usable_ops,
        }


def _normalize_backend_name(name: str | None) -> str:
    normalized = (
        (name or os.environ.get("FQ_OPERATOR_BACKEND") or "pytorch").strip().lower()
    )
    if normalized in {"", "none", "off", "native", "torch"}:
        return "pytorch"
    if normalized in {"flaggems", "flag_gems", "gems"}:
        return "flaggems"
    return normalized


def _split_ops(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    parts: Iterable[str]
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    else:
        parts = (str(item) for item in raw)
    return tuple(
        dict.fromkeys(
            FLAGGEMS_OP_ALIASES.get(part.strip(), part.strip())
            for part in parts
            if part and part.strip()
        )
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _import_flag_gems() -> tuple[Any | None, BaseException | None]:
    try:
        return importlib.import_module("flag_gems"), None
    except BaseException as exc:  # pragma: no cover - depends on optional runtime.
        return None, exc


def _extract_flaggems_catalog_keys(module: Any) -> tuple[str, ...]:
    collected: list[str] = []
    by_func = getattr(module, "FULL_CONFIG_BY_FUNC", None)
    if isinstance(by_func, Mapping):
        collected.extend(str(key) for key in by_func.keys())
    full_config = getattr(module, "_FULL_CONFIG", None)
    if full_config is not None:
        try:
            for item in full_config:
                if not item or len(item) < 2:
                    continue
                collected.append(str(item[0]))
                func = item[1]
                collected.append(
                    func.__name__ if hasattr(func, "__name__") else str(func)
                )
        except TypeError:
            pass
    if not collected and hasattr(module, "all_registered_keys"):
        try:
            collected.extend(str(item) for item in module.all_registered_keys())
        except BaseException:
            pass
    return tuple(sorted(dict.fromkeys(item for item in collected if item)))


def _active_flaggems_registered_keys(module: Any) -> tuple[str, ...]:
    if not hasattr(module, "all_registered_keys"):
        return ()
    try:
        return tuple(str(item) for item in module.all_registered_keys())
    except BaseException:
        return ()


def _availability_from_module(module: Any) -> OperatorBackendAvailability:
    keys = _extract_flaggems_catalog_keys(module)
    return OperatorBackendAvailability(
        name="flaggems",
        available=True,
        module_version=str(getattr(module, "__version__", "")) or None,
        device=str(getattr(module, "device", "")) or None,
        vendor=str(getattr(module, "vendor_name", "")) or None,
        registered_op_count=len(keys),
        registered_keys=keys,
    )


def _probe_flaggems_in_subprocess(
    timeout_seconds: float,
) -> OperatorBackendAvailability:
    code = (
        "import json\n"
        "import torch\n"
        "import flag_gems\n"
        "keys = []\n"
        "by_func = getattr(flag_gems, 'FULL_CONFIG_BY_FUNC', None)\n"
        "if isinstance(by_func, dict):\n"
        "    keys.extend(str(item) for item in by_func.keys())\n"
        "full_config = getattr(flag_gems, '_FULL_CONFIG', None)\n"
        "if full_config is not None:\n"
        "    for item in full_config:\n"
        "        if not item or len(item) < 2:\n"
        "            continue\n"
        "        keys.append(str(item[0]))\n"
        "        func = item[1]\n"
        "        keys.append(func.__name__ if hasattr(func, '__name__') else str(func))\n"
        "if not keys and hasattr(flag_gems, 'all_registered_keys'):\n"
        "    try:\n"
        "        keys.extend(str(item) for item in flag_gems.all_registered_keys())\n"
        "    except Exception:\n"
        "        pass\n"
        "keys = sorted(dict.fromkeys(item for item in keys if item))\n"
        "print(json.dumps({\n"
        "    'version': str(getattr(flag_gems, '__version__', '') or ''),\n"
        "    'device': str(getattr(flag_gems, 'device', '') or ''),\n"
        "    'vendor': str(getattr(flag_gems, 'vendor_name', '') or ''),\n"
        "    'keys': keys,\n"
        "}))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason=f"FlagGems import probe timed out after {timeout_seconds:.1f}s.",
        )
    except BaseException as exc:  # pragma: no cover - platform dependent.
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason=f"{type(exc).__name__}: {exc}",
        )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or f"probe exited with code {completed.returncode}"
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason=message,
        )
    try:
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason=f"Invalid FlagGems probe output: {exc}",
        )
    keys = tuple(str(item) for item in payload.get("keys", ()))
    return OperatorBackendAvailability(
        name="flaggems",
        available=True,
        module_version=str(payload.get("version") or "") or None,
        device=str(payload.get("device") or "") or None,
        vendor=str(payload.get("vendor") or "") or None,
        registered_op_count=len(keys),
        registered_keys=keys,
    )


def flaggems_availability(
    *, include_registered_keys: bool = True
) -> OperatorBackendAvailability:
    """Inspect whether FlagGems is importable and which ops it advertises."""

    loaded = sys.modules.get("flag_gems", ...)
    if loaded is None:
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason="flag_gems import is blocked in sys.modules",
        )
    if loaded is not ...:
        return _availability_from_module(loaded)

    try:
        spec = importlib.util.find_spec("flag_gems")
    except (ImportError, ValueError) as exc:
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason=f"{type(exc).__name__}: {exc}",
        )
    if spec is None:
        return OperatorBackendAvailability(
            name="flaggems",
            available=False,
            reason="flag_gems is not importable",
        )
    timeout = float(os.environ.get("FQ_FLAGGEMS_PROBE_TIMEOUT_SECONDS", "10"))
    availability = _probe_flaggems_in_subprocess(timeout)
    if not include_registered_keys and availability.available:
        return OperatorBackendAvailability(
            name=availability.name,
            available=True,
            reason=availability.reason,
            module_version=availability.module_version,
            device=availability.device,
            vendor=availability.vendor,
            registered_op_count=None,
        )
    return availability


def _known_flaggems_catalog() -> set[str]:
    return set(FLAGGEMS_SAFE_OPS) | set(FLAGGEMS_EXPERIMENTAL_OPS)


def plan_operator_replacements(
    backend: str = "flaggems",
    *,
    requested_ops: Iterable[str] | None = None,
    include_experimental: bool | None = None,
    availability: OperatorBackendAvailability | None = None,
) -> OperatorReplacementPlan:
    """Classify which native PyTorch ops can be delegated to an operator backend."""

    backend = _normalize_backend_name(backend)
    include_experimental = (
        _env_bool("FQ_FLAGGEMS_INCLUDE_EXPERIMENTAL", False)
        if include_experimental is None
        else bool(include_experimental)
    )
    requested = _split_ops(requested_ops)
    if not requested:
        env_ops = _split_ops(os.environ.get("FQ_FLAGGEMS_INCLUDE"))
        requested = env_ops or FLAGGEMS_SAFE_OPS
    if include_experimental:
        requested = tuple(dict.fromkeys(tuple(requested) + FLAGGEMS_EXPERIMENTAL_OPS))

    if backend != "flaggems":
        return OperatorReplacementPlan(
            backend=backend,
            requested_ops=tuple(requested),
            runtime_replaceable_ops=(),
            catalog_safe_ops=(),
            experimental_ops=(),
            unavailable_ops=tuple(requested),
            non_operator_requirements=(),
            notes=(f"Operator backend {backend!r} has no FlagQuantum adapter.",),
        )

    availability = availability or flaggems_availability()
    runtime_keys = set(availability.registered_keys)
    catalog = _known_flaggems_catalog()
    safe = set(FLAGGEMS_SAFE_OPS)
    experimental = set(FLAGGEMS_EXPERIMENTAL_OPS)
    non_operator = set(FLAGGEMS_NON_OPERATOR_REQUIREMENTS)

    runtime_replaceable = []
    catalog_safe = []
    experimental_requested = []
    unavailable = []
    non_operator_requested = []
    for op in requested:
        if (
            op in non_operator
            or op.startswith("jax.")
            or op.startswith("torch.distributed")
        ):
            non_operator_requested.append(op)
        elif (
            availability.available
            and op in runtime_keys
            and (op in safe or include_experimental)
        ):
            runtime_replaceable.append(op)
        elif op in safe:
            catalog_safe.append(op)
        elif op in experimental:
            experimental_requested.append(op)
        elif op in catalog:
            experimental_requested.append(op)
        else:
            unavailable.append(op)

    notes = [
        "FlagGems is an ATen operator layer for the native PyTorch path; it does not replace JAX kernels or distributed sharding.",
        "Use experimental ops only after precision/autograd preflight passes on the target accelerator.",
    ]
    if not availability.available:
        notes.append(f"Runtime import unavailable: {availability.reason}")
    if experimental_requested:
        notes.append(
            "Experimental ops need FlagQuantum-specific validation before default enablement."
        )
    if non_operator_requested:
        notes.append(
            "Non-operator requirements need FlagQuantum runtime work or upstream feature requests, not ATen replacement."
        )

    return OperatorReplacementPlan(
        backend="flaggems",
        requested_ops=tuple(requested),
        runtime_replaceable_ops=tuple(runtime_replaceable),
        catalog_safe_ops=tuple(catalog_safe),
        experimental_ops=tuple(experimental_requested),
        unavailable_ops=tuple(unavailable),
        non_operator_requirements=tuple(non_operator_requested),
        notes=tuple(notes),
    )


def flaggems_preflight(
    *,
    requested_ops: Iterable[str] | None = None,
    include_experimental: bool | None = None,
) -> dict[str, Any]:
    """Return a serializable FlagGems availability and replacement report."""

    availability = flaggems_availability()
    plan = plan_operator_replacements(
        "flaggems",
        requested_ops=requested_ops,
        include_experimental=include_experimental,
        availability=availability,
    )
    return {
        "availability": availability.summary(),
        "replacement_plan": plan.summary(),
        "replacement_guidance": {
            "safe_first_wave": FLAGGEMS_SAFE_OPS,
            "experimental_needs_validation": {
                op: FLAGGEMS_FEATURE_REQUESTS[op] for op in FLAGGEMS_EXPERIMENTAL_OPS
            },
            "not_solved_by_operator_backend": {
                op: FLAGGEMS_FEATURE_REQUESTS[op]
                for op in FLAGGEMS_NON_OPERATOR_REQUIREMENTS
            },
        },
        "native_pytorch_hotspots": dict(FLAGGEMS_NATIVE_PYTORCH_HOTSPOTS),
    }


@contextmanager
def operator_backend(
    name: str | None = None,
    *,
    include: Iterable[str] | str | None = None,
    include_experimental: bool | None = None,
    strict: bool | None = None,
    record: bool = False,
    once: bool = False,
    path: str | None = None,
) -> Iterator[OperatorBackendSession]:
    """Temporarily enable an optional PyTorch operator backend.

    ``strict=False`` keeps local development easy: if FlagGems is not installed,
    execution continues with PyTorch and the yielded session reports
    ``enabled=False``.  Use ``strict=True`` in CI or performance runs that must
    prove FlagGems is actually active.
    """

    normalized = _normalize_backend_name(name)
    strict = _env_bool("FQ_FLAGGEMS_STRICT", False) if strict is None else bool(strict)
    if normalized == "pytorch":
        availability = OperatorBackendAvailability(name="pytorch", available=True)
        plan = OperatorReplacementPlan(
            backend="pytorch",
            requested_ops=(),
            runtime_replaceable_ops=(),
            catalog_safe_ops=(),
            experimental_ops=(),
            unavailable_ops=(),
            non_operator_requirements=(),
            notes=("Native PyTorch ATen operators are used without replacement.",),
        )
        yield OperatorBackendSession("pytorch", False, availability, plan)
        return

    if normalized != "flaggems":
        raise ValueError("operator backend must be 'pytorch' or 'flaggems'.")

    availability = flaggems_availability()
    plan = plan_operator_replacements(
        "flaggems",
        requested_ops=_split_ops(include),
        include_experimental=include_experimental,
        availability=availability,
    )
    if not availability.available:
        if strict:
            raise RuntimeError(
                f"FlagGems operator backend is unavailable: {availability.reason}"
            )
        yield OperatorBackendSession("flaggems", False, availability, plan)
        return

    module, error = _import_flag_gems()
    if module is None:
        if strict:
            raise RuntimeError(
                f"FlagGems operator backend is unavailable: {error}"
            ) from error
        yield OperatorBackendSession("flaggems", False, availability, plan)
        return

    enabled_ops = tuple(
        op
        for op in plan.enabled_ops
        if op not in plan.experimental_ops or include_experimental
    )
    if not enabled_ops:
        if strict:
            raise RuntimeError(
                "No FlagGems operator could be enabled for the requested plan."
            )
        yield OperatorBackendSession("flaggems", False, availability, plan)
        return

    manager = module.use_gems(
        include=list(enabled_ops), record=record, once=once, path=path
    )
    with manager if manager is not None else nullcontext():
        active_keys = _active_flaggems_registered_keys(module)
        if strict and not active_keys:
            raise RuntimeError(
                "FlagGems imported but did not register any requested ATen ops. "
                f"Requested include={enabled_ops!r}."
            )
        yield OperatorBackendSession("flaggems", True, availability, plan, active_keys)


@contextmanager
def operator_backend_from_env(**kwargs: Any) -> Iterator[OperatorBackendSession]:
    """Enable the operator backend requested by ``FQ_OPERATOR_BACKEND``."""

    with operator_backend(os.environ.get("FQ_OPERATOR_BACKEND"), **kwargs) as session:
        yield session


def _torch_dtype_from_string(dtype: str | Any) -> Any:
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).removeprefix("torch.").lower()
    mapping = {
        "complex64": torch.complex64,
        "cfloat": torch.complex64,
        "complex128": torch.complex128,
        "cdouble": torch.complex128,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported validation dtype {dtype!r}.")
    return mapping[normalized]


def _real_objective(value: Any) -> Any:
    import torch

    tensor = value if hasattr(value, "sum") else torch.as_tensor(value)
    if torch.is_complex(tensor):
        tensor = torch.real(tensor)
    return tensor.sum()


def _run_flaggems_op_smoke(op: str, *, device: str, dtype: Any) -> None:
    import torch

    torch_device = torch.device(device)
    dtype = _torch_dtype_from_string(dtype)
    is_complex_dtype = dtype in {torch.complex64, torch.complex128}
    real_dtype = (
        torch.float32
        if dtype in {torch.complex64, torch.float16, torch.bfloat16, torch.float32}
        else torch.float64
    )

    def tensor(shape: tuple[int, ...], *, requires_grad: bool = True) -> torch.Tensor:
        base = torch.linspace(
            -0.4,
            0.7,
            steps=max(1, int(torch.tensor(shape).prod().item())),
            device=torch_device,
        )
        base = base.reshape(shape).to(real_dtype)
        if is_complex_dtype:
            imag = (
                torch.linspace(0.2, -0.3, steps=base.numel(), device=torch_device)
                .reshape(shape)
                .to(real_dtype)
            )
            out = torch.complex(base, imag).to(dtype)
        else:
            out = base.to(dtype)
        return out.detach().clone().requires_grad_(requires_grad)

    if op == "abs":
        x = tensor((4, 4))
        objective = torch.abs(x).sum()
    elif op == "add":
        x = tensor((4, 4))
        y = tensor((4, 4))
        objective = _real_objective(x + y)
    elif op == "addmm":
        bias = tensor((4, 4))
        a = tensor((4, 3))
        b = tensor((3, 4))
        objective = _real_objective(torch.addmm(bias, a, b))
    elif op == "bmm":
        a = tensor((2, 4, 3))
        b = tensor((2, 3, 5))
        objective = _real_objective(torch.bmm(a, b))
    elif op == "dot":
        x = tensor((8,))
        y = tensor((8,))
        objective = _real_objective(torch.dot(x, y))
    elif op == "gather":
        x = tensor((3, 4))
        index = torch.tensor([[0, 2], [1, 3], [0, 1]], device=torch_device)
        objective = _real_objective(torch.gather(x, 1, index))
    elif op == "index_select":
        x = tensor((5, 4))
        index = torch.tensor([0, 2, 4], device=torch_device)
        objective = _real_objective(torch.index_select(x, 0, index))
    elif op == "kron":
        a = tensor((2, 2))
        b = tensor((2, 2))
        objective = _real_objective(torch.kron(a, b))
    elif op == "mm":
        a = tensor((4, 3))
        b = tensor((3, 5))
        objective = _real_objective(a @ b)
    elif op == "mul":
        x = tensor((4, 4))
        y = tensor((4, 4))
        objective = _real_objective(x * y)
    elif op == "outer":
        x = tensor((4,))
        y = tensor((5,))
        objective = _real_objective(torch.outer(x, y))
    elif op == "sum":
        x = tensor((4, 4))
        objective = _real_objective(torch.sum(x))
    elif op == "vdot":
        x = tensor((8,))
        y = tensor((8,))
        objective = _real_objective(torch.vdot(x, y))
    elif op == "where_self":
        condition = torch.tensor([[True, False], [False, True]], device=torch_device)
        x = tensor((2, 2))
        y = tensor((2, 2))
        objective = _real_objective(torch.where(condition, x, y))
    elif op == "zeros":
        objective = _real_objective(
            torch.zeros((4, 4), dtype=dtype, device=torch_device)
        )
    elif op == "zeros_like":
        x = tensor((4, 4), requires_grad=False)
        objective = _real_objective(torch.zeros_like(x))
    else:
        raise ValueError(f"No FlagGems smoke validator is implemented for op {op!r}.")

    if getattr(objective, "requires_grad", False):
        torch.autograd.backward(objective)
    if torch_device.type == "cuda":
        cuda_platform = get_platform_runtime("cuda")
        if cuda_platform.is_available():
            cuda_platform.synchronize(torch_device)


def validate_flaggems_ops(
    ops: Iterable[str] | str | None = None,
    *,
    device: str = "cuda",
    dtype: str | Any = "complex64",
    include_experimental: bool = False,
) -> OperatorValidationResult:
    """Validate selected FlagGems ops on a target device and dtype.

    The validation runs tiny forward/backward smoke tests under a one-op
    FlagGems context.  This catches target-runtime issues such as CUDA kernels
    that do not support complex64 before a full quantum benchmark starts.
    """

    requested = _split_ops(ops) or FLAGGEMS_SAFE_OPS
    passed: list[str] = []
    failed: dict[str, str] = {}
    for op in requested:
        try:
            with operator_backend(
                "flaggems",
                include=[op],
                include_experimental=include_experimental,
                strict=True,
            ):
                _run_flaggems_op_smoke(op, device=device, dtype=dtype)
            passed.append(op)
        except BaseException as exc:  # pragma: no cover - hardware dependent.
            failed[op] = f"{type(exc).__name__}: {exc}"
    return OperatorValidationResult(
        backend="flaggems",
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        requested_ops=tuple(requested),
        passed_ops=tuple(passed),
        failed_ops=failed,
    )


__all__ = [
    "FLAGGEMS_EXPERIMENTAL_OPS",
    "FLAGGEMS_NATIVE_PYTORCH_HOTSPOTS",
    "FLAGGEMS_FEATURE_REQUESTS",
    "FLAGGEMS_NON_OPERATOR_REQUIREMENTS",
    "FLAGGEMS_SAFE_OPS",
    "OperatorBackendAvailability",
    "OperatorBackendSession",
    "OperatorReplacementPlan",
    "OperatorValidationResult",
    "flaggems_availability",
    "flaggems_preflight",
    "operator_backend",
    "operator_backend_from_env",
    "plan_operator_replacements",
    "validate_flaggems_ops",
]
