"""Runtime backend and precision configuration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ..core.runtime_config import get_runtime_config, runtime_config, set_runtime_config
from .backend_registry import (
    backend_execution_options,
    get_active_backend,
    get_backend_capabilities,
    resolve_dtype,
    set_active_backend,
)


def set_backend(name: str = "pytorch") -> str:
    """Select the native tensor backend for the current runtime context."""

    try:
        selected = set_active_backend(name).name
        set_runtime_config(get_runtime_config().with_overrides(backend=selected))
        return selected
    except KeyError as exc:
        raise NotImplementedError(
            "FlagQuantum native core currently executes with registered tensor "
            "backends. Add JAX, TensorFlow, or accelerator-specific adapters "
            "behind the backend registry before selecting them."
        ) from exc


def get_backend() -> str:
    """Return the active native tensor backend name."""

    return get_active_backend()


def set_dtype(dtype: str = "complex64") -> tuple[str, str]:
    """Select complex and matching real dtypes for the current context."""

    real, complex_ = resolve_dtype(dtype)
    complex_dtype = str(complex_).removeprefix("torch.")
    real_dtype = str(real).removeprefix("torch.")
    set_runtime_config(get_runtime_config().with_overrides(complex_dtype=complex_dtype))
    return complex_dtype, real_dtype


def get_dtype() -> tuple[str, str]:
    """Return the configured complex and real dtype names."""

    config = get_runtime_config()
    return config.complex_dtype, config.real_dtype


@contextmanager
def runtime_backend(name: str = "pytorch") -> Iterator[str]:
    """Temporarily select a registered native tensor backend."""

    capabilities = get_backend_capabilities(name)
    with runtime_config(backend=capabilities.name):
        yield capabilities.name


@contextmanager
def runtime_dtype(dtype: str = "complex64") -> Iterator[tuple[str, str]]:
    """Temporarily select complex and matching real dtypes."""

    real, complex_ = resolve_dtype(dtype)
    complex_name = str(complex_).removeprefix("torch.")
    real_name = str(real).removeprefix("torch.")
    with runtime_config(complex_dtype=complex_name):
        yield complex_name, real_name


__all__ = (
    "get_backend",
    "backend_execution_options",
    "get_backend_capabilities",
    "get_dtype",
    "runtime_backend",
    "runtime_dtype",
    "set_backend",
    "set_dtype",
)
