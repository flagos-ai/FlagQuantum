"""Immutable, serializable runtime configuration with scoped overrides."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator, Mapping

RUNTIME_CONFIG_VERSION = "1.0"


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution policy copied across compile, task and process boundaries."""

    backend: str = "pytorch"
    device: str = "cpu"
    complex_dtype: str = "complex64"
    real_dtype: str = "float32"
    jax_enable_x64: bool = False
    jax_matmul_precision: str = "highest"
    drawing_style: str = "black_white"
    version: str = RUNTIME_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.version != RUNTIME_CONFIG_VERSION:
            raise ValueError(f"unsupported runtime config version {self.version!r}")
        if self.complex_dtype not in {"complex64", "complex128"}:
            raise ValueError("complex_dtype must be complex64 or complex128")
        expected_real = "float64" if self.complex_dtype == "complex128" else "float32"
        if self.real_dtype != expected_real:
            raise ValueError(
                f"{self.complex_dtype} requires real_dtype={expected_real!r}"
            )
        if self.jax_enable_x64 != (self.complex_dtype == "complex128"):
            raise ValueError("jax_enable_x64 must agree with complex_dtype")

    def with_overrides(self, **changes: Any) -> "RuntimeConfig":
        if "complex_dtype" in changes and "real_dtype" not in changes:
            changes["real_dtype"] = (
                "float64" if changes["complex_dtype"] == "complex128" else "float32"
            )
        if "complex_dtype" in changes and "jax_enable_x64" not in changes:
            changes["jax_enable_x64"] = changes["complex_dtype"] == "complex128"
        return replace(self, **changes)

    def to_manifest(self) -> dict[str, Any]:
        return {"schema": "flagquantum_runtime_config", **asdict(self)}

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "RuntimeConfig":
        values = dict(manifest)
        if values.pop("schema", None) != "flagquantum_runtime_config":
            raise ValueError("invalid runtime configuration manifest schema")
        return cls(**values)

    @property
    def cache_key(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())


_CURRENT_CONFIG: ContextVar[RuntimeConfig] = ContextVar(
    "flagquantum_runtime_config", default=RuntimeConfig()
)


def get_runtime_config() -> RuntimeConfig:
    return _CURRENT_CONFIG.get()


def set_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig")
    _CURRENT_CONFIG.set(config)
    return config


@contextmanager
def runtime_config(
    config: RuntimeConfig | None = None, **overrides: Any
) -> Iterator[RuntimeConfig]:
    """Apply a task-local override and restore it safely for nested contexts."""

    selected = config or get_runtime_config()
    if overrides:
        selected = selected.with_overrides(**overrides)
    token = _CURRENT_CONFIG.set(selected)
    try:
        yield selected
    finally:
        _CURRENT_CONFIG.reset(token)


__all__ = [
    "RUNTIME_CONFIG_VERSION",
    "RuntimeConfig",
    "get_runtime_config",
    "runtime_config",
    "set_runtime_config",
]
