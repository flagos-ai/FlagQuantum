"""Single internal resolver for all execution-option sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from .options import ExecutionOptions

_DEFAULTS: dict[str, object] = {
    "mode": "auto",
    "backend": "auto",
    "device": "auto",
    "target": "auto",
    "batch_size": 1,
    "precision": "complex64",
    "shots": None,
    "seed": None,
    "memory_limit_bytes": None,
    "require_gradients": False,
    "allow_approximate": False,
    "allow_backend_fallback": False,
}


@dataclass(frozen=True, slots=True)
class ResolvedExecutionOptions:
    """Internal concrete options plus field-level provenance."""

    mode: str
    backend: str
    device: str
    target: str
    batch_size: int
    precision: str
    shots: int | None
    seed: int | None
    memory_limit_bytes: int | None
    require_gradients: bool
    allow_approximate: bool
    allow_backend_fallback: bool
    sources: tuple[tuple[str, str], ...]

    def source_for(self, field_name: str) -> str:
        try:
            return dict(self.sources)[field_name]
        except KeyError as exc:
            raise KeyError(f"unknown resolved execution field {field_name!r}") from exc

    def to_execution_options(self) -> ExecutionOptions:
        values = {
            field.name: getattr(self, field.name) for field in fields(ExecutionOptions)
        }
        return ExecutionOptions(**values)


def resolve_execution_options(
    options: ExecutionOptions | None = None,
    *,
    policy_options: ExecutionOptions | None = None,
    program_constraints: ExecutionOptions | None = None,
    runtime_config: Any | None = None,
) -> ResolvedExecutionOptions:
    """Resolve all supported sources using the approved field precedence."""

    _require_options("options", options)
    _require_options("policy_options", policy_options)
    _require_options("program_constraints", program_constraints)
    config_options = runtime_config_to_execution_options(runtime_config)
    values = dict(_DEFAULTS)
    sources = {name: "framework_defaults" for name in values}
    for source_name, overlay in (
        ("runtime_config", config_options),
        ("program_constraints", program_constraints),
        ("runtime_policy", policy_options),
        ("call", options),
    ):
        if overlay is None:
            continue
        for name, value in asdict(overlay).items():
            if value is not None:
                values[name] = value
                sources[name] = source_name
    _validate_program_batch(values, program_constraints)
    return ResolvedExecutionOptions(
        **values,
        sources=tuple((name, sources[name]) for name in _DEFAULTS),
    )


def runtime_config_to_execution_options(config: Any | None) -> ExecutionOptions:
    """Adapt the task-local RuntimeConfig without leaking expert-only fields."""

    if config is None:
        from ..core.runtime_config import get_runtime_config

        config = get_runtime_config()
    required = ("backend", "device", "complex_dtype")
    if any(not hasattr(config, name) for name in required):
        raise TypeError("runtime_config must be a RuntimeConfig-compatible object")
    return ExecutionOptions(
        backend=str(config.backend),
        device=str(config.device),
        precision=str(config.complex_dtype),
    )


def circuit_execution_constraints(circuit: Any) -> ExecutionOptions:
    """Extract only stable program constraints from a Circuit-like object."""

    if not hasattr(circuit, "bsz"):
        return ExecutionOptions()
    dtype = str(getattr(circuit, "dtype", "")).removeprefix("torch.") or None
    device = str(getattr(circuit, "device", "")) or None
    return ExecutionOptions(
        batch_size=int(circuit.bsz),
        device=device,
        precision=dtype,
    )


def _require_options(name: str, value: object) -> None:
    if value is not None and not isinstance(value, ExecutionOptions):
        raise TypeError(f"{name} must be an ExecutionOptions or None")


def _validate_program_batch(
    values: Mapping[str, object], constraints: ExecutionOptions | None
) -> None:
    if constraints is None or constraints.batch_size is None:
        return
    if values["batch_size"] != constraints.batch_size:
        raise ValueError(
            "batch_size conflicts with the program batch constraint: "
            f"{values['batch_size']} != {constraints.batch_size}"
        )


__all__ = (
    "ResolvedExecutionOptions",
    "circuit_execution_constraints",
    "resolve_execution_options",
    "runtime_config_to_execution_options",
)
