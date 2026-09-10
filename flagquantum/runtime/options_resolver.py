"""Single internal resolver for all execution-option sources."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, TypeVar

from .options import ExecutionOptions

_OptionValue = TypeVar("_OptionValue")


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
    overlays = (
        ("call", options),
        ("runtime_policy", policy_options),
        ("program_constraints", program_constraints),
        ("runtime_config", config_options),
    )
    sources: list[tuple[str, str]] = []

    def resolve_field(
        name: str,
        getter: Callable[[ExecutionOptions], _OptionValue | None],
        default: _OptionValue,
    ) -> _OptionValue:
        for source_name, overlay in overlays:
            if overlay is not None and (value := getter(overlay)) is not None:
                sources.append((name, source_name))
                return value
        sources.append((name, "framework_defaults"))
        return default

    resolved = ResolvedExecutionOptions(
        mode=resolve_field("mode", lambda item: item.mode, "auto"),
        backend=resolve_field("backend", lambda item: item.backend, "auto"),
        device=resolve_field("device", lambda item: item.device, "auto"),
        target=resolve_field("target", lambda item: item.target, "auto"),
        batch_size=resolve_field("batch_size", lambda item: item.batch_size, 1),
        precision=resolve_field("precision", lambda item: item.precision, "complex64"),
        shots=resolve_field("shots", lambda item: item.shots, None),
        seed=resolve_field("seed", lambda item: item.seed, None),
        memory_limit_bytes=resolve_field(
            "memory_limit_bytes", lambda item: item.memory_limit_bytes, None
        ),
        require_gradients=resolve_field(
            "require_gradients", lambda item: item.require_gradients, False
        ),
        allow_approximate=resolve_field(
            "allow_approximate", lambda item: item.allow_approximate, False
        ),
        allow_backend_fallback=resolve_field(
            "allow_backend_fallback", lambda item: item.allow_backend_fallback, False
        ),
        sources=tuple(sources),
    )
    _validate_program_batch(resolved, program_constraints)
    _validate_program_precision(resolved, program_constraints)
    return resolved


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

    dtype = str(getattr(circuit, "dtype", "")).removeprefix("torch.") or None
    if not hasattr(circuit, "bsz"):
        # CircuitIR carries a numerical precision constraint even though it has
        # no live Circuit device or ``bsz`` attribute. Ignoring it would let the
        # framework complex64 default silently downcast a complex128 program.
        if dtype is not None:
            return ExecutionOptions(precision=dtype)
        return ExecutionOptions()
    device = str(getattr(circuit, "device", "")) or None
    return ExecutionOptions(
        batch_size=int(circuit.bsz),
        device=device,
        precision=dtype,
    )


def _require_options(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, ExecutionOptions):
        raise TypeError(f"{name} must be an ExecutionOptions or None")
    if type(value) is ExecutionOptions:
        return
    supported = {field.name for field in fields(ExecutionOptions)}
    for field in fields(value):
        if field.name not in supported and getattr(value, field.name) is not None:
            raise TypeError(
                f"{name} contains unsupported execution field {field.name!r}"
            )


def _validate_program_batch(
    values: ResolvedExecutionOptions, constraints: ExecutionOptions | None
) -> None:
    if constraints is None or constraints.batch_size is None:
        return
    if values.batch_size != constraints.batch_size:
        raise ValueError(
            "batch_size conflicts with the program batch constraint: "
            f"{values.batch_size} != {constraints.batch_size}"
        )


def _validate_program_precision(
    values: ResolvedExecutionOptions, constraints: ExecutionOptions | None
) -> None:
    if constraints is None or constraints.precision != "complex128":
        return
    if values.precision == "complex64":
        raise ValueError(
            "precision=complex64 would demote a complex128 program; construct "
            "the program with complex64 precision instead"
        )


__all__ = (
    "ResolvedExecutionOptions",
    "circuit_execution_constraints",
    "resolve_execution_options",
    "runtime_config_to_execution_options",
)
