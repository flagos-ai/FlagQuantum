"""Backend-neutral execution options proposed for the Stable Core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

EXECUTION_OPTIONS_SCHEMA = "flagquantum.execution_options"
EXECUTION_OPTIONS_VERSION = "1.0"

_MODES = frozenset({"auto", "statevector", "mps", "tensor_network", "density_matrix"})
_TARGETS = frozenset({"auto", "state", "expectation", "samples", "amplitudes"})
_PRECISIONS = frozenset({"complex64", "complex128"})


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Immutable field-level overrides for planning and execution.

    ``None`` always means that this layer does not specify the field. Concrete
    framework defaults are applied only by the shared options resolver.
    """

    mode: str | None = None
    backend: str | None = None
    device: str | None = None
    target: str | None = None
    batch_size: int | None = None
    precision: str | None = None
    shots: int | None = None
    seed: int | None = None
    memory_limit_bytes: int | None = None
    require_gradients: bool | None = None
    allow_approximate: bool | None = None
    allow_backend_fallback: bool | None = None

    def __post_init__(self) -> None:
        _validate_optional_choice("mode", self.mode, _MODES)
        _validate_optional_string("backend", self.backend)
        _validate_optional_string("device", self.device)
        _validate_optional_choice("target", self.target, _TARGETS)
        _validate_optional_choice("precision", self.precision, _PRECISIONS)
        _validate_optional_integer("batch_size", self.batch_size, minimum=1)
        _validate_optional_integer("shots", self.shots, minimum=1)
        _validate_optional_integer("seed", self.seed, minimum=0)
        _validate_optional_integer(
            "memory_limit_bytes", self.memory_limit_bytes, minimum=1
        )
        _validate_optional_bool("require_gradients", self.require_gradients)
        _validate_optional_bool("allow_approximate", self.allow_approximate)
        _validate_optional_bool("allow_backend_fallback", self.allow_backend_fallback)

    def to_dict(self) -> dict[str, object]:
        """Serialize without erasing ``None`` inheritance semantics."""

        return {
            "schema": EXECUTION_OPTIONS_SCHEMA,
            "version": EXECUTION_OPTIONS_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionOptions":
        """Deserialize one exact v1 options payload, failing on schema drift."""

        if not isinstance(payload, Mapping):
            raise TypeError("execution options payload must be a mapping")
        values = dict(payload)
        if values.pop("schema", None) != EXECUTION_OPTIONS_SCHEMA:
            raise ValueError("invalid execution options schema")
        if values.pop("version", None) != EXECUTION_OPTIONS_VERSION:
            raise ValueError("unsupported execution options version")
        expected = {field.name for field in fields(cls)}
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown:
            raise ValueError(
                "unknown execution options field(s): " + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "missing execution options field(s): " + ", ".join(sorted(missing))
            )
        return cls(**values)


def _validate_optional_string(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _validate_optional_choice(
    name: str, value: object, choices: frozenset[str]
) -> None:
    _validate_optional_string(name, value)
    if value is not None and value not in choices:
        rendered = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {rendered}")


def _validate_optional_integer(name: str, value: object, *, minimum: int) -> None:
    if value is None:
        return
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer or None")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _validate_optional_bool(name: str, value: object) -> None:
    if value is not None and type(value) is not bool:
        raise TypeError(f"{name} must be a bool or None")


__all__ = (
    "EXECUTION_OPTIONS_SCHEMA",
    "EXECUTION_OPTIONS_VERSION",
    "ExecutionOptions",
)
