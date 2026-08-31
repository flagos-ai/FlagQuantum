"""Serializable runtime policy owned by ``fq.Module``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .options import ExecutionOptions

ObservableKind = Literal["z", "z_sum", "hamiltonian"]


def _module_execution_defaults() -> ExecutionOptions:
    return ExecutionOptions(mode="statevector", backend="pytorch")


@dataclass(frozen=True)
class RuntimePolicy:
    """Module execution options plus training-specific observable policy."""

    execution_options: ExecutionOptions = field(
        default_factory=_module_execution_defaults
    )
    observable: ObservableKind = "z"
    observable_wires: tuple[int, ...] = (0,)
    correctness_debug: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.execution_options, ExecutionOptions):
            raise TypeError("execution_options must be an ExecutionOptions")
        if self.observable not in {"z", "z_sum", "hamiltonian"}:
            raise ValueError(f"unsupported fq.Module observable {self.observable!r}")
        if not self.observable_wires or any(wire < 0 for wire in self.observable_wires):
            raise ValueError("observable_wires must contain non-negative wire indices")
        if type(self.correctness_debug) is not bool:
            raise TypeError("correctness_debug must be a bool")

    @property
    def mode(self) -> str:
        return self.execution_options.mode or "statevector"

    @property
    def backend(self) -> str:
        return self.execution_options.backend or "pytorch"

    @property
    def allow_backend_fallback(self) -> bool:
        return bool(self.execution_options.allow_backend_fallback)

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_options": self.execution_options.to_dict(),
            "observable": self.observable,
            "observable_wires": list(self.observable_wires),
            "correctness_debug": self.correctness_debug,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePolicy":
        if not isinstance(payload, Mapping):
            raise TypeError("runtime policy payload must be a mapping")
        values = dict(payload)
        expected = {
            "execution_options",
            "observable",
            "observable_wires",
            "correctness_debug",
        }
        unknown = set(values) - expected
        if unknown:
            raise ValueError(
                "unknown runtime policy field(s): " + ", ".join(sorted(unknown))
            )
        return cls(
            execution_options=ExecutionOptions.from_dict(values["execution_options"]),
            observable=values.get("observable", "z"),
            observable_wires=tuple(values.get("observable_wires", (0,))),
            correctness_debug=values.get("correctness_debug", False),
        )


__all__ = ("RuntimePolicy",)
