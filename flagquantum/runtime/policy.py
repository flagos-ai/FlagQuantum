"""Serializable runtime policy owned by ``fq.Module``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..core._qubit_aliases import OMITTED, Omitted, warn_qubit_alias
from ..errors import ValidationError
from .options import ExecutionOptions

ObservableKind = Literal["z", "z_sum", "hamiltonian"]


def _module_execution_defaults() -> ExecutionOptions:
    return ExecutionOptions(mode="statevector", backend="pytorch")


@dataclass(frozen=True, init=False)
class RuntimePolicy:
    """Module execution options plus training-specific observable policy."""

    execution_options: ExecutionOptions = field(
        default_factory=_module_execution_defaults
    )
    observable: ObservableKind = "z"
    observable_qubits: tuple[int, ...] = (0,)
    correctness_debug: bool = False

    def __init__(
        self,
        execution_options: ExecutionOptions | Omitted = OMITTED,
        observable: ObservableKind = "z",
        observable_qubits: tuple[int, ...] | Omitted = OMITTED,
        correctness_debug: bool = False,
        *,
        observable_wires: tuple[int, ...] | Omitted = OMITTED,
    ) -> None:
        if not isinstance(observable_wires, Omitted):
            if not isinstance(observable_qubits, Omitted):
                raise TypeError("pass observable_qubits or observable_wires, not both")
            warn_qubit_alias("observable_wires", "observable_qubits")
            observable_qubits = observable_wires
        selected = (0,) if isinstance(observable_qubits, Omitted) else observable_qubits
        options = (
            _module_execution_defaults()
            if isinstance(execution_options, Omitted)
            else execution_options
        )
        object.__setattr__(self, "execution_options", options)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "observable_qubits", selected)
        object.__setattr__(self, "correctness_debug", correctness_debug)
        self.__post_init__()

    @property
    def observable_wires(self) -> tuple[int, ...]:
        """Deprecated accessor for configurations created before qubit naming."""
        warn_qubit_alias("observable_wires", "observable_qubits")
        return self.observable_qubits

    def __post_init__(self) -> None:
        if not isinstance(self.execution_options, ExecutionOptions):
            raise TypeError("execution_options must be an ExecutionOptions")
        if self.observable not in {"z", "z_sum", "hamiltonian"}:
            raise ValidationError(
                f"unsupported fq.Module observable {self.observable!r}"
            )
        if not self.observable_qubits or any(
            wire < 0 for wire in self.observable_qubits
        ):
            raise ValidationError(
                "observable_qubits must contain non-negative wire indices"
            )
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
            "schema": "flagquantum.runtime_policy",
            "version": "2.0",
            "execution_options": self.execution_options.to_dict(),
            "observable": self.observable,
            "observable_qubits": list(self.observable_qubits),
            "correctness_debug": self.correctness_debug,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePolicy":
        if not isinstance(payload, Mapping):
            raise TypeError("runtime policy payload must be a mapping")
        values = dict(payload)
        schema = values.pop("schema", None)
        version = values.pop("version", None)
        if schema is not None or version is not None:
            if schema != "flagquantum.runtime_policy" or version != "2.0":
                raise ValidationError("unsupported runtime policy schema/version")
        if "observable_wires" in values:
            if "observable_qubits" in values:
                raise ValidationError("conflicting runtime policy qubit fields")
            values["observable_qubits"] = values.pop("observable_wires")
        expected = {
            "execution_options",
            "observable",
            "observable_qubits",
            "correctness_debug",
        }
        unknown = set(values) - expected
        if unknown:
            raise ValidationError(
                "unknown runtime policy field(s): " + ", ".join(sorted(unknown))
            )
        return cls(
            execution_options=ExecutionOptions.from_dict(values["execution_options"]),
            observable=values.get("observable", "z"),
            observable_qubits=tuple(values.get("observable_qubits", (0,))),
            correctness_debug=values.get("correctness_debug", False),
        )


__all__ = ("RuntimePolicy",)
