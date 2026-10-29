"""Pure hard limits shared by candidate selection and executable inspection."""

from dataclasses import dataclass

from .._hybrid.verifier import Diagnostic
from ..resource_estimation import ResourceEstimate


@dataclass(frozen=True)
class CircuitLimits:
    """Whole-program ceilings; neither ranking preferences nor search budgets.

    The caller supplies an estimate of the exact scope being checked. Workspace
    ownership requires allocation history and is not inferred from this record.
    """

    max_qubits: int | None = None
    max_operations: int | None = None
    max_depth: int | None = None

    def __post_init__(self) -> None:
        for name, limit in (
            ("max_qubits", self.max_qubits),
            ("max_operations", self.max_operations),
            ("max_depth", self.max_depth),
        ):
            if limit is not None and (
                type(limit) is not int or limit < (1 if name == "max_qubits" else 0)
            ):
                raise ValueError(
                    f"{name} requires a nonnegative plain integer; qubits must be positive"
                )

    def violations(self, resources: ResourceEstimate) -> tuple[Diagnostic, ...]:
        return tuple(
            Diagnostic(
                f"synthesis.constraints.{name}", f"{name}={limit}, actual={actual}"
            )
            for name, limit, actual in (
                ("max_qubits", self.max_qubits, resources.n_qubits),
                ("max_operations", self.max_operations, resources.n_operations),
                ("max_depth", self.max_depth, resources.depth),
            )
            if limit is not None and actual > limit
        )
