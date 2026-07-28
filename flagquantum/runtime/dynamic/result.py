"""Dynamic execution result implementation."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class DynamicExecutionResult:
    """Trajectory result containing final samples and classical registers."""

    samples: torch.Tensor
    classical_bits: torch.Tensor
    final_states: torch.Tensor
    shots: int
    seed: int | None
    execution_semantics: str = "local_statevector_trajectory"
    mid_circuit_measurements_available: bool = True
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_samples(self) -> torch.Tensor:
        return self.samples

    @property
    def classical_register(self) -> torch.Tensor:
        return self.classical_bits

    @property
    def mid_circuit_measurements(self) -> torch.Tensor | None:
        return self.classical_bits if self.mid_circuit_measurements_available else None

    def to_execution_result(self) -> Any:
        """Project dynamic shots into the canonical execution result contract."""

        from ..result import ExecutionResult, MeasurementResult

        measurements = [
            MeasurementResult(
                "sample",
                tuple(range(self.samples.shape[-1])),
                self.samples,
                shots=self.shots,
                metadata={"stage": "final"},
            )
        ]
        if self.mid_circuit_measurements_available:
            measurements.append(
                MeasurementResult(
                    "mid_circuit_measurement",
                    tuple(range(self.classical_bits.shape[-1])),
                    self.classical_bits,
                    shots=self.shots,
                )
            )
        return ExecutionResult(
            samples=self.samples,
            measurements=tuple(measurements),
            runtime={
                "mode": self.execution_semantics,
                "shots": self.shots,
                "seed": self.seed,
                "mid_circuit_measurements_available": (
                    self.mid_circuit_measurements_available
                ),
            },
            provenance=dict(self.provider_metadata),
        )

__all__ = ("DynamicExecutionResult",)
