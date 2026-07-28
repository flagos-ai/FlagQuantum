"""Canonical execution result contract and native-result normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..core.contracts import AccuracyContract, RuntimePlanContract
from .result_adapters import LiveRuntimeSummary


@dataclass(frozen=True)
class MeasurementResult:
    """One backend-neutral result produced from a ``MeasurementNode``."""

    kind: str
    wires: tuple[int, ...]
    value: Any
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    """Stable result shape shared by local, distributed, and adapter paths."""

    value: torch.Tensor | None = None
    state: torch.Tensor | None = None
    samples: torch.Tensor | None = None
    measurements: tuple[MeasurementResult, ...] = ()
    plan: RuntimePlanContract | Any | None = None
    accuracy: AccuracyContract = AccuracyContract()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    compatibility: Mapping[str, Any] = field(default_factory=dict)

    def to(self, *args: Any, **kwargs: Any) -> "ExecutionResult":
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        return ExecutionResult(
            value=move(self.value),
            state=move(self.state),
            samples=move(self.samples),
            measurements=tuple(
                MeasurementResult(
                    kind=item.kind,
                    wires=item.wires,
                    value=(
                        item.value.to(*args, **kwargs)
                        if isinstance(item.value, torch.Tensor)
                        else item.value
                    ),
                    shots=item.shots,
                    metadata=item.metadata,
                    statistics=item.statistics,
                )
                for item in self.measurements
            ),
            plan=self.plan,
            accuracy=self.accuracy,
            metrics=self.metrics,
            provenance=self.provenance,
            runtime=self.runtime,
            compatibility=self.compatibility,
        )

    def detach(self) -> "ExecutionResult":
        return self.to_detached(copy=False)

    def to_detached(self, *, copy: bool = False) -> "ExecutionResult":
        def detach(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            result = value.detach()
            return result.clone() if copy else result

        return ExecutionResult(
            value=detach(self.value),
            state=detach(self.state),
            samples=detach(self.samples),
            measurements=tuple(
                MeasurementResult(
                    kind=item.kind,
                    wires=item.wires,
                    value=(
                        detach(item.value)
                        if isinstance(item.value, torch.Tensor)
                        else item.value
                    ),
                    shots=item.shots,
                    metadata=item.metadata,
                    statistics=item.statistics,
                )
                for item in self.measurements
            ),
            plan=self.plan,
            accuracy=self.accuracy,
            metrics=self.metrics,
            provenance=self.provenance,
            runtime=self.runtime,
            compatibility=self.compatibility,
        )

    def to_statevector(self) -> torch.Tensor:
        """Return a statevector when the selected backend exposes one."""

        if isinstance(self.state, torch.Tensor):
            return self.state
        native = self.__dict__.get("_native_output")
        converter = getattr(native, "to_statevector", None)
        if callable(converter):
            value = converter()
            if isinstance(value, torch.Tensor):
                return value
        state_getter = getattr(native, "state", None)
        if callable(state_getter):
            value = state_getter()
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError(
            "execution result does not expose a statevector; use backend-native "
            "measurements or request a state-producing execution mode"
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate backend-specific compatibility attributes to native output."""

        native = self.__dict__.get("_native_output")
        if native is not None and hasattr(native, name):
            return getattr(native, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def summary(self) -> dict[str, Any]:
        runtime = dict(self.runtime)
        summary = {
            "has_value": self.value is not None,
            "has_state": self.state is not None,
            "has_samples": self.samples is not None,
            "measurement_count": len(self.measurements),
            "has_plan": self.plan is not None,
            "accuracy": self.accuracy.to_dict(),
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
            "runtime": runtime,
            "compatibility": dict(self.compatibility),
        }
        conflicts = tuple(sorted(set(summary).intersection(runtime)))
        summary.update(
            (key, value) for key, value in runtime.items() if key not in summary
        )
        if conflicts:
            summary["runtime_projection_conflicts"] = conflicts
        return summary


def normalize_execution_result(
    output: Any,
    *,
    mode: str,
    plan: Any | None = None,
) -> ExecutionResult:
    """Normalize a legacy runtime return without materializing hidden state."""

    if isinstance(output, ExecutionResult):
        return output
    native_state = getattr(output, "state", None)
    state = (
        output
        if isinstance(output, torch.Tensor)
        else native_state if isinstance(native_state, torch.Tensor) else None
    )
    samples = getattr(output, "samples", None)
    if not isinstance(samples, torch.Tensor):
        samples = None
    summary = getattr(output, "summary", None)
    runtime: Mapping[str, Any] = (
        LiveRuntimeSummary(output, {"mode": mode})
        if callable(summary)
        else {"mode": mode}
    )
    result = ExecutionResult(
        state=state,
        samples=samples,
        plan=plan,
        runtime=runtime,
        compatibility={
            "legacy_return_normalized": True,
            "source_type": type(output).__name__,
            "full_state_materialized_by_adapter": False,
        },
    )
    object.__setattr__(result, "_native_output", output)
    return result


__all__ = ("ExecutionResult", "MeasurementResult", "normalize_execution_result")
