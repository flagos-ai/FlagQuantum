"""Canonical execution result contract and native-result normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias, overload

import torch

from ..core.contracts import AccuracyContract, RuntimePlanContract
from ..errors import ExecutionError
from .result_adapters import LiveRuntimeSummary

if TYPE_CHECKING:
    from .execution_plan import ExecutionPlan
    from .executors.mps.production import MPSProductionPlan

EXECUTION_RESULT_SUMMARY_SCHEMA = "flagquantum.execution_result.summary"
EXECUTION_RESULT_SUMMARY_VERSION = "1.0"
EXECUTION_DIAGNOSTICS_SCHEMA = "flagquantum.execution_diagnostics"
EXECUTION_DIAGNOSTICS_VERSION = "1.0"
MeasurementValue: TypeAlias = torch.Tensor | list[dict[str | int, int]]


@dataclass(frozen=True)
class MeasurementResult:
    """One backend-neutral result produced from a ``MeasurementNode``."""

    kind: str
    wires: tuple[int, ...]
    value: MeasurementValue
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    statistics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return "1.0"


@dataclass(frozen=True)
class ExecutionResult:
    """Stable result shape shared by local, distributed, and adapter paths."""

    value: torch.Tensor | None = None
    state: torch.Tensor | None = None
    samples: torch.Tensor | None = None
    measurements: tuple[MeasurementResult, ...] = ()
    plan: ExecutionPlan | RuntimePlanContract | MPSProductionPlan | None = None
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
        @overload
        def detach(value: torch.Tensor) -> torch.Tensor: ...

        @overload
        def detach(value: None) -> None: ...

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
        raise ExecutionError(
            "execution result does not expose a statevector; use backend-native "
            "measurements or request a state-producing execution mode"
        )

    def statevector(self) -> torch.Tensor:
        """Return the statevector or fail clearly when it was not produced."""

        return self.to_statevector()

    def require_samples(self) -> torch.Tensor:
        """Return samples or fail clearly when no sample request was executed."""

        if isinstance(self.samples, torch.Tensor):
            return self.samples
        raise ExecutionError("execution result does not contain samples")

    def require_value(self) -> torch.Tensor:
        """Return the differentiable module value or fail when it is absent."""

        if isinstance(self.value, torch.Tensor):
            return self.value
        raise ExecutionError("execution result does not contain a module value")

    @property
    def probabilities(self) -> torch.Tensor:
        """Return the unique requested probability tensor."""

        return self._unique_tensor_measurement("probabilities")

    @property
    def counts(self) -> list[dict[str | int, int]]:
        """Return the unique requested outcome counts."""

        matches = tuple(
            item for item in self.measurements if item.kind in {"counts", "counts_ps"}
        )
        if not matches:
            raise ExecutionError("execution result does not contain counts")
        if len(matches) > 1:
            raise ExecutionError(
                "execution result contains multiple counts outputs; "
                "select one with measurement(...)"
            )
        value = matches[0].value
        if not isinstance(value, list):
            raise ExecutionError("counts output has an invalid result type")
        return value

    @property
    def expectations(self) -> tuple[torch.Tensor, ...]:
        """Return requested expectation values in output order."""

        return tuple(value for _, _, value in self._expectation_outputs())

    def _unique_tensor_measurement(self, kind: str) -> torch.Tensor:
        matches = tuple(item for item in self.measurements if item.kind == kind)
        if not matches:
            raise ExecutionError(f"execution result does not contain {kind}")
        if len(matches) > 1:
            raise ExecutionError(
                f"execution result contains multiple {kind} outputs; "
                "select one with measurement(...)"
            )
        value = matches[0].value
        if not isinstance(value, torch.Tensor):
            raise ExecutionError(f"{kind} output has an invalid result type")
        return value

    def _expectation_outputs(self) -> tuple[tuple[int, str | None, torch.Tensor], ...]:
        grouped: dict[int, list[MeasurementResult]] = {}
        legacy_index = 0
        for item in self.measurements:
            if not item.kind.startswith("expectation"):
                continue
            index = item.metadata.get("fq_output_index")
            if index is None:
                index = legacy_index
                legacy_index += 1
            grouped.setdefault(int(index), []).append(item)
        outputs: list[tuple[int, str | None, torch.Tensor]] = []
        for index, items in sorted(grouped.items()):
            value: torch.Tensor | None = None
            for item in items:
                if not isinstance(item.value, torch.Tensor):
                    raise ExecutionError(
                        "expectation output has an invalid result type"
                    )
                coefficient = float(item.metadata.get("fq_coefficient", 1.0))
                term = coefficient * item.value
                value = term if value is None else value + term
            assert value is not None
            outputs.append((index, items[0].metadata.get("fq_output_name"), value))
        return tuple(outputs)

    def measurement(self, selector: int | str) -> MeasurementResult:
        """Return one requested measurement by position, name, or unique kind."""

        if type(selector) is int:
            try:
                return self.measurements[selector]
            except IndexError as exc:
                raise ExecutionError(
                    f"measurement index {selector} is outside the result"
                ) from exc
        if not isinstance(selector, str) or not selector:
            raise TypeError(
                "measurement selector must be an integer or non-empty string"
            )
        named = tuple(
            item for item in self.measurements if item.metadata.get("name") == selector
        )
        matches = named or tuple(
            item for item in self.measurements if item.kind == selector
        )
        if not matches:
            raise ExecutionError(f"execution result has no measurement {selector!r}")
        if len(matches) > 1:
            raise ExecutionError(
                f"measurement selector {selector!r} is ambiguous; use an index or "
                "unique metadata name"
            )
        return matches[0]

    def expectation(self, selector: int | str | None = None) -> torch.Tensor:
        """Return one expectation tensor from the requested measurements."""

        outputs = self._expectation_outputs()
        if not outputs:
            raise ExecutionError("execution result does not contain an expectation")
        if selector is None:
            if len(outputs) != 1:
                raise ExecutionError(
                    "execution result contains multiple expectations; select one by "
                    "output index or name"
                )
            return outputs[0][2]
        if type(selector) is int:
            matches = tuple(item for item in outputs if item[0] == selector)
        elif isinstance(selector, str) and selector:
            matches = tuple(item for item in outputs if item[1] == selector)
        else:
            raise TypeError(
                "expectation selector must be an integer or non-empty string"
            )
        if len(matches) != 1:
            message = "ambiguous" if matches else "missing"
            raise ExecutionError(f"expectation selector {selector!r} is {message}")
        return matches[0][2]

    def native(self) -> object:
        """Return the explicitly unstable backend-native output when available."""

        native = self.__dict__.get("_native_output")
        if native is None:
            raise ExecutionError(
                "execution result does not retain a backend-native output"
            )
        return native

    def diagnostics(self) -> dict[str, object]:
        """Return the versioned diagnostics envelope.

        Keys inside each diagnostic section remain additive and are not a
        substitute for stable result accessors.
        """

        return {
            "schema": EXECUTION_DIAGNOSTICS_SCHEMA,
            "version": EXECUTION_DIAGNOSTICS_VERSION,
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
            "runtime": dict(self.runtime),
            "compatibility": dict(self.compatibility),
        }

    def summary(self) -> dict[str, Any]:
        runtime = dict(self.runtime)
        summary = {
            "schema": EXECUTION_RESULT_SUMMARY_SCHEMA,
            "version": EXECUTION_RESULT_SUMMARY_VERSION,
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
    """Adapt a backend-native return without materializing hidden state."""

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
            "native_output_adapted": True,
            "source_type": type(output).__name__,
            "full_state_materialized_by_adapter": False,
        },
    )
    object.__setattr__(result, "_native_output", output)
    return result


__all__ = (
    "EXECUTION_DIAGNOSTICS_SCHEMA",
    "EXECUTION_DIAGNOSTICS_VERSION",
    "EXECUTION_RESULT_SUMMARY_SCHEMA",
    "EXECUTION_RESULT_SUMMARY_VERSION",
    "ExecutionResult",
    "MeasurementResult",
    "normalize_execution_result",
)
