"""FlagQuantum stable public API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .circuit import Circuit
    from .core.ir import CircuitIR, MeasurementNode
    from .noise import NoiseModel
    from .runtime.contracts import ExecutionOptions, ExecutionResult
    from .runtime.execution_plan import ExecutionPlan

from .version import __version__

__author__ = "FlagQuantum Team"
__license__ = "Apache-2.0"

# Stable root surface for the first public alpha. Names absent from this tuple
# are not available from the package root.
__all__ = (
    "Circuit",
    "CircuitIR",
    "ExecutionOptions",
    "ExecutionPlan",
    "ExecutionResult",
    "IRSerializationError",
    "IRValidationError",
    "IR_VERSION",
    "Instruction",
    "MeasurementNode",
    "MeasurementResult",
    "Module",
    "ObservableNode",
    "Parameter",
    "ParameterExpression",
    "RuntimePolicy",
    "TrainingResult",
    "compile",
    "plan",
    "run",
    "train",
    "__version__",
    "experimental",
)


def compile(
    program: Any,
    *,
    compiler: str | None = None,
    target: str | Mapping[str, Any] | None = None,
) -> Any:
    """Compile a circuit with FlagQuantum or one named installed compiler."""

    return import_module(".api", __name__).compile_program(
        program, compiler=compiler, target=target
    )


def run(
    program_or_plan: Circuit | CircuitIR | ExecutionPlan,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: NoiseModel | None = None,
    compiler: str | None = None,
    target: str | None = None,
    shots: int | None = None,
) -> ExecutionResult:
    """Execute locally, or compile and execute on one named remote target."""

    return import_module(".api", __name__).run_program(
        program_or_plan,
        options=options,
        measurements=measurements,
        noise_model=noise_model,
        compiler=compiler,
        target=target,
        shots=shots,
    )


def __getattr__(name: str) -> Any:
    if name == "experimental":
        return import_module(".experimental", __name__)
    if name == "Circuit":
        return getattr(import_module(".circuit", __name__), name)
    if name in {
        "CircuitIR",
        "IRSerializationError",
        "IRValidationError",
        "IR_VERSION",
        "Instruction",
        "MeasurementNode",
        "ObservableNode",
    }:
        return getattr(import_module(".core.ir", __name__), name)
    if name in {"Parameter", "ParameterExpression"}:
        return getattr(import_module(".core.parameters", __name__), name)
    if name == "ExecutionPlan":
        return getattr(import_module(".runtime.execution_plan", __name__), name)
    if name in {
        "ExecutionOptions",
        "ExecutionResult",
        "MeasurementResult",
        "Module",
        "RuntimePolicy",
    }:
        return getattr(import_module(".runtime.contracts", __name__), name)
    if name in {"TrainingResult", "train"}:
        return getattr(import_module(".runtime.training", __name__), name)
    if name == "plan":
        return getattr(import_module(".runtime.planner", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    private_names = {name for name in globals() if name.startswith("_")}
    return sorted(private_names | set(__all__))
