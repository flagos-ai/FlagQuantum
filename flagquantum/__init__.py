"""FlagQuantum stable public API."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

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

    ir = import_module(".core.ir", __name__).ensure_circuit_ir(program)
    if compiler is None or compiler == "flagquantum":
        if target is not None:
            raise ValueError(
                "fq.compile target selection requires a named external compiler; "
                "use flagquantum.compiler.compile for manual topology compilation"
            )
        return import_module(".compiler", __name__).compile(ir)
    if not isinstance(compiler, str) or not compiler.strip():
        raise TypeError("compiler must be a non-empty installed compiler name")

    resolved_target: Mapping[str, Any] | None
    if isinstance(target, str):
        provider_name, separator, backend = target.partition(":")
        if separator != ":" or not provider_name or not backend:
            raise ValueError("target must use the form 'provider:backend'")
        if provider_name.lower() != "quafu":
            raise ValueError(f"unsupported compiler target provider {provider_name!r}")
        provider = import_module(".remote", __name__).QuafuProvider()
        resolved_target = {
            "provider": "quafu",
            "backend": backend,
            "chip_info": provider.fetch_chip_info(backend),
        }
    elif target is None:
        resolved_target = None
    elif isinstance(target, Mapping):
        resolved_target = dict(target)
    else:
        raise TypeError("target must be 'provider:backend', a mapping, or None")

    extensions = import_module(".ecosystem.extensions", __name__)
    return extensions.compile_with_extension(
        ir,
        extension=compiler.strip(),
        target=resolved_target,
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
    if name == "run":
        return getattr(import_module(".runtime.execution", __name__), name)
    if name in {"TrainingResult", "train"}:
        return getattr(import_module(".runtime.training", __name__), name)
    if name == "plan":
        return getattr(import_module(".runtime.planner", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    private_names = {name for name in globals() if name.startswith("_")}
    return sorted(private_names | set(__all__))
