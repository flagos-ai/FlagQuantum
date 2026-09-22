"""Fail-closed export from FlagQuantum IR to CUDA-Q kernels."""

from __future__ import annotations

import math
from importlib import import_module
from numbers import Number
from typing import Any

from ...core.ir import Instruction, ensure_circuit_ir
from ...core.operator_schema import get_operator_schema
from ...core.parameters import Parameter, ParameterExpression
from ._version import installed_cudaq_version
from .models import (
    CudaqConversionError,
    CudaqConversionIssue,
    CudaqConversionReport,
    CudaqDependencyError,
    CudaqExportResult,
)

_SUPPORTED = frozenset(
    {"x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap"}
)


def _cudaq() -> Any:
    try:
        return import_module("cudaq")
    except ImportError as exc:
        raise CudaqDependencyError(
            "CUDA-Q export requires the optional heterogeneous toolchain; install "
            "it with `pip install 'flagquantum[cudaq]'`."
        ) from exc


def _issue(
    issues: list[CudaqConversionIssue],
    code: str,
    message: str,
    *,
    index: int | None = None,
    name: str | None = None,
) -> None:
    issues.append(CudaqConversionIssue(code, message, "error", index, name))


def _real_scalar(
    value: Any,
    issues: list[CudaqConversionIssue],
    index: int,
    name: str,
) -> float | None:
    if isinstance(value, (Parameter, ParameterExpression)) or not (
        isinstance(value, Number) or hasattr(value, "item")
    ):
        _issue(
            issues,
            "symbolic_parameter_not_supported",
            "Bind FlagQuantum symbolic parameters before CUDA-Q export.",
            index=index,
            name=name,
        )
        return None
    try:
        converted = complex(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        _issue(
            issues,
            "non_real_parameter",
            f"parameter value {value!r} must be a real scalar",
            index=index,
            name=name,
        )
        return None
    if converted.imag != 0:
        _issue(
            issues,
            "non_real_parameter",
            f"parameter value {value!r} must be real",
            index=index,
            name=name,
        )
        return None
    if not math.isfinite(converted.real):
        _issue(
            issues,
            "non_finite_parameter",
            f"parameter value {value!r} must be finite",
            index=index,
            name=name,
        )
        return None
    return float(converted.real)


def _validate(
    program: Any, version: str | None
) -> tuple[Any, list[tuple[Instruction, tuple[float, ...]]], CudaqConversionReport]:
    ir = ensure_circuit_ir(program)
    issues: list[CudaqConversionIssue] = []
    lowered: list[tuple[Instruction, tuple[float, ...]]] = []
    if ir.observables or ir.measurements:
        _issue(
            issues,
            "measurement_not_supported",
            "Measurements and observables require an execution plan and are not part of CUDA-Q kernel export v1.",
        )
    for index, instruction in enumerate(ir.instructions):
        if instruction.name not in _SUPPORTED or instruction.matrix is not None:
            _issue(
                issues,
                "unsupported_operation",
                f"FlagQuantum operation {instruction.name!r} has no CUDA-Q export v1 lowering.",
                index=index,
                name=instruction.name,
            )
            continue
        schema = get_operator_schema(instruction.name)
        assert schema is not None
        parameters: list[float] = []
        for parameter in schema.parameters:
            scalar = _real_scalar(
                instruction.params.get(parameter), issues, index, instruction.name
            )
            if scalar is None:
                break
            parameters.append(scalar)
        else:
            lowered.append((instruction, tuple(parameters)))
    report = CudaqConversionReport("to_cudaq", version, tuple(issues))
    if not report.lossless:
        codes = ", ".join(issue.code for issue in report.issues)
        raise CudaqConversionError(
            "CUDA-Q export is not lossless; lossy export is intentionally unsupported. "
            f"Issues: {codes}",
            report,
        )
    return ir, lowered, report


def _append(
    kernel: Any, qubits: Any, instruction: Instruction, parameters: tuple[float, ...]
) -> None:
    method = getattr(kernel, instruction.name)
    operands = tuple(qubits[wire] for wire in instruction.wires)
    if parameters:
        method(*parameters, *operands)
    else:
        method(*operands)


def export_cudaq(program: Any, *, allow_lossy: bool = False) -> CudaqExportResult:
    """Export a static unitary program to a CUDA-Q dynamic-builder kernel.

    The ``allow_lossy`` argument satisfies the shared adapter protocol but does
    not weaken this boundary: CUDA-Q export v1 always fails closed.
    """

    del allow_lossy
    cudaq = _cudaq()
    version = installed_cudaq_version(cudaq)
    ir, lowered, report = _validate(program, version)
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(ir.n_wires)
    for instruction, parameters in lowered:
        _append(kernel, qubits, instruction, parameters)
    return CudaqExportResult(kernel, report)


def to_cudaq(program: Any, *, allow_lossy: bool = False) -> Any:
    """Return the CUDA-Q kernel produced by :func:`export_cudaq`."""

    return export_cudaq(program, allow_lossy=allow_lossy).kernel


__all__ = ("export_cudaq", "to_cudaq")
