"""Read and select Python source without executing the captured function."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Sequence
from typing import Any

from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    CaptureDiagnostic,
    HybridCaptureError,
)
from flagquantum.compiler.frontends._structured.frontend.program import _Capture
from flagquantum.compiler.frontends._structured.model import (
    HybridProgram,
    IRType,
    SourceLocation,
)


def _function_node(
    source: str, *, function_name: str | None, filename: str
) -> ast.FunctionDef:
    try:
        module = ast.parse(textwrap.dedent(source), filename=filename)
    except SyntaxError as exc:
        location = SourceLocation(
            filename,
            1 if exc.lineno is None else exc.lineno,
            0 if exc.offset is None else max(0, exc.offset - 1),
        )
        raise HybridCaptureError(
            CaptureDiagnostic("python.syntax", exc.msg, location)
        ) from exc
    functions = [item for item in module.body if isinstance(item, ast.FunctionDef)]
    if function_name is not None:
        functions = [item for item in functions if item.name == function_name]
    if len(functions) != 1:
        location = SourceLocation(filename, 1, 0)
        raise HybridCaptureError(
            CaptureDiagnostic(
                "function.selection",
                "source must contain exactly one selected synchronous function",
                location,
            )
        )
    return functions[0]


def capture_source(
    source: str,
    input_types: Sequence[IRType],
    *,
    function_name: str | None = None,
    filename: str = "<string>",
    line_offset: int = 0,
) -> HybridProgram:
    """Capture supported source without executing it or evaluating tensor values."""

    function = _function_node(source, function_name=function_name, filename=filename)
    return _Capture(filename=filename, line_offset=line_offset).capture(
        function, tuple(input_types)
    )


def capture_function(
    function: Callable[..., Any], input_types: Sequence[IRType]
) -> HybridProgram:
    """Capture a Python function's source without invoking the function."""

    try:
        source_lines, start_line = inspect.getsourcelines(function)
        filename = inspect.getsourcefile(function) or "<unknown>"
    except (OSError, TypeError) as exc:
        raise HybridCaptureError(
            CaptureDiagnostic(
                "function.source",
                "source is unavailable for the supplied function",
                SourceLocation("<unknown>", 1, 0),
            )
        ) from exc
    return capture_source(
        "".join(source_lines),
        input_types,
        function_name=function.__name__,
        filename=filename,
        line_offset=start_line - 1,
    )
