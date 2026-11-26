"""Capture state, SSA emission and statement dispatch for hybrid source."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any, NoReturn

from flagquantum.compiler.frontends._structured.frontend import (
    control,
    expressions,
    quantum,
)
from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    CaptureDiagnostic,
    HybridCaptureError,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import _Binding
from flagquantum.compiler.frontends._structured.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from flagquantum.compiler.frontends._structured.verifier import verify_program


class _Capture:
    def __init__(self, *, filename: str, line_offset: int) -> None:
        self.filename = filename
        self.line_offset = line_offset
        self.scope = "entry"
        self.environment: dict[str, _Binding] = {}
        self.operations: list[Operation] = []
        self.effect: Value | None = None
        self._value_counters: dict[str, int] = {}
        self._region_counter = 0

    def location(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            self.filename,
            self.line_offset + int(getattr(node, "lineno", 1)),
            int(getattr(node, "col_offset", 0)),
        )

    def fail(self, node: ast.AST, code: str, message: str) -> NoReturn:
        raise HybridCaptureError(CaptureDiagnostic(code, message, self.location(node)))

    def new_value(self, value_type: IRType, *, scope: str | None = None) -> Value:
        owner = self.scope if scope is None else scope
        index = self._value_counters.get(owner, 0)
        self._value_counters[owner] = index + 1
        return Value(ValueId(owner, index), value_type)

    def emit(
        self,
        node: ast.AST,
        name: str,
        *,
        operands: Sequence[Value] = (),
        result_types: Sequence[IRType] = (),
        attributes: dict[str, Any] | None = None,
        regions: Sequence[Region] = (),
    ) -> tuple[Value, ...]:
        results = tuple(self.new_value(value_type) for value_type in result_types)
        self.operations.append(
            Operation(
                name,
                operands=tuple(operands),
                results=results,
                attributes={} if attributes is None else attributes,
                regions=tuple(regions),
                location=self.location(node),
            )
        )
        return results

    def capture(
        self, function: ast.FunctionDef, input_types: Sequence[IRType]
    ) -> HybridProgram:
        if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            self.fail(
                function,
                "function.signature",
                "variadic and keyword-only arguments are unsupported",
            )
        arguments = function.args.posonlyargs + function.args.args
        if function.args.defaults or function.args.kw_defaults:
            self.fail(
                function, "function.defaults", "default arguments are unsupported"
            )
        if len(arguments) != len(input_types):
            self.fail(
                function,
                "function.input_types",
                f"expected {len(arguments)} input type(s), got {len(input_types)}",
            )

        block_arguments = []
        for argument, value_type in zip(arguments, input_types, strict=True):
            value = self.new_value(value_type)
            self.environment[argument.arg] = value
            block_arguments.append(value)
        self.effect = self.new_value(QUANTUM_EFFECT)
        block_arguments.append(self.effect)

        body = function.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        self.capture_statements(body, allow_return=True)
        if not self.operations or self.operations[-1].name != "program.return":
            self.fail(
                function,
                "function.return",
                "captured function must end with one supported scalar return",
            )
        program = HybridProgram(
            function.name,
            Region((Block(tuple(block_arguments), tuple(self.operations)),)),
            location=self.location(function),
        )
        return verify_program(program)

    def capture_statements(
        self, statements: Sequence[ast.stmt], *, allow_return: bool
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.Assign):
                self.capture_assignment(statement)
            elif isinstance(statement, ast.Expr):
                if not isinstance(statement.value, ast.Call):
                    self.fail(
                        statement,
                        "statement.expression",
                        "only supported quantum calls may be used as statements",
                    )
                quantum.capture_quantum_call(self, statement.value)
            elif isinstance(statement, ast.If):
                control.capture_if(self, statement)
            elif isinstance(statement, ast.For):
                control.capture_for(self, statement)
            elif isinstance(statement, ast.Return):
                if not allow_return:
                    self.fail(
                        statement,
                        "control.return",
                        "return inside structured control flow is unsupported",
                    )
                quantum.capture_return(self, statement)
            elif isinstance(statement, ast.Pass):
                continue
            elif isinstance(statement, ast.While):
                self.fail(statement, "control.while", "while is unsupported")
            elif isinstance(statement, (ast.Break, ast.Continue)):
                self.fail(
                    statement,
                    "control.loop_exit",
                    "break and continue are unsupported",
                )
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                self.fail(statement, "effect.import", "imports are unsupported")
            elif isinstance(statement, (ast.Global, ast.Nonlocal)):
                self.fail(
                    statement,
                    "effect.external_state",
                    "global and nonlocal state are unsupported",
                )
            elif isinstance(statement, (ast.Try, ast.Raise, ast.Assert)):
                self.fail(statement, "effect.exception", "exceptions are unsupported")
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                self.fail(
                    statement, "effect.context", "context managers are unsupported"
                )
            elif isinstance(statement, (ast.AugAssign, ast.AnnAssign, ast.Delete)):
                self.fail(
                    statement,
                    "effect.mutation",
                    "mutation and annotated assignment are unsupported",
                )
            else:
                self.fail(
                    statement,
                    "statement.unsupported",
                    f"unsupported statement {type(statement).__name__}",
                )

    def capture_assignment(self, statement: ast.Assign) -> None:
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], ast.Name
        ):
            self.fail(
                statement,
                "effect.mutation",
                "assignment is limited to one local name",
            )
        self.environment[statement.targets[0].id] = expressions.emit_expression(
            self, statement.value
        )

    def require_effect(self, node: ast.AST) -> Value:
        if self.effect is None:
            self.fail(
                node, "quantum.effect", "quantum effect has already been consumed"
            )
        return self.effect

    def _next_region_serial(self) -> int:
        serial = self._region_counter
        self._region_counter += 1
        return serial
