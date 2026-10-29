"""Capture a bounded unsigned Python subset without executing function bodies."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn

from .._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from .._hybrid.verifier import Diagnostic, HybridVerificationError
from .profile import _positive_integer, uint_type, verify_construction
from .ranges import unsigned_width


@dataclass(frozen=True)
class _Source:
    node: ast.FunctionDef
    filename: str
    first_line: int
    indentation: int

    def location(self, node: ast.expr | ast.stmt) -> SourceLocation:
        return SourceLocation(
            self.filename,
            self.first_line + node.lineno - 1,
            self.indentation + node.col_offset + 1,
        )


def _error(
    source: _Source, node: ast.expr | ast.stmt, code: str, message: str
) -> NoReturn:
    loc = source.location(node)
    raise HybridVerificationError(
        (
            Diagnostic(
                f"construction.python.{code}",
                f"{loc.filename}:{loc.line}:{loc.column}: {message}",
            ),
        )
    )


def _source(function: Callable[..., object], *, decorated: bool = False) -> _Source:
    if not inspect.isfunction(function) or hasattr(function, "__wrapped__"):
        raise TypeError(
            "capture requires a Python function, not a callable object or method"
        )
    try:
        lines, first = inspect.getsourcelines(function)
    except (OSError, TypeError) as error:
        raise ValueError(
            "Python source is unavailable; define the function in a .py file or a "
            "notebook kernel that retains cell source. There is no execution fallback."
        ) from error
    text = "".join(lines)
    if len(text) > 262144:
        raise ValueError("function source exceeds the 262144-character safety cap")
    indentation = min(len(s) - len(s.lstrip()) for s in lines if s.strip())
    try:
        tree = ast.parse(textwrap.dedent(text))
    except (SyntaxError, RecursionError) as error:
        raise ValueError(
            "source cannot be parsed; simplify the function definition"
        ) from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("capture requires one ordinary function definition")
    node = tree.body[0]
    source = _Source(
        node, inspect.getsourcefile(function) or "<python>", first, indentation
    )
    if sum(1 for _ in ast.walk(node)) > 4096:
        _error(
            source, node, "budget", "source exceeds 4096 AST nodes; split the function"
        )
    if node.name != function.__name__:
        _error(
            source, node, "source", "source name disagrees with the supplied function"
        )
    if node.decorator_list and not (
        decorated
        and len(node.decorator_list) == 1
        and (
            isinstance(node.decorator_list[0], ast.Name)
            and node.decorator_list[0].id == "qfunc"
            or isinstance(node.decorator_list[0], ast.Attribute)
            and node.decorator_list[0].attr == "qfunc"
        )
    ):
        _error(
            source,
            node,
            "decorator",
            "only the explicit trial qfunc decorator is supported",
        )
    args = node.args
    for annotation in (
        node.returns,
        *(arg.annotation for arg in (*args.posonlyargs, *args.args)),
    ):
        if annotation is not None and not (
            isinstance(annotation, ast.Name)
            and annotation.id == "int"
            or isinstance(annotation, ast.Constant)
            and annotation.value == "int"
        ):
            _error(
                source,
                annotation,
                "annotation",
                "only optional int annotations are supported; quantum roles and other annotation semantics require a separate contract",
            )
    if (
        args.vararg
        or args.kwarg
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
    ):
        _error(
            source,
            node,
            "signature",
            "use positional parameters without defaults or variadic arguments",
        )
    statements = node.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    if not statements or not isinstance(statements[-1], ast.Return):
        _error(source, node, "return", "end the function with one return expression")
    for statement in statements[:-1]:
        if not isinstance(statement, ast.Assign):
            _error(
                source,
                statement,
                "statement",
                "use single assignments and one final return; branches, loops and side effects are unsupported",
            )
    return source


@dataclass(frozen=True)
class _QuantumFunction:
    source: _Source

    def __call__(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("qfunc is a captured quantum definition; pass it to synthesize")


def qfunc(function: Callable[..., object]) -> _QuantumFunction:
    """Snapshot a definition's source early and prevent accidental host execution."""
    return _QuantumFunction(_source(function, decorated=True))


Definition = Callable[..., object] | _QuantumFunction
_Expression = tuple[Value, int]


def capture(
    function: Definition,
    inputs: Mapping[str, int],
    *,
    constants: Mapping[str, int] | None = None,
    functions: Sequence[Definition] = (),
) -> HybridProgram:
    """Infer exact widths into existing SSA; explicitly supplied helpers inline.

    The subset is nonnegative integer constants, +, *, literal **2, local single
    assignments, direct helper calls and one return. Constants bind declared entry
    parameters by value. Globals, closures and arbitrary Python are not executed.
    """
    definitions = (function, *functions)
    sources: dict[str, _Source] = {}
    source_nodes = 0
    for definition in definitions:
        source = (
            definition.source
            if isinstance(definition, _QuantumFunction)
            else _source(definition)
        )
        source_nodes += sum(1 for _ in ast.walk(source.node))
        if source_nodes > 4096:
            _error(
                source,
                source.node,
                "budget",
                "registered definitions exceed 4096 source nodes; register only the needed helpers",
            )
        if source.node.name in sources:
            _error(
                source,
                source.node,
                "definition",
                f"duplicate definition {source.node.name!r}; provide each helper once",
            )
        sources[source.node.name] = source
    main = next(iter(sources.values()))
    parameters = tuple(
        arg.arg for arg in (*main.node.args.posonlyargs, *main.node.args.args)
    )
    classic = dict(constants or {})
    if not inputs or any(not isinstance(name, str) for name in (*inputs, *classic)):
        raise ValueError(
            "provide at least one named quantum input and string parameter names"
        )
    if "result" in inputs:
        raise ValueError(
            "input name 'result' is reserved for the return value; rename the parameter"
        )
    if set(inputs).intersection(classic) or set(inputs) | set(classic) != set(
        parameters
    ):
        _error(
            main,
            main.node,
            "parameters",
            "inputs and constants must disjointly cover every entry parameter; fix missing or unknown names",
        )
    for width in inputs.values():
        if type(width) is not int:
            raise TypeError("input width must be a plain Python integer")
        _positive_integer(width, "input width")
        if width > 256:
            raise ValueError("input width exceeds the 256-bit construction safety cap")
    operations: list[Operation] = []
    counter = 0

    def value(width: int | None = None) -> Value:
        nonlocal counter
        result = Value(
            ValueId(main.node.name, counter),
            INDEX if width is None else uint_type(width),
        )
        counter += 1
        return result

    arguments = tuple(
        Value(ValueId(name, index), uint_type(inputs[name]))
        for index, name in enumerate(n for n in parameters if n in inputs)
    )
    counter = len(arguments)
    environment: dict[str, _Expression] = {
        name: (argument, (1 << inputs[name]) - 1)
        for name, argument in zip(
            (n for n in parameters if n in inputs), arguments, strict=True
        )
    }

    def emit(operation: Operation, source: _Source, node: ast.expr | ast.stmt) -> None:
        if len(operations) >= 4095:
            _error(
                source,
                node,
                "budget",
                "expanded source exceeds 4096 operations; reduce helper expansion",
            )
        operations.append(operation)

    def constant(
        number: object, source: _Source, node: ast.expr | ast.stmt
    ) -> _Expression:
        if type(number) is not int or number < 0:
            _error(
                source,
                node,
                "constant",
                "use a nonnegative Python integer; trainable tensors and lossy conversion are unsupported",
            )
        if number.bit_length() > 256:
            _error(source, node, "budget", "integer exceeds the 256-bit safety cap")
        result = value()
        emit(
            Operation(
                "arith.constant",
                results=(result,),
                attributes={"value": number},
                location=source.location(node),
            ),
            source,
            node,
        )
        return result, number

    for name, number in classic.items():
        environment[name] = constant(number, main, main.node)
    stack: list[str] = []

    def expression(
        node: ast.expr, env: Mapping[str, _Expression], source: _Source, depth: int = 0
    ) -> _Expression:
        if depth > 64:
            _error(
                source,
                node,
                "budget",
                "expression nesting exceeds 64; split it into local assignments",
            )
        if isinstance(node, ast.Name):
            if node.id not in env:
                _error(
                    source,
                    node,
                    "name",
                    f"unknown name {node.id!r}; declare a parameter or assign it once locally",
                )
            return env[node.id]
        if isinstance(node, ast.Constant):
            return constant(node.value, source, node)
        if isinstance(node, ast.BinOp):
            left = expression(node.left, env, source, depth + 1)
            if isinstance(node.op, ast.Pow):
                if (
                    not isinstance(node.right, ast.Constant)
                    or type(node.right.value) is not int
                    or node.right.value != 2
                ):
                    _error(
                        source,
                        node,
                        "power",
                        "only literal **2 is supported; use an explicit supported product",
                    )
                right, opcode = left, "qint.multiply"
            elif isinstance(node.op, (ast.Add, ast.Mult)):
                right = expression(node.right, env, source, depth + 1)
                opcode = "qint.add" if isinstance(node.op, ast.Add) else "qint.multiply"
            else:
                _error(
                    source,
                    node,
                    "operator",
                    "only +, * and literal **2 are supported; unsigned subtraction/division/modulo need a separate contract",
                )
            bound = left[1] + right[1] if opcode == "qint.add" else left[1] * right[1]
            if unsigned_width(bound) > 256:
                _error(
                    source, node, "budget", "expression exceeds the 256-bit safety cap"
                )
            if left[0].type == right[0].type == INDEX:
                return constant(bound, source, node)
            result = value(unsigned_width(bound))
            emit(
                Operation(
                    opcode,
                    (left[0], right[0]),
                    (result,),
                    location=source.location(node),
                ),
                source,
                node,
            )
            return result, bound
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            callee = sources.get(node.func.id)
            if callee is None:
                _error(
                    source,
                    node,
                    "call",
                    f"unknown helper {node.func.id!r}; register it explicitly with functions=(helper,)",
                )
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                _error(
                    source,
                    node,
                    "call",
                    "helper calls accept positional arguments without unpacking",
                )
            args = tuple(expression(arg, env, source, depth + 1) for arg in node.args)
            names = tuple(
                arg.arg
                for arg in (*callee.node.args.posonlyargs, *callee.node.args.args)
            )
            if len(args) != len(names):
                _error(
                    source,
                    node,
                    "call",
                    f"{node.func.id!r} needs {len(names)} positional arguments",
                )
            return body(callee, dict(zip(names, args, strict=True)))
        _error(
            source,
            node,
            "syntax",
            f"unsupported {type(node).__name__}; use named integer values, arithmetic and explicitly registered direct helpers",
        )

    def body(source: _Source, env: dict[str, _Expression]) -> _Expression:
        name = source.node.name
        if name in stack or len(stack) >= 16:
            _error(
                source,
                source.node,
                "recursion",
                "recursive or overly deep helper calls are unsupported; use an acyclic definition set",
            )
        stack.append(name)
        statements = source.node.body
        for statement in statements:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, ast.Return):
                if statement.value is None:
                    _error(
                        source,
                        statement,
                        "return",
                        "return one supported integer expression",
                    )
                result = expression(statement.value, env, source)
                stack.pop()
                return result
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                _error(
                    source,
                    statement,
                    "assignment",
                    "use one local name = expression; tuple, attribute and augmented assignments are unsupported",
                )
            target = statement.targets[0].id
            if target in env:
                _error(
                    source,
                    statement,
                    "assignment",
                    f"{target!r} is already bound; assign a new local name",
                )
            env[target] = expression(statement.value, env, source)
        raise AssertionError("validated source did not return")

    result, _ = body(main, environment)
    if result.type == INDEX:
        _error(
            main,
            main.node.body[-1],
            "return",
            "the entry must return a quantum integer expression; a fixed value can be written as x*0 + value",
        )
    emit(
        Operation("qint.return", (result,), location=main.location(main.node.body[-1])),
        main,
        main.node,
    )
    program = HybridProgram(
        main.node.name, Region((Block(arguments, tuple(operations)),))
    )
    return verify_construction((program,), entry=program.name)
