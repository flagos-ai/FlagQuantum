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


def _annotation_name(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return None


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
    annotations = (
        (node.returns, {"int", "bool"}),
        *((arg.annotation, {"int"}) for arg in (*args.posonlyargs, *args.args)),
    )
    for annotation, allowed in annotations:
        if annotation is not None and _annotation_name(annotation) not in allowed:
            _error(
                source,
                annotation,
                "annotation",
                "use optional int parameters and int/bool return annotations; quantum roles require a separate contract",
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
        if isinstance(statement, ast.For):
            if (
                "range" in function.__code__.co_freevars
                or dict.get(function.__globals__, "range", range) is not range
            ):
                _error(
                    source,
                    statement,
                    "loop.range",
                    "builtin range cannot be replaced by a global or closure binding",
                )
            if (
                not isinstance(statement.target, ast.Name)
                or not isinstance(statement.iter, ast.Call)
                or not isinstance(statement.iter.func, ast.Name)
                or statement.iter.func.id != "range"
                or not 1 <= len(statement.iter.args) <= 3
                or statement.iter.keywords
                or any(isinstance(arg, ast.Starred) for arg in statement.iter.args)
                or statement.orelse
            ):
                _error(
                    source,
                    statement,
                    "loop.range",
                    "use one index and builtin range with 1-3 positional arguments, without loop else",
                )
            if any(not isinstance(child, ast.Assign) for child in statement.body):
                _error(
                    source,
                    statement,
                    "loop.body",
                    "loop bodies accept assignments only; nested loops, branches, return and effects are unsupported",
                )
        elif not isinstance(statement, ast.Assign):
            _error(
                source,
                statement,
                "statement",
                "use single assignments, bounded range loops and one final return; branches, dynamic loops and side effects are unsupported",
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

    The subset includes unsigned arithmetic, comparison chains, one-bit
    and/or/not, conditional values, static ranges, assignments, helpers and return.
    Constants bind declared entry parameters by value. Globals, closures and arbitrary Python are not executed.
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
    loop_depth = 0
    loop_iterations = 0

    def one_bit(operand: _Expression, source: _Source, node: ast.expr) -> _Expression:
        if operand[1] > 1 or (
            operand[0].type != INDEX and operand[0].type != uint_type(1)
        ):
            _error(
                source,
                node,
                "boolean",
                "and/or/not require one-bit unsigned operands or numeric 0/1; compare multi-bit integers explicitly",
            )
        return operand

    def negate(operand: _Expression, source: _Source, node: ast.expr) -> _Expression:
        if operand[0].type == INDEX:
            return constant(1 - operand[1], source, node)
        zero = constant(0, source, node)
        result = value(1)
        emit(
            Operation(
                "qint.equal",
                (operand[0], zero[0]),
                (result,),
                location=source.location(node),
            ),
            source,
            node,
        )
        return result, 1

    def conjoin(
        left: _Expression, right: _Expression, source: _Source, node: ast.expr
    ) -> _Expression:
        bound = left[1] * right[1]
        if left[0].type == right[0].type == INDEX:
            return constant(bound, source, node)
        result = value(1)
        emit(
            Operation(
                "qint.multiply",
                (left[0], right[0]),
                (result,),
                location=source.location(node),
            ),
            source,
            node,
        )
        return result, bound

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
        if isinstance(node, ast.IfExp):
            predicate = expression(node.test, env, source, depth + 1)
            if predicate[1] > 1 or (
                predicate[0].type != INDEX and predicate[0].type != uint_type(1)
            ):
                _error(
                    source,
                    node.test,
                    "conditional",
                    "conditional predicate requires one bit or numeric 0/1; compare multi-bit integers explicitly",
                )
            yes = expression(node.body, env, source, depth + 1)
            no = expression(node.orelse, env, source, depth + 1)
            if predicate[0].type == INDEX:
                return yes if predicate[1] else no
            bound = max(yes[1], no[1])
            result = value(unsigned_width(bound))
            emit(
                Operation(
                    "qint.select",
                    (predicate[0], yes[0], no[0]),
                    (result,),
                    location=source.location(node),
                ),
                source,
                node,
            )
            return result, bound
        if isinstance(node, ast.Compare):
            if any(
                not isinstance(
                    op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                )
                for op in node.ops
            ):
                _error(
                    source,
                    node,
                    "comparison",
                    "use unsigned ==, !=, <, <=, > or >= comparisons; identity and membership are unsupported",
                )
            previous = expression(node.left, env, source, depth + 1)
            combined: _Expression | None = None
            for op, child in zip(node.ops, node.comparators, strict=True):
                following = expression(child, env, source, depth + 1)
                left, right = previous, following
                equal = isinstance(op, (ast.Eq, ast.NotEq))
                if isinstance(op, (ast.Gt, ast.LtE)):
                    left, right = right, left
                if left[0].type == right[0].type == INDEX:
                    comparison = constant(
                        int(left[1] == right[1] if equal else left[1] < right[1]),
                        source,
                        node,
                    )
                else:
                    result = value(1)
                    emit(
                        Operation(
                            "qint.equal" if equal else "qint.less",
                            (left[0], right[0]),
                            (result,),
                            location=source.location(node),
                        ),
                        source,
                        node,
                    )
                    comparison = result, 1
                if isinstance(op, (ast.NotEq, ast.LtE, ast.GtE)):
                    comparison = negate(comparison, source, node)
                combined = (
                    comparison
                    if combined is None
                    else conjoin(combined, comparison, source, node)
                )
                previous = following
            assert combined is not None
            return combined
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = one_bit(
                expression(node.operand, env, source, depth + 1), source, node.operand
            )
            return negate(operand, source, node)
        if isinstance(node, ast.BoolOp):
            left = one_bit(
                expression(node.values[0], env, source, depth + 1),
                source,
                node.values[0],
            )
            for child in node.values[1:]:
                right = one_bit(
                    expression(child, env, source, depth + 1), source, child
                )
                if isinstance(node.op, ast.Or):
                    left, right = negate(left, source, node), negate(
                        right, source, node
                    )
                left = conjoin(left, right, source, node)
                if isinstance(node.op, ast.Or):
                    left = negate(left, source, node)
            return left
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
            if node.func.id in env:
                _error(
                    source,
                    node,
                    "call",
                    "helper names cannot be shadowed by value bindings",
                )
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

    def assignment(
        statement: ast.stmt,
        env: dict[str, _Expression],
        source: _Source,
        *,
        updates: frozenset[str] = frozenset(),
        written: set[str] | None = None,
    ) -> None:
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
        if written is not None and (target == "range" or target in sources):
            _error(
                source,
                statement,
                "loop.binding",
                "loop locals cannot shadow range or registered helpers",
            )
        if target in env and (
            target not in updates or written is not None and target in written
        ):
            _error(
                source,
                statement,
                "assignment",
                f"{target!r} is already bound; assign a new local name",
            )
        env[target] = expression(statement.value, env, source)
        if written is not None:
            written.add(target)

    def static_loop(
        statement: ast.For, env: dict[str, _Expression], source: _Source
    ) -> None:
        nonlocal loop_depth, loop_iterations, counter
        if loop_depth:
            _error(
                source,
                statement,
                "loop.nested",
                "only one active loop is supported, including helper bodies",
            )
        assert isinstance(statement.target, ast.Name) and isinstance(
            statement.iter, ast.Call
        )
        index = statement.target.id
        if index in env or index in sources or index == "range":
            _error(
                source,
                statement.target,
                "loop.binding",
                "use a fresh scoped loop index without shadowing inputs, locals or helpers",
            )
        if "range" in env or "range" in sources:
            _error(
                source, statement, "loop.range", "builtin range must not be shadowed"
            )
        before_bounds = len(operations)
        bounds = tuple(expression(arg, env, source) for arg in statement.iter.args)
        if any(item[0].type != INDEX for item in bounds) or any(
            op.name != "arith.constant" for op in operations[before_bounds:]
        ):
            _error(
                source,
                statement.iter,
                "loop.range",
                "range arguments must be explicit classical integers, not quantum values",
            )
        numbers = tuple(item[1] for item in bounds)
        if len(numbers) == 1:
            start, stop, step = 0, numbers[0], 1
        elif len(numbers) == 2:
            start, stop, step = numbers[0], numbers[1], 1
        else:
            start, stop, step = numbers
        if step < 1:
            _error(source, statement.iter, "loop.range", "range step must be positive")
        iterations = max(0, (stop - start + step - 1) // step)
        if iterations > 256 - loop_iterations:
            _error(
                source,
                statement,
                "loop.budget",
                "capture exceeds 256 total loop iterations; reduce static ranges",
            )
        loop_iterations += iterations
        parameters = {
            arg.arg for arg in (*source.node.args.posonlyargs, *source.node.args.args)
        }
        carried = frozenset(env).difference(parameters)
        before_ops, before_counter = len(operations), counter
        loop_depth += 1
        try:
            # Empty loops still validate source, but their provisional SSA cannot escape.
            for number in range(start, stop, step) if iterations else (start,):
                local = dict(env)
                local[index] = constant(number, source, statement.target)
                written: set[str] = set()
                for child in statement.body:
                    assignment(child, local, source, updates=carried, written=written)
                if iterations:
                    for name in written.intersection(carried):
                        env[name] = local[name]
        finally:
            loop_depth -= 1
            if not iterations:
                del operations[before_ops:]
                counter = before_counter

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
                if _annotation_name(source.node.returns) == "bool" and (
                    result[1] > 1
                    or result[0].type != INDEX
                    and result[0].type.parameters != (1,)
                ):
                    _error(
                        source,
                        statement,
                        "return",
                        "bool return requires a one-bit result; return a comparison or use int",
                    )
                stack.pop()
                return result
            if isinstance(statement, ast.For):
                static_loop(statement, env, source)
            else:
                assignment(statement, env, source)
        raise AssertionError("validated source did not return")

    result, _ = body(main, environment)
    if result.type == INDEX:
        _error(
            main,
            main.node.body[-1],
            "return",
            "the entry must return a quantum integer expression; a fixed value can be written as x*0 + value",
        )
    used = {operand.id for op in operations for operand in op.operands}
    used.add(result.id)
    operations[:] = [
        op
        for op in operations
        if op.name != "arith.constant" or op.results[0].id in used
    ]
    emit(
        Operation("qint.return", (result,), location=main.location(main.node.body[-1])),
        main,
        main.node,
    )
    program = HybridProgram(
        main.node.name, Region((Block(arguments, tuple(operations)),))
    )
    return verify_construction((program,), entry=program.name)
