"""Resolve native annotation symbols and validate quantum function signatures."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Literal

from .frontend_python import _error, _QuantumFunction, _Source
from .program_types import Const, Output, QArray, QBit, QNum, _width_spec


@dataclass(frozen=True)
class _Port:
    name: str
    role: Literal["output", "const", "inout"]
    kind: Literal["qbit", "qnum", "qarray"]
    width: int | None


def _resolve(
    definition: _QuantumFunction,
    node: ast.expr,
    depth: int = 0,
    *,
    annotation: bool = False,
) -> object:
    source = definition.source
    function = definition.function
    if depth >= 32:
        _error(source, node, "budget", "qualified symbol depth exceeds 32")
    if not isinstance(function, FunctionType):
        _error(
            source,
            node,
            "annotation",
            "program capture requires a retained Python namespace",
        )
    if isinstance(node, ast.Name):
        if (
            annotation
            and definition.annotation_bindings is not None
            and node.id in definition.annotation_bindings
        ):
            return definition.annotation_bindings[node.id]
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        ):
            if name == node.id:
                try:
                    return cell.cell_contents
                except ValueError:
                    _error(
                        source, node, "name", "the referenced closure binding is empty"
                    )
        if node.id in function.__globals__:
            return function.__globals__[node.id]
        if annotation and node.id in {"float", "list"}:
            import builtins

            namespace = function.__globals__.get("__builtins__", builtins)
            values = vars(namespace) if type(namespace) is ModuleType else namespace
            return values.get(node.id) if isinstance(values, dict) else None
        return None
    if isinstance(node, ast.Attribute):
        parent = _resolve(definition, node.value, depth + 1, annotation=annotation)
        if type(parent) is ModuleType:
            return vars(parent).get(node.attr)
    _error(
        source,
        node,
        "name",
        "use a static imported symbol, not dynamic attribute access",
    )


def _annotation_node(source: _Source, node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            parsed = ast.parse(node.value, mode="eval").body
        except (SyntaxError, RecursionError):
            _error(source, node, "annotation", "invalid quoted quantum annotation")
        if sum(1 for _ in ast.walk(parsed)) > 4096:
            _error(
                source, node, "budget", "quantum annotation exceeds the source node cap"
            )
        ast.increment_lineno(parsed, node.lineno - 1)
        return parsed
    return node


def program_annotation_names(source: _Source) -> set[str]:
    return {
        node.id
        for arg in (*source.node.args.posonlyargs, *source.node.args.args)
        if arg.annotation is not None
        for node in ast.walk(_annotation_node(source, arg.annotation))
        if isinstance(node, ast.Name)
    }


def classical_parameters(definition: _QuantumFunction) -> dict[str, bool]:
    """Identify finite real scalars and one-dimensional real array annotations."""
    parameters: dict[str, bool] = {}
    for argument in (
        *definition.source.node.args.posonlyargs,
        *definition.source.node.args.args,
    ):
        if argument.annotation is None:
            continue
        node = _annotation_node(definition.source, argument.annotation)
        if isinstance(node, (ast.Name, ast.Attribute)):
            if _resolve(definition, node, annotation=True) is float:
                parameters[argument.arg] = False
        elif isinstance(node, ast.Subscript):
            if (
                _resolve(definition, node.value, annotation=True) is list
                and isinstance(node.slice, (ast.Name, ast.Attribute))
                and _resolve(definition, node.slice, annotation=True) is float
            ):
                parameters[argument.arg] = True
    return parameters


def program_ports(definition: _QuantumFunction) -> tuple[_Port, ...]:
    source = definition.source
    ports: list[_Port] = []
    classical = classical_parameters(definition)
    for argument in (*source.node.args.posonlyargs, *source.node.args.args):
        if argument.arg in classical:
            continue
        annotation = argument.annotation
        if annotation is None:
            _error(
                source,
                source.node,
                "annotation",
                "declare an Output/Const port or a plain quantum helper port",
            )
        node = _annotation_node(source, annotation)
        marker = (
            _resolve(definition, node.value, annotation=True)
            if isinstance(node, ast.Subscript)
            else None
        )
        role: Literal["output", "const", "inout"] = "inout"
        if marker is Output or marker is Const:
            role = "output" if marker is Output else "const"
            assert isinstance(node, ast.Subscript)
            quantum = node.slice
        else:
            quantum = node
        width = None
        if isinstance(quantum, ast.Subscript):
            kind = _resolve(definition, quantum.value, annotation=True)
            if kind is QArray:
                if (
                    not isinstance(quantum.slice, (ast.Name, ast.Attribute))
                    or _resolve(definition, quantum.slice, annotation=True) is not QBit
                ):
                    _error(
                        source,
                        annotation,
                        "annotation",
                        "the array profile supports QArray[QBit] only",
                    )
            elif kind is not QNum:
                _error(
                    source,
                    annotation,
                    "annotation",
                    "only QNum accepts numeric type parameters",
                )
            if kind is QNum:
                sizes = (
                    quantum.slice.elts
                    if isinstance(quantum.slice, ast.Tuple)
                    else (quantum.slice,)
                )
                if any(not isinstance(size, ast.Constant) for size in sizes):
                    _error(
                        source,
                        annotation,
                        "annotation",
                        "numeric widths/sign/fraction must be literals",
                    )
                spec = tuple(
                    size.value for size in sizes if isinstance(size, ast.Constant)
                )
                try:
                    width = _width_spec(spec)
                except (TypeError, ValueError) as error:
                    _error(source, annotation, "annotation", str(error))
        else:
            kind = _resolve(definition, quantum, annotation=True)
            if kind is QArray:
                _error(
                    source,
                    annotation,
                    "annotation",
                    "QArray requires the QBit element parameter",
                )
        if kind is not QNum and kind is not QBit and kind is not QArray:
            _error(
                source,
                annotation,
                "annotation",
                "program ports require QNum, QBit or QArray[QBit]",
            )
        ports.append(
            _Port(
                argument.arg,
                role,
                "qbit" if kind is QBit else "qarray" if kind is QArray else "qnum",
                1 if kind is QBit else width,
            )
        )
    returned = source.node.returns
    if returned is not None and not (
        isinstance(returned, ast.Constant) and returned.value in (None, "None")
    ):
        _error(
            source,
            returned,
            "annotation",
            "program outputs use signature ports; return annotation is None",
        )
    statements = source.node.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    for index, statement in enumerate(statements):
        if (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.op, ast.BitOr)
            and isinstance(statement.target, ast.Name)
        ):
            continue
        if isinstance(statement, (ast.With, ast.For)):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            continue
        if (
            isinstance(statement, ast.Return)
            and index == len(statements) - 1
            and (
                statement.value is None
                or isinstance(statement.value, ast.Constant)
                and statement.value.value is None
            )
        ):
            continue
        _error(
            source,
            statement,
            "program.statement",
            "use allocate, hadamard_transform, fresh |= arithmetic and static qfunc calls",
        )
    return tuple(ports)
