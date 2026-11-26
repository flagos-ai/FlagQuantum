"""Capture scalar expressions, tensor indices and quantum operands."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.model import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    IRType,
    Value,
    scalar_type,
)

if TYPE_CHECKING:
    from flagquantum.compiler.frontends._structured.frontend.program import _Capture


@dataclass(frozen=True)
class _TensorView:
    base: Value
    indices: tuple[Value, ...] = ()

    @property
    def shape(self) -> tuple[int | None, ...]:
        shape = self.base.type.parameters[1]
        return tuple(shape[len(self.indices) :])


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


_Binding = Value | _TensorView


def emit_expression(
    owner: _Capture, node: ast.expr, *, expected_type: IRType | None = None
) -> Value:
    if isinstance(node, ast.Name):
        binding = owner.environment.get(node.id)
        if binding is None:
            owner.fail(node, "name.undefined", f"undefined name {node.id!r}")
        if isinstance(binding, _TensorView):
            owner.fail(
                node,
                "tensor.view_value",
                "a tensor-axis view must be indexed or enumerated",
            )
        return binding
    if isinstance(node, ast.Constant):
        return emit_constant(owner, node, expected_type)
    if isinstance(node, ast.Subscript):
        shape_value = emit_shape_dimension(owner, node)
        if shape_value is not None:
            return shape_value
        return emit_tensor_extract(owner, node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        left = emit_expression(owner, node.left)
        right = emit_expression(owner, node.right, expected_type=left.type)
        operation = "arith.add" if isinstance(node.op, ast.Add) else "arith.rem"
        return owner.emit(
            node,
            operation,
            operands=(left, right),
            result_types=(left.type,),
        )[0]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        reject_inline_measurement(owner, node)
        operand = emit_expression(owner, node.operand)
        if operand.type != BOOL:
            owner.fail(node, "arith.bool_type", "not requires one bool operand")
        return owner.emit(
            node,
            "arith.not",
            operands=(operand,),
            result_types=(BOOL,),
        )[0]
    if isinstance(node, ast.BoolOp):
        reject_inline_measurement(owner, node)
        if not isinstance(node.op, (ast.And, ast.Or)) or len(node.values) < 2:
            owner.fail(
                node,
                "arith.boolean_expression",
                "boolean expression requires and/or with at least two operands",
            )
        operation = "arith.and" if isinstance(node.op, ast.And) else "arith.or"
        spelling = "and" if isinstance(node.op, ast.And) else "or"
        result = emit_expression(owner, node.values[0])
        if result.type != BOOL:
            owner.fail(node, "arith.bool_type", f"{spelling} requires bool operands")
        for value_node in node.values[1:]:
            right = emit_expression(owner, value_node)
            if right.type != BOOL:
                owner.fail(
                    node, "arith.bool_type", f"{spelling} requires bool operands"
                )
            result = owner.emit(
                node,
                operation,
                operands=(result, right),
                result_types=(BOOL,),
            )[0]
        return result
    if isinstance(node, ast.Compare):
        reject_inline_measurement(owner, node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            owner.fail(node, "arith.comparison", "chained comparisons are unsupported")
        predicates = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }
        predicate = predicates.get(type(node.ops[0]))
        if predicate is None:
            owner.fail(node, "arith.comparison", "comparison operator is unsupported")
        left = emit_expression(owner, node.left)
        right = emit_expression(owner, node.comparators[0], expected_type=left.type)
        return owner.emit(
            node,
            "arith.cmp",
            operands=(left, right),
            result_types=(BOOL,),
            attributes={"predicate": predicate},
        )[0]
    if isinstance(node, ast.Call):
        name = _call_name(node).lower()
        if name == "measure":
            return capture_measurement(owner, node)
        if name in {"item", "detach", "numpy", "tolist"}:
            owner.fail(
                node,
                "tensor.host_conversion",
                f"tensor host conversion {name!r} is unsupported",
            )
        if name == "mod" and len(node.args) == 2 and not node.keywords:
            left = emit_expression(owner, node.args[0])
            right = emit_expression(owner, node.args[1], expected_type=left.type)
            return owner.emit(
                node,
                "arith.rem",
                operands=(left, right),
                result_types=(left.type,),
            )[0]
        owner.fail(node, "call.unsupported", f"unsupported call {_call_name(node)!r}")
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        owner.fail(node, "expression.comprehension", "comprehensions are unsupported")
    if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
        owner.fail(node, "effect.suspension", "yield and await are unsupported")
    owner.fail(
        node,
        "expression.unsupported",
        f"unsupported expression {type(node).__name__}",
    )


def reject_inline_measurement(owner: _Capture, node: ast.expr) -> None:
    if any(
        isinstance(child, ast.Call)
        and _call_name(child).replace("_", "").lower() == "measure"
        for child in ast.walk(node)
    ):
        owner.fail(
            node,
            "quantum.measurement_expression",
            "measurements must be assigned before boolean composition",
        )


def emit_constant(
    owner: _Capture, node: ast.Constant, expected_type: IRType | None
) -> Value:
    raw = node.value
    if expected_type is not None:
        value_type = expected_type
    elif isinstance(raw, bool):
        value_type = BOOL
    elif isinstance(raw, int):
        value_type = INDEX
    elif isinstance(raw, float):
        value_type = scalar_type("float64")
    else:
        owner.fail(
            node,
            "constant.type",
            f"unsupported constant type {type(raw).__name__}",
        )
    return owner.emit(
        node,
        "arith.constant",
        result_types=(value_type,),
        attributes={"value": raw},
    )[0]


def resolve_tensor_view(owner: _Capture, node: ast.expr) -> _TensorView:
    if isinstance(node, ast.Name):
        binding = owner.environment.get(node.id)
        if isinstance(binding, _TensorView):
            return binding
        if isinstance(binding, Value) and binding.type.kind == "tensor":
            return _TensorView(binding)
    owner.fail(node, "tensor.iterable", "expected a statically ranked tensor axis")


def emit_shape_dimension(owner: _Capture, node: ast.Subscript) -> Value | None:
    attribute = node.value
    if not isinstance(attribute, ast.Attribute) or attribute.attr != "shape":
        return None
    view = resolve_tensor_view(owner, attribute.value)
    dimension = literal_non_negative_int(owner, node.slice, "tensor.shape_dimension")
    if dimension >= len(view.shape):
        owner.fail(node, "tensor.shape_dimension", "tensor dimension is out of range")
    return emit_dimension(owner, node, view, dimension)


def emit_dimension(
    owner: _Capture, node: ast.AST, view: _TensorView, dimension: int = 0
) -> Value:
    size = view.shape[dimension]
    if size is not None:
        constant = ast.Constant(value=size)
        ast.copy_location(constant, node)
        return emit_constant(owner, constant, INDEX)
    absolute_dimension = len(view.indices) + dimension
    dimension_node = ast.Constant(value=absolute_dimension)
    ast.copy_location(dimension_node, node)
    dimension_value = emit_constant(owner, dimension_node, INDEX)
    return owner.emit(
        node,
        "tensor.dim",
        operands=(view.base, dimension_value),
        result_types=(INDEX,),
    )[0]


def emit_tensor_extract(owner: _Capture, node: ast.Subscript) -> Value:
    view = resolve_tensor_view(owner, node.value)
    slices = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
    indices = tuple(emit_index(owner, item) for item in slices)
    all_indices = view.indices + indices
    shape = view.base.type.parameters[1]
    if len(all_indices) != len(shape):
        owner.fail(
            node,
            "tensor.extract_rank",
            "first profile requires scalar extraction with one index per rank",
        )
    dtype = view.base.type.parameters[0]
    return owner.emit(
        node,
        "tensor.extract",
        operands=(view.base, *all_indices),
        result_types=(scalar_type(dtype),),
    )[0]


def emit_index(owner: _Capture, node: ast.expr) -> Value:
    value = emit_expression(owner, node, expected_type=INDEX)
    if value.type != INDEX:
        owner.fail(node, "index.type", "wire and tensor indices must have index type")
    return value


def capture_measurement(owner: _Capture, call: ast.Call) -> Value:
    if call.args:
        owner.fail(
            call,
            "quantum.arguments",
            "measure accepts only the wires keyword",
        )
    wire = dynamic_wires(owner, call, expected=1)[0]
    measured, effect = owner.emit(
        call,
        "quantum.measure",
        operands=(wire, owner.require_effect(call)),
        result_types=(BOOL, QUANTUM_EFFECT),
    )
    owner.effect = effect
    return measured


def wires_node(owner: _Capture, call: ast.Call) -> ast.expr:
    matches = [item.value for item in call.keywords if item.arg == "wires"]
    unknown = [item.arg for item in call.keywords if item.arg != "wires"]
    if unknown or len(matches) != 1:
        owner.fail(call, "quantum.wires", "exactly one wires keyword is required")
    return matches[0]


def static_wires(owner: _Capture, call: ast.Call) -> tuple[int, ...]:
    node = wires_node(owner, call)
    if isinstance(node, ast.Call) and _call_name(node) == "range":
        try:
            arguments = [ast.literal_eval(item) for item in node.args]
            wires = tuple(range(*arguments))
        except (TypeError, ValueError) as exc:
            owner.fail(node, "quantum.wires", f"wires range must be static: {exc}")
    elif isinstance(node, (ast.List, ast.Tuple)):
        wires = tuple(
            literal_non_negative_int(owner, item, "quantum.wires") for item in node.elts
        )
    else:
        wires = (literal_non_negative_int(owner, node, "quantum.wires"),)
    if not wires or len(set(wires)) != len(wires):
        owner.fail(node, "quantum.wires", "static wires must be unique and non-empty")
    return wires


def dynamic_wires(
    owner: _Capture, call: ast.Call, *, expected: int
) -> tuple[Value, ...]:
    node = wires_node(owner, call)
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else (node,)
    if len(items) != expected:
        owner.fail(node, "quantum.wires", f"expected {expected} wire expression(s)")
    return tuple(emit_index(owner, item) for item in items)


def literal_non_negative_int(owner: _Capture, node: ast.expr, code: str) -> int:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        owner.fail(node, code, "expected a static non-negative integer")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        owner.fail(node, code, "expected a static non-negative integer")
    return value


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    _TensorView,
    _call_name,
    emit_expression,
    reject_inline_measurement,
    emit_constant,
    resolve_tensor_view,
    emit_shape_dimension,
    emit_dimension,
    emit_tensor_extract,
    emit_index,
    capture_measurement,
    wires_node,
    static_wires,
    dynamic_wires,
    literal_non_negative_int,
):
    _identity_export.__module__ = "flagquantum.compiler._hybrid.frontend.expressions"
del _identity_export

_alias_module("flagquantum.compiler._hybrid.frontend.expressions", __name__)
