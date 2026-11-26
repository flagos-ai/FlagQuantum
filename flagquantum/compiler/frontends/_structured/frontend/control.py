"""Capture structured regions and their explicitly carried bindings."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.frontend import expressions
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    _Binding,
    _call_name,
    _TensorView,
)
from flagquantum.compiler.frontends._structured.model import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    Block,
    IRType,
    Operation,
    Region,
    Value,
    scalar_type,
)

if TYPE_CHECKING:
    from flagquantum.compiler.frontends._structured.frontend.program import _Capture


def capture_if(owner: _Capture, statement: ast.If) -> None:
    condition = expressions.emit_expression(owner, statement.test)
    if condition.type != BOOL:
        owner.fail(statement.test, "control.condition", "if condition must be bool")
    input_effect = owner.require_effect(statement)
    saved_environment = owner.environment
    carried_names = structured_carried_names(
        owner, (*statement.body, *statement.orelse), saved_environment
    )
    carried_inputs = tuple(
        require_classical_carry(owner, statement, name, saved_environment[name])
        for name in carried_names
    )
    serial = owner._next_region_serial()
    then_region = capture_branch_region(
        owner,
        statement.body,
        f"if{serial}.then",
        origin=statement,
        carried_names=carried_names,
        carried_inputs=carried_inputs,
    )
    else_region = capture_branch_region(
        owner,
        statement.orelse,
        f"if{serial}.else",
        origin=statement,
        carried_names=carried_names,
        carried_inputs=carried_inputs,
    )
    results = tuple(
        owner.new_value(value.type) for value in (*carried_inputs, input_effect)
    )
    owner.operations.append(
        Operation(
            "scf.if",
            operands=(condition, *carried_inputs, input_effect),
            results=results,
            regions=(then_region, else_region),
            location=owner.location(statement),
        )
    )
    owner.environment.update(zip(carried_names, results[:-1], strict=False))
    owner.effect = results[-1]


def capture_branch_region(
    owner: _Capture,
    statements: Sequence[ast.stmt],
    scope: str,
    *,
    origin: ast.AST,
    carried_names: tuple[str, ...],
    carried_inputs: tuple[Value, ...],
) -> Region:
    saved_scope = owner.scope
    saved_environment = owner.environment
    saved_operations = owner.operations
    saved_effect = owner.effect
    block_carried = tuple(
        owner.new_value(value.type, scope=scope) for value in carried_inputs
    )
    block_effect = owner.new_value(QUANTUM_EFFECT, scope=scope)
    owner.scope = scope
    owner.environment = dict(saved_environment)
    owner.environment.update(zip(carried_names, block_carried, strict=True))
    owner.operations = []
    owner.effect = block_effect
    try:
        owner.capture_statements(statements, allow_return=False)
        reject_outer_binding_writes(
            owner,
            saved_environment,
            origin,
            allowed=frozenset(carried_names),
        )
        location_node = statements[-1] if statements else origin
        carried_outputs = tuple(
            require_matching_carry(
                owner,
                location_node,
                name,
                carried_input.type,
                owner.environment.get(name),
            )
            for name, carried_input in zip(carried_names, carried_inputs, strict=True)
        )
        output_effect = owner.require_effect(location_node)
        owner.operations.append(
            Operation(
                "scf.yield",
                operands=(*carried_outputs, output_effect),
                location=owner.location(location_node),
            )
        )
        return Region(
            (
                Block(
                    (*block_carried, block_effect),
                    tuple(owner.operations),
                ),
            )
        )
    finally:
        owner.scope = saved_scope
        owner.environment = saved_environment
        owner.operations = saved_operations
        owner.effect = saved_effect


def structured_carried_names(
    owner: _Capture, statements: Sequence[ast.stmt], outer: dict[str, _Binding]
) -> tuple[str, ...]:
    names: list[str] = []
    for statement in statements:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in outer:
                    if target.id not in names:
                        names.append(target.id)
        elif isinstance(statement, ast.If):
            nested = structured_carried_names(
                owner, (*statement.body, *statement.orelse), outer
            )
            for name in nested:
                if name not in names:
                    names.append(name)
        elif isinstance(statement, ast.For):
            nested = structured_carried_names(owner, statement.body, outer)
            for name in nested:
                if name not in names:
                    names.append(name)
    return tuple(names)


def capture_for(owner: _Capture, statement: ast.For) -> None:
    if statement.orelse:
        owner.fail(statement, "control.for_else", "for-else is unsupported")
    lower, upper, step, descriptor = loop_bounds(owner, statement.iter)
    input_effect = owner.require_effect(statement)
    saved_environment = owner.environment
    carried_names = loop_carried_names(owner, statement, saved_environment)
    carried_inputs = tuple(
        require_classical_carry(owner, statement, name, saved_environment[name])
        for name in carried_names
    )
    serial = owner._next_region_serial()
    loop_scope = f"for{serial}"
    iteration = owner.new_value(INDEX, scope=loop_scope)
    block_carried = tuple(
        owner.new_value(value.type, scope=loop_scope) for value in carried_inputs
    )
    block_effect = owner.new_value(QUANTUM_EFFECT, scope=loop_scope)

    saved_scope = owner.scope
    saved_operations = owner.operations
    saved_effect = owner.effect
    owner.scope = loop_scope
    owner.environment = dict(saved_environment)
    owner.environment.update(zip(carried_names, block_carried, strict=True))
    owner.operations = []
    owner.effect = block_effect
    try:
        bind_loop_target(owner, statement.target, descriptor, iteration)
        owner.capture_statements(statement.body, allow_return=False)
        reject_outer_binding_writes(
            owner,
            saved_environment,
            statement,
            allowed=frozenset(carried_names),
        )
        carried_outputs = tuple(
            require_matching_carry(
                owner,
                statement,
                name,
                carried_input.type,
                owner.environment.get(name),
            )
            for name, carried_input in zip(carried_names, carried_inputs, strict=True)
        )
        output_effect = owner.require_effect(statement)
        owner.operations.append(
            Operation(
                "scf.yield",
                operands=(*carried_outputs, output_effect),
                location=owner.location(statement),
            )
        )
        body = Region(
            (
                Block(
                    (iteration, *block_carried, block_effect),
                    tuple(owner.operations),
                ),
            )
        )
    finally:
        owner.scope = saved_scope
        owner.environment = saved_environment
        owner.operations = saved_operations
        owner.effect = saved_effect

    results = tuple(
        owner.new_value(value.type) for value in (*carried_inputs, input_effect)
    )
    owner.operations.append(
        Operation(
            "scf.for",
            operands=(lower, upper, step, *carried_inputs, input_effect),
            results=results,
            regions=(body,),
            location=owner.location(statement),
        )
    )
    owner.environment.update(zip(carried_names, results[:-1], strict=False))
    owner.effect = results[-1]


def loop_carried_names(
    owner: _Capture, statement: ast.For, outer: dict[str, _Binding]
) -> tuple[str, ...]:
    target_names = {
        node.id for node in ast.walk(statement.target) if isinstance(node, ast.Name)
    }
    shadowed = sorted(target_names & outer.keys())
    if shadowed:
        owner.fail(
            statement.target,
            "control.target_shadow",
            "loop targets may not overwrite outer bindings: " + ", ".join(shadowed),
        )
    return structured_carried_names(owner, statement.body, outer)


def require_classical_carry(
    owner: _Capture, node: ast.AST, name: str, binding: _Binding
) -> Value:
    if (
        not isinstance(binding, Value)
        or binding.type.linear
        or binding.type.kind not in {"scalar", "index", "bool"}
    ):
        owner.fail(
            node,
            "control.classical_carry_type",
            f"carried binding {name!r} must be scalar, index, or bool",
        )
    return binding


def require_matching_carry(
    owner: _Capture,
    node: ast.AST,
    name: str,
    expected_type: IRType,
    binding: _Binding | None,
) -> Value:
    if not isinstance(binding, Value) or binding.type != expected_type:
        owner.fail(
            node,
            "control.classical_carry_type",
            f"carried binding {name!r} must preserve type {expected_type}",
        )
    return binding


def loop_bounds(
    owner: _Capture, node: ast.expr
) -> tuple[Value, Value, Value, tuple[str, _TensorView | None]]:
    if isinstance(node, ast.Call) and _call_name(node) == "range":
        if node.keywords or not 1 <= len(node.args) <= 3:
            owner.fail(node, "control.range", "range accepts one to three arguments")
        raw = list(node.args)
        if len(raw) == 1:
            raw.insert(0, ast.copy_location(ast.Constant(value=0), node))
        if len(raw) == 2:
            raw.append(ast.copy_location(ast.Constant(value=1), node))
        if isinstance(raw[2], ast.Constant) and raw[2].value == 0:
            owner.fail(node, "control.range_step", "range step cannot be zero")
        bounds = tuple(expressions.emit_index(owner, item) for item in raw)
        return bounds[0], bounds[1], bounds[2], ("range", None)

    if isinstance(node, ast.Call) and _call_name(node) == "enumerate":
        if node.keywords or len(node.args) != 1:
            owner.fail(node, "control.enumerate", "enumerate accepts one tensor axis")
        view = expressions.resolve_tensor_view(owner, node.args[0])
        kind = "enumerate"
    else:
        view = expressions.resolve_tensor_view(owner, node)
        kind = "tensor"
    if not view.shape:
        owner.fail(node, "tensor.iterable", "cannot iterate a scalar tensor view")
    zero = ast.copy_location(ast.Constant(value=0), node)
    one = ast.copy_location(ast.Constant(value=1), node)
    return (
        expressions.emit_constant(owner, zero, INDEX),
        expressions.emit_dimension(owner, node, view),
        expressions.emit_constant(owner, one, INDEX),
        (kind, view),
    )


def bind_loop_target(
    owner: _Capture,
    target: ast.expr,
    descriptor: tuple[str, _TensorView | None],
    iteration: Value,
) -> None:
    kind, view = descriptor
    if kind == "range":
        if not isinstance(target, ast.Name):
            owner.fail(target, "control.target", "range loop target must be one name")
        owner.environment[target.id] = iteration
        return
    assert view is not None
    element_view = _TensorView(view.base, view.indices + (iteration,))
    if kind == "tensor":
        if not isinstance(target, ast.Name):
            owner.fail(target, "control.target", "tensor loop target must be one name")
        owner.environment[target.id] = element_view
        return
    if (
        not isinstance(target, (ast.Tuple, ast.List))
        or len(target.elts) != 2
        or not all(isinstance(item, ast.Name) for item in target.elts)
    ):
        owner.fail(
            target,
            "control.target",
            "enumerate target must contain index and tensor element names",
        )
    index_name, element_name = target.elts
    assert isinstance(index_name, ast.Name) and isinstance(element_name, ast.Name)
    owner.environment[index_name.id] = iteration
    if element_view.shape:
        owner.environment[element_name.id] = element_view
    else:
        dtype = view.base.type.parameters[0]
        extracted = owner.emit(
            target,
            "tensor.extract",
            operands=(view.base, *element_view.indices),
            result_types=(scalar_type(dtype),),
        )[0]
        owner.environment[element_name.id] = extracted


def reject_outer_binding_writes(
    owner: _Capture,
    outer: dict[str, _Binding],
    node: ast.AST,
    *,
    allowed: frozenset[str] = frozenset(),
) -> None:
    changed = sorted(
        name
        for name, binding in outer.items()
        if name not in allowed and owner.environment.get(name) != binding
    )
    if changed:
        owner.fail(
            node,
            "control.classical_carry",
            "classical values written inside control flow require explicit "
            f"region-carried results: {', '.join(changed)}",
        )


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    capture_if,
    capture_branch_region,
    structured_carried_names,
    capture_for,
    loop_carried_names,
    require_classical_carry,
    require_matching_carry,
    loop_bounds,
    bind_loop_target,
    reject_outer_binding_writes,
):
    _identity_export.__module__ = "flagquantum.compiler._hybrid.frontend.control"
del _identity_export

_alias_module("flagquantum.compiler._hybrid.frontend.control", __name__)
