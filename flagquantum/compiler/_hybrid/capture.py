"""Restricted, non-executing Python AST capture for private hybrid programs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, NoReturn, Sequence

from .model import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
    scalar_type,
)
from .verifier import verify_program


@dataclass(frozen=True)
class CaptureDiagnostic:
    """One source-located restricted-capture failure."""

    code: str
    message: str
    location: SourceLocation


class HybridCaptureError(ValueError):
    """Raised when Python source exceeds the supported capture profile."""

    def __init__(self, diagnostic: CaptureDiagnostic) -> None:
        self.diagnostic = diagnostic
        location = diagnostic.location
        super().__init__(
            f"{location.filename}:{location.line}:{location.column}: "
            f"{diagnostic.code}: {diagnostic.message}"
        )


@dataclass(frozen=True)
class _TensorView:
    base: Value
    indices: tuple[Value, ...] = ()

    @property
    def shape(self) -> tuple[int | None, ...]:
        shape = self.base.type.parameters[1]
        return tuple(shape[len(self.indices) :])


_Binding = Value | _TensorView


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


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
        for argument, value_type in zip(arguments, input_types):
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
                self.capture_quantum_call(statement.value)
            elif isinstance(statement, ast.If):
                self.capture_if(statement)
            elif isinstance(statement, ast.For):
                self.capture_for(statement)
            elif isinstance(statement, ast.Return):
                if not allow_return:
                    self.fail(
                        statement,
                        "control.return",
                        "return inside structured control flow is unsupported",
                    )
                self.capture_return(statement)
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
        self.environment[statement.targets[0].id] = self.emit_expression(
            statement.value
        )

    def emit_expression(
        self, node: ast.expr, *, expected_type: IRType | None = None
    ) -> Value:
        if isinstance(node, ast.Name):
            binding = self.environment.get(node.id)
            if binding is None:
                self.fail(node, "name.undefined", f"undefined name {node.id!r}")
            if isinstance(binding, _TensorView):
                self.fail(
                    node,
                    "tensor.view_value",
                    "a tensor-axis view must be indexed or enumerated",
                )
            return binding
        if isinstance(node, ast.Constant):
            return self.emit_constant(node, expected_type)
        if isinstance(node, ast.Subscript):
            shape_value = self.emit_shape_dimension(node)
            if shape_value is not None:
                return shape_value
            return self.emit_tensor_extract(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
            left = self.emit_expression(node.left)
            right = self.emit_expression(node.right, expected_type=left.type)
            operation = "arith.add" if isinstance(node.op, ast.Add) else "arith.rem"
            return self.emit(
                node,
                operation,
                operands=(left, right),
                result_types=(left.type,),
            )[0]
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                self.fail(
                    node, "arith.comparison", "chained comparisons are unsupported"
                )
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
                self.fail(
                    node, "arith.comparison", "comparison operator is unsupported"
                )
            left = self.emit_expression(node.left)
            right = self.emit_expression(node.comparators[0], expected_type=left.type)
            return self.emit(
                node,
                "arith.cmp",
                operands=(left, right),
                result_types=(BOOL,),
                attributes={"predicate": predicate},
            )[0]
        if isinstance(node, ast.Call):
            name = _call_name(node).lower()
            if name == "measure":
                return self.capture_measurement(node)
            if name in {"item", "detach", "numpy", "tolist"}:
                self.fail(
                    node,
                    "tensor.host_conversion",
                    f"tensor host conversion {name!r} is unsupported",
                )
            if name == "mod" and len(node.args) == 2 and not node.keywords:
                left = self.emit_expression(node.args[0])
                right = self.emit_expression(node.args[1], expected_type=left.type)
                return self.emit(
                    node,
                    "arith.rem",
                    operands=(left, right),
                    result_types=(left.type,),
                )[0]
            self.fail(
                node, "call.unsupported", f"unsupported call {_call_name(node)!r}"
            )
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            self.fail(
                node, "expression.comprehension", "comprehensions are unsupported"
            )
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            self.fail(node, "effect.suspension", "yield and await are unsupported")
        self.fail(
            node,
            "expression.unsupported",
            f"unsupported expression {type(node).__name__}",
        )

    def emit_constant(self, node: ast.Constant, expected_type: IRType | None) -> Value:
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
            self.fail(
                node,
                "constant.type",
                f"unsupported constant type {type(raw).__name__}",
            )
        return self.emit(
            node,
            "arith.constant",
            result_types=(value_type,),
            attributes={"value": raw},
        )[0]

    def resolve_tensor_view(self, node: ast.expr) -> _TensorView:
        if isinstance(node, ast.Name):
            binding = self.environment.get(node.id)
            if isinstance(binding, _TensorView):
                return binding
            if isinstance(binding, Value) and binding.type.kind == "tensor":
                return _TensorView(binding)
        self.fail(node, "tensor.iterable", "expected a statically ranked tensor axis")

    def emit_shape_dimension(self, node: ast.Subscript) -> Value | None:
        attribute = node.value
        if not isinstance(attribute, ast.Attribute) or attribute.attr != "shape":
            return None
        view = self.resolve_tensor_view(attribute.value)
        dimension = self.literal_non_negative_int(node.slice, "tensor.shape_dimension")
        if dimension >= len(view.shape):
            self.fail(
                node, "tensor.shape_dimension", "tensor dimension is out of range"
            )
        return self.emit_dimension(node, view, dimension)

    def emit_dimension(
        self, node: ast.AST, view: _TensorView, dimension: int = 0
    ) -> Value:
        size = view.shape[dimension]
        if size is not None:
            constant = ast.Constant(value=size)
            ast.copy_location(constant, node)
            return self.emit_constant(constant, INDEX)
        absolute_dimension = len(view.indices) + dimension
        dimension_node = ast.Constant(value=absolute_dimension)
        ast.copy_location(dimension_node, node)
        dimension_value = self.emit_constant(dimension_node, INDEX)
        return self.emit(
            node,
            "tensor.dim",
            operands=(view.base, dimension_value),
            result_types=(INDEX,),
        )[0]

    def emit_tensor_extract(self, node: ast.Subscript) -> Value:
        view = self.resolve_tensor_view(node.value)
        slices = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        indices = tuple(self.emit_index(item) for item in slices)
        all_indices = view.indices + indices
        shape = view.base.type.parameters[1]
        if len(all_indices) != len(shape):
            self.fail(
                node,
                "tensor.extract_rank",
                "first profile requires scalar extraction with one index per rank",
            )
        dtype = view.base.type.parameters[0]
        return self.emit(
            node,
            "tensor.extract",
            operands=(view.base, *all_indices),
            result_types=(scalar_type(dtype),),
        )[0]

    def emit_index(self, node: ast.expr) -> Value:
        value = self.emit_expression(node, expected_type=INDEX)
        if value.type != INDEX:
            self.fail(
                node, "index.type", "wire and tensor indices must have index type"
            )
        return value

    def capture_if(self, statement: ast.If) -> None:
        condition = self.emit_expression(statement.test)
        if condition.type != BOOL:
            self.fail(statement.test, "control.condition", "if condition must be bool")
        input_effect = self.require_effect(statement)
        saved_environment = self.environment
        carried_names = self.structured_carried_names(
            (*statement.body, *statement.orelse), saved_environment
        )
        carried_inputs = tuple(
            self.require_classical_carry(statement, name, saved_environment[name])
            for name in carried_names
        )
        serial = self._next_region_serial()
        then_region = self.capture_branch_region(
            statement.body,
            f"if{serial}.then",
            origin=statement,
            carried_names=carried_names,
            carried_inputs=carried_inputs,
        )
        else_region = self.capture_branch_region(
            statement.orelse,
            f"if{serial}.else",
            origin=statement,
            carried_names=carried_names,
            carried_inputs=carried_inputs,
        )
        results = tuple(
            self.new_value(value.type) for value in (*carried_inputs, input_effect)
        )
        self.operations.append(
            Operation(
                "scf.if",
                operands=(condition, *carried_inputs, input_effect),
                results=results,
                regions=(then_region, else_region),
                location=self.location(statement),
            )
        )
        self.environment.update(zip(carried_names, results[:-1]))
        self.effect = results[-1]

    def capture_branch_region(
        self,
        statements: Sequence[ast.stmt],
        scope: str,
        *,
        origin: ast.AST,
        carried_names: tuple[str, ...],
        carried_inputs: tuple[Value, ...],
    ) -> Region:
        saved_scope = self.scope
        saved_environment = self.environment
        saved_operations = self.operations
        saved_effect = self.effect
        block_carried = tuple(
            self.new_value(value.type, scope=scope) for value in carried_inputs
        )
        block_effect = self.new_value(QUANTUM_EFFECT, scope=scope)
        self.scope = scope
        self.environment = dict(saved_environment)
        self.environment.update(zip(carried_names, block_carried))
        self.operations = []
        self.effect = block_effect
        try:
            self.capture_statements(statements, allow_return=False)
            self.reject_outer_binding_writes(
                saved_environment,
                origin,
                allowed=frozenset(carried_names),
            )
            location_node = statements[-1] if statements else origin
            carried_outputs = tuple(
                self.require_matching_carry(
                    location_node,
                    name,
                    carried_input.type,
                    self.environment.get(name),
                )
                for name, carried_input in zip(carried_names, carried_inputs)
            )
            output_effect = self.require_effect(location_node)
            self.operations.append(
                Operation(
                    "scf.yield",
                    operands=(*carried_outputs, output_effect),
                    location=self.location(location_node),
                )
            )
            return Region(
                (
                    Block(
                        (*block_carried, block_effect),
                        tuple(self.operations),
                    ),
                )
            )
        finally:
            self.scope = saved_scope
            self.environment = saved_environment
            self.operations = saved_operations
            self.effect = saved_effect

    def structured_carried_names(
        self, statements: Sequence[ast.stmt], outer: dict[str, _Binding]
    ) -> tuple[str, ...]:
        names: list[str] = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id in outer:
                        if target.id not in names:
                            names.append(target.id)
            elif isinstance(statement, ast.If):
                nested = self.structured_carried_names(
                    (*statement.body, *statement.orelse), outer
                )
                for name in nested:
                    if name not in names:
                        names.append(name)
            elif isinstance(statement, ast.For):
                nested = self.structured_carried_names(statement.body, outer)
                for name in nested:
                    if name not in names:
                        names.append(name)
        return tuple(names)

    def capture_for(self, statement: ast.For) -> None:
        if statement.orelse:
            self.fail(statement, "control.for_else", "for-else is unsupported")
        lower, upper, step, descriptor = self.loop_bounds(statement.iter)
        input_effect = self.require_effect(statement)
        saved_environment = self.environment
        carried_names = self.loop_carried_names(statement, saved_environment)
        carried_inputs = tuple(
            self.require_classical_carry(statement, name, saved_environment[name])
            for name in carried_names
        )
        serial = self._next_region_serial()
        loop_scope = f"for{serial}"
        iteration = self.new_value(INDEX, scope=loop_scope)
        block_carried = tuple(
            self.new_value(value.type, scope=loop_scope) for value in carried_inputs
        )
        block_effect = self.new_value(QUANTUM_EFFECT, scope=loop_scope)

        saved_scope = self.scope
        saved_operations = self.operations
        saved_effect = self.effect
        self.scope = loop_scope
        self.environment = dict(saved_environment)
        self.environment.update(zip(carried_names, block_carried))
        self.operations = []
        self.effect = block_effect
        try:
            self.bind_loop_target(statement.target, descriptor, iteration)
            self.capture_statements(statement.body, allow_return=False)
            self.reject_outer_binding_writes(
                saved_environment,
                statement,
                allowed=frozenset(carried_names),
            )
            carried_outputs = tuple(
                self.require_matching_carry(
                    statement,
                    name,
                    carried_input.type,
                    self.environment.get(name),
                )
                for name, carried_input in zip(carried_names, carried_inputs)
            )
            output_effect = self.require_effect(statement)
            self.operations.append(
                Operation(
                    "scf.yield",
                    operands=(*carried_outputs, output_effect),
                    location=self.location(statement),
                )
            )
            body = Region(
                (
                    Block(
                        (iteration, *block_carried, block_effect),
                        tuple(self.operations),
                    ),
                )
            )
        finally:
            self.scope = saved_scope
            self.environment = saved_environment
            self.operations = saved_operations
            self.effect = saved_effect

        results = tuple(
            self.new_value(value.type) for value in (*carried_inputs, input_effect)
        )
        self.operations.append(
            Operation(
                "scf.for",
                operands=(lower, upper, step, *carried_inputs, input_effect),
                results=results,
                regions=(body,),
                location=self.location(statement),
            )
        )
        self.environment.update(zip(carried_names, results[:-1]))
        self.effect = results[-1]

    def loop_carried_names(
        self, statement: ast.For, outer: dict[str, _Binding]
    ) -> tuple[str, ...]:
        target_names = {
            node.id for node in ast.walk(statement.target) if isinstance(node, ast.Name)
        }
        shadowed = sorted(target_names & outer.keys())
        if shadowed:
            self.fail(
                statement.target,
                "control.target_shadow",
                "loop targets may not overwrite outer bindings: " + ", ".join(shadowed),
            )
        return self.structured_carried_names(statement.body, outer)

    def require_classical_carry(
        self, node: ast.AST, name: str, binding: _Binding
    ) -> Value:
        if (
            not isinstance(binding, Value)
            or binding.type.linear
            or binding.type.kind not in {"scalar", "index", "bool"}
        ):
            self.fail(
                node,
                "control.classical_carry_type",
                f"carried binding {name!r} must be scalar, index, or bool",
            )
        return binding

    def require_matching_carry(
        self,
        node: ast.AST,
        name: str,
        expected_type: IRType,
        binding: _Binding | None,
    ) -> Value:
        if not isinstance(binding, Value) or binding.type != expected_type:
            self.fail(
                node,
                "control.classical_carry_type",
                f"carried binding {name!r} must preserve type {expected_type}",
            )
        return binding

    def loop_bounds(
        self, node: ast.expr
    ) -> tuple[Value, Value, Value, tuple[str, _TensorView | None]]:
        if isinstance(node, ast.Call) and _call_name(node) == "range":
            if node.keywords or not 1 <= len(node.args) <= 3:
                self.fail(node, "control.range", "range accepts one to three arguments")
            raw = list(node.args)
            if len(raw) == 1:
                raw.insert(0, ast.copy_location(ast.Constant(value=0), node))
            if len(raw) == 2:
                raw.append(ast.copy_location(ast.Constant(value=1), node))
            if isinstance(raw[2], ast.Constant) and raw[2].value == 0:
                self.fail(node, "control.range_step", "range step cannot be zero")
            bounds = tuple(self.emit_index(item) for item in raw)
            return bounds[0], bounds[1], bounds[2], ("range", None)

        if isinstance(node, ast.Call) and _call_name(node) == "enumerate":
            if node.keywords or len(node.args) != 1:
                self.fail(
                    node, "control.enumerate", "enumerate accepts one tensor axis"
                )
            view = self.resolve_tensor_view(node.args[0])
            kind = "enumerate"
        else:
            view = self.resolve_tensor_view(node)
            kind = "tensor"
        if not view.shape:
            self.fail(node, "tensor.iterable", "cannot iterate a scalar tensor view")
        zero = ast.copy_location(ast.Constant(value=0), node)
        one = ast.copy_location(ast.Constant(value=1), node)
        return (
            self.emit_constant(zero, INDEX),
            self.emit_dimension(node, view),
            self.emit_constant(one, INDEX),
            (kind, view),
        )

    def bind_loop_target(
        self,
        target: ast.expr,
        descriptor: tuple[str, _TensorView | None],
        iteration: Value,
    ) -> None:
        kind, view = descriptor
        if kind == "range":
            if not isinstance(target, ast.Name):
                self.fail(
                    target, "control.target", "range loop target must be one name"
                )
            self.environment[target.id] = iteration
            return
        assert view is not None
        element_view = _TensorView(view.base, view.indices + (iteration,))
        if kind == "tensor":
            if not isinstance(target, ast.Name):
                self.fail(
                    target, "control.target", "tensor loop target must be one name"
                )
            self.environment[target.id] = element_view
            return
        if (
            not isinstance(target, (ast.Tuple, ast.List))
            or len(target.elts) != 2
            or not all(isinstance(item, ast.Name) for item in target.elts)
        ):
            self.fail(
                target,
                "control.target",
                "enumerate target must contain index and tensor element names",
            )
        index_name, element_name = target.elts
        assert isinstance(index_name, ast.Name) and isinstance(element_name, ast.Name)
        self.environment[index_name.id] = iteration
        if element_view.shape:
            self.environment[element_name.id] = element_view
        else:
            dtype = view.base.type.parameters[0]
            extracted = self.emit(
                target,
                "tensor.extract",
                operands=(view.base, *element_view.indices),
                result_types=(scalar_type(dtype),),
            )[0]
            self.environment[element_name.id] = extracted

    def capture_quantum_call(self, call: ast.Call) -> None:
        name = _call_name(call).replace("_", "").lower()
        if name == "angleembedding":
            if len(call.args) != 1:
                self.fail(call, "quantum.arguments", "AngleEmbedding requires data")
            data = self.emit_expression(call.args[0])
            wires = self.static_wires(call)
            self.effect = self.emit(
                call,
                "quantum.angle_embedding",
                operands=(data, self.require_effect(call)),
                result_types=(QUANTUM_EFFECT,),
                attributes={"wires": wires},
            )[0]
            return
        if name in {"rx", "ry"}:
            if len(call.args) != 1:
                self.fail(
                    call, "quantum.arguments", f"{name.upper()} requires one parameter"
                )
            parameter = self.emit_expression(call.args[0])
            wire = self.dynamic_wires(call, expected=1)[0]
            self.effect = self.emit(
                call,
                f"quantum.{name}",
                operands=(parameter, wire, self.require_effect(call)),
                result_types=(QUANTUM_EFFECT,),
            )[0]
            return
        if name in {"cx", "cnot"}:
            control, target = self.dynamic_wires(call, expected=2)
            self.effect = self.emit(
                call,
                "quantum.cx",
                operands=(control, target, self.require_effect(call)),
                result_types=(QUANTUM_EFFECT,),
            )[0]
            return
        if name in {"h", "x"}:
            if call.args:
                self.fail(
                    call,
                    "quantum.arguments",
                    f"{name.upper()} accepts only the wires keyword",
                )
            wire = self.dynamic_wires(call, expected=1)[0]
            self.effect = self.emit(
                call,
                f"quantum.{name}",
                operands=(wire, self.require_effect(call)),
                result_types=(QUANTUM_EFFECT,),
            )[0]
            return
        if name == "measure":
            self.fail(
                call,
                "quantum.measurement_value",
                "measure must be assigned to a local name",
            )
        self.fail(
            call,
            "quantum.unsupported",
            f"unsupported quantum call {_call_name(call)!r}",
        )

    def capture_measurement(self, call: ast.Call) -> Value:
        if call.args:
            self.fail(
                call,
                "quantum.arguments",
                "measure accepts only the wires keyword",
            )
        wire = self.dynamic_wires(call, expected=1)[0]
        measured, effect = self.emit(
            call,
            "quantum.measure",
            operands=(wire, self.require_effect(call)),
            result_types=(BOOL, QUANTUM_EFFECT),
        )
        self.effect = effect
        return measured

    def wires_node(self, call: ast.Call) -> ast.expr:
        matches = [item.value for item in call.keywords if item.arg == "wires"]
        unknown = [item.arg for item in call.keywords if item.arg != "wires"]
        if unknown or len(matches) != 1:
            self.fail(call, "quantum.wires", "exactly one wires keyword is required")
        return matches[0]

    def static_wires(self, call: ast.Call) -> tuple[int, ...]:
        node = self.wires_node(call)
        if isinstance(node, ast.Call) and _call_name(node) == "range":
            try:
                arguments = [ast.literal_eval(item) for item in node.args]
                wires = tuple(range(*arguments))
            except (TypeError, ValueError) as exc:
                self.fail(node, "quantum.wires", f"wires range must be static: {exc}")
        elif isinstance(node, (ast.List, ast.Tuple)):
            wires = tuple(
                self.literal_non_negative_int(item, "quantum.wires")
                for item in node.elts
            )
        else:
            wires = (self.literal_non_negative_int(node, "quantum.wires"),)
        if not wires or len(set(wires)) != len(wires):
            self.fail(
                node, "quantum.wires", "static wires must be unique and non-empty"
            )
        return wires

    def dynamic_wires(self, call: ast.Call, *, expected: int) -> tuple[Value, ...]:
        node = self.wires_node(call)
        items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else (node,)
        if len(items) != expected:
            self.fail(node, "quantum.wires", f"expected {expected} wire expression(s)")
        return tuple(self.emit_index(item) for item in items)

    def capture_return(self, statement: ast.Return) -> None:
        if isinstance(statement.value, ast.Name):
            result = self.emit_expression(statement.value)
            if result.type != BOOL:
                self.fail(
                    statement.value,
                    "function.return",
                    "a direct return is limited to a measurement bool",
                )
            effect = self.require_effect(statement)
            self.effect = None
            self.operations.append(
                Operation(
                    "program.return",
                    operands=(result, effect),
                    location=self.location(statement),
                )
            )
            return
        if statement.value is None or not isinstance(statement.value, ast.Call):
            self.fail(
                statement,
                "function.return",
                "return must contain an expectation or measurement bool",
            )
        call = statement.value
        name = _call_name(call).replace("_", "").lower()
        if (
            name not in {"expval", "expectation"}
            or len(call.args) != 1
            or call.keywords
        ):
            self.fail(
                call,
                "quantum.expectation",
                "return must call expval or expectation with one observable",
            )
        terms = self.parse_observable(call.args[0])
        result = self.emit(
            call,
            "quantum.expectation",
            operands=(self.require_effect(call),),
            result_types=(scalar_type("float64"),),
            attributes={"terms": terms},
        )[0]
        self.effect = None
        self.operations.append(
            Operation(
                "program.return",
                operands=(result,),
                location=self.location(statement),
            )
        )

    def parse_observable(self, node: ast.expr) -> tuple[tuple[str, int], ...]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.parse_observable(node.left) + self.parse_observable(node.right)
        if isinstance(node, ast.Call):
            name = _call_name(node).replace("_", "").lower()
            paulis = {"paulix": "x", "pauliy": "y", "pauliz": "z"}
            if name in paulis and len(node.args) == 1 and not node.keywords:
                return (
                    (
                        paulis[name],
                        self.literal_non_negative_int(
                            node.args[0], "quantum.observable_wire"
                        ),
                    ),
                )
        try:
            literal = ast.literal_eval(node)
            terms = tuple((str(name).lower(), int(wire)) for name, wire in literal)
        except (TypeError, ValueError):
            self.fail(
                node, "quantum.observable", "observable expression is unsupported"
            )
        if not terms or any(
            name not in {"x", "y", "z"} or wire < 0 for name, wire in terms
        ):
            self.fail(node, "quantum.observable", "observable terms are invalid")
        return terms

    def literal_non_negative_int(self, node: ast.expr, code: str) -> int:
        try:
            value = ast.literal_eval(node)
        except (TypeError, ValueError):
            self.fail(node, code, "expected a static non-negative integer")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            self.fail(node, code, "expected a static non-negative integer")
        return value

    def require_effect(self, node: ast.AST) -> Value:
        if self.effect is None:
            self.fail(
                node, "quantum.effect", "quantum effect has already been consumed"
            )
        return self.effect

    def reject_outer_binding_writes(
        self,
        outer: dict[str, _Binding],
        node: ast.AST,
        *,
        allowed: frozenset[str] = frozenset(),
    ) -> None:
        changed = sorted(
            name
            for name, binding in outer.items()
            if name not in allowed and self.environment.get(name) != binding
        )
        if changed:
            self.fail(
                node,
                "control.classical_carry",
                "classical values written inside control flow require explicit "
                f"region-carried results: {', '.join(changed)}",
            )

    def _next_region_serial(self) -> int:
        serial = self._region_counter
        self._region_counter += 1
        return serial


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
