"""Capture coherent conditions through the existing checked register effects."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from .._hybrid.model import Operation, Value, ValueId
from .frontend_cnf import capture_cnf_phase
from .frontend_controlled_effects import capture_controlled_effects
from .frontend_increment import capture_increment_call, is_increment_helper
from .frontend_matrix import capture_controlled_unitary
from .frontend_numeric import has_numeric_reads, needs_numeric, reject_numeric_reads
from .frontend_phase import capture_polynomial_phase
from .frontend_ports import program_ports
from .frontend_python import _error, _QuantumFunction
from .frontend_views import gate_target, locations
from .program_types import (
    hadamard_transform,
    inplace_prepare_amplitudes,
    inplace_prepare_state,
    phase,
    qft,
    r,
    unitary,
    x,
)
from .register_profile import register_type, register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def is_composite_predicate(expression: ast.expr) -> bool:
    """Recognize comparison leaves without recursively walking user input."""
    pending = [expression]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Compare):
            continue
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, ast.Invert):
            pending.append(current.operand)
        elif isinstance(current, ast.BinOp) and isinstance(
            current.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            pending.extend((current.left, current.right))
        else:
            return False
    return True


def capture_control(owner: _Body, node: ast.Call) -> None:
    """Retain scalar/predicate behavior and admit direct all-ones controlled X."""
    if len(node.args) != 2 or not isinstance(node.args[1], ast.Lambda):
        _error(
            owner.source,
            node,
            "program.control",
            "expected control(bit, lambda: operation)",
        )
    condition, callback = node.args
    assert isinstance(callback, ast.Lambda)
    numeric_condition = has_numeric_reads(owner, condition)
    numeric_predicate = needs_numeric(owner, condition) and is_composite_predicate(
        condition
    )
    if numeric_condition and not is_composite_predicate(condition):
        _error(
            owner.source,
            condition,
            "program.numeric",
            "numeric reads in controls require an explicit bounded comparison",
        )
    reject_numeric_reads(owner, callback, "coherent callback")
    args = callback.args
    body = callback.body
    if (
        args.posonlyargs
        or args.args
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or not isinstance(body, ast.Call)
        or body.keywords
        or any(isinstance(arg, ast.Starred) for arg in body.args)
    ):
        _error(
            owner.source,
            callback,
            "program.control",
            "expected one static operation in a zero-argument lambda",
        )
    assert isinstance(body, ast.Call)
    if isinstance(body.func, ast.Name) and body.func.id in owner.ports:
        _error(
            owner.source,
            body.func,
            "program.control",
            "a quantum port cannot shadow a callable",
        )
    operation = owner.resolve_callable(body.func)
    increment = is_increment_helper(operation)
    composable = not increment and (
        operation in (inplace_prepare_state, inplace_prepare_amplitudes)
        or isinstance(operation, _QuantumFunction)
        and operation.source.program
        and any(port.role == "inout" for port in program_ports(operation))
        and all(port.role == "inout" for port in program_ports(operation))
    )
    if operation is unitary:
        capture_controlled_unitary(owner, condition, body)
        return
    if operation is x and not is_composite_predicate(condition):
        capture_controlled_x(owner, condition, body)
        return
    if (
        owner.aliases is not None
        and operation not in (phase, r, x)
        and not increment
        and not composable
    ):
        _error(
            owner.source,
            body,
            "program.borrow",
            "mutable helpers allow phase, predicate-X or direct Pauli-axis rotation control",
        )
    phase_constant = None
    if operation is phase and len(body.args) == 2:
        phase_constant = owner.phase_number(body.args[0])
        if capture_cnf_phase(owner, condition, body, phase_constant):
            return

    def needs_predicate_phase_capture(expression: ast.expr) -> bool:
        """Route bounded compound predicates through the shared phase kernel."""
        return any(
            isinstance(
                child, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.Mod, ast.Invert, ast.Sub)
            )
            for child in ast.walk(expression)
        )

    if (
        operation is phase
        and len(body.args) == 2
        and not numeric_condition
        and not (numeric_predicate and phase_constant is not None)
        and (
            owner.aliases is not None
            or isinstance(condition, ast.Subscript)
            or needs_predicate_phase_capture(condition)
            or phase_constant is None
            and (
                not isinstance(body.args[0], ast.Name)
                or any(
                    child.id == body.args[0].id
                    for child in ast.walk(condition)
                    if isinstance(child, ast.Name)
                )
            )
        )
    ):
        expression, coefficient = body.args
        angle = owner.phase_number(coefficient)
        if angle is None:
            _error(
                owner.source,
                coefficient,
                "program.control",
                "expected a finite static phase coefficient",
            )
        capture_polynomial_phase(owner, expression, angle, body, condition=condition)
        return

    predicate_computed = (
        numeric_predicate
        or isinstance(condition, ast.Compare)
        or (
            (operation in (r, qft, x) or increment or composable)
            and is_composite_predicate(condition)
        )
    )
    scalar_location = (operation is r or increment or composable) and (
        isinstance(condition, ast.Subscript) or owner.aliases is not None
    )
    predicate_index = 0
    if predicate_computed:
        graph, inputs = owner.capture_expression(condition, node)
        if sum(register_width(value) for value in inputs) > 8:
            _error(
                owner.source,
                condition,
                "program.control",
                "control predicate exceeds eight input bits",
            )
        predicate = Value(ValueId(owner.name, owner.counter), register_type(1))
        owner.counter += 1
        owner.operations.append(
            Operation(
                "qreg.local",
                results=(predicate,),
                attributes={"name": f"control.{predicate.id.index}"},
                location=owner.source.location(node),
            )
        )
        owner.emit("qreg.compute", node, (predicate, *inputs), callee=graph.name)
    elif scalar_location:
        predicate, predicate_index = gate_target(owner, condition)
        inputs = (predicate,)
    else:
        predicate = owner.register(condition)
        inputs = (predicate,)
    if composable:
        if not scalar_location and register_width(predicate) != 1:
            _error(
                owner.source,
                condition,
                "program.control",
                "callback requires a scalar condition or bounded predicate",
            )
        capture_controlled_effects(owner, body, predicate, predicate_index, inputs)
    elif increment:
        assert isinstance(operation, _QuantumFunction)
        if not scalar_location and register_width(predicate) != 1:
            _error(
                owner.source,
                condition,
                "program.control",
                "increment requires a scalar condition or bounded predicate",
            )
        capture_increment_call(
            owner, body, operation, (predicate, predicate_index), inputs
        )
    elif operation is x and len(body.args) == 1:
        target, index = gate_target(owner, body.args[0])
        if target in inputs:
            _error(
                owner.source,
                node,
                "program.control",
                "condition and target must be disjoint",
            )
        owner.emit("qreg.control_x", node, (predicate, target), indices=(0, index))
    elif (operation is hadamard_transform or operation is qft) and len(body.args) == 1:
        target = owner.register(body.args[0])
        if target in inputs:
            _error(
                owner.source,
                node,
                "program.control",
                "condition and target must be disjoint",
            )
        opcode = "qreg.control_qft" if operation is qft else "qreg.control_h"
        owner.emit(opcode, node, (predicate, target))
    elif operation is phase and len(body.args) == 2:
        expression, coefficient = body.args
        angle = owner.phase_number(coefficient)
        if angle is None:
            _error(
                owner.source,
                coefficient,
                "program.control",
                "expected a finite static phase coefficient",
            )
        constant = phase_constant
        operands: tuple[Value, ...]
        if constant is None:
            target = owner.register(expression)
            if target in inputs:
                _error(
                    owner.source,
                    node,
                    "program.control",
                    "condition and target must be disjoint",
                )
            operands = (predicate, target)
        else:
            operands = (predicate,)
            angle *= constant
            if not math.isfinite(angle):
                _error(
                    owner.source,
                    expression,
                    "program.control",
                    "controlled phase angle exceeds finite range",
                )
        owner.emit("qreg.control_phase", node, operands, angle=angle)
    elif operation is r and len(body.args) == 3:
        theta = owner.phase_number(body.args[0])
        phi = owner.phase_number(body.args[1])
        if theta is None or phi is None:
            _error(
                owner.source,
                body,
                "program.control",
                "controlled R requires finite static angles",
            )
        angle = math.remainder(phi, 2 * math.pi)
        if abs(angle) == math.pi / 2:
            axis = "Y"
            sign = 1.0 if angle > 0 else -1.0
        elif angle == 0.0 or abs(angle) == math.pi:
            axis = "X"
            sign = 1.0 if math.cos(angle) > 0 else -1.0
        else:
            _error(
                owner.source,
                body.args[1],
                "program.control",
                "controlled R supports only Pauli X or Y axes",
            )
        target, index = gate_target(owner, body.args[2])
        if not scalar_location and register_width(predicate) != 1:
            _error(
                owner.source,
                condition,
                "program.control",
                "controlled R requires a one-bit condition or a bounded predicate",
            )
        if target in inputs:
            _error(
                owner.source,
                node,
                "program.control",
                "condition and target must be disjoint",
            )
        owner.emit(
            "qreg.pauli_rotation",
            node,
            (target, predicate),
            support=((index, axis),),
            angle=sign * theta / 2,
            control_index=predicate_index,
        )
    elif isinstance(operation, _QuantumFunction) and operation.source.program:
        owner.call(body, control_bit=predicate, condition_inputs=inputs)
    else:
        _error(
            owner.source,
            body,
            "program.control",
            "expected phase, Hadamard, QFT, Pauli-axis R or a supported qfunc call",
        )

    if predicate_computed:
        owner.emit("qreg.uncompute", node, (predicate,))
        owner.emit("qreg.release", node, (predicate,))


def capture_controlled_x(owner: _Body, condition: ast.expr, body: ast.Call) -> None:
    """Resolve distinct initialized qubits; verifier enforces target mutability."""
    if len(body.args) != 1:
        _error(owner.source, body, "program.signature", "controlled X takes one target")
    controls = locations(owner, condition)
    target = gate_target(owner, body.args[0])
    if (
        not controls
        or len(controls) > 256
        or target in controls
        or len(set(controls)) != len(controls)
    ):
        _error(
            owner.source,
            body,
            "program.control",
            "use 1..256 distinct controls and a disjoint scalar X target",
        )
    ports = (*controls, target)
    owner.emit(
        "qreg.control_x",
        body,
        tuple(value for value, _ in ports),
        indices=tuple(bit for _, bit in ports),
    )
