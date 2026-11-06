"""Checked Qmod control and reversible scopes."""

from __future__ import annotations

import math
from itertools import groupby
from typing import TYPE_CHECKING

from ...compiler._construction.ranges import unsigned_width
from ...compiler._construction.register_profile import (
    register_type,
)
from ...compiler._hybrid.model import (
    Operation,
    Value,
    ValueId,
)
from .adapter_model import (
    _INTRINSICS,
    _bound_helper,
    _Expression,
    _literal_angle,
)
from .bindings import (
    lower_binding_scope,
)
from .callback_arrays import (
    lower_callback_control,
    lower_selected_effects,
)
from .controlled_evolution import controlled_evolution
from .fixed_xor import lower_fixed_xor
from .indexed_control import (
    lower_constant_phase_guard,
    lower_guarded_field_increment,
    lower_indexed_control,
    lower_nested_indexed_control,
    lower_zero_reflection_control,
)
from .masked_selection import (
    emit_selected,
    lower_masked_selection,
)
from .matrices import lower_unitary
from .parity_updates import lower_masked_word_xor
from .scope_control import lower_fourier_control
from .selector_rotations import (
    lower_guarded_selector,
)
from .syntax import (
    Expression,
    Statement,
    Token,
    reject,
)
from .unitary_effects import (
    emit_controlled_pauli,
    is_reversible_scope,
    lower_reversible_scope,
)

if TYPE_CHECKING:
    from .adapter import _Adapt


def coherent_helper_control(self: _Adapt, statement: Statement, index: str) -> bool:
    condition = statement.arguments[0]
    if condition.token.text in {"==", "!="} and len(condition.children) == 2:
        source, expected = condition.children
        if expected.children or expected.token.text not in {"0", "1"}:
            return False
        positive = int(expected.token.text) ^ (condition.token.text == "!=")
    elif condition.token.text == "[" or (
        not condition.children
        and self.widths.get(condition.token.text) == 1
        and self.parameters[condition.token.text].kind in {"qbit", "qnum"}
    ):
        source, positive = condition, 1
    else:
        return False
    if source.token.text == "[":
        control = self.gate_target(source)
    elif not source.children:
        parameter = self.parameters.get(source.token.text)
        if (
            parameter is None
            or parameter.kind not in {"qbit", "qnum"}
            or self.widths[source.token.text] != 1
        ):
            return False
        control = self.array_view(source)[0]
    else:
        return False
    body = statement.control_body
    assert body is not None
    branches: tuple[tuple[int, tuple[Statement, ...]], ...] = ((positive, body),)
    if statement.control_else is not None:
        branches += ((1 - branches[0][0], statement.control_else),)

    # Nested control blocks enter this path when they contain a helper composition.
    def contains_composition(children: tuple[Statement, ...]) -> bool:
        return any(
            child.inverse_body is not None
            or child.repeat_index is not None
            or child.within is not None
            or child.token.text in {"+=", "unitary"}
            or child.token.text in self.definitions
            or (
                child.control_body is not None
                and contains_composition(child.control_body)
            )
            or (
                child.control_else is not None
                and contains_composition(child.control_else)
            )
            for child in children
        )

    if not any(contains_composition(children) for _, children in branches):
        return False
    for value, children in branches:
        start, incoming = len(self.operations), self.effect
        for ordinal, child in enumerate(children):
            if (
                child.token.text == "IDENTITY"
                and len(child.arguments) == 1
                and control in self.array_view(child.arguments[0])
            ):
                reject(child.token, "alias", "control and target must be disjoint")
            self.statement(child, f"{index}.control{value}.{ordinal}")
        expanded = self.operations[start:]
        del self.operations[start:]
        self.effect = incoming
        if not value:
            self.effect_op(
                "qreg.gate",
                statement.token,
                (control[0],),
                gate="x",
                index=control[1],
            )
        for (name, operands, attrs), run in groupby(
            expanded, key=lambda op: (op.name, op.operands[1:], dict(op.attributes))
        ):
            if (
                name == "qreg.unitary"
                and "control_index" not in attrs
                and "power" not in attrs
                and control[0] not in operands
            ):
                power = sum(1 for _ in run)
                for start in range(0, power, 128):
                    self.effect_op(
                        name,
                        statement.token,
                        (*operands, control[0]),
                        **attrs,
                        control_index=control[1],
                        power=min(128, power - start),
                    )
                continue
            for effect in run:
                if (
                    "control_state" in attrs
                    or name == "qreg.inplace_prepare_amplitudes"
                    or name == "qreg.gate"
                    and attrs.get("gate") in {"phase", "z", "s"}
                ):
                    emit_selected(self, statement.token, (control,), 1, (effect,))
                    continue
                if name == "qreg.control_x" and "indices" in attrs:
                    locations = tuple(zip(operands, attrs["indices"], strict=True))
                elif name == "qreg.gate" and attrs["gate"] in {"i", "x"}:
                    locations = ((operands[0], attrs["index"]),)
                    if attrs["gate"] == "i":
                        if control in locations:
                            reject(
                                statement.token,
                                "alias",
                                "control and target must be disjoint",
                            )
                        self.effect_op(name, statement.token, operands, **attrs)
                        continue
                elif name == "qreg.pauli_rotation":
                    selectors = (
                        (control, (operands[1], attrs["control_index"]))
                        if "control_index" in attrs
                        else (control,)
                    )
                    emit_controlled_pauli(
                        self,
                        statement.token,
                        operands[0],
                        attrs["support"],
                        attrs["angle"],
                        selectors,
                    )
                    continue
                elif name == "qreg.phase" or (
                    name == "qreg.control_phase" and "indices" in attrs
                ):
                    phase_inputs = tuple(
                        zip(operands, attrs.get("indices", ()), strict=True)
                    )
                    if control in phase_inputs:
                        reject(
                            statement.token,
                            "alias",
                            "control and target must be disjoint",
                        )
                    selected = (control, *phase_inputs)
                    self.effect_op(
                        "qreg.control_phase",
                        statement.token,
                        tuple(value for value, _ in selected),
                        indices=tuple(wire for _, wire in selected),
                        angle=attrs["angle"],
                    )
                    continue
                elif name == "qreg.gate" and attrs["gate"] in {
                    "rx",
                    "ry",
                    "rz",
                    "h",
                }:
                    destination = operands[0]
                    wire = attrs["index"]
                    if control == (destination, wire):
                        reject(
                            statement.token,
                            "alias",
                            "control and target must be disjoint",
                        )
                    if attrs["gate"] == "h":
                        self.effect_op(
                            "qreg.control_phase",
                            statement.token,
                            (control[0], destination),
                            indices=(control[1], wire),
                            angle=math.pi,
                        )
                        axis, angle = "Y", math.pi / 4
                    else:
                        axis, angle = attrs["gate"][-1].upper(), attrs["angle"] / 2
                        if attrs.get("inverse"):
                            angle = -angle
                    support = ((wire, axis),)
                    if control[0] == destination:
                        emit_controlled_pauli(
                            self,
                            statement.token,
                            destination,
                            support,
                            angle,
                            (control,),
                        )
                    else:
                        self.pauli_exponential(
                            statement.token,
                            destination,
                            support,
                            angle,
                            control=control,
                        )
                    continue
                elif name == "qreg.pair_gate" and attrs["gate"] in {"cx", "swap"}:
                    left, right = zip(
                        operands, attrs.get("indices", (0, 0)), strict=True
                    )
                    if control in (left, right):
                        reject(
                            statement.token,
                            "alias",
                            f"controlled {attrs['gate'].upper()} requires distinct locations",
                        )
                    if attrs["gate"] == "swap":
                        self.emit_indexed_x(statement.token, (right, left))
                    self.emit_indexed_x(statement.token, (control, left, right))
                    if attrs["gate"] == "swap":
                        self.emit_indexed_x(statement.token, (right, left))
                    continue
                else:
                    reject(
                        statement.token,
                        "control",
                        "controlled helper requires reversible X, indexed controlled-X, SWAP or a disjoint unitary",
                    )
                self.emit_indexed_x(statement.token, (control, *locations))
        if not value:
            self.effect_op(
                "qreg.gate",
                statement.token,
                (control[0],),
                gate="x",
                index=control[1],
            )
    return True


def controlled_rz_branches(self: _Adapt, statement: Statement) -> bool:
    yes, no = statement.control_body, statement.control_else
    if yes is None or no is None or len(yes) != 1 or len(no) != 1:
        return False
    if any(
        action.token.text != "RZ" or len(action.arguments) != 2
        for action in (*yes, *no)
    ):
        return False
    condition = statement.arguments[0]
    negative = Expression(Token("~", condition.token.location), (condition,))
    for branch, guard in ((yes[0], condition), (no[0], negative)):
        angle, marker = branch.arguments
        location = self.gate_target(marker, allow_numeric=True)
        half = Expression(Token("0.5", marker.token.location))
        weighted = Expression(Token("-", marker.token.location), (marker, half))
        self.real_phase(
            branch.token,
            weighted,
            angle,
            condition=guard,
            disjoint_condition_target=location,
        )
    return True


def scalar_control(self: _Adapt, statement: Statement, index: str) -> None:
    token = statement.token
    body = statement.control_body
    if len(statement.arguments) != 1 or body is None:
        reject(token, "control", "control requires one condition and one operation")
    if any(
        child.token.text == "RESET"
        for child in (*body, *(statement.control_else or ()))
    ):
        reject(
            token,
            "control",
            "RESET is nonunitary and cannot be coherently controlled",
        )
    condition = statement.arguments[0]
    if (
        len(body) == 1
        and body[0].token.text == "^="
        and len(body[0].arguments) == 2
        and statement.control_else is None
        and lower_fixed_xor(
            self,
            body[0].token,
            body[0].arguments[0],
            body[0].arguments[1],
            guard=condition,
        )
    ):
        return
    if lower_masked_word_xor(self, statement):
        return
    if lower_guarded_selector(self, statement, index):
        return
    if lower_callback_control(self, statement):
        return
    if controlled_evolution(self, statement, index):
        return
    if lower_fourier_control(self, statement):
        return
    if lower_guarded_field_increment(self, statement):
        return
    if self.controlled_rz_branches(statement):
        return
    if lower_constant_phase_guard(self, statement):
        return
    if lower_masked_selection(self, statement):
        return
    if self.coherent_helper_control(statement, index):
        return
    if (
        len(body) == 1
        and body[0].token.text == "unitary"
        and statement.control_else is None
        and condition.token.text == "=="
    ):
        selector, literal = condition.children
        controls = self.array_view(selector)
        lower_unitary(
            self,
            body[0].token,
            body[0].arguments,
            selection=(controls, self.static_integer(literal)),
        )
        return
    if (
        len(body) == 1
        and statement.control_else is None
        and condition.token.text == "=="
        and not condition.children[0].children
        and (self.widths.get(condition.children[0].token.text) or 0) > 1
        and (
            body[0].token.text in {"grover_diffuser", "inplace_prepare_state"}
            or is_reversible_scope(body[0])
            or (
                (helper := self.definitions.get(body[0].token.text)) is not None
                and _bound_helper(helper)
            )
        )
    ):
        start, incoming = len(self.operations), self.effect
        self.statement(body[0], f"{index}.select")
        effects = tuple(self.operations[start:])
        del self.operations[start:]
        self.effect = incoming
        if is_reversible_scope(body[0]):
            selector, literal = condition.children
            controls = self.array_view(selector)
            expected = self.static_integer(literal)
            if not 1 <= len(controls) <= 7 or not 0 <= expected < 1 << len(controls):
                reject(token, "control", "selected scope equality must fit its word")
            emit_selected(self, token, controls, expected, effects)
        else:
            lower_selected_effects(self, token, condition, effects)
        return
    if len(body) != 1:
        reject(token, "control", "control requires one condition and one operation")
    if statement.control_else is not None:
        reject(token, "control", "control else requires a supported reversible helper")
    operation = body[0]
    if operation.control_body is not None:
        if lower_nested_indexed_control(self, condition, operation):
            return
        reject(token, "scope", "nested scalar control requires function scope")
    if lower_zero_reflection_control(self, condition, operation):
        return
    if lower_indexed_control(self, condition, operation):
        return
    if condition.token.text == ":":
        if operation.token.text != "X" or len(operation.arguments) != 1:
            reject(token, "control", "slice control requires one X target")
        controls = self.array_view(condition)
        target_location = self.gate_target(operation.arguments[0])
        self.emit_indexed_x(token, (*controls, target_location))
        return
    args = operation.arguments
    if (
        operation.token.text == "phase"
        and len(args) == 2
        and (
            args[0].children
            or args[1].token.text in self.classical_values
            or args[1].token.text in self.classical_integers
            or args[1].token.text == "["
            or any(
                parameter.record is not None for parameter in self.parameters.values()
            )
        )
    ):
        self.real_phase(operation.token, args[0], args[1], condition=condition)
        return
    flip_phase = (
        operation.token.text == "phase"
        and len(args) == 1
        and not args[0].children
        and args[0].token.text == "pi"
        and "pi" not in self.parameters
    )
    if flip_phase and condition.token.text == "[":
        register, wire = self.gate_target(condition)
        self.effect_op("qreg.gate", token, (register,), gate="z", index=wire)
        return
    parameter = self.parameters.get(condition.token.text)
    if not condition.children and parameter is not None and parameter.kind == "qbit[]":
        if flip_phase:
            if parameter.name.text in self.views:
                view = self.array_view(condition)
                if len(view) > 8:
                    phase_target, phase_bit = view[-1]
                    self.effect_op(
                        "qreg.gate",
                        token,
                        (phase_target,),
                        gate="h",
                        index=phase_bit,
                    )
                    self.emit_indexed_x(token, view)
                    self.effect_op(
                        "qreg.gate",
                        token,
                        (phase_target,),
                        gate="h",
                        index=phase_bit,
                    )
                else:
                    self.effect_op(
                        "qreg.control_phase",
                        token,
                        tuple(value for value, _ in view),
                        indices=tuple(bit for _, bit in view),
                        angle=math.pi,
                    )
            else:
                control = self.register(parameter.name, allow_qubit_array=True)
                self.effect_op("qreg.array_phase_flip", token, (control,))
            return
        if operation.token.text != "X" or len(operation.arguments) != 1:
            reject(token, "control", "array control requires a single scalar X target")
        target = self.atom(operation.arguments[0])
        if target.text == parameter.name.text:
            reject(token, "control", "condition and target must be disjoint")
        control = self.register(parameter.name, allow_qubit_array=True)
        destination, _ = self.gate_target(operation.arguments[0])
        self.effect_op("qreg.control_x", token, (control, destination))
        return
    expression = _Expression(
        condition, self.widths, f"{self.scope}.control{index}", self.signed
    )
    if (
        condition in expression.signed_words
        or expression.fractional_bits.get(condition, 0)
        or not expression.quantum
        or unsigned_width(expression.bound) != 1
    ):
        reject(token, "control", "control requires a quantum one-bit condition")
    inputs = tuple(
        self.register(self.parameters[name].name) for name in expression.names
    )
    if sum(value.type.parameters[0] for value in inputs) > 8:
        reject(token, "control", "control predicate exceeds eight input bits")
    targets: tuple[Value, ...] = ()
    angle = None
    if operation.token.text in {"H", "hadamard_transform"} and len(args) == 1:
        targets = (self.register(self.atom(args[0])),)
        opcode = "qreg.control_h"
    elif flip_phase:
        angle = math.pi
        opcode = "qreg.control_phase"
    elif operation.token.text == "phase" and len(args) == 2:
        coefficient = args[1]
        if (
            not coefficient.children
            and coefficient.token.text == "pi"
            and "pi" not in self.parameters
        ):
            angle = math.pi
        else:
            angle = _literal_angle(coefficient, operation.token)
        value = args[0]
        if not value.children and value.token.text in {"0", "1"}:
            angle *= int(value.token.text)
        else:
            targets = (self.register(self.atom(value)),)
        opcode = "qreg.control_phase"
    elif operation.token.text not in _INTRINSICS and operation.token.text not in {
        "measure",
        "release",
        "consume",
    }:
        opcode = "qreg.control_call"
    else:
        reject(token, "control", "expected a scalar Hadamard or phase")
    if any(target in inputs for target in targets):
        if opcode == "qreg.control_phase" and len(args) == 2:
            self.real_phase(operation.token, args[0], args[1], condition=condition)
            return
        reject(token, "control", "condition and target must be disjoint")
    computed = bool(condition.children)
    if computed:
        graph = expression.function()
        self.functions.append(graph)
        predicate = Value(ValueId(self.scope, self.counter), register_type(1))
        self.counter += 1
        self.append_op(
            token,
            Operation(
                "qreg.local",
                results=(predicate,),
                attributes={"name": f"control.{predicate.id.index}"},
                location=token.location,
            ),
        )
        self.effect_op("qreg.compute", token, (predicate, *inputs), callee=graph.name)
    else:
        predicate = inputs[0]
    phase_attributes = {} if angle is None else {"angle": angle}
    if opcode == "qreg.control_call":
        self.call(operation.token, args, control_bit=predicate, condition_inputs=inputs)
    else:
        self.effect_op(opcode, token, (predicate, *targets), **phase_attributes)
    if computed:
        self.effect_op("qreg.uncompute", token, (predicate,))
        self.effect_op("qreg.release", token, (predicate,))


def within(self: _Adapt, statement: Statement, index: str) -> None:
    if lower_binding_scope(self, statement, index):
        return
    if not statement.within:
        reject(
            statement.token,
            "scope",
            "within requires at least one local computation",
        )
    if (
        len(statement.within) == 1
        and statement.within[0].token.text == "H"
        and len(statement.apply) == 1
    ):
        action = statement.apply[0]
        helper = self.definitions.get(action.token.text)
        if helper is not None and _bound_helper(helper) and action.control_body is None:
            self.statement(statement.within[0], index)
            self.call(action.token, action.arguments)
            self.statement(statement.within[0], index)
            return
    if is_reversible_scope(statement):
        lower_reversible_scope(self, statement, index)
        return
    parent_scope = self.local_scope
    self.local_scope = []
    try:
        for position, child in enumerate(statement.within):
            if child.token.text != "=":
                reject(
                    child.token,
                    "scope",
                    "within admits only quantum-expression local assignments",
                )
            target = self.atom(child.arguments[0])
            if (
                target.text not in self.parameters
                or self.parameters[target.text].role != "local"
            ):
                reject(target, "scope", "within prepares declared locals only")
            self.statement(child, f"{index}.within{position}")
        for position, child in enumerate(statement.apply):
            if child.within is not None:
                self.within(child, f"{index}.apply{position}")
            elif child.token.text == "phase":
                self.statement(child, f"{index}.apply{position}")
            elif child.token.text in _INTRINSICS or child.token.text == "=":
                reject(
                    child.token,
                    "scope",
                    "apply admits phase and Const-only helper calls",
                )
            else:
                self.call(child.token, child.arguments, const_only=True)
        for name, token in reversed(self.local_scope):
            value = self.registers[name]
            self.effect_op("qreg.uncompute", token, (value,))
            self.effect_op("qreg.release", token, (value,))
    finally:
        for name, _ in self.local_scope:
            self.registers.pop(name, None)
        self.local_scope = parent_scope
