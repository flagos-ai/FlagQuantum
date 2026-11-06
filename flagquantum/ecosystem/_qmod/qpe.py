"""Static ordinary QPE composition over existing checked register effects."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from ...compiler._hybrid.model import Operation, Value
from .angles import REAL_LITERAL
from .syntax import Expression, Function, Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt
    from .syntax import Parser


def parse_qpe_call(
    parser: Parser, token: Token, depth: int, *, reversible_preparation: bool = False
) -> Statement:
    if depth and not (reversible_preparation and token.text == "qpe_flexible"):
        reject(token, "scope", f"{token.text} requires function scope")
    parser.take("(")
    parser.take("lambda")
    parser.take("(")
    power = parser.name() if token.text == "qpe_flexible" else None
    parser.take(")")
    body = parser.block(allow_scopes=False, scope_depth=1, power_body=power is not None)
    parser.take(",")
    destination = Expression(parser.name())
    parser.take(")")
    parser.take(";")
    return (
        Statement(token, (destination,), power_callback=(power, body))
        if power is not None
        else Statement(token, (destination,), qpe_body=body)
    )


def is_modular_power(statement: Statement) -> bool:
    if statement.power_callback is None:
        return False
    _, body = statement.power_callback
    return len(body) == 1 and body[0].token.text == "modular_multiply_constant_inplace"


def lower_modular_qpe(owner: _Adapt, statement: Statement) -> None:
    token = statement.token
    assert statement.power_callback is not None
    power, body = statement.power_callback
    if len(body[0].arguments) != 3:
        reject(
            token,
            "qpe.modular",
            "modular power requires modulus, base**power and target",
        )
    modulus_arg, multiplier, target = body[0].arguments
    if (
        multiplier.token.text != "**"
        or multiplier.children[1].children
        or multiplier.children[1].token.text != power.text
    ):
        reject(
            token,
            "qpe.modular",
            "modular power requires a static base raised to the power parameter",
        )
    modulus = owner.static_integer(modulus_arg)
    base = owner.static_integer(multiplier.children[0])
    name = owner.atom(statement.arguments[0])
    parameter = owner.parameters.get(name.text)
    width = owner.widths.get(name.text)
    if (
        parameter is None
        or parameter.kind != "qphase"
        or width is None
        or not 1 <= width <= 10
    ):
        reject(
            name, "qpe.modular", "modular QPE requires one to ten unsigned phase bits"
        )
    phase = owner.register(name, allow_fractional_output=True)
    last = next(
        (op for op in reversed(owner.operations) if phase in op.operands[1:]), None
    )
    if not (
        last is not None
        and last.name == "qreg.allocate"
        or last is None
        and phase in owner.borrowed_zero_phases
    ):
        reject(name, "qpe", "phase output must be initialized by allocate and unused")
    target_name = owner.atom(target)
    if (
        target_name.text in owner.signed
        or owner.parameters.get(target_name.text) is None
        or owner.parameters[target_name.text].kind != "qnum"
    ):
        reject(
            target_name,
            "qpe.modular",
            "modular QPE residue requires an unsigned numeric register",
        )
    residue = owner.register(target_name)
    if residue == phase:
        reject(target_name, "qpe.alias", "phase counter and residue must be disjoint")
    for bit in range(width):
        owner.effect_op("qreg.gate", token, (phase,), gate="h", index=bit)
    owner.effect_op(
        "qreg.modular_exp", token, (residue, phase), modulus=modulus, base=base
    )
    owner.effect_op("qreg.inverse_qft", token, (phase,))
    owner.fractional_bits[name.text] = width


def lower_modular_loop(owner: _Adapt, statement: Statement, count: int) -> bool:
    if not count or statement.repeat_index is None or len(statement.repeat_body) != 1:
        return False
    selector = statement.repeat_body[0]
    if (
        selector.control_body is None
        or selector.control_else is not None
        or len(selector.control_body) != 1
        or len(selector.arguments) != 1
    ):
        return False
    action = selector.control_body[0]
    if (
        action.token.text != "modular_multiply_constant_inplace"
        or len(action.arguments) != 3
    ):
        return False
    (condition,) = selector.arguments
    modulus, multiplier, target = action.arguments
    index = statement.repeat_index.text

    def atom(node: Expression, name: str) -> bool:
        return not node.children and node.token.text == name

    if (
        condition.token.text != "["
        or not atom(condition.children[1], index)
        or multiplier.token.text != "**"
    ):
        return False
    exponent, base, power = condition.children[0], *multiplier.children
    if (
        power.token.text != "**"
        or not atom(power.children[0], "2")
        or not atom(power.children[1], index)
    ):
        return False
    exponent_name = owner.atom(exponent)
    exponent_port = owner.parameters.get(exponent_name.text)
    if exponent_port is None or exponent_port.kind != "qbit[]":
        reject(
            exponent_name, "modular", "modular bit loop requires an indexed qubit array"
        )
    counter = owner.register(exponent_name, allow_qubit_array=True)
    if count != register_width(counter):
        reject(
            statement.token,
            "modular",
            "modular bit loop must cover its whole exponent register",
        )
    target_name = owner.atom(target)
    target_port = owner.parameters.get(target_name.text)
    if (
        target_port is None
        or target_port.kind not in {"qnum", "qbit[]"}
        or target_name.text in owner.signed
    ):
        reject(
            target_name, "modular", "modular bit loop requires unsigned residue storage"
        )
    residue = owner.register(target_name, allow_qubit_array=True)
    owner.effect_op(
        "qreg.modular_exp",
        action.token,
        (residue, counter),
        modulus=owner.static_integer(modulus),
        base=owner.static_integer(base),
    )
    return True


def lower_qpe(owner: _Adapt, statement: Statement, index: str) -> None:
    token = statement.token
    name = owner.atom(statement.arguments[0])
    parameter = owner.parameters.get(name.text)
    if (
        parameter is None
        or parameter.kind not in {"qphase", "qphase_signed"}
        or parameter.width is None
    ):
        reject(name, "qpe", "expected an all-fractional phase output")
    width = parameter.width
    if width > 8:
        reject(name, "qpe", "phase estimation supports at most eight phase qubits")
    phase = owner.register(name, allow_fractional_output=True)
    last = next(
        (op for op in reversed(owner.operations) if phase in op.operands[1:]), None
    )
    if last is None or last.name != "qreg.allocate":
        reject(name, "qpe", "phase output must be initialized by allocate and unused")
    start, incoming = len(owner.operations), owner.effect
    assert statement.qpe_body is not None
    for ordinal, child in enumerate(statement.qpe_body):
        owner.statement(child, f"{index}.qpe{ordinal}")
    body = tuple(owner.operations[start:])
    del owner.operations[start:]
    owner.effect = incoming
    if not body:
        reject(token, "qpe", "QPE requires a nonempty checked unitary callback")
    allowed = {
        "qreg.gate",
        "qreg.pair_gate",
        "qreg.phase",
        "qreg.control_phase",
        "qreg.control_x",
        "qreg.array_phase_flip",
        "qreg.pauli_rotation",
        "qreg.unitary",
        "qreg.inplace_prepare_amplitudes",
    }
    for op in body:
        if phase in op.operands[1:]:
            reject(
                token, "qpe.alias", "phase counter and callback data must be disjoint"
            )
        if op.name not in allowed or (
            op.name == "qreg.phase" and "callee" in op.attributes
        ):
            reject(
                token,
                "qpe.escape",
                "QPE callback requires checked pure unitary effects; no allocation, drop or Output escape",
            )
    for bit in range(width):
        owner.effect_op("qreg.gate", token, (phase,), gate="h", index=bit)
    for bit in range(width):
        if len(body) == 1 and body[0].name == "qreg.unitary":
            op = body[0]
            owner.effect_op(
                op.name,
                token,
                (*op.operands[1:], phase),
                **dict(op.attributes),
                control_index=bit,
                power=1 << bit,
            )
        else:
            for _ in range(1 << bit):
                for op in body:
                    _controlled(owner, token, phase, bit, op)
    owner.effect_op("qreg.inverse_qft", token, (phase,))
    owner.fractional_bits[name.text] = width


def _controlled(
    owner: _Adapt, token: Token, control: Value, bit: int, op: Operation
) -> None:
    values, attrs = op.operands[1:], op.attributes

    def phase(locations: tuple[tuple[Value, int], ...], angle: float) -> None:
        all_locations = ((control, bit), *locations)
        owner.effect_op(
            "qreg.control_phase",
            token,
            tuple(value for value, _ in all_locations),
            indices=tuple(index for _, index in all_locations),
            angle=angle,
        )

    if op.name == "qreg.phase":
        phase((), attrs["angle"])
    elif op.name == "qreg.gate":
        target = values[0]
        gate, wire = attrs["gate"], attrs["index"]
        direction = -1 if attrs.get("inverse") else 1
        if gate == "x":
            owner.effect_op(
                "qreg.control_x", token, (control, target), indices=(bit, wire)
            )
        elif gate in {"z", "s", "phase", "i"}:
            angle = {"z": math.pi, "s": math.pi / 2, "i": 0.0}.get(
                gate, attrs.get("angle", 0.0)
            )
            phase(((target, wire),), direction * angle)
        elif gate in {"h", "rx", "ry", "rz", "sx"}:
            if gate == "h":
                phase(((target, wire),), math.pi)
                axis, angle = "Y", math.pi / 4
            elif gate == "sx":
                phase((), direction * math.pi / 4)
                axis, angle = "X", direction * math.pi / 4
            else:
                axis, angle = gate[-1].upper(), direction * attrs["angle"] / 2
            owner.pauli_exponential(
                token, target, ((wire, axis),), angle, control=(control, bit)
            )
        else:
            reject(token, "qpe", "unsupported gate in QPE callback")
    elif op.name in {"qreg.control_phase", "qreg.control_x"} and "indices" in attrs:
        indices = (bit, *attrs["indices"])
        owner.effect_op(
            op.name,
            token,
            (control, *values),
            indices=indices,
            **({"angle": attrs["angle"]} if op.name == "qreg.control_phase" else {}),
        )
    elif op.name == "qreg.array_phase_flip":
        phase(tuple((values[0], i) for i in range(register_width(values[0]))), math.pi)
    elif (
        op.name == "qreg.pauli_rotation"
        and "control_index" not in attrs
        or op.name == "qreg.unitary"
        and "control_index" not in attrs
    ):
        owner.effect_op(
            op.name, token, (*values, control), **dict(attrs), control_index=bit
        )
    elif op.name == "qreg.inplace_prepare_amplitudes" and "control_count" not in attrs:
        locations = (
            tuple(zip(values, attrs["indices"], strict=True))
            if "indices" in attrs
            else tuple(
                (values[0], i) for i in reversed(range(register_width(values[0])))
            )
        )
        selected = ((control, bit), *locations)
        owner.effect_op(
            op.name,
            token,
            tuple(value for value, _ in selected),
            **{**attrs, "control_count": 1, "indices": tuple(i for _, i in selected)},
        )
    elif op.name == "qreg.pair_gate" and attrs["gate"] == "cry":
        control_wire, target_wire = attrs.get("indices", (0, 0))
        angle = attrs["angle"] / 2
        for radians in (angle, -angle):
            owner.effect_op(
                "qreg.gate",
                token,
                (values[1],),
                gate="ry",
                index=target_wire,
                angle=radians,
            )
            owner.effect_op(
                "qreg.control_x",
                token,
                (control, *values),
                indices=(bit, control_wire, target_wire),
            )
    elif op.name == "qreg.pair_gate" and attrs["gate"] in {"cx", "cz"}:
        coordinates = tuple(zip(values, attrs.get("indices", (0, 0)), strict=True))
        if attrs["gate"] == "cz":
            phase(coordinates, math.pi)
        else:
            owner.effect_op(
                "qreg.control_x",
                token,
                (control, *values),
                indices=(bit, *(i for _, i in coordinates)),
            )
    else:
        reject(token, "qpe", "unsupported controlled operation in QPE callback")


def infer_phase_allocation(
    owner: _Adapt, token: Token, args: tuple[Expression, ...], size: int | None
) -> bool:
    if len(args) != 4 or size is None:
        return False
    sign, fraction = args[1:3]
    if sign.children or sign.token.text not in {"True", "False", "SIGNED", "UNSIGNED"}:
        return False
    point = owner.static_integer(fraction)
    if point != size:
        return False
    target = owner.atom(args[-1])
    parameter = owner.parameters.get(target.text)
    if (
        parameter is None
        or parameter.kind not in {"qnum", "qphase", "qphase_signed"}
        or parameter.width not in {None, size}
        or not (
            parameter.role == "output"
            or parameter.role == "local"
            and parameter.kind == "qphase"
            and owner.scope == "main"
        )
        or (owner.scope != "main" and not owner.helper_stack)
    ):
        reject(token, "qpe", "all-fractional allocation requires a matching Output")
    signed = sign.token.text in {"True", "SIGNED"}
    if parameter.kind in {"qphase", "qphase_signed"} and signed != (
        parameter.kind == "qphase_signed"
    ):
        reject(
            sign.token,
            "precision",
            "fixed-point allocation signedness must agree with its type",
        )
    if signed and owner.helper_stack:
        reject(
            sign.token, "precision", "signed fixed-point helper ports are not admitted"
        )
    owner.parameters[target.text] = replace(
        parameter, kind="qphase_signed" if signed else "qphase", width=size
    )
    owner.widths[target.text] = size
    if parameter.width is None:
        owner.inferred[target.text] = size
    if signed:
        owner.signed.add(target.text)
    return True


def parse_fraction(
    parser: Parser, width: int, allow_phase: bool, *, allow_signed_integer: bool
) -> tuple[str | None, int]:
    sign = parser.current
    if sign.text not in (
        {"UNSIGNED", "SIGNED", "True", "False"} if allow_phase else {"False"}
    ):
        reject(sign, "type", "signed modifiers are not admitted")
    signed = sign.text in {"SIGNED", "True"}
    parser.take(sign.text)
    parser.take(",")
    fraction = parser.current
    value = (
        int(fraction.text)
        if fraction.text.isdecimal()
        else parser.integer_values.get(fraction.text)
    )
    if allow_phase and value == width:
        parser.take(fraction.text)
        parser.take(">")
        return ("qphase_signed" if signed else "qphase"), 0
    if allow_signed_integer and value == 0 and signed:
        parser.take(fraction.text)
        parser.take(">")
        return "qnum_signed", 0
    if allow_phase and not signed and value is not None and 0 < value < width:
        parser.take(fraction.text)
        parser.take(">")
        return "qnum", value
    if value != 0 or signed:
        reject(fraction, "type", "fractional modifiers are not admitted")
    parser.take(fraction.text)
    return None, 0


def _atom(node: Expression, name: str) -> bool:
    return not node.children and node.token.text == name


def _plain(statement: Statement) -> bool:
    return statement == Statement(statement.token, statement.arguments)


def _power_degree(node: Expression, name: str) -> int | None:
    if not node.children:
        if node.token.text == name:
            return 1
        if node.token.text == "pi" or REAL_LITERAL.fullmatch(node.token.text):
            return 0
    if node.token.text == "-" and len(node.children) == 1:
        return _power_degree(node.children[0], name)
    if node.token.text == "*" and len(node.children) == 2:
        left, right = (_power_degree(child, name) for child in node.children)
        if left is not None and right is not None:
            return left + right
    return None


def power_action(helper: Function) -> Statement:
    """Check the borrowed signature and power form even for unused definitions."""
    ports = {p.role: p for p in helper.parameters}
    if (
        helper.permutation
        or helper.disable_permutation_check is not None
        or helper.disable_const_checks is not None
        or len(helper.parameters) != 3
        or len({p.name.text for p in helper.parameters}) != 3
        or {role: port.kind for role, port in ports.items()}
        != {"integer": "int", "operator": "SparsePauliOp", "inout": "qbit[]"}
        or any(p.width is not None or p.record is not None for p in ports.values())
        or len(helper.statements) != 1
    ):
        reject(
            helper.name,
            "qpe",
            "typed power helper needs int, SparsePauliOp and borrowed array ports",
        )
    (action,) = helper.statements
    if (
        not _plain(action)
        or action.token.text != "suzuki_trotter"
        or len(action.arguments) != 5
    ):
        reject(
            action.token,
            "qpe",
            "typed power helper requires a single Suzuki-Trotter call",
        )
    operator, time, order, repetitions, target = action.arguments
    if (
        not _atom(operator, ports["operator"].name.text)
        or not _atom(target, ports["inout"].name.text)
        or not _atom(order, "1")
        or not _atom(repetitions, "1")
        or _power_degree(time, ports["integer"].name.text) != 1
        or any(p.name.text == "pi" for p in ports.values())
    ):
        reject(
            action.token,
            "qpe",
            "typed power helper must forward its ports with a linear power and a single first-order step",
        )
    return action


def specialize_power_body(
    owner: _Adapt, body: tuple[Statement, ...], power: Token
) -> tuple[Statement, ...]:
    if len(body) != 1:
        return body
    (call,) = body
    helper = owner.definitions.get(call.token.text)
    if helper is None or not any(p.role == "operator" for p in helper.parameters):
        return body
    action = power_action(helper)
    if (
        not _plain(call)
        or call.token.text in owner.parameters
        or len(call.arguments) != 3
    ):
        reject(
            call.token, "qpe", "typed power call requires three unshadowed arguments"
        )
    for port, argument in zip(helper.parameters, call.arguments, strict=True):
        if port.role == "integer" and not _atom(argument, power.text):
            reject(
                argument.token, "qpe", "typed power call must forward its lexical power"
            )
        if port.role == "operator":
            operator = owner.constants.get(argument.token.text)
            if (
                argument.children
                or argument.token.text in owner.parameters
                or operator is None
                or operator.token.text != "SparsePauliOp"
            ):
                reject(
                    argument.token,
                    "qpe",
                    "typed power call requires a static SparsePauliOp constant",
                )
    substitution = dict(
        zip((p.name.text for p in helper.parameters), call.arguments, strict=True)
    )

    def substitute(node: Expression) -> Expression:
        if not node.children:
            return substitution.get(node.token.text, node)
        return replace(
            node, children=tuple(substitute(child) for child in node.children)
        )

    return (
        replace(action, arguments=tuple(substitute(arg) for arg in action.arguments)),
    )
