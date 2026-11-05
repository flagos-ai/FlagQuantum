"""Static ordinary QPE composition over existing checked register effects."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from ...compiler._hybrid.model import Operation, Value
from .syntax import Expression, Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt
    from .syntax import Parser


def parse_qpe_call(parser: Parser, token: Token, depth: int) -> Statement:
    if depth:
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
    if fraction.children:
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
        or parameter.role != "output"
        or owner.scope != "main"
    ):
        reject(
            token, "qpe", "all-fractional allocation requires a matching main output"
        )
    signed = sign.token.text in {"True", "SIGNED"}
    owner.parameters[target.text] = replace(
        parameter, kind="qphase_signed" if signed else "qphase", width=size
    )
    owner.widths[target.text] = size
    if parameter.width is None:
        owner.inferred[target.text] = size
    if signed:
        owner.signed.add(target.text)
    return True


def parse_fraction(parser: Parser, width: int, allow_phase: bool) -> str | None:
    sign = parser.current
    if sign.text not in ({"UNSIGNED", "SIGNED", "False"} if allow_phase else {"False"}):
        reject(sign, "type", "signed modifiers are not admitted")
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
        return "qphase_signed" if sign.text == "SIGNED" else "qphase"
    if value != 0 or sign.text == "SIGNED":
        reject(fraction, "type", "fractional modifiers are not admitted")
    parser.take(fraction.text)
    return None
