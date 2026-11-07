"""Qmod gates emitted through the function context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import (
    equatorial_rotation,
    euler_rotation,
    register_width,
)
from ...compiler._hybrid.model import (
    Value,
)
from .adapter_model import (
    _GATE_CALLS,
    _PAIR_CALLS,
)
from .matrices import lower_unitary
from .syntax import (
    ANGLE_CALLS,
    Statement,
    reject,
)

if TYPE_CHECKING:
    from .context import _Adapt


def adapt_gates(context: _Adapt, statement: Statement) -> None:
    token, args = statement.token, statement.arguments
    if token.text == "modular_exp":
        if len(args) != 4 or any(
            arg.children or not arg.token.text.isdecimal() for arg in args[:2]
        ):
            reject(
                token,
                "signature",
                "modular_exp requires literal modulus and base, target and exponent",
            )
        modulus, base = (int(arg.token.text) for arg in args[:2])
        names = tuple(context.atom(arg) for arg in args[2:])
        registers = tuple(context.register(name) for name in names)
        if any(
            context.parameters[name.text].kind != "qnum" or name.text in context.signed
            for name in names
        ):
            reject(token, "type", "modular_exp requires unsigned qnum registers")
        context.effect_op(
            "qreg.modular_exp", token, registers, modulus=modulus, base=base
        )
    elif token.text == "assign_phase_table":
        if len(args) != 2:
            reject(token, "signature", "phase table requires angles and a quantum word")
        angles = context.real_array(args[0])
        locations = context.array_view(args[1])
        if not 1 <= len(locations) <= 7 or len(angles) != 1 << len(locations):
            reject(
                token,
                "width",
                "phase table size must match a one-to-seven-bit word",
            )
        if len(locations) != len(set(locations)):
            reject(token, "alias", "phase table target locations must be disjoint")
        registers, indices = zip(*locations, strict=True)
        for word, angle in enumerate(angles):
            if abs(angle) > 2**53:
                reject(token, "phase", "phase table angles must fit finite phase range")
            if angle or word == 0:
                context.effect_op(
                    "qreg.control_phase",
                    token,
                    registers,
                    indices=indices,
                    control_state=word,
                    angle=angle,
                )
    elif token.text == "IDENTITY":
        if len(args) != 1:
            reject(token, "signature", "IDENTITY requires one register")
        indexed = args[0].token.text == "[" and not args[0].children[0].children
        locations = (
            (context.gate_target(args[0]),) if indexed else context.array_view(args[0])
        )
        root = args[0].children[0] if indexed else args[0]
        port = context.parameters.get(root.token.text) if not root.children else None
        checked: set[Value] = set()
        for value, bit in locations:
            if value not in checked:
                if port is not None and port.role == "const":
                    context.effect_op(
                        "qreg.control_phase",
                        token,
                        (value,),
                        indices=(bit,),
                        angle=0.0,
                    )
                else:
                    context.effect_op("qreg.gate", token, (value,), gate="i", index=bit)
                checked.add(value)
    elif token.text == "qft":
        if len(args) != 1:
            reject(token, "signature", "qft requires one qubit array")
        target = context.atom(args[0])
        fourier_parameter = context.parameters.get(target.text)
        kind = fourier_parameter.kind if fourier_parameter is not None else None
        if target.text in context.fractional_bits:
            reject(target, "precision", "a fractional result is a terminal output")
        register = context.register(
            target, allow_qubit_array=True, allow_fractional_output=kind == "qphase"
        )
        if kind not in {"qbit[]", "qnum", "qphase"}:
            reject(target, "type", "qft requires a qubit array or integer register")
        context.effect_op("qreg.qft", token, (register,))
    elif token.text == "qct_qst_type1":
        if len(args) != 1:
            reject(token, "signature", "Type I transform requires one data register")
        data = context.register(context.atom(args[0]), allow_qubit_array=True)
        context.effect_op("qreg.type1_transform", token, (data,))
    elif token.text in {"qct_type2", "qst_type2", "qct_qst_type2"}:
        kind = {
            "qct_type2": "cosine",
            "qst_type2": "sine",
            "qct_qst_type2": "combined",
        }[token.text]
        if len(args) != 1 + (kind == "combined"):
            reject(
                token,
                "signature",
                "Type II transform requires a data register and, for the combined form, a selector",
            )
        data = context.register(context.atom(args[0]), allow_qubit_array=True)
        selectors: tuple[Value, ...] = ()
        if kind == "combined":
            context.atom(args[1])
            selector, _ = context.gate_target(args[1])
            selectors = (selector,)
        context.effect_op("qreg.type2_transform", token, (data, *selectors), kind=kind)
    elif token.text == "apply_to_all":
        if (
            len(args) != 2
            or args[0].children
            or args[0].token.text not in {"H", "X", "Z"}
        ):
            reject(
                token,
                "signature",
                "apply_to_all currently requires H, X or Z and a register",
            )
        for value, wire in context.array_view(args[1]):
            context.effect_op(
                "qreg.gate",
                token,
                (value,),
                gate=args[0].token.text.lower(),
                index=wire,
            )
    elif token.text == "unitary":
        lower_unitary(context, token, args)
    elif token.text == "linear_pauli_rotations":
        context.linear_pauli_rotations(token, args)
    elif token.text == "single_pauli_exponent":
        context.single_pauli_exponent(token, args)
    elif token.text in {"suzuki_trotter", "multi_suzuki_trotter"}:
        context.suzuki_trotter(token, args)
    elif token.text == "exponentiate":
        context.exponentiate(token, args)
    elif token.text == "qdrift":
        context.qdrift(token, args)
    elif token.text == "RZZ":
        if len(args) != 2:
            reject(token, "signature", "RZZ requires an angle and a qubit array")
        angle = context.angle(args[0], token)
        view = context.array_view(args[1])
        if len(view) != 2:
            reject(args[1].token, "type", "RZZ requires a two-qubit array or view")
        if len(set(view)) != 2:
            reject(args[1].token, "alias", "RZZ requires distinct qubit locations")
        (first, first_index), (second, second_index) = view
        context.effect_op(
            "qreg.pair_gate",
            token,
            (first, second),
            gate="rzz",
            indices=(first_index, second_index),
            angle=angle,
        )
    elif token.text == "U":
        if len(args) != 5:
            reject(token, "signature", "U requires four angles and a qbit target")
        theta, phi, lam, gamma = (context.angle(arg, token) for arg in args[:4])
        target = context.atom(args[4])
        destination = context.register(target)
        if context.parameters[target.text].kind != "qbit":
            reject(target, "type", "U requires a scalar qbit target")
        for gate, angle in euler_rotation(theta, phi, lam):
            context.effect_op(
                "qreg.gate", token, (destination,), gate=gate, index=0, angle=angle
            )
        context.effect_op("qreg.phase", token, (), angle=gamma)
    elif token.text == "R":
        if len(args) != 3:
            reject(token, "signature", "R requires two angles and a qbit target")
        theta, phi = (context.angle(arg, token) for arg in args[:2])
        target = context.atom(args[2])
        destination = context.register(target)
        if context.parameters[target.text].kind != "qbit":
            reject(target, "type", "R requires a scalar qbit target")
        for gate, angle in equatorial_rotation(theta, phi):
            context.effect_op(
                "qreg.gate", token, (destination,), gate=gate, index=0, angle=angle
            )
    elif token.text in ANGLE_CALLS or token.text in _PAIR_CALLS:
        pair = token.text in _PAIR_CALLS
        angled = token.text in ANGLE_CALLS
        if len(args) != (2 if pair else 1) + angled:
            reject(
                token,
                "signature",
                f"{token.text} has the wrong number of arguments",
            )
        attributes: dict[str, object] = {"gate": token.text.lower()}
        if angled:
            attributes["angle"] = context.angle(args[0], token)
        targets = args[int(angled) :]
        locations = tuple(
            context.gate_target(arg, allow_numeric=token.text in {"CX", "CZ"})
            for arg in targets
        )
        registers, indices = zip(*locations, strict=True)
        if pair and (
            any(arg.token.text == "[" for arg in targets)
            or any(register_width(register) != 1 for register in registers)
        ):
            attributes["indices"] = indices
        if not pair:
            attributes["index"] = indices[0]
        context.effect_op(
            "qreg.pair_gate" if pair else "qreg.gate",
            token,
            registers,
            **attributes,
        )
    elif token.text in _GATE_CALLS:
        if len(args) != 1:
            reject(token, "signature", f"{token.text} requires one output variable")
        if token.text == "hadamard_transform":
            if args[0].token.text in {".", "["}:
                gate_locations = context.array_view(args[0], allow_phase=True)
            else:
                target = context.atom(args[0])
                if target.text in context.views:
                    gate_locations = context.array_view(args[0])
                else:
                    if target.text in context.fractional_bits:
                        reject(
                            target,
                            "precision",
                            "a fractional result is a terminal output",
                        )
                    hadamard_parameter = context.parameters.get(target.text)
                    destination = context.register(
                        target,
                        allow_qubit_array=True,
                        allow_fractional_output=hadamard_parameter is not None
                        and hadamard_parameter.kind == "qphase",
                    )
                    gate_locations = tuple(
                        (destination, bit) for bit in range(register_width(destination))
                    )
        else:
            gate_locations = (
                context.gate_target(
                    args[0], allow_numeric=token.text in {"H", "X", "Z"}
                ),
            )
        for destination, bit in gate_locations:
            if token.text == "Y":
                for gate, gate_attributes in (
                    ("s", {"inverse": True}),
                    ("x", {}),
                    ("s", {}),
                ):
                    context.effect_op(
                        "qreg.gate",
                        token,
                        (destination,),
                        gate=gate,
                        index=bit,
                        **gate_attributes,
                    )
                continue
            context.effect_op(
                "qreg.gate",
                token,
                (destination,),
                gate=(
                    "h" if token.text == "hadamard_transform" else token.text.lower()
                ),
                index=bit,
            )
    else:
        context.call(token, args)
