"""Qmod statement dispatch to verified register effects."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.ranges import unsigned_width
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
    _INTRINSICS,
    _PAIR_CALLS,
    _STATIC_COMPARISONS,
    _STATIC_GATES,
    _UNIFORM_CALLS,
    MAX_LOOP_ITERATIONS,
    _Expression,
    _literal_angle,
)
from .amplification import lower_amplification
from .bindings import (
    publish_partition,
)
from .callback_arrays import (
    indexed_callback,
)
from .diffusion import lower_diffuser
from .fixed_xor import lower_fixed_xor
from .hea import lower_hea
from .local_cleanup import lower_free
from .matrices import lower_unitary
from .parity_updates import lower_indexed_parity
from .preparation import lower_inplace_preparation
from .qpe import (
    infer_phase_allocation,
    lower_qpe,
)
from .qsvt import lower_qsvt_inversion, lower_qsvt_lcu
from .real_matrices import lower_matrix_iteration
from .selector_rotations import (
    lower_select_rotation,
)
from .swap_test import lower_swap_test
from .syntax import (
    ANGLE_CALLS,
    Expression,
    Parameter,
    Statement,
    Token,
    exact_unsigned_literal,
    reject,
)
from .unitary_effects import (
    emit_inverse,
)
from .variational import lower_qaoa_penalty

if TYPE_CHECKING:
    from .adapter import _Adapt


def statement(self: _Adapt, statement: Statement, index: str) -> None:
    token, args = statement.token, statement.arguments
    if statement.callback_index is not None:
        for effect in indexed_callback(self, statement):
            self.effect_op(effect.name, token, effect.operands[1:], **effect.attributes)
        return
    if statement.amplification is not None:
        lower_amplification(self, token, args[0], statement.amplification)
        return
    if token.text == "RESET":
        if len(args) != 1:
            reject(token, "signature", "RESET requires one initialized bit")
        value, wire = self.gate_target(args[0])
        self.effect_op("qreg.reset", token, (value,), index=wire)
        return
    if token.text == "swap_test":
        lower_swap_test(self, token, args)
        return
    if token.text == "$projected_rz":
        phase_angle, marker, condition = args
        location = self.gate_target(marker, allow_numeric=True)

        def literal(text: str, *children: Expression) -> Expression:
            return Expression(Token(text, token.location), children)

        # XOR conjugation maps the RZ exponent to t - 1/2 + b(1 - 2t).
        weighted = literal("-", literal("1"), literal("*", literal("2"), marker))
        self.real_phase(
            token,
            weighted,
            phase_angle,
            condition=condition,
            disjoint_condition_target=location,
        )
        self.real_phase(token, literal("-", marker, literal("0.5")), phase_angle)
        return
    if token.text == "qsvt_lcu":
        lower_qsvt_lcu(self, token, args)
        return
    if token.text == "qsvt_inversion":
        lower_qsvt_inversion(self, token, args)
        return
    if token.text == "grover_diffuser":
        lower_diffuser(self, token, args)
        return
    if token.text == "multiswap":
        if len(args) != 2:
            reject(token, "signature", "multiswap requires two register views")
        left_view, right_view = (self.array_view(arg) for arg in args)
        if len(left_view) != len(right_view) or len(
            set((*left_view, *right_view))
        ) != len(left_view) + len(right_view):
            reject(token, "alias", "multiswap needs equally sized disjoint views")
        for (first, i), (second, j) in zip(left_view, right_view, strict=True):
            self.effect_op(
                "qreg.pair_gate",
                token,
                (first, second),
                gate="swap",
                indices=(i, j),
            )
        return
    if token.text == "inplace_prepare_state":
        lower_inplace_preparation(self, token, args)
        return
    if token.text == "select_rotation":
        lower_select_rotation(self, token, args)
        return
    if token.text == "qaoa_penalty":
        lower_qaoa_penalty(self, token, args)
        return
    if token.text == "full_hea":
        lower_hea(self, token, args)
        return
    if token.text == "CPHASE":
        if len(args) != 3:
            reject(token, "signature", "CPHASE requires an angle and two qubits")
        locations = tuple(self.gate_target(arg) for arg in args[1:])
        if len(set(locations)) != 2:
            reject(token, "alias", "CPHASE requires distinct qubit locations")
        self.effect_op(
            "qreg.control_phase",
            token,
            tuple(value for value, _ in locations),
            indices=tuple(index for _, index in locations),
            angle=self.angle(args[0], token),
        )
        return
    if statement.map_callback is not None:
        variable, body = statement.map_callback
        name = variable.text
        if (
            name in self.parameters
            or name in _INTRINSICS
            or name in self.constants
            or name in self.definitions
            or name in self.loop_indices
        ):
            reject(variable, "name", "map callback cannot shadow a bound symbol")
        if any(
            child.declaration is not None
            or child.token.text not in _GATE_CALLS | {"RX", "RY", "RZ"}
            for child in body
        ):
            reject(token, "scope", "map callback admits only single-qubit gates")
        locations = self.array_view(args[0])
        self.parameters[name] = Parameter(variable, "qbit", 1, "inout")
        self.widths[name] = 1
        try:
            for ordinal, location in enumerate(locations):
                self.views[name] = (location,)
                for gate_index, child in enumerate(body):
                    self.statement(child, f"{index}.map{ordinal}.{gate_index}")
        finally:
            self.views.pop(name, None)
            self.parameters.pop(name)
            self.widths.pop(name)
        return
    if token.text == "->":
        if self.scope == "main" and publish_partition(self, token, args[0], args[1]):
            return
        self.bind_registers(token, args[0], args[1], publish_output=True)
        return
    if token.text == "+=":
        self.increment(token, args[0], args[1])
        return
    if statement.matrix_row is not None:
        if any(variable.text in _INTRINSICS for variable in statement.matrix_row):
            reject(token, "name", "real matrix row names cannot shadow intrinsics")
        lower_matrix_iteration(self, statement, index, MAX_LOOP_ITERATIONS)
        return
    if token.text in self.zero_callbacks:
        if statement != Statement(token, ()) or args:
            reject(
                token,
                "callback.signature",
                "zero-argument callbacks do not accept operands or nested scopes",
            )
        for op in self.zero_callbacks[token.text]:
            self.effect_op(op.name, token, op.operands[1:], **op.attributes)
        return
    if statement.static_if_body is not None:
        if any(
            item.declaration is not None
            or (
                item.token.text not in _STATIC_GATES
                and item.static_if_body is None
                and item.matrix_row is None
                and item.control_body is None
                and item.token.text not in self.definitions
            )
            for item in statement.static_if_body
        ):
            reject(token, "scope", "static if bodies admit only gates")
        condition = args[0]
        left, right = (self.static_integer(item) for item in condition.children)
        if _STATIC_COMPARISONS[condition.token.text](left, right):
            for ordinal, child in enumerate(statement.static_if_body):
                self.statement(child, f"{index}.if{ordinal}")
        return
    if statement.inverse_body is not None:
        body = statement.inverse_body
        allowed = _GATE_CALLS | {
            "PHASE",
            "IDENTITY",
            "RX",
            "RY",
            "RZ",
            "inplace_encode_on_bloch",
            "inplace_prepare_state",
            "inplace_prepare_amplitudes",
            "qft",
        }
        if not body or any(
            child.token.text not in allowed
            and child.token.text not in self.definitions
            and child.token.text not in self.callback_bindings
            and child.token.text not in self.zero_callbacks
            and child.inverse_body is None
            and child.control_body is None
            for child in body
        ):
            reject(
                token,
                "scope",
                "invert requires static single-qubit gates or reversible helpers",
            )
        start, incoming = len(self.operations), self.effect
        for ordinal, child in enumerate(body):
            self.statement(child, f"{index}.invert{ordinal}")
        forward = self.operations[start:]
        del self.operations[start:]
        self.effect = incoming
        emit_inverse(self, token, forward)
    elif statement.qpe_body is not None:
        lower_qpe(self, statement, index)
    elif statement.power_callback is not None:
        self.qpe_flexible(statement)
    elif statement.repeat_index is not None:
        self.repeat(statement, index)
    elif statement.declaration is not None:
        parameter = statement.declaration
        name = parameter.name.text
        if parameter.fractional_bits:
            reject(
                parameter.name,
                "precision",
                "fixed-point locals require a checked computed rotation",
            )
        if name in self.parameters or name in _INTRINSICS or name in self.constants:
            reject(
                parameter.name,
                "name",
                "local names cannot shadow parameters, locals or intrinsics",
            )
        width = parameter.width
        if parameter.width_expression is not None:
            width = self.static_integer(parameter.width_expression)
            if not 1 <= width <= 256:
                reject(parameter.name, "width", "local width must be in 1..256")
        self.parameters[name] = parameter
        self.widths[name] = width
    elif statement.control_body is not None:
        self.scalar_control(statement, index)
    elif statement.within is not None:
        self.within(statement, index)
    elif token.text == "=":
        target = self.atom(args[0])
        if (
            self.helper_stack
            and not args[1].children
            and args[1].token.text in self.parameters
            and self.parameters[args[1].token.text].kind == "qnum"
            and args[1].token.text not in self.signed
            and self.parameters.get(target.text) is not None
            and self.parameters[target.text].role == "output"
            and self.parameters[target.text].kind == "qnum"
            and target.text not in self.signed
        ):
            view = self.array_view(args[1])
            register = self.register(target, len(view))
            if any(value == register for value, _ in view):
                reject(target, "alias", "copy requires disjoint input and Output")
            self.effect_op("qreg.allocate", token, (register,))
            for bit, control in enumerate(view):
                self.emit_indexed_x(token, (control, (register, bit)))
            return
        if "." in args[1].token.text and not args[1].children:
            reject(
                args[1].token,
                "operator",
                "decimal assignment requires min/max, subtraction or multiplication",
            )
        if (
            args[1].token.text == "-"
            and len(args[1].children) == 1
            and target.text in self.parameters
            and self.parameters[target.text].kind == "qbit"
        ):
            reject(args[1].token, "expression", "negation requires a numeric output")
        expression = _Expression(
            args[1], self.widths, f"{self.scope}.expr{index}", self.signed
        )
        if self.local_scope is not None and not expression.quantum:
            reject(
                token,
                "scope",
                "within preparation requires a quantum-dependent expression",
            )
        inferred_width = (
            expression.word_widths[args[1]]
            if args[1].token.text in {"%", "-"}
            else unsigned_width(expression.bound)
        )
        destination = self.register(target, inferred_width)
        if args[1] in expression.signed_words:
            if (
                self.scope != "main"
                or self.local_scope is not None
                or self.parameters[target.text].kind != "qnum"
                or register_width(destination) != expression.word_widths[args[1]]
            ):
                reject(
                    target,
                    "type",
                    "signed results require a matching-width main output",
                )
            self.signed.add(target.text)

        fractional_bits = expression.fractional_bits.get(args[1], 0)
        if fractional_bits:
            if (
                self.scope != "main"
                or self.local_scope is not None
                or self.parameters[target.text].kind != "qnum"
                or self.parameters[target.text].width is not None
            ):
                reject(
                    target,
                    "precision",
                    "fractional results require an inferred-width main qnum output",
                )
        if expression.quantum:
            graph = expression.function()
            self.functions.append(graph)
            inputs = tuple(
                self.register(self.parameters[name].name) for name in expression.names
            )
            self.effect_op(
                "qreg.compute", token, (destination, *inputs), callee=graph.name
            )
        else:
            self.effect_op(
                "qreg.initialize", token, (destination,), value=expression.bound
            )
        if fractional_bits:
            self.fractional_bits[target.text] = fractional_bits
    elif token.text == "$kickback_phase":
        condition, kickback_coefficient = args
        self.real_phase(
            token,
            Expression(replace(token, text="1")),
            kickback_coefficient,
            condition=condition,
            phase_sign=True,
        )
    elif token.text == "phase":
        default_coefficient = len(args) == 1
        if default_coefficient:
            args = (*args, Expression(Token("1.0", token.location)))
        if len(args) != 2 or "phase" in self.parameters:
            reject(
                token,
                "signature",
                "phase requires an expression and unshadowed pi coefficient",
            )
        pending = [args[0]]
        indexed = False
        while pending:
            node = pending.pop()
            if node.token.text == "[":
                indexed = True
                break
            pending.extend(node.children)
        if (
            default_coefficient
            or bool(args[1].children)
            or args[1].token.text.isdecimal()
            or indexed
            or args[1].token.text == "["
            or any(
                parameter.record is not None for parameter in self.parameters.values()
            )
            or args[1].token.text in self.classical_values
            or args[1].token.text in self.classical_integers
            or (
                self.helper_stack
                and (
                    self.classical_values
                    or self.classical_arrays
                    or self.classical_integers
                    or self.integer_arrays
                )
            )
        ):
            self.real_phase(token, args[0], args[1])
            return
        coefficient = args[1]
        angle = None
        if "." in coefficient.token.text:
            angle = _literal_angle(coefficient, token)
        elif (
            coefficient.children
            or coefficient.token.text != "pi"
            or "pi" in self.parameters
        ):
            reject(
                coefficient.token,
                "coefficient",
                "phase requires unshadowed pi or a decimal literal",
            )
        expression = _Expression(
            args[0], self.widths, f"{self.scope}.phase{index}", self.signed
        )
        if (
            args[0] in expression.signed_words
            or expression.fractional_bits.get(args[0], 0)
            or not expression.quantum
            or (angle is None and unsigned_width(expression.bound) != 1)
        ):
            reject(
                args[0].token,
                "phase",
                "phase requires a quantum expression and pi requires a one-bit result",
            )
        graph = expression.function()
        self.functions.append(graph)
        inputs = tuple(
            self.register(self.parameters[name].name) for name in expression.names
        )
        phase_attributes = {} if angle is None else {"angle": angle}
        self.effect_op(
            "qreg.phase", token, inputs, callee=graph.name, **phase_attributes
        )
    elif token.text == "^=":
        if not (
            lower_indexed_parity(self, token, args[0], args[1])
            or lower_fixed_xor(self, token, args[0], args[1])
        ):
            self.xor_update(token, args[0], args[1])
    elif token.text in {
        "encode_in_angle",
        "encode_on_bloch",
        "inplace_encode_on_bloch",
    }:
        self.encode_data(token, args)
    elif token.text == "allocate":
        if len(args) not in {1, 2, 4}:
            reject(
                token,
                "signature",
                "allocate accepts an output, size and output, or size, signedness, zero and output",
            )
        size = None
        if len(args) > 1:
            if (
                args[0].token.text == "."
                and len(args[0].children) == 2
                and args[0].children[1].token.text in {"size", "len"}
            ):
                size = self.static_integer(args[0])
            elif not args[0].children:
                name = args[0].token.text
                size = self.classical_integers.get(name)
                if size is None:
                    size = exact_unsigned_literal(name)
            elif args[0].token.text in {"+", "-", "*", "%", "**", "$ceil_log2"}:
                size = self.static_integer(args[0])
            if size is None:
                reject(
                    args[0].token,
                    "width",
                    "allocation size must be a literal or record size",
                )
            if not 1 <= size <= 256:
                reject(
                    args[0].token,
                    "width",
                    "allocation size must be a literal in 1..256",
                )
        target = self.atom(args[-1])
        if self.helper_stack and (
            self.parameters.get(target.text) is None
            or not (
                self.parameters[target.text].role == "output"
                or self.parameters[target.text].role == "local"
                and self.parameters[target.text].kind == "qbit"
            )
        ):
            reject(target, "free", "bound allocation requires a local qbit")
        phase_allocation = infer_phase_allocation(self, token, args, size)
        phase_parameter = self.parameters.get(target.text)
        destination = self.register(
            target,
            size,
            allow_qubit_array=True,
            allocate_local=True,
            allow_fractional_output=(
                phase_parameter is not None
                and phase_parameter.kind in {"qphase", "qphase_signed"}
                and target.text not in self.fractional_bits
            ),
        )
        if size is not None and self.widths[target.text] != size:
            reject(
                target,
                "width",
                "allocation size must agree with the declared width",
            )
        if len(args) == 4 and not phase_allocation:
            sign, fraction = args[1:3]
            if (
                sign.children
                or sign.token.text not in {"True", "False", "SIGNED", "UNSIGNED"}
                or fraction.children
                or fraction.token.text != "0"
                or self.parameters[target.text].kind != "qnum"
            ):
                reject(
                    token,
                    "type",
                    "typed allocate requires a qnum, literal signedness and zero fractional bits",
                )
            if sign.token.text in {"True", "SIGNED"}:
                if self.scope != "main" or self.local_scope is not None:
                    reject(
                        token,
                        "scope",
                        "signed allocation is supported only at main scope",
                    )
                self.signed.add(target.text)
            elif target.text in self.signed:
                reject(
                    sign.token,
                    "type",
                    "allocation signedness must agree with the declared type",
                )
        self.effect_op("qreg.allocate", token, (destination,))
    elif token.text in {
        "prepare_amplitudes",
        "inplace_prepare_amplitudes",
        "prepare_state",
    }:
        self.prepare_state_vector(token, args)
    elif token.text == "prepare_exponential_state":
        self.prepare_exponential_state(token, args)
    elif token.text == "modular_exp":
        if len(args) != 4 or any(
            arg.children or not arg.token.text.isdecimal() for arg in args[:2]
        ):
            reject(
                token,
                "signature",
                "modular_exp requires literal modulus and base, target and exponent",
            )
        modulus, base = (int(arg.token.text) for arg in args[:2])
        names = tuple(self.atom(arg) for arg in args[2:])
        registers = tuple(self.register(name) for name in names)
        if any(
            self.parameters[name.text].kind != "qnum" or name.text in self.signed
            for name in names
        ):
            reject(token, "type", "modular_exp requires unsigned qnum registers")
        self.effect_op("qreg.modular_exp", token, registers, modulus=modulus, base=base)
    elif token.text == "free":
        if len(args) != 1:
            reject(token, "signature", "free requires one local qbit")
        lower_free(self, self.atom(args[0]))
    elif token.text == "drop":
        if len(args) != 1 or self.scope != "main" or self.local_scope is not None:
            reject(token, "drop", "drop requires one local register in main")
        target = self.atom(args[0])
        drop_parameter = self.parameters.get(target.text)
        if (
            drop_parameter is None
            or drop_parameter.role != "local"
            or drop_parameter.kind not in {"qbit", "qnum", "qbit[]", "qphase"}
        ):
            reject(target, "drop", "drop requires a local register")
        destination = self.register(
            target,
            allow_qubit_array=True,
            allow_fractional_output=drop_parameter.kind == "qphase",
        )
        self.effect_op("qreg.drop", token, (destination,))
        self.dropped.add(target.text)
    elif token.text == "assign_phase_table":
        if len(args) != 2:
            reject(token, "signature", "phase table requires angles and a quantum word")
        angles = self.real_array(args[0])
        locations = self.array_view(args[1])
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
                self.effect_op(
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
            (self.gate_target(args[0]),) if indexed else self.array_view(args[0])
        )
        root = args[0].children[0] if indexed else args[0]
        port = self.parameters.get(root.token.text) if not root.children else None
        checked: set[Value] = set()
        for value, bit in locations:
            if value not in checked:
                if port is not None and port.role == "const":
                    self.effect_op(
                        "qreg.control_phase",
                        token,
                        (value,),
                        indices=(bit,),
                        angle=0.0,
                    )
                else:
                    self.effect_op("qreg.gate", token, (value,), gate="i", index=bit)
                checked.add(value)
    elif token.text == "qft":
        if len(args) != 1:
            reject(token, "signature", "qft requires one qubit array")
        target = self.atom(args[0])
        fourier_parameter = self.parameters.get(target.text)
        kind = fourier_parameter.kind if fourier_parameter is not None else None
        if target.text in self.fractional_bits:
            reject(target, "precision", "a fractional result is a terminal output")
        register = self.register(
            target, allow_qubit_array=True, allow_fractional_output=kind == "qphase"
        )
        if kind not in {"qbit[]", "qnum", "qphase"}:
            reject(target, "type", "qft requires a qubit array or integer register")
        self.effect_op("qreg.qft", token, (register,))
    elif token.text == "qct_qst_type1":
        if len(args) != 1:
            reject(token, "signature", "Type I transform requires one data register")
        data = self.register(self.atom(args[0]), allow_qubit_array=True)
        self.effect_op("qreg.type1_transform", token, (data,))
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
        data = self.register(self.atom(args[0]), allow_qubit_array=True)
        selectors: tuple[Value, ...] = ()
        if kind == "combined":
            self.atom(args[1])
            selector, _ = self.gate_target(args[1])
            selectors = (selector,)
        self.effect_op("qreg.type2_transform", token, (data, *selectors), kind=kind)
    elif token.text == "prepare_bell_state":
        if len(args) != 2:
            reject(token, "signature", "prepare_bell_state requires a state and output")
        state, output = args
        if state.children or state.token.text not in {"1", "2"}:
            reject(
                state.token,
                "prepare",
                "only literal Bell states 1 and 2 are admitted",
            )
        target = self.atom(output)
        if (
            target.text not in self.parameters
            or self.parameters[target.text].kind != "qbit[]"
        ):
            reject(target, "type", "prepare_bell_state requires a qubit array output")
        destination = self.register(target, 2, allow_qubit_array=True)
        if self.widths[target.text] != 2:
            reject(target, "width", "Bell preparation requires exactly two qubits")
        self.effect_op("qreg.prepare_ghz", token, (destination,))
        self.effect_op(
            "qreg.gate",
            token,
            (destination,),
            gate="z" if state.token.text == "1" else "x",
            index=0,
        )
    elif token.text == "prepare_ghz_state":
        if len(args) != 2:
            reject(token, "signature", "prepare_ghz_state requires a size and output")
        size_arg, output = args
        if size_arg.children or not size_arg.token.text.isdecimal():
            reject(size_arg.token, "width", "GHZ size requires a literal in 1..256")
        size = int(size_arg.token.text)
        if not 1 <= size <= 256:
            reject(size_arg.token, "width", "GHZ size requires a literal in 1..256")
        target = self.atom(output)
        if (
            target.text not in self.parameters
            or self.parameters[target.text].kind != "qbit[]"
        ):
            reject(target, "type", "prepare_ghz_state requires a qubit array output")
        destination = self.register(target, size, allow_qubit_array=True)
        if self.widths[target.text] != size:
            reject(target, "width", "GHZ size must agree with the output width")
        self.effect_op("qreg.prepare_ghz", token, (destination,))
    elif token.text in _UNIFORM_CALLS:
        trimmed = token.text == "prepare_uniform_trimmed_state"
        if len(args) != (2 if trimmed else 3):
            reject(token, "signature", "invalid uniform preparation arity")
        bounds = args[:-1]
        if any(bound.children or not bound.token.text.isdecimal() for bound in bounds):
            reject(
                token,
                "signature",
                "uniform preparation requires literal integer bounds",
            )
        start = 0 if trimmed else int(bounds[0].token.text)
        stop = int(bounds[-1].token.text)
        destination = self.register(self.atom(args[-1]), allow_qubit_array=True)
        self.effect_op(
            "qreg.uniform_interval", token, (destination,), start=start, stop=stop
        )
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
        for value, wire in self.array_view(args[1]):
            self.effect_op(
                "qreg.gate",
                token,
                (value,),
                gate=args[0].token.text.lower(),
                index=wire,
            )
    elif token.text == "unitary":
        lower_unitary(self, token, args)
    elif token.text == "linear_pauli_rotations":
        self.linear_pauli_rotations(token, args)
    elif token.text == "single_pauli_exponent":
        self.single_pauli_exponent(token, args)
    elif token.text in {"suzuki_trotter", "multi_suzuki_trotter"}:
        self.suzuki_trotter(token, args)
    elif token.text == "exponentiate":
        self.exponentiate(token, args)
    elif token.text == "qdrift":
        self.qdrift(token, args)
    elif token.text == "RZZ":
        if len(args) != 2:
            reject(token, "signature", "RZZ requires an angle and a qubit array")
        angle = self.angle(args[0], token)
        view = self.array_view(args[1])
        if len(view) != 2:
            reject(args[1].token, "type", "RZZ requires a two-qubit array or view")
        if len(set(view)) != 2:
            reject(args[1].token, "alias", "RZZ requires distinct qubit locations")
        (first, first_index), (second, second_index) = view
        self.effect_op(
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
        theta, phi, lam, gamma = (self.angle(arg, token) for arg in args[:4])
        target = self.atom(args[4])
        destination = self.register(target)
        if self.parameters[target.text].kind != "qbit":
            reject(target, "type", "U requires a scalar qbit target")
        for gate, angle in euler_rotation(theta, phi, lam):
            self.effect_op(
                "qreg.gate", token, (destination,), gate=gate, index=0, angle=angle
            )
        self.effect_op("qreg.phase", token, (), angle=gamma)
    elif token.text == "R":
        if len(args) != 3:
            reject(token, "signature", "R requires two angles and a qbit target")
        theta, phi = (self.angle(arg, token) for arg in args[:2])
        target = self.atom(args[2])
        destination = self.register(target)
        if self.parameters[target.text].kind != "qbit":
            reject(target, "type", "R requires a scalar qbit target")
        for gate, angle in equatorial_rotation(theta, phi):
            self.effect_op(
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
            attributes["angle"] = self.angle(args[0], token)
        targets = args[int(angled) :]
        locations = tuple(
            self.gate_target(arg, allow_numeric=token.text in {"CX", "CZ"})
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
        self.effect_op(
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
                gate_locations = self.array_view(args[0])
            else:
                target = self.atom(args[0])
                if target.text in self.views:
                    gate_locations = self.array_view(args[0])
                else:
                    if target.text in self.fractional_bits:
                        reject(
                            target,
                            "precision",
                            "a fractional result is a terminal output",
                        )
                    hadamard_parameter = self.parameters.get(target.text)
                    destination = self.register(
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
                self.gate_target(args[0], allow_numeric=token.text in {"H", "X", "Z"}),
            )
        for destination, bit in gate_locations:
            if token.text == "Y":
                for gate, gate_attributes in (
                    ("s", {"inverse": True}),
                    ("x", {}),
                    ("s", {}),
                ):
                    self.effect_op(
                        "qreg.gate",
                        token,
                        (destination,),
                        gate=gate,
                        index=bit,
                        **gate_attributes,
                    )
                continue
            self.effect_op(
                "qreg.gate",
                token,
                (destination,),
                gate=(
                    "h" if token.text == "hadamard_transform" else token.text.lower()
                ),
                index=bit,
            )
    else:
        self.call(token, args)
