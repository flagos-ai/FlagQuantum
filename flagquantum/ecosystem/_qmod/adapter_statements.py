"""Qmod statement dispatch to verified register effects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapter_model import (
    _GATE_CALLS,
    _INTRINSICS,
    _STATIC_COMPARISONS,
    _STATIC_GATES,
    _UNIFORM_CALLS,
    MAX_LOOP_ITERATIONS,
)
from .amplification import lower_amplification
from .bindings import (
    publish_partition,
)
from .callback_arrays import (
    indexed_callback,
)
from .diffusion import lower_diffuser
from .hea import lower_hea
from .preparation import lower_inplace_preparation
from .qpe import (
    lower_qpe,
)
from .qsvt import lower_qsvt_inversion, lower_qsvt_lcu
from .real_matrices import lower_matrix_iteration
from .scoped_amplification import lower_scoped_amplification
from .selector_rotations import (
    lower_select_rotation,
)
from .swap_test import lower_swap_test
from .syntax import (
    Expression,
    Parameter,
    Statement,
    Token,
    reject,
)
from .unitary_effects import (
    emit_inverse,
)
from .variational import lower_qaoa_penalty

if TYPE_CHECKING:
    from .context import _Adapt


from .adapter_assignments import adapt_assignments
from .adapter_gates import adapt_gates
from .adapter_storage import adapt_storage


def statement(self: _Adapt, statement: Statement, index: str) -> None:
    token, args = statement.token, statement.arguments
    if statement.callback_index is not None:
        for effect in indexed_callback(self, statement):
            self.effect_op(effect.name, token, effect.operands[1:], **effect.attributes)
        return
    if statement.amplification is not None:
        lower_amplification(self, token, args[0], statement.amplification)
        return
    if token.text == "exact_amplitude_amplification":
        lower_scoped_amplification(self, token, args)
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
    elif token.text in ("$kickback_phase", "=", "^=", "phase"):
        adapt_assignments(self, statement, index)
    elif (
        token.text
        in (
            "allocate",
            "drop",
            "encode_in_angle",
            "encode_on_bloch",
            "free",
            "inplace_encode_on_bloch",
            "inplace_prepare_amplitudes",
            "prepare_amplitudes",
            "prepare_bell_state",
            "prepare_exponential_state",
            "prepare_ghz_state",
            "prepare_state",
        )
        or token.text in _UNIFORM_CALLS
    ):
        adapt_storage(self, statement)
    else:
        adapt_gates(self, statement)
