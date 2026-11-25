"""Qmod storage emitted through the function context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapter_model import (
    _UNIFORM_CALLS,
)
from .local_cleanup import lower_free
from .qpe import (
    infer_phase_allocation,
)
from .syntax import (
    Statement,
    exact_unsigned_literal,
    reject,
)

if TYPE_CHECKING:
    from .context import _Adapt


def adapt_storage(context: _Adapt, statement: Statement) -> None:
    token, args = statement.token, statement.arguments
    if token.text in {
        "encode_in_angle",
        "encode_on_bloch",
        "inplace_encode_on_bloch",
    }:
        context.encode_data(token, args)
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
                size = context.static_integer(args[0])
            elif not args[0].children:
                name = args[0].token.text
                size = context.classical_integers.get(name)
                if size is None:
                    size = exact_unsigned_literal(name)
            elif args[0].token.text in {"+", "-", "*", "%", "**", "$ceil_log2"}:
                size = context.static_integer(args[0])
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
        target = context.atom(args[-1])
        if context.helper_stack and (
            context.parameters.get(target.text) is None
            or not (
                context.parameters[target.text].role == "output"
                or context.parameters[target.text].role == "local"
                and context.parameters[target.text].kind == "qbit"
            )
        ):
            reject(target, "free", "bound allocation requires a local qbit")
        phase_allocation = infer_phase_allocation(context, token, args, size)
        phase_parameter = context.parameters.get(target.text)
        destination = context.register(
            target,
            size,
            allow_qubit_array=True,
            allocate_local=True,
            allow_fractional_output=(
                phase_parameter is not None
                and phase_parameter.kind in {"qphase", "qphase_signed"}
                and target.text not in context.fractional_bits
            ),
        )
        if size is not None and context.widths[target.text] != size:
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
                or context.parameters[target.text].kind != "qnum"
            ):
                reject(
                    token,
                    "type",
                    "typed allocate requires a qnum, literal signedness and zero fractional bits",
                )
            if sign.token.text in {"True", "SIGNED"}:
                if context.scope != "main" or context.local_scope is not None:
                    reject(
                        token,
                        "scope",
                        "signed allocation is supported only at main scope",
                    )
                context.signed.add(target.text)
            elif target.text in context.signed:
                reject(
                    sign.token,
                    "type",
                    "allocation signedness must agree with the declared type",
                )
        context.effect_op("qreg.allocate", token, (destination,))
    elif token.text in {
        "prepare_amplitudes",
        "inplace_prepare_amplitudes",
        "prepare_state",
    }:
        context.prepare_state_vector(token, args)
    elif token.text == "prepare_exponential_state":
        context.prepare_exponential_state(token, args)
    elif token.text == "free":
        if len(args) != 1:
            reject(token, "signature", "free requires one local qbit")
        lower_free(context, context.atom(args[0]))
    elif token.text == "drop":
        if len(args) != 1 or context.scope != "main" or context.local_scope is not None:
            reject(token, "drop", "drop requires one local register in main")
        target = context.atom(args[0])
        drop_parameter = context.parameters.get(target.text)
        if (
            drop_parameter is None
            or drop_parameter.role != "local"
            or drop_parameter.kind not in {"qbit", "qnum", "qbit[]", "qphase"}
        ):
            reject(target, "drop", "drop requires a local register")
        destination = context.register(
            target,
            allow_qubit_array=True,
            allow_fractional_output=drop_parameter.kind == "qphase",
        )
        context.effect_op("qreg.drop", token, (destination,))
        context.dropped.add(target.text)
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
        target = context.atom(output)
        if (
            target.text not in context.parameters
            or context.parameters[target.text].kind != "qbit[]"
        ):
            reject(target, "type", "prepare_bell_state requires a qubit array output")
        destination = context.register(target, 2, allow_qubit_array=True)
        if context.widths[target.text] != 2:
            reject(target, "width", "Bell preparation requires exactly two qubits")
        context.effect_op("qreg.prepare_ghz", token, (destination,))
        context.effect_op(
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
        target = context.atom(output)
        if (
            target.text not in context.parameters
            or context.parameters[target.text].kind != "qbit[]"
        ):
            reject(target, "type", "prepare_ghz_state requires a qubit array output")
        destination = context.register(target, size, allow_qubit_array=True)
        if context.widths[target.text] != size:
            reject(target, "width", "GHZ size must agree with the output width")
        context.effect_op("qreg.prepare_ghz", token, (destination,))
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
        destination = context.register(context.atom(args[-1]), allow_qubit_array=True)
        context.effect_op(
            "qreg.uniform_interval", token, (destination,), start=start, stop=stop
        )
