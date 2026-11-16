"""Capture static qfunc composition and specialize complete numeric formats."""

from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_numeric import has_numeric, has_numeric_reads
from .frontend_ports import classical_parameters, program_ports
from .frontend_python import _error, _QuantumFunction
from .frontend_views import inline_mutable

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_helper_call(
    owner: _Body,
    node: ast.Call,
    function: _QuantumFunction,
    *,
    inverse: bool,
    control_bit: Value | None,
    condition_inputs: tuple[Value, ...],
) -> None:
    """Validate borrowing and numeric formats before emitting the existing call."""
    formal = program_ports(function)
    numeric_call = any(has_numeric(port) for port in formal) or any(
        has_numeric_reads(owner, argument) for argument in node.args
    )
    if numeric_call and (inverse or control_bit is not None):
        _error(
            owner.source,
            node,
            "program.numeric",
            "numeric helpers require an unconditional forward call",
        )
    if any(port.role == "inout" for port in formal):
        if inverse or control_bit is not None:
            _error(
                owner.source,
                node,
                "program.borrow",
                "mutable helpers are not admitted under adjoint/control",
            )
        inline_mutable(owner, node, function, formal)
        return
    if classical_parameters(function):
        _error(
            owner.source,
            node,
            "program.binding",
            "parameterized helpers require plain writable quantum ports",
        )
    if owner.aliases is not None:
        _error(
            owner.source,
            node,
            "program.borrow",
            "mutable helper calls cannot initialize or publish Outputs",
        )
    if control_bit is not None and any(port.role != "const" for port in formal):
        _error(
            owner.source,
            node,
            "program.control",
            "controlled helpers require Const-only ports",
        )
    if len(node.args) != len(formal):
        _error(
            owner.source,
            node,
            "program.signature",
            "qfunc argument count disagrees",
        )
    widths: list[int | None] = []
    formats: list[tuple[bool, int]] = []
    for argument, port in zip(node.args, formal, strict=True):
        if not isinstance(argument, ast.Name) or argument.id not in owner.ports:
            _error(
                owner.source,
                argument,
                "program.port",
                "qfunc actuals must be declared quantum ports",
            )
        if owner.ports[argument.id].kind != port.kind:
            _error(
                owner.source,
                argument,
                "program.kind",
                "qfunc actual and formal quantum kinds must agree",
            )
        actual = owner.ports[argument.id]
        formats.append((actual.signed, actual.fractional_bits))
        widths.append(owner.widths[argument.id])
        if port.role == "const":
            owner.register(argument)
    program, resolved = owner.owner.build(function, tuple(widths), tuple(formats))
    reads: list[Value] = []
    writes: list[Value] = []
    for argument, port in zip(node.args, resolved, strict=True):
        assert isinstance(argument, ast.Name)
        actual = owner.ports[argument.id]
        if port.role == "output":
            if (
                actual.width is not None
                and argument.id not in owner.inferred_numeric
                and (
                    actual.signed,
                    actual.fractional_bits,
                )
                != (port.signed, port.fractional_bits)
            ):
                _error(
                    owner.source,
                    argument,
                    "program.numeric.format",
                    "declared and inferred helper output formats must agree",
                )
            owner.ports[argument.id] = replace(
                actual, signed=port.signed, fractional_bits=port.fractional_bits
            )
        (reads if port.role == "const" else writes).append(
            owner.register(argument, port.width)
        )
    call_attributes: dict[str, object] = {"callee": program.name}
    if inverse:
        call_attributes["inverse"] = True
    if control_bit is None:
        owner.emit("qreg.call", node, (*reads, *writes), **call_attributes)
    else:
        if len(set(reads)) != len(reads) or set(reads).intersection(condition_inputs):
            _error(
                owner.source,
                node,
                "program.control",
                "controlled helper arguments must be disjoint",
            )
        owner.emit("qreg.control_call", node, (control_bit, *reads), **call_attributes)
