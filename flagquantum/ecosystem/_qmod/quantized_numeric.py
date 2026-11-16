"""Bounded lexical numeric programs with a proved discard environment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING

from ...compiler._construction.fixed_point import (
    LITERAL_BITS,
    FixedPointBranches,
    FixedPointExpression,
    FixedPointScalar,
)
from ...compiler._construction.fixed_point_program import (
    FixedPointScope,
    emit_fixed_point_tables,
)
from .angles import REAL_LITERAL
from .syntax import Expression, Function, Parameter, Statement, Token, reject

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Value
    from .context import _Adapt


@dataclass(frozen=True)
class NumericProgram:
    bits: int
    outputs: tuple[FixedPointScalar, ...]


def _literal(node: Expression) -> Fraction | None:
    if not node.children and REAL_LITERAL.fullmatch(node.token.text):
        number = float(node.token.text)
        if not math.isfinite(number) or abs(number) > 2**20:
            reject(
                node.token,
                "precision",
                "numeric literal exceeds the bounded fixed-point range",
            )
        return Fraction(number)
    if node.token.text == "-" and len(node.children) == 1:
        value = _literal(node.children[0])
        return None if value is None else -value
    return None


_ARITHMETIC: FixedPointExpression[Expression] = FixedPointExpression(
    text=lambda node: node.token.text,
    children=lambda node: node.children,
    literal=_literal,
    reject=lambda node, code, message: reject(node.token, code, message),
)


class _Planner:
    def __init__(self, definitions: Mapping[str, Function]) -> None:
        self.definitions = definitions
        self.branches = FixedPointBranches()
        self.steps = 0
        self.used: set[str] = set()

    def function(
        self,
        syntax: Function,
        inputs: Mapping[str, FixedPointScalar],
        active: tuple[str, ...],
    ) -> tuple[FixedPointScalar, ...]:
        name = syntax.name.text
        if (
            name in active
            or len(active) >= 12
            or syntax.disable_const_checks
            or syntax.disable_permutation_check
        ):
            reject(
                syntax.name,
                "ownership",
                "quantized helpers must be bounded and acyclic",
            )
        self.used.add(name)
        roles = {port.name.text: port.role for port in syntax.parameters}
        if len(roles) != len(syntax.parameters) or any(
            port.kind != "qnum"
            or port.width is not None
            or port.record is not None
            or port.role not in {"output", "const", "inout"}
            for port in syntax.parameters
        ):
            reject(
                syntax.name, "type", "quantized helpers require inferred numeric ports"
            )
        scope = FixedPointScope(self.branches, roles, inputs)
        initialized = scope.values

        def reserve(node: Expression) -> str:
            if node.children:
                reject(
                    node.token, "ownership", "quantized assignment needs a whole word"
                )
            return scope.reserve(
                node.token.text,
                lambda code, message: reject(node.token, code, message),
            )

        def read(node: Expression) -> FixedPointScalar:
            if node.children:
                reject(node.token, "ownership", "quantized read needs a whole word")
            return scope.read(
                node.token.text,
                lambda code, message: reject(node.token, code, message),
            )

        for statement in syntax.statements:
            self.steps += 1
            token, args = statement.token, statement.arguments
            if self.steps > 256:
                reject(
                    token,
                    "budget.precision",
                    "quantized helper expansion exceeds 256 statements",
                )
            if statement.declaration is not None:
                parameter = statement.declaration
                if (
                    parameter.kind != "qnum"
                    or parameter.width is not None
                    or parameter.name.text in scope.roles
                ):
                    reject(
                        parameter.name,
                        "type",
                        "quantized local must be a fresh inferred qnum",
                    )
                scope.declare(
                    parameter.name.text,
                    partial(reject, parameter.name),
                )
                continue
            if statement != Statement(token, args):
                reject(
                    token,
                    "scope",
                    "quantized numeric evaluation requires straight-line ownership",
                )
            if syntax.permutation and token.text != "=":
                reject(
                    token,
                    "permutation",
                    "quantized qperm may only compute fresh numeric results",
                )
            if token.text == "=" and len(args) == 2:
                target = reserve(args[0])
                initialized[target] = _ARITHMETIC.evaluate(args[1], initialized)
            elif token.text == "allocate" and len(args) == 2 and _literal(args[0]) == 1:
                target = reserve(args[1])
                scope.allocate(target, partial(reject, token))
            elif token.text == "hadamard_transform" and len(args) == 1:
                if args[0].children:
                    reject(token, "ownership", "quantized Hadamard needs a whole word")
                scope.hadamard(
                    args[0].token.text,
                    partial(reject, token),
                )
            elif token.text == "prepare_state" and len(args) == 3:
                vector, bound, destination = args
                if (
                    _literal(bound) != 0
                    or vector.token.text != "["
                    or tuple(map(_literal, vector.children))
                    != (Fraction(1, 2), Fraction(1, 2))
                ):
                    reject(
                        token,
                        "amplitudes",
                        "quantized preparation needs an exact uniform bit",
                    )
                scope.prepare_uniform(
                    reserve(destination),
                    partial(reject, token),
                )
            elif token.text == "drop" and len(args) == 1:
                read(args[0])
                scope.drop(
                    args[0].token.text,
                    partial(reject, token),
                )
            elif token.text in self.definitions:
                helper = self.definitions[token.text]
                if len(args) != len(helper.parameters) or len(
                    {arg.token.text for arg in args}
                ) != len(args):
                    reject(
                        token,
                        "alias",
                        "quantized helper actuals must be disjoint whole words",
                    )
                child_inputs = {}
                child_outputs = []
                for port, actual in zip(helper.parameters, args, strict=True):
                    if port.role == "output":
                        child_outputs.append(reserve(actual))
                    else:
                        child_inputs[port.name.text] = read(actual)
                result = self.function(helper, child_inputs, (*active, name))
                initialized.update(zip(child_outputs, result, strict=True))
            else:
                reject(
                    token, "scope", "unsupported effect in quantized numeric program"
                )
        results = []
        for port in syntax.parameters:
            if port.role != "output":
                continue
            value = initialized.get(port.name.text)
            if value is None:
                reject(
                    port.name,
                    "ownership",
                    "quantized helper must initialize its Outputs",
                )
            results.append(value)
        return tuple(results)


def try_numeric_program(definitions: Mapping[str, Function]) -> NumericProgram | None:
    main = definitions["main"]
    if (
        len(main.parameters) < 2
        or any(
            port.role != "output" or port.kind != "qnum" or port.width is not None
            for port in main.parameters
        )
        or not any(statement.token.text == "drop" for statement in main.statements)
    ):
        return None

    def non_dyadic(node: Expression) -> bool:
        literal = _literal(node)
        if literal is not None:
            return literal.denominator > 1 << LITERAL_BITS
        return any(non_dyadic(child) for child in node.children)

    if not any(
        function.permutation
        and any(
            port.role == "const" and port.kind == "qnum" and port.width is None
            for port in function.parameters
        )
        and any(
            statement.token.text == "=" and non_dyadic(statement.arguments[-1])
            for statement in function.statements
        )
        for function in definitions.values()
    ):
        return None
    planner = _Planner(definitions)
    outputs = planner.function(main, {}, ())
    if planner.used != definitions.keys() or not planner.branches.bits:
        reject(
            main.name,
            "ownership",
            "quantized profile requires one closed reachable program",
        )
    planner.branches.verify_discarded(
        lambda code, message: reject(main.name, code, message)
    )
    return NumericProgram(planner.branches.bits, outputs)


def emit_numeric_program(owner: _Adapt, program: NumericProgram) -> None:
    token = owner.syntax.name
    branch = Token("$quantized_branch", token.location)
    owner.parameters[branch.text] = Parameter(branch, "qbit[]", program.bits, "local")
    owner.widths[branch.text] = program.bits
    selector = owner.register(branch, allow_qubit_array=True, allocate_local=True)
    owner.effect_op("qreg.allocate", token, (selector,))
    reads = tuple((selector, bit) for bit in range(program.bits))
    for _, bit in reads:
        owner.effect_op("qreg.gate", token, (selector,), index=bit, gate="h")
    outputs = []
    for port, output in zip(owner.syntax.parameters, program.outputs, strict=True):
        target = owner.register(port.name, output.width, allow_fractional_output=True)
        if output.fraction:
            owner.fractional_bits[port.name.text] = output.fraction
        if output.signed:
            owner.signed.add(port.name.text)
        owner.effect_op("qreg.allocate", token, (target,))
        outputs.append((output, tuple((target, bit) for bit in range(output.width))))

    def flip(target: tuple[Value, int]) -> None:
        owner.effect_op("qreg.gate", token, (target[0],), gate="x", index=target[1])

    def controlled(
        controls: tuple[tuple[Value, int], ...], pattern: int, target: tuple[Value, int]
    ) -> None:
        combined = (*controls, target)
        owner.effect_op(
            "qreg.control_x",
            token,
            tuple(value for value, _ in combined),
            indices=tuple(bit for _, bit in combined),
            control_state=pattern,
        )

    emit_fixed_point_tables(reads, outputs, flip=flip, controlled=controlled)
    owner.effect_op("qreg.drop", token, (selector,))
