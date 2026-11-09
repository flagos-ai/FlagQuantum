"""Bounded fixed-point preparation proofs for coherent amplification."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from operator import add, and_, or_, sub
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Value
from .amplification import checked_probabilities
from .basis_regions import (
    _COMPARISONS,
    Effect,
    Frame,
    Region,
    Scalar,
    View,
    _Guards,
    _name,
)
from .basis_regions import (
    expression as basis_expression,
)
from .preparation import exponential_angles
from .syntax import Expression, Function, Parameter, Record, Statement, Token, reject

if TYPE_CHECKING:
    from .context import _Adapt


_Number = Callable[[int], float]


def _constant(value: float) -> _Number:
    return lambda row: value


class FixedRegion(Region):
    allocation_arities = frozenset({1, 2})

    def __init__(
        self,
        definitions: Mapping[str, Function],
        records: Mapping[str, Record],
        precision: int,
    ) -> None:
        super().__init__(definitions, records)
        self.precision = precision

    def fresh(self, width: int, token: Token) -> tuple[int, ...]:
        if not 1 <= width <= 24 or self.next_wire + width > 128:
            reject(
                token, "budget.proof", "fixed-point preparation exceeds its wire bound"
            )
        bits = tuple(range(self.next_wire, self.next_wire + width))
        self.next_wire += width
        return bits

    def bind(
        self,
        frame: Frame,
        label: str,
        bits: tuple[int, ...],
        token: Token,
        actual: View | None = None,
    ) -> None:
        port = frame.declarations.get(label)
        if port is None or label in frame.views or len(bits) != len(set(bits)):
            reject(
                token,
                "ownership",
                "fixed-point preparation requires fresh disjoint binding",
            )
        if port.record is not None:
            if port.width != len(bits):
                reject(token, "width", "fixed-point record view width differs")
            offset = 0
            for leaf in self.records[port.record].leaves(self.records, label + "."):
                if leaf.width is None or leaf.kind not in {
                    "qnum[]",
                    "qnum",
                    "qnum_signed",
                    "qbit",
                    "qbit[]",
                }:
                    reject(
                        token, "proof", "fixed-point proof needs fixed record fields"
                    )
                total = leaf.width * (leaf.length or 1)
                frame.views[leaf.name.text] = View(
                    bits[offset : offset + total],
                    leaf.kind == "qnum_signed",
                    leaf.fractional_bits,
                    leaf.width if leaf.length is not None else 1,
                )
                offset += total
            if offset != len(bits):
                reject(
                    token, "width", "fixed-point record fields do not cover all wires"
                )
        if port.kind == "qnum[]":
            element_width = actual.element_width if actual is not None else 1
            if port.length is None or len(bits) != element_width * port.length:
                reject(token, "width", "numeric array has incompatible width")
            frame.views[label] = View(bits, element_width=element_width)
            return
        if port.width is not None and port.width != len(bits):
            reject(token, "width", "fixed-point scalar view width differs")
        if port.kind == "qnum" and port.width is None and actual is not None:
            frame.views[label] = actual
            return
        if (
            port.fractional_bits
            and actual is not None
            and (actual.fraction != port.fractional_bits or not actual.signed)
        ):
            reject(token, "precision", "fixed-point actual and formal formats differ")
        frame.views[label] = View(
            bits, port.kind == "qnum_signed", port.fractional_bits
        )

    def view(self, node: Expression, frame: Frame) -> View:
        op, children = node.token.text, node.children
        if op in {"[", ":"} and len(children) in {2, 3}:
            source = self.view(children[0], frame)
            width = source.element_width
            size = len(source.bits) // width
            left = self.expression(children[1], frame).static(node.token)
            right = (
                self.expression(children[2], frame).static(node.token)
                if op == ":"
                else left + 1
            )
            if not 0 <= left < right <= size:
                reject(node.token, "width", "fixed-point view is out of bounds")
            bits = source.bits[left * width : right * width]
            return View(bits, element_width=width if op == ":" else 1)
        if op == "{" and children:
            bits = tuple(
                bit for item in children for bit in self.view(item, frame).bits
            )
            if len(bits) != len(set(bits)):
                reject(node.token, "alias", "fixed-point packed view overlaps")
            return View(bits)
        label = _name(node)
        if label not in frame.views:
            reject(
                node.token, "ownership", "fixed-point view requires initialized storage"
            )
        return frame.views[label]

    def number(self, node: Expression, frame: Frame) -> tuple[_Number, frozenset[int]]:
        op, children = node.token.text, node.children
        if not children:
            if op in frame.reals:
                real = frame.reals[op]
                if not isinstance(real, float):
                    reject(node.token, "proof", "real array needs an index")
                return _constant(real), frozenset()
            if op in frame.classical:
                return _constant(float(frame.classical[op])), frozenset()
            try:
                value = float(op)
            except ValueError:
                pass
            else:
                if not math.isfinite(value):
                    reject(node.token, "proof", "fixed-point real must be finite")
                return _constant(value), frozenset()
        if op == "-" and len(children) == 1:
            child, reads = self.number(children[0], frame)
            return lambda row: -child(row), reads
        register = self.view(node, frame)
        return lambda row: math.ldexp(
            register.read(row), -register.fraction
        ), frozenset(register.bits)

    def expression(self, node: Expression, frame: Frame) -> Scalar:
        op, children = node.token.text, node.children
        if (
            op == "."
            and len(children) == 2
            and children[1].token.text in {"len", "size"}
        ):
            register = self.view(children[0], frame)
            value = len(register.bits)
            if children[1].token.text == "len":
                value //= register.element_width
            return Scalar(lambda row: value)
        if op in _COMPARISONS and len(children) == 2:
            left, right = children
            first, a = (
                self.number(left, frame)
                if left.token.text not in {"&", "|"}
                else self.as_number(left, frame)
            )
            second, b = (
                self.number(right, frame)
                if right.token.text not in {"&", "|"}
                else self.as_number(right, frame)
            )
            compare = _COMPARISONS[op]
            return Scalar(lambda row: int(compare(first(row), second(row))), a | b)
        if op in {"&", "|", "+", "-"} and len(children) == 2:
            first_int, second_int = (
                self.expression(child, frame) for child in children
            )
            operator = {"&": and_, "|": or_, "+": add, "-": sub}[op]
            return Scalar(
                lambda row: operator(first_int.evaluate(row), second_int.evaluate(row)),
                first_int.reads | second_int.reads,
            )
        if op in {"[", ".", "{", ":"} or (not children and op in frame.views):
            register = self.view(node, frame)
            if register.fraction:
                reject(
                    node.token,
                    "precision",
                    "fixed-point values need an explicit numeric operation",
                )
            return Scalar(register.read, frozenset(register.bits))
        return basis_expression(node, frame)

    def as_number(
        self, node: Expression, frame: Frame
    ) -> tuple[_Number, frozenset[int]]:
        raw = self.expression(node, frame)
        return lambda row: float(raw.evaluate(row)), raw.reads

    def affine(self, node: Expression, frame: Frame) -> tuple[float, dict[View, float]]:
        op, children = node.token.text, node.children
        if op == "sqrt" and len(children) == 1:
            value, reads = self.number(children[0], frame)
            if reads or value(0) < 0:
                reject(
                    node.token,
                    "proof",
                    "square root must be a static nonnegative value",
                )
            return math.sqrt(value(0)), {}
        if op == "-" and len(children) == 1:
            offset, terms = self.affine(children[0], frame)
            return -offset, {register: -factor for register, factor in terms.items()}
        if op in {"+", "-", "*"} and len(children) == 2:
            (a, left), (b, right) = (self.affine(child, frame) for child in children)
            if op == "*":
                if left and right:
                    reject(
                        node.token,
                        "proof",
                        "fixed-point proof admits affine updates only",
                    )
                return a * b, {
                    reg: coef * (b if left else a)
                    for reg, coef in (left or right).items()
                }
            sign = 1 if op == "+" else -1
            terms = dict(left)
            for register, coefficient in right.items():
                terms[register] = terms.get(register, 0) + sign * coefficient
            return a + sign * b, terms
        if op in {"[", "."} or (not children and op in frame.views):
            register = self.view(node, frame)
            if register.fraction or register.element_width != 1:
                reject(
                    node.token,
                    "proof",
                    "fixed-point affine inputs must be integral words",
                )
            return 0.0, {register: 1.0}
        value, reads = self.number(node, frame)
        if reads:
            reject(node.token, "proof", "fixed-point affine constant depends on a wire")
        return value(0), {}

    def update_value(self, source: Expression, frame: Frame, target: View) -> Scalar:
        if not target.fraction:
            return self.expression(source, frame)
        if target.fraction != self.precision:
            reject(
                source.token,
                "precision",
                "fixed-point target must use the declared machine precision",
            )
        offset, terms = self.affine(source, frame)
        scaled = [
            (reg, math.ldexp(factor, self.precision)) for reg, factor in terms.items()
        ]
        intercept = math.ldexp(offset, self.precision)
        if any(
            not math.isfinite(value) or abs(value) >= 2**31
            for value in [intercept, *(factor for _, factor in scaled)]
        ):
            reject(
                source.token,
                "budget.number",
                "fixed-point affine coefficient exceeds its bound",
            )
        coefficients = [(reg, math.trunc(value)) for reg, value in scaled]
        constant = math.trunc(intercept)
        return Scalar(
            lambda row: constant
            + sum(reg.read(row) * coef for reg, coef in coefficients),
            frozenset(bit for reg in terms for bit in reg.bits),
        )

    def bind_real(
        self, frame: Frame, label: str, node: Expression, caller: Frame
    ) -> None:
        scalar, reads = self.number(node, caller)
        if reads:
            reject(node.token, "proof", "fixed-point classical real must be static")
        frame.reals[label] = scalar(0)

    def allocation_width(
        self, args: tuple[Expression, ...], label: str, frame: Frame, token: Token
    ) -> int:
        if len(args) == 1:
            port = frame.declarations.get(label)
            if port is None or port.width is None:
                reject(token, "width", "fixed-point allocation needs a declared width")
            return port.width
        return super().allocation_width(args, label, frame, token)

    def transfer_width(self, port: Parameter | None, frame: Frame) -> int | None:
        if port is not None and port.width_expression is not None:
            return self.expression(port.width_expression, frame).static(port.name)
        return super().transfer_width(port, frame)

    def transfer(self, source: Expression, target: Expression, frame: Frame) -> None:
        # Resolve a derived destination width while its source is still in scope.
        label = _name(target) if target.token.text != "{" else ""
        port = frame.declarations.get(label)
        if port is not None and (
            port.width_expression is not None or port.width is None
        ):
            frame.declarations[label] = replace(
                port,
                width=(
                    self.transfer_width(port, frame)
                    or len(self.view(source, frame).bits)
                ),
                width_expression=None,
            )
        super().transfer(source, target, frame)

    def compute_within(
        self,
        statement: Statement,
        frame: Frame,
        guards: _Guards,
        stack: tuple[str, ...],
    ) -> list[Effect]:
        assert statement.within is not None
        if not statement.within or statement.within[0].token.text != "->":
            return super().compute_within(statement, frame, guards, stack)
        source = statement.within[0].arguments[0]
        bits = frozenset(self.view(source, frame).bits)
        original = frame.forbidden
        frame.forbidden -= bits
        try:
            effects = self.block(statement.within, frame, (), stack)
        finally:
            frame.forbidden = original
        if any(
            effect.gate is None
            or effect.gate.name != "X"
            or not set(effect.gate.bits) <= bits
            for effect in effects
        ):
            reject(
                statement.token,
                "proof",
                "const reinterpretation admits only paired bit flips",
            )
        return effects

    def rotate(
        self, token: Token, wire: int, angle: float, frame: Frame, guards: _Guards
    ) -> Effect:
        mask = 1 << wire
        cosine, sine = math.cos(angle / 2), math.sin(angle / 2)

        def terms(row: int, reverse: bool) -> tuple[tuple[int, complex], ...]:
            sign = (-1 if row & mask else 1) * (-1 if reverse else 1)
            return ((row, complex(cosine)), (row ^ mask, complex(sign * sine)))

        def refuse(row: int) -> int:
            reject(token, "proof", "rotation requires coherent preparation proof")

        return self.mutation(
            token, frame, guards, frozenset({wire}), Effect(refuse, refuse, terms)
        )

    def extra(
        self, statement: Statement, frame: Frame, guards: _Guards
    ) -> list[Effect]:
        token, args = statement.token, statement.arguments
        if token.text in {"RY", "prepare_exponential_state"} and len(args) == 2:
            scalar, reads = self.number(args[0], frame)
            register = self.view(args[1], frame)
            if reads or len(register.bits) > 8:
                reject(
                    token,
                    "proof",
                    "coherent preparation needs a bounded static rotation",
                )
            value = scalar(0)
            if token.text == "RY" and len(register.bits) == 1:
                return [self.rotate(token, register.bits[0], value, frame, guards)]
            if token.text == "prepare_exponential_state":
                angles = exponential_angles(value, len(register.bits))
                return [
                    self.rotate(token, bit, angle, frame, guards)
                    for bit, angle in zip(register.bits, angles, strict=True)
                ]
        if token.text == "inplace_prepare_state" and len(args) == 3:
            register = self.view(args[2], frame)
            probs = checked_probabilities(args[0], args[1], {})
            if len(register.bits) > 4 or len(probs) != 1 << len(register.bits):
                reject(
                    token,
                    "proof",
                    "local probability preparation exceeds the proof bound",
                )
            amplitudes = tuple(math.sqrt(prob) for prob in probs)

            def terms(row: int, reverse: bool) -> tuple[tuple[int, complex], ...]:
                if reverse or register.read(row):
                    reject(
                        token,
                        "proof",
                        "probability preparation requires fresh zero storage",
                    )
                return tuple(
                    (register.write(row, word), complex(coefficient))
                    for word, coefficient in enumerate(amplitudes)
                    if coefficient
                )

            def refuse(row: int) -> int:
                reject(
                    token,
                    "proof",
                    "probability preparation requires coherent verification",
                )

            return [
                self.mutation(
                    token,
                    frame,
                    guards,
                    frozenset(register.bits),
                    Effect(refuse, refuse, terms),
                )
            ]
        return super().extra(statement, frame, guards)


@dataclass(frozen=True)
class FixedPreparation:
    amplitudes: tuple[float, ...]
    width: int
    rounds: int


def _entry_layout(
    main: Function, token: Token
) -> tuple[Statement, Statement, Expression, Parameter]:
    """Check one allocated packed workspace, one scalar output and terminal drop."""
    operations = tuple(stmt for stmt in main.statements if stmt.declaration is None)
    if len(operations) != 4:
        reject(
            token,
            "proof",
            "bounded amplification needs one prepared scope and a terminal drop",
        )
    workspace, output, scope, drop = operations
    if (
        workspace.token.text != "allocate"
        or len(workspace.arguments) != 1
        or output.token.text != "allocate"
        or len(output.arguments) != 1
        or scope.within is None
        or len(scope.within) != 1
        or len(scope.apply) != 1
        or scope.within[0].token.text != "->"
        or scope.apply[0].token.text != "amplitude_amplification"
        or drop.token.text != "drop"
        or len(drop.arguments) != 1
        or _name(drop.arguments[0]) != _name(workspace.arguments[0])
    ):
        reject(
            token,
            "proof",
            "bounded amplification needs allocated storage and a terminal drop",
        )
    [public] = [port for port in main.parameters if port.role == "output"]
    if public.kind != "qbit" or public.name.text != _name(output.arguments[0]):
        reject(
            public.name, "proof", "bounded amplification requires one indicator output"
        )
    return scope, scope.apply[0], drop.arguments[0], public


def _oracle_bit(
    region: FixedRegion, function: Function, packed: Expression, frame: Frame
) -> int:
    if (
        len(function.parameters) != 1
        or function.parameters[0].role != "const"
        or len(function.statements) != 1
    ):
        reject(
            function.name, "proof", "bounded amplification requires a scalar Z oracle"
        )
    [step] = function.statements
    if (
        step.token.text != "Z"
        or step != Statement(step.token, step.arguments)
        or len(step.arguments) != 1
    ):
        reject(step.token, "proof", "bounded amplification requires a scalar Z oracle")
    argument = region.view(packed, frame)
    [port] = function.parameters
    oracle_frame = Frame({port.name.text: port}, {}, {}, frozenset(argument.bits))
    region.bind(oracle_frame, port.name.text, argument.bits, packed.token)
    target = region.view(step.arguments[0], oracle_frame)
    if len(target.bits) != 1:
        reject(step.token, "proof", "bounded amplification oracle must target one bit")
    return target.bits[0]


def build_fixed_preparation(
    definitions: Mapping[str, Function],
    records: Mapping[str, Record],
    precision: int,
    classical: Mapping[str, int],
) -> FixedPreparation:
    main = definitions["main"]
    outputs = tuple(port for port in main.parameters if port.role == "output")
    if len(outputs) != 1 or main.permutation:
        reject(
            main.name, "proof", "bounded amplification requires one indicator output"
        )
    token = main.name
    scope, call, workspace, public = _entry_layout(main, token)
    if call != Statement(call.token, call.arguments) or len(call.arguments) != 4:
        reject(
            call.token,
            "signature",
            "bounded amplification requires rounds, oracle, preparation and target",
        )
    count, oracle_ref, prep_ref, packed = call.arguments
    region = FixedRegion(definitions, records, precision)
    declarations = {port.name.text: port for port in main.parameters}
    for step in main.statements:
        if step.declaration is not None:
            declarations[step.declaration.name.text] = step.declaration
    frame = Frame(declarations, {}, dict(classical), frozenset())
    rounds = region.expression(count, frame).static(count.token)
    if not 0 <= rounds <= 4:
        reject(
            count.token,
            "budget.proof",
            "bounded coherent amplification admits at most four rounds",
        )
    # The first two allocation calls have been checked by _entry_layout.
    entry = tuple(step for step in main.statements if step.declaration is None)
    region.block(entry[:2], frame, (), ("main",))
    indicator = frame.views[public.name.text].bits[0]
    environment = region.view(workspace, frame).bits
    assert scope.within is not None
    region.block(scope.within, frame, (), ("main",))
    all_data = frozenset((*environment, indicator))
    packed_view = region.view(packed, frame)
    if set(packed_view.bits) != all_data or len(packed_view.bits) != len(all_data):
        reject(
            packed.token, "alias", "amplification must cover all allocated input wires"
        )
    oracle = definitions.get(_name(oracle_ref))
    preparation = definitions.get(_name(prep_ref))
    if oracle is None or preparation is None:
        reject(call.token, "proof", "bounded amplification requires declared callbacks")
    if _oracle_bit(region, oracle, packed, frame) != indicator:
        reject(
            oracle.name, "proof", "bounded amplification oracle must mark the indicator"
        )
    tape = region.invoke(preparation, (packed,), frame, (), ("main",))
    state: dict[int, complex] = {0: 1 + 0j}
    for instruction in tape:
        updated: dict[int, complex] = {}
        for row, amplitude in state.items():
            for output, coefficient in instruction.terms(row, False):
                updated[output] = updated.get(output, 0j) + amplitude * coefficient
        state = {row: amplitude for row, amplitude in updated.items() if amplitude}
        if len(state) > 4096:
            reject(
                call.token,
                "budget.proof",
                "bounded coherent preparation exceeds 4096 support words",
            )
    data_mask = sum(1 << bit for bit in all_data)
    if any(row & ~data_mask or amplitude.imag for row, amplitude in state.items()):
        reject(
            call.token,
            "proof",
            "bounded coherent preparation must release all private work",
        )
    if not math.isclose(
        math.hypot(*(amplitude.real for amplitude in state.values())),
        1,
        abs_tol=1e-12,
        rel_tol=0,
    ):
        reject(call.token, "proof", "bounded coherent preparation must be normalized")
    environment_mask = data_mask ^ (1 << indicator)
    labels = {row & environment_mask for row in state}
    keep = list(environment)
    mask = environment_mask
    for bit in reversed(environment):
        smaller = mask ^ (1 << bit)
        if len({label & smaller for label in labels}) == len(labels):
            mask = smaller
            keep.remove(bit)
    width = len(keep) + 1
    if not 1 <= width <= 12:
        reject(
            call.token,
            "budget.proof",
            "bounded coherent preparation needs at most twelve retained qubits",
        )
    amplitudes = [0.0] * (1 << width)
    for row, amplitude in state.items():
        environment_word = sum(
            (row >> bit & 1) << offset for offset, bit in enumerate(keep)
        )
        amplitudes[environment_word | ((row >> indicator & 1) << (width - 1))] = (
            amplitude.real
        )
    return FixedPreparation(tuple(amplitudes), width, rounds)


def emit_fixed_preparation(owner: _Adapt, plan: FixedPreparation) -> None:
    token = owner.syntax.name
    [public] = [port for port in owner.syntax.parameters if port.role == "output"]
    output = owner.register(public.name)
    owner.effect_op("qreg.allocate", token, (output,))
    storage = Token("$prepared_environment", token.location)
    width = plan.width - 1
    locations: tuple[tuple[Value, int], ...] = ((output, 0),)
    private: Value | None = None
    if width:
        owner.parameters[storage.text] = Parameter(storage, "qbit[]", width, "local")
        owner.widths[storage.text] = width
        private = owner.register(storage, allow_qubit_array=True, allocate_local=True)
        owner.effect_op("qreg.allocate", token, (private,))
        locations += tuple((private, i) for i in reversed(range(width)))
    operands, indices = tuple(v for v, _ in locations), tuple(i for _, i in locations)
    half = len(plan.amplitudes) // 2
    pivot = 2 * math.atan2(
        math.hypot(*plan.amplitudes[half:]), math.hypot(*plan.amplitudes[:half])
    )
    # The wide real ladder begins with the indicator rotation R. Every
    # remaining gate B leaves the indicator unchanged, so A = B R and
    # A† Z A = R† Z R. Move all rounds to before A. The environment is
    # still zero at that point, and the all-zero reflection is Z on the
    # indicator. This keeps the signed good/bad amplitudes and avoids
    # materializing inverse copies of the whole preparation.
    if plan.rounds:
        owner.effect_op(
            "qreg.gate",
            token,
            (output,),
            gate="ry",
            index=0,
            angle=2 * plan.rounds * pivot,
        )
    owner.effect_op(
        "qreg.inplace_prepare_amplitudes",
        token,
        operands,
        indices=indices,
        amplitudes=plan.amplitudes,
        bound=0.0,
    )
    if private is not None:
        owner.effect_op("qreg.drop", token, (private,))
