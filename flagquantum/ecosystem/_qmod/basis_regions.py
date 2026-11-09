"""Prove a closed coherent preparation using bounded reversible regions."""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from operator import eq, ge, gt, le, lt, ne
from typing import TYPE_CHECKING

from ...compiler._hybrid.verifier import HybridVerificationError
from .integers import checked_integer, integer_operation
from .masked_selection import emit_truth_xor
from .syntax import Expression, Function, Parameter, Record, Statement, Token, reject
from .unary_selection_proof import _children, _reachable

if TYPE_CHECKING:
    from .context import _Adapt


@dataclass(frozen=True)
class Scalar:
    evaluate: Callable[[int], int]
    reads: frozenset[int] = frozenset()

    def static(self, token: Token) -> int:
        if self.reads:
            reject(token, "proof", "expected a static integer")
        return self.evaluate(0)


@dataclass(frozen=True)
class View:
    bits: tuple[int, ...]
    signed: bool = False
    fraction: int = 0
    element_width: int = 1

    def read(self, row: int) -> int:
        word = sum((row >> bit & 1) << offset for offset, bit in enumerate(self.bits))
        if self.signed and word >> (len(self.bits) - 1):
            return word - (1 << len(self.bits))
        return word

    def write(self, row: int, word: int) -> int:
        for offset, bit in enumerate(self.bits):
            row ^= ((word >> offset ^ row >> bit) & 1) << bit
        return row


@dataclass
class Frame:
    declarations: dict[str, Parameter]
    views: dict[str, View]
    classical: dict[str, int]
    forbidden: frozenset[int]
    reals: dict[str, float | tuple[float, ...]] = field(default_factory=dict)


def _name(node: Expression) -> str:
    if not node.children:
        return node.token.text
    if node.token.text == "." and len(node.children) == 2:
        return ".".join(_name(child) for child in node.children)
    reject(node.token, "proof", "expected a whole static register name")


_COMPARISONS = {"==": eq, "!=": ne, ">=": ge, ">": gt, "<=": le, "<": lt}
_ARITHMETIC = frozenset({"+", "-", "*", "%", "**"})


def view(node: Expression, frame: Frame) -> View:
    if node.token.text == "[" and len(node.children) == 2:
        value, index = node.children
        source = view(value, frame)
        offset = expression(index, frame).static(index.token)
        if not 0 <= offset < len(source.bits):
            reject(index.token, "width", "static proof index is out of bounds")
        return View(source.bits[offset : offset + 1])
    label = _name(node)
    if label not in frame.views:
        reject(node.token, "ownership", "static proof needs initialized storage")
    return frame.views[label]


def expression(node: Expression, frame: Frame) -> Scalar:
    token, children = node.token, node.children
    op = token.text
    if not children and op.isdecimal():
        value = checked_integer(int(op), token)
        return Scalar(lambda row: value)
    if not children and op in frame.classical:
        value = frame.classical[op]
        return Scalar(lambda row: value)
    if op in {"[", "."} or not children:
        if (
            not children
            and op in frame.declarations
            and frame.declarations[op].kind == "qphase"
        ):
            reject(token, "proof", "fractional ports may be read only as bit views")
        register = view(node, frame)
        return Scalar(register.read, frozenset(register.bits))
    if op == "-" and len(children) == 1:
        operand = expression(children[0], frame)
        return Scalar(lambda row: -operand.evaluate(row), operand.reads)
    if len(children) != 2:
        reject(token, "proof", "static proof does not support this expression")
    left, right = (expression(child, frame) for child in children)
    reads = left.reads | right.reads
    if op == "mod_inverse":
        a, modulus = left.static(token), right.static(token)
        if not 1 < modulus <= 256:
            reject(token, "proof", "modular inverse requires a bounded modulus")
        try:
            value = pow(a, -1, modulus)
        except ValueError:
            reject(token, "proof", "modular inverse has no solution")
        return Scalar(lambda row: value)
    if op in _ARITHMETIC:
        return Scalar(
            lambda row: integer_operation(
                token, left.evaluate(row), right.evaluate(row)
            ),
            reads,
        )
    if op in _COMPARISONS:
        comparison = _COMPARISONS[op]
        return Scalar(
            lambda row: int(comparison(left.evaluate(row), right.evaluate(row))), reads
        )
    if op == "and":
        return Scalar(
            lambda row: int(bool(left.evaluate(row) and right.evaluate(row))), reads
        )
    reject(token, "proof", "static proof does not support this operator")


@dataclass(frozen=True)
class Gate:
    name: str
    bits: tuple[int, ...]
    angle: float = 0.0

    def inverse(self) -> Gate:
        return replace(self, angle=-self.angle)


@dataclass(frozen=True)
class Effect:
    forward: Callable[[int], int]
    reverse: Callable[[int], int]
    amplitude: Callable[[int, bool], tuple[tuple[int, complex], ...]] | None = None
    gate: Gate | None = None
    target: View | None = None
    value: Scalar | None = None
    update: str | None = None
    guards: tuple[tuple[Scalar, bool], ...] = ()

    def terms(self, row: int, reverse: bool) -> tuple[tuple[int, complex], ...]:
        if self.amplitude is not None:
            return self.amplitude(row, reverse)
        return (((self.reverse if reverse else self.forward)(row), 1 + 0j),)

    def inverse(self) -> Effect:
        gate = self.gate.inverse() if self.gate else None
        handler = self.amplitude
        return Effect(
            self.reverse,
            self.forward,
            (
                None
                if handler is None
                else lambda row, reverse: handler(row, not reverse)
            ),
            gate,
            self.target,
            self.value,
            (
                {"+=": "-=", "-=": "+="}.get(self.update, self.update)
                if self.update
                else None
            ),
            self.guards,
        )

    def controlled(self, guards: tuple[tuple[Scalar, bool], ...]) -> Effect:
        def enabled(row: int) -> bool:
            return all(
                bool(condition.evaluate(row)) == expected
                for condition, expected in guards
            )

        return Effect(
            lambda row: self.forward(row) if enabled(row) else row,
            lambda row: self.reverse(row) if enabled(row) else row,
            (
                None
                if self.amplitude is None
                else lambda row, reverse: (
                    self.terms(row, reverse) if enabled(row) else ((row, 1 + 0j),)
                )
            ),
            self.gate,
            self.target,
            self.value,
            self.update,
            (*self.guards, *guards),
        )


_Guards = tuple[tuple[Scalar, bool], ...]
_Kernel = Callable[[int, bool, bool], int]


class Region:
    def view(self, node: Expression, frame: Frame) -> View:
        return view(node, frame)

    def expression(self, node: Expression, frame: Frame) -> Scalar:
        return expression(node, frame)

    def __init__(
        self, functions: Mapping[str, Function], records: Mapping[str, Record]
    ) -> None:
        self.functions = functions
        self.records = records
        self.next_wire = 0
        self.visits = 0
        self.quantum = False
        self.kernels: dict[tuple[str, tuple[tuple[str, int], ...]], _Kernel] = {}

    def fresh(self, width: int, token: Token) -> tuple[int, ...]:
        if not 1 <= width <= 16 or self.next_wire + width > 2048:
            reject(token, "budget.proof", "static proof exceeds the wire bound")
        result = tuple(range(self.next_wire, self.next_wire + width))
        self.next_wire += width
        return result

    def bind(
        self,
        frame: Frame,
        label: str,
        bits: tuple[int, ...],
        token: Token,
        actual: View | None = None,
    ) -> None:
        port = frame.declarations.get(label)
        if (
            port is None
            or label in frame.views
            or port.fractional_bits
            or (port.width is not None and port.width != len(bits))
        ):
            reject(
                token,
                "ownership",
                "static proof binding must be fresh with matching width",
            )
        if port.record is not None:
            offset = 0
            for leaf in self.records[port.record].leaves(self.records, label + "."):
                if leaf.width is None or leaf.length is not None:
                    reject(
                        token, "proof", "static proof needs fixed scalar record leaves"
                    )
                frame.views[leaf.name.text] = View(
                    bits[offset : offset + leaf.width], leaf.kind == "qnum_signed"
                )
                offset += leaf.width
            if offset != len(bits):
                reject(token, "width", "static proof record widths do not match")
        frame.views[label] = View(bits, port.kind == "qnum_signed")

    @staticmethod
    def remove(frame: Frame, label: str) -> View:
        prior = frame.views.pop(label)
        for field_name in tuple(frame.views):
            if field_name.startswith(label + "."):
                del frame.views[field_name]
        return prior

    def mutation(
        self,
        token: Token,
        frame: Frame,
        guards: tuple[tuple[Scalar, bool], ...],
        writes: frozenset[int],
        effect: Effect,
    ) -> Effect:
        if writes & frame.forbidden or any(writes & guard.reads for guard, _ in guards):
            reject(
                token,
                "alias",
                "reversible proof cannot overwrite a const port or a guard",
            )
        return effect.controlled(guards) if guards else effect

    @staticmethod
    def clean(register: View, token: Token) -> Effect:
        def check(row: int) -> int:
            if register.read(row):
                reject(token, "ownership", "reversible proof requires zero at free")
            return row

        return Effect(check, check)

    def invoke(
        self,
        function: Function,
        nodes: tuple[Expression, ...],
        caller: Frame,
        guards: tuple[tuple[Scalar, bool], ...],
        stack: tuple[str, ...],
    ) -> list[Effect]:
        name = function.name.text
        if name in stack or len(stack) >= 24 or len(nodes) != len(function.parameters):
            reject(
                function.name, "proof", "static proof needs a bounded closed call graph"
            )
        if function.disable_const_checks is not None:
            reject(function.name, "proof", "static proof requires checked const ports")
        declarations = {port.name.text: port for port in function.parameters}
        for statement in function.statements:
            if statement.declaration is not None:
                port = statement.declaration
                if port.name.text in declarations:
                    reject(
                        port.name, "name", "static proof requires unique declarations"
                    )
                declarations[port.name.text] = port
        frame = Frame(declarations, {}, {}, caller.forbidden)
        borrowed = {}
        outputs = {}
        seen: set[int] = set()
        for port, node in zip(function.parameters, nodes, strict=True):
            label = port.name.text
            if port.role == "integer" and port.kind == "int":
                frame.classical[label] = self.expression(node, caller).static(
                    node.token
                )
            elif port.role == "real":
                self.bind_real(frame, label, node, caller)
            elif port.role == "output":
                outputs[label] = node
            elif port.role in {"inout", "const"}:
                register = self.view(node, caller)
                if seen.intersection(register.bits):
                    reject(
                        node.token, "alias", "static proof call ports must be disjoint"
                    )
                seen.update(register.bits)
                self.bind(frame, label, register.bits, node.token, actual=register)
                borrowed[label] = frame.views[label]
                if port.role == "const":
                    frame.forbidden |= frozenset(register.bits)
            else:
                reject(port.name, "proof", "unsupported proof port")
        effects = self.block(function.statements, frame, guards, (*stack, name))
        live = {label for label in frame.views if "." not in label}
        if live != (borrowed.keys() | outputs.keys()):
            reject(
                function.name,
                "ownership",
                "static proof requires all local storage to be released",
            )
        for label, register in borrowed.items():
            if frame.views[label] != register:
                reject(
                    function.name,
                    "ownership",
                    "static proof must restore each borrowed binding",
                )
        for label, node in outputs.items():
            self.bind(caller, _name(node), frame.views[label].bits, node.token)
        return effects

    def block(
        self,
        statements: tuple[Statement, ...],
        frame: Frame,
        guards: tuple[tuple[Scalar, bool], ...],
        stack: tuple[str, ...],
    ) -> list[Effect]:
        effects: list[Effect] = []
        for statement in statements:
            token, args = statement.token, statement.arguments
            op = token.text
            self.visits += 1
            if self.visits > 50_000:
                reject(
                    token, "budget.proof", "static proof exceeds its statement bound"
                )
            if statement.declaration is not None:
                continue
            if statement.repeat_index is not None:
                index = statement.repeat_index
                count = self.expression(args[0], frame).static(token)
                if not 0 <= count <= self.max_repeat or index.text in frame.classical:
                    reject(
                        token,
                        "budget.proof",
                        "static proof loop requires a bounded fresh index",
                    )
                for i in range(count):
                    frame.classical[index.text] = i
                    effects.extend(
                        self.block(statement.repeat_body, frame, guards, stack)
                    )
                frame.classical.pop(index.text, None)
                continue
            if statement.static_if_body is not None:
                if self.expression(args[0], frame).static(token):
                    effects.extend(
                        self.block(statement.static_if_body, frame, guards, stack)
                    )
                continue
            if statement.control_body is not None:
                condition = self.expression(args[0], frame)
                original = dict(frame.views)
                effects.extend(
                    self.block(
                        statement.control_body,
                        frame,
                        (*guards, (condition, True)),
                        stack,
                    )
                )
                if frame.views != original:
                    reject(
                        token,
                        "ownership",
                        "static proof control cannot escape bindings",
                    )
                if statement.control_else is not None:
                    effects.extend(
                        self.block(
                            statement.control_else,
                            frame,
                            (*guards, (condition, False)),
                            stack,
                        )
                    )
                    if frame.views != original:
                        reject(
                            token,
                            "ownership",
                            "static proof control cannot escape bindings",
                        )
                continue
            if statement.within is not None:
                outer_views = dict(frame.views)
                first_temporary = self.next_wire
                compute = self.compute_within(statement, frame, guards, stack)
                after_compute = self.next_wire
                within_views = dict(frame.views)
                apply = self.block(statement.apply, frame, guards, stack)
                self.check_workspace(
                    statement, compute, apply, first_temporary, after_compute
                )
                if frame.views != within_views:
                    reject(
                        token,
                        "ownership",
                        "static proof apply cannot move scoped bindings",
                    )
                frame.views = outer_views
                effects.extend(compute)
                effects.extend(apply)
                effects.extend(effect.inverse() for effect in reversed(compute))
                continue
            if statement.inverse_body is not None:
                original = dict(frame.views)
                inverse = self.block(statement.inverse_body, frame, guards, stack)
                if frame.views != original:
                    reject(
                        token,
                        "ownership",
                        "static proof inverse cannot escape bindings",
                    )
                effects.extend(effect.inverse() for effect in reversed(inverse))
                continue
            if statement.skip_control is not None and self.quantum:
                effects.extend(self.block(statement.skip_control, frame, (), stack))
                continue
            if statement != Statement(token, args):
                reject(token, "proof", "unsupported reversible proof scope")
            if op in self.functions:
                helper = self.functions[op]
                if helper.disable_permutation_check is not None and not self.quantum:
                    effects.extend(self.quantum_call(helper, args, frame, guards))
                else:
                    effects.extend(self.invoke(helper, args, frame, guards, stack))
            elif op == "allocate" and len(args) in self.allocation_arities:
                label = _name(args[-1])
                if len(args) == 4:
                    sign, fraction = args[1:3]
                    port = frame.declarations[label]
                    if (
                        sign.token.text != "False"
                        or self.expression(fraction, frame).static(token) != port.width
                        or port.kind != "qphase"
                    ):
                        reject(
                            token,
                            "proof",
                            "unsupported fixed-point allocation in proof prelude",
                        )
                width = self.allocation_width(args, label, frame, token)
                self.bind(frame, label, self.fresh(width, token), token)
                effects.append(self.clean(frame.views[label], token))
            elif op == "free" and len(args) == 1:
                register = self.view(args[0], frame)
                if set(register.bits) & frame.forbidden:
                    reject(token, "ownership", "static proof cannot free a const port")
                self.remove(frame, _name(args[0]))
                effects.append(self.clean(register, token))
            elif op == "->" and len(args) == 2:
                self.transfer(args[0], args[1], frame)
            elif op in {"^=", "+="} and len(args) == 2:
                target = self.view(args[0], frame)
                value = self.update_value(args[1], frame, target)
                writes = frozenset(target.bits)
                if writes & value.reads:
                    reject(
                        token, "alias", "static proof update must have disjoint sources"
                    )
                if op == "^=":

                    def xor(
                        row: int, register: View = target, source: Scalar = value
                    ) -> int:
                        return register.write(
                            row, register.read(row) ^ source.evaluate(row)
                        )

                    effect = Effect(xor, xor, target=target, value=value, update="^=")
                else:

                    def add(
                        row: int, register: View = target, source: Scalar = value
                    ) -> int:
                        return register.write(
                            row, register.read(row) + source.evaluate(row)
                        )

                    def subtract(
                        row: int, register: View = target, source: Scalar = value
                    ) -> int:
                        return register.write(
                            row, register.read(row) - source.evaluate(row)
                        )

                    effect = Effect(
                        add, subtract, target=target, value=value, update="+="
                    )
                effects.append(self.mutation(token, frame, guards, writes, effect))
            elif op in {"X", "SWAP"}:
                registers = tuple(self.view(node, frame) for node in args)
                bits = tuple(bit for register in registers for bit in register.bits)
                if (
                    len(registers) != (1 if op == "X" else 2)
                    or len(registers) != len(bits)
                    or len(bits) != len(set(bits))
                ):
                    reject(
                        token,
                        "alias",
                        "static proof needs distinct scalar gate operands",
                    )
                if op == "X":

                    def flip(row: int, target: int = bits[0]) -> int:
                        return row ^ (1 << target)

                else:

                    def swap(
                        row: int, left: int = bits[0], right: int = bits[1]
                    ) -> int:
                        mask = (row >> left ^ row >> right) & 1
                        return row ^ (mask << left) ^ (mask << right)

                transform = flip if op == "X" else swap
                effects.append(
                    self.mutation(
                        token,
                        frame,
                        guards,
                        frozenset(bits),
                        Effect(transform, transform, gate=Gate(op, bits)),
                    )
                )
            elif self.quantum and op in {"H", "PHASE", "CPHASE"}:
                effects.append(self.phase_gate(op, token, args, frame, guards))
            else:
                effects.extend(self.extra(statement, frame, guards))
        return effects

    max_repeat = 16
    allocation_arities = frozenset({2, 4})

    def allocation_width(
        self, args: tuple[Expression, ...], label: str, frame: Frame, token: Token
    ) -> int:
        return self.expression(args[0], frame).static(token)

    def update_value(self, source: Expression, frame: Frame, target: View) -> Scalar:
        return self.expression(source, frame)

    def bind_real(
        self, frame: Frame, label: str, node: Expression, caller: Frame
    ) -> None:
        reject(node.token, "proof", "unsupported real proof port")

    def compute_within(
        self,
        statement: Statement,
        frame: Frame,
        guards: _Guards,
        stack: tuple[str, ...],
    ) -> list[Effect]:
        assert statement.within is not None
        return self.block(
            statement.within, frame, () if self.quantum else guards, stack
        )

    def check_workspace(
        self,
        statement: Statement,
        compute: list[Effect],
        apply: list[Effect],
        first_temporary: int,
        after_compute: int,
    ) -> None:
        pass

    def extra(
        self, statement: Statement, frame: Frame, guards: _Guards
    ) -> list[Effect]:
        reject(statement.token, "proof", "unsupported reversible proof statement")

    def transfer(self, source: Expression, target: Expression, frame: Frame) -> None:
        sources = source.children if source.token.text == "{" else (source,)
        targets = target.children if target.token.text == "{" else (target,)
        inputs = tuple(_name(node) for node in sources)
        outputs = tuple(_name(node) for node in targets)
        if len(set((*inputs, *outputs))) != len(inputs) + len(outputs):
            reject(target.token, "alias", "static proof transfer requires unique names")
        bits = tuple(bit for node in sources for bit in self.view(node, frame).bits)
        for label in inputs:
            self.remove(frame, label)
        offset = 0
        for label in outputs:
            port = frame.declarations.get(label)
            width = self.transfer_width(port, frame)
            if width is None or offset + width > len(bits):
                reject(
                    target.token,
                    "width",
                    "static proof transfer needs known output widths",
                )
            self.bind(frame, label, bits[offset : offset + width], target.token)
            offset += width
        if offset != len(bits):
            reject(target.token, "width", "static proof transfer must move all wires")

    def transfer_width(self, port: Parameter | None, frame: Frame) -> int | None:
        return port.width if port is not None else None

    def phase_gate(
        self,
        op: str,
        token: Token,
        args: tuple[Expression, ...],
        frame: Frame,
        guards: _Guards,
    ) -> Effect:
        targets = args if op == "H" else args[1:]
        registers = tuple(self.view(arg, frame) for arg in targets)
        bits = tuple(bit for register in registers for bit in register.bits)
        required = 2 if op == "CPHASE" else 1
        if (
            len(registers) != required
            or len(bits) != required
            or len(set(bits)) != required
        ):
            reject(token, "proof", "bounded quantum gate needs distinct scalar wires")
        angle = 0.0
        if op == "H":
            mask = 1 << bits[0]
            scale = math.sqrt(0.5)

            def hadamard_terms(
                row: int, reverse: bool
            ) -> tuple[tuple[int, complex], ...]:
                return ((row, -scale if row & mask else scale), (row ^ mask, scale))

            writes = frozenset(bits)
        else:
            angle = self.angle(args[0], frame)
            phase = cmath.exp(1j * angle)
            mask = sum(1 << bit for bit in bits)

            def phase_terms(row: int, reverse: bool) -> tuple[tuple[int, complex], ...]:
                return (
                    (
                        row,
                        (
                            (phase.conjugate() if reverse else phase)
                            if row & mask == mask
                            else 1 + 0j
                        ),
                    ),
                )

            writes = frozenset()
            if set(bits) & frame.forbidden:
                reject(token, "alias", "bounded quantum gate changes a const phase")

        def fail(row: int) -> int:
            reject(token, "proof", "quantum gate must be inside a verified kernel")

        return self.mutation(
            token,
            frame,
            guards,
            writes,
            Effect(
                fail,
                fail,
                hadamard_terms if op == "H" else phase_terms,
                Gate(op, bits, angle),
            ),
        )

    def angle(self, node: Expression, frame: Frame) -> float:
        token, children = node.token, node.children
        op = token.text
        if op == "qft_const_adder_phase" and len(children) == 3:
            i, value, width = (
                self.expression(child, frame).static(token) for child in children
            )
            if not 0 <= i < width <= 8:
                reject(token, "proof", "Fourier phase index or width is out of range")
            period = 1 << (width - i)
            return math.tau * (value % period) / period
        if op == "pi" and not children:
            return math.pi
        if op == "/" and len(children) == 2:
            left, right = children
            divisor = self.expression(right, frame).static(token)
            if divisor <= 0:
                reject(token, "proof", "Fourier phase requires a positive divisor")
            return self.angle(left, frame) / divisor
        return float(self.expression(node, frame).static(token))

    def quantum_call(
        self,
        helper: Function,
        args: tuple[Expression, ...],
        frame: Frame,
        guards: _Guards,
    ) -> list[Effect]:
        if len(args) != len(helper.parameters):
            reject(
                helper.name, "signature", "quantum proof helper argument count differs"
            )
        classical: dict[str, int] = {}
        writes: set[int] = set()
        bits: list[int] = []
        for port, node in zip(helper.parameters, args, strict=True):
            name = port.name.text
            if port.role == "integer":
                classical[name] = self.expression(node, frame).static(node.token)
            elif port.role in {"inout", "const"}:
                register = self.view(node, frame)
                if port.width != len(register.bits) or set(register.bits).intersection(
                    bits
                ):
                    reject(
                        node.token,
                        "alias",
                        "quantum proof requires exact disjoint port views",
                    )
                bits.extend(register.bits)
                if port.role == "inout":
                    writes.update(register.bits)
            else:
                reject(node.token, "proof", "quantum proof admits borrowed ports only")
        if len(bits) > 8:
            reject(
                helper.name,
                "budget.proof",
                "quantum proof is limited to eight live input bits",
            )
        key = helper.name.text, tuple(classical.items())
        if key not in self.kernels:
            self.kernels[key] = self.build_kernel(helper, classical)
        kernel = self.kernels[key]
        register = View(tuple(bits))
        if writes & frame.forbidden or any(writes & guard.reads for guard, _ in guards):
            reject(
                helper.name, "alias", "quantum proof may not overwrite a guard or const"
            )

        def apply(row: int, reverse: bool) -> int:
            active = all(bool(guard.evaluate(row)) == value for guard, value in guards)
            word = register.read(row)
            return register.write(row, kernel(word, active, reverse))

        return [Effect(lambda row: apply(row, False), lambda row: apply(row, True))]

    def build_kernel(self, helper: Function, classical: dict[str, int]) -> _Kernel:
        evaluator = Region(self.functions, self.records)
        evaluator.quantum = True
        enabled = evaluator.fresh(1, helper.name)[0]
        declarations = {port.name.text: port for port in helper.parameters}
        frame = Frame(declarations, {}, classical, frozenset({enabled}))
        operands = tuple(Expression(port.name) for port in helper.parameters)
        for port in helper.parameters:
            if port.role != "integer":
                if port.width is None:
                    reject(port.name, "proof", "quantum proof needs fixed-width ports")
                evaluator.bind(
                    frame,
                    port.name.text,
                    evaluator.fresh(port.width, port.name),
                    port.name,
                )
        all_bits = evaluator.next_wire
        guard = Scalar(View((enabled,)).read, frozenset({enabled}))
        instructions = evaluator.invoke(helper, operands, frame, ((guard, True),), ())
        if evaluator.next_wire > 16 or len(instructions) > 1024:
            reject(
                helper.name, "budget.proof", "quantum kernel exceeds its proof budget"
            )
        inverse = [effect.inverse() for effect in reversed(instructions)]
        cache: dict[tuple[int, bool, bool], int] = {}

        def apply(word: int, active: bool, reverse: bool) -> int:
            key = word, active, reverse
            if key not in cache:
                state = {(word << 1) | active: 1 + 0j}
                loss = 0.0
                for instruction in inverse if reverse else instructions:
                    updated: dict[int, complex] = {}
                    for row, coeff in state.items():
                        for out, phase in instruction.terms(row, False):
                            updated[out] = updated.get(out, 0j) + coeff * phase
                    # Account for discarded rounding tails by the sum of their
                    # magnitudes. Subsequent unitary steps preserve the norm.
                    threshold = 1e-14
                    loss += sum(
                        abs(coeff)
                        for coeff in updated.values()
                        if abs(coeff) <= threshold
                    )
                    state = {
                        row: coeff
                        for row, coeff in updated.items()
                        if abs(coeff) > threshold
                    }
                    if len(state) > 256:
                        reject(
                            helper.name,
                            "budget.proof",
                            "quantum kernel amplitude support exceeds its bound",
                        )
                if len(state) != 1:
                    reject(
                        helper.name,
                        "proof",
                        "quantum kernel is not a basis map at a reachable input",
                    )
                ((row, coeff),) = state.items()
                if (
                    row >> all_bits
                    or row & 1 != active
                    or abs(coeff - 1) + loss > 1e-11
                ):
                    reject(
                        helper.name,
                        "proof",
                        "quantum kernel must have clean scratch and no relative phase",
                    )
                if not active and row >> 1 != word:
                    reject(
                        helper.name,
                        "proof",
                        "skip-control subgraph must be identity on inactive input",
                    )
                cache[key] = row >> 1
            return cache[key]

        return apply


@dataclass(frozen=True)
class ClosedRegionPlan:
    ports: tuple[tuple[Parameter, View], ...]
    controls: tuple[int, ...]
    targets: tuple[tuple[int, tuple[int, ...]], ...]
    suffix: tuple[Gate, ...]


def _main_body(functions: Mapping[str, Function]) -> Function:
    main = functions["main"]
    while len(main.statements) == 1:
        statement = main.statements[0]
        helper = functions.get(statement.token.text)
        if (
            helper is None
            or helper is main
            or helper.permutation
            or len(helper.parameters) != len(main.parameters)
            or statement != Statement(statement.token, statement.arguments)
            or len(statement.arguments) != len(main.parameters)
            or any(
                argument.children
                or argument.token.text != public.name.text
                or replace(private, name=public.name) != public
                or private.name.text != public.name.text
                for argument, public, private in zip(
                    statement.arguments, main.parameters, helper.parameters, strict=True
                )
            )
        ):
            break
        return replace(main, statements=helper.statements)
    return main


def _region_calls(function: Function) -> set[str]:
    pending = list(function.statements)
    calls = set()
    while pending:
        statement = pending.pop()
        calls.add(statement.token.text)
        pending.extend(_children(statement))
        pending.extend(statement.skip_control or ())
    return calls


def try_closed_region(
    functions: Mapping[str, Function], records: Mapping[str, Record]
) -> ClosedRegionPlan | None:
    if not any(function.disable_permutation_check for function in functions.values()):
        return None
    main = _main_body(functions)
    calls = {name: _region_calls(fn) for name, fn in functions.items()}
    reachable = _reachable("main", calls)
    if set(functions) != reachable:
        return None
    if main.permutation or any(
        port.role != "output" or port.width is None for port in main.parameters
    ):
        return None
    first = next(
        (
            i
            for i, statement in enumerate(main.statements)
            if statement.token.text not in {"allocate", "^="}
        ),
        0,
    )
    end = next(
        (
            i
            for i, statement in enumerate(main.statements)
            if statement.inverse_body is not None
        ),
        0,
    )
    if not 0 < first < end:
        return None
    region = Region(functions, records)
    frame = Frame(
        {port.name.text: port for port in main.parameters}, {}, {}, frozenset()
    )
    try:
        prefix = region.block(main.statements[:first], frame, (), ("main",))
    except HybridVerificationError:
        return None
    ports = tuple(
        (port, frame.views[port.name.text])
        for port in main.parameters
        if port.name.text in frame.views
    )
    n_public = region.next_wire
    if len(ports) != len(main.parameters) or not 1 <= n_public <= 16:
        return None
    initial = 0
    for effect in prefix:
        initial = effect.forward(initial)
    region.quantum = True
    controls: list[int] = []
    head = first
    for statement in main.statements[first:end]:
        before = region.next_wire
        previous = dict(frame.views)
        try:
            candidate = region.block((statement,), frame, (), ("main",))
        except HybridVerificationError:
            region.next_wire = before
            frame.views = previous
            break
        if not candidate or any(
            effect.gate is None or effect.gate.name != "H" for effect in candidate
        ):
            region.next_wire = before
            frame.views = previous
            break
        controls.extend(
            effect.gate.bits[0] for effect in candidate if effect.gate is not None
        )
        head += 1
    if (
        not 1 <= len(controls) <= 6
        or len(set(controls)) != len(controls)
        or initial & sum(1 << bit for bit in controls)
        or head == end
    ):
        return None
    region.quantum = False
    basis = region.block(main.statements[head:end], frame, (), ("main",))
    if not region.kernels or not basis:
        return None
    if any(frame.views.get(port.name.text) != register for port, register in ports):
        reject(
            main.name, "ownership", "closed region must retain public output ownership"
        )
    region.quantum = True
    suffix = region.block(main.statements[end:], frame, (), ("main",))
    if any(effect.gate is None for effect in suffix):
        reject(
            main.name, "proof", "closed region tail must be a static sequence of gates"
        )
    gates = tuple(effect.gate for effect in suffix if effect.gate is not None)
    if any(bit >= n_public for gate in gates for bit in gate.bits):
        reject(main.name, "proof", "closed region tail cannot touch temporary storage")
    selectors = View(tuple(controls))
    rows = []
    for word in range(1 << len(controls)):
        start = selectors.write(initial, word)
        row = start
        for effect in basis:
            row = effect.forward(row)
        if row >> n_public or selectors.read(row) != word:
            reject(
                main.name,
                "proof",
                "closed region must preserve selectors and release all scratch",
            )
        rows.append(row)
        for effect in reversed(basis):
            row = effect.reverse(row)
        if row != start:
            reject(
                main.name,
                "proof",
                "closed region did not reverse its reachable basis row",
            )
    targets = tuple(
        (wire, tuple((row >> wire) & 1 for row in rows))
        for wire in range(n_public)
        if wire not in controls
    )
    return ClosedRegionPlan(ports, tuple(controls), targets, gates)


def emit_closed_region(owner: _Adapt, plan: ClosedRegionPlan) -> None:
    token = owner.syntax.name
    locations = {}
    for port, view in plan.ports:
        register = owner.register(
            port.name, allow_qubit_array=True, allow_fractional_output=True
        )
        owner.effect_op("qreg.allocate", token, (register,))
        locations.update(
            {wire: (register, index) for index, wire in enumerate(view.bits)}
        )
    controls = tuple(locations[wire] for wire in plan.controls)
    for register, index in controls:
        owner.effect_op("qreg.gate", token, (register,), gate="h", index=index)
    for wire, table in plan.targets:
        emit_truth_xor(owner, token, controls, locations[wire], list(table))
    for gate in plan.suffix:
        values = tuple(locations[wire][0] for wire in gate.bits)
        indices = tuple(locations[wire][1] for wire in gate.bits)
        if gate.name in {"X", "H"}:
            owner.effect_op(
                "qreg.gate", token, values, gate=gate.name.lower(), index=indices[0]
            )
        elif gate.name == "SWAP":
            owner.effect_op(
                "qreg.pair_gate", token, values, gate="swap", indices=indices
            )
        elif gate.name == "CPHASE":
            owner.effect_op(
                "qreg.control_phase", token, values, indices=indices, angle=gate.angle
            )
        elif gate.name == "PHASE":
            owner.effect_op(
                "qreg.gate",
                token,
                values,
                gate="phase",
                index=indices[0],
                angle=gate.angle,
            )
        else:
            reject(token, "proof", "unsupported verified tail gate")
