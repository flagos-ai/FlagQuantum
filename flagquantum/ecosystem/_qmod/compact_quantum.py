"""Experimental checked scoped unitary tape for large repeated Qmod programs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import groupby
from typing import TYPE_CHECKING

from ...compiler.state_preparation_synthesis import _synthesize_real_state
from ...core._composition import controlled_instruction
from ...core.controlled import _clean_mcx_instructions
from ...core.ir import Instruction
from .basis_regions import (
    Effect,
    Frame,
    Gate,
    Scalar,
    View,
    _Guards,
    _name,
    _region_calls,
)
from .fixed_regions import FixedRegion, _Number
from .syntax import Expression, Function, Record, Statement, Token, reject
from .unary_selection_proof import _reachable

if TYPE_CHECKING:
    from .adapter import _Adapt


_Mask = tuple[tuple[int, int], ...]


class GateRegion(FixedRegion):
    max_repeat = 128

    def __init__(
        self, definitions: Mapping[str, Function], records: Mapping[str, Record]
    ) -> None:
        super().__init__(definitions, records, 8)
        self.quantum = True

    def fresh(self, width: int, token: Token) -> tuple[int, ...]:
        if not 1 <= width <= 24 or self.next_wire + width > 16384:
            reject(token, "budget.proof", "scoped unitary proof exceeds its wire bound")
        bits = tuple(range(self.next_wire, self.next_wire + width))
        self.next_wire += width
        return bits

    def view(self, node: Expression, frame: Frame) -> View:
        if node.token.text == ".":
            label = _name(node)
            if label not in frame.views:
                members = [
                    view
                    for key, view in frame.views.items()
                    if key.startswith(label + ".")
                ]
                if members:
                    bits = tuple(bit for member in members for bit in member.bits)
                    return View(bits)
        return super().view(node, frame)

    def number(self, node: Expression, frame: Frame) -> tuple[_Number, frozenset[int]]:
        if node.token.text == "pi" and not node.children:
            return lambda row: math.pi, frozenset()
        if node.token.text == "[" and len(node.children) == 2:
            source, index = node.children
            if not source.children and isinstance(
                values := frame.reals.get(source.token.text), tuple
            ):
                offset = self.expression(index, frame).static(index.token)
                if not 0 <= offset < len(values):
                    reject(index.token, "proof", "real array index outside bounds")
                return lambda row: values[offset], frozenset()
        return super().number(node, frame)

    def bind_real(
        self, frame: Frame, label: str, node: Expression, caller: Frame
    ) -> None:
        if frame.declarations[label].kind == "real[]":
            values = caller.reals.get(node.token.text) if not node.children else None
            if values is None and node.token.text == "[]":
                values = tuple(
                    self.number(item, caller)[0](0) for item in node.children
                )
            if not isinstance(values, tuple) or not 1 <= len(values) <= 256:
                reject(
                    node.token,
                    "proof",
                    "scoped unitary needs a bounded static real vector",
                )
            frame.reals[label] = values
            return
        super().bind_real(frame, label, node, caller)

    def compute_within(
        self,
        statement: Statement,
        frame: Frame,
        guards: _Guards,
        stack: tuple[str, ...],
    ) -> list[Effect]:
        assert statement.within is not None
        original = frame.forbidden
        frame.forbidden = frozenset()
        try:
            effects = self.block(statement.within, frame, (), stack)
        finally:
            frame.forbidden = original
        for step in effects:
            writes = (
                set(step.gate.bits)
                if step.gate is not None
                else set(step.target.bits if step.target is not None else ())
            )
            if writes & original:
                # Const may acquire phase through an X/predicate conjugation.
                # Every predicate for such a write reads only protected wires.
                reads = set(step.value.reads if step.value is not None else ())
                reads.update(bit for cond, _ in step.guards for bit in cond.reads)
                if (
                    step.update not in {None, "^="}
                    or (step.gate is not None and step.gate.name != "X")
                    or reads - original
                ):
                    reject(
                        statement.token,
                        "proof",
                        "scoped Const conjugation must preserve its basis labels",
                    )
        return effects

    def check_workspace(
        self,
        statement: Statement,
        compute: list[Effect],
        apply: list[Effect],
        first_temporary: int,
        after_compute: int,
    ) -> None:
        if first_temporary == after_compute:
            return
        reads = set().union(*(all_wires(step) for step in compute))
        if any(reads & write_wires(step) for step in apply):
            reject(
                statement.token,
                "proof",
                "scoped apply may not change a computed work dependency",
            )

    def extra(
        self, statement: Statement, frame: Frame, guards: _Guards
    ) -> list[Effect]:
        token, args = statement.token, statement.arguments
        if token.text == "RZ" and len(args) == 2:
            angle, reads = self.number(args[0], frame)
            target = self.view(args[1], frame)
            if reads or len(target.bits) != 1:
                reject(token, "proof", "scoped RZ needs static angle and scalar target")
            return [
                Effect(
                    lambda row: row,
                    lambda row: row,
                    gate=Gate("RZ", target.bits, angle(0)),
                ).controlled(guards)
            ]
        if token.text == "U" and len(args) == 5:
            coefficients = [self.number(arg, frame) for arg in args[:4]]
            if any(reads for _, reads in coefficients) or any(
                number(0) != 0 for number, _ in coefficients[:3]
            ):
                reject(token, "proof", "scoped phase U must be scalar")
            target = self.view(args[-1], frame)
            if len(target.bits) != 1:
                reject(token, "proof", "scoped U needs scalar target")
            return [
                Effect(
                    lambda row: row,
                    lambda row: row,
                    gate=Gate("GLOBAL", (), coefficients[3][0](0)),
                ).controlled(guards)
            ]
        if token.text == "inplace_prepare_amplitudes" and len(args) == 3:
            vector, bound, arg = args
            target = self.view(arg, frame)
            entries = tuple(self.number(item, frame) for item in vector.children)
            weights = tuple(number(0) for number, _ in entries)
            if (
                len(target.bits) > 2
                or len(weights) != 1 << len(target.bits)
                or any(reads for _, reads in entries)
                or self.number(bound, frame)[0](0) != 0
                or abs(math.hypot(*weights) - 1) > 1e-12
            ):
                reject(
                    token,
                    "proof",
                    "scoped real preparation must be normalized and at most two bits",
                )
            return [
                self.mutation(
                    token,
                    frame,
                    guards,
                    frozenset(gate.wires),
                    Effect(
                        lambda row: row,
                        lambda row: row,
                        gate=Gate(
                            gate.name.upper(), gate.wires, gate.params.get("theta", 0.0)
                        ),
                    ),
                )
                for gate in _synthesize_real_state(
                    weights, tuple(reversed(target.bits))
                )
            ]
        return super().extra(statement, frame, guards)


def write_wires(effect: Effect) -> set[int]:
    if effect.target is not None:
        return set(effect.target.bits)
    if effect.gate is None or effect.gate.name in {"RZ", "PHASE", "CPHASE", "GLOBAL"}:
        return set()
    if effect.gate.name == "CX":
        return {effect.gate.bits[-1]}
    return set(effect.gate.bits)


def all_wires(effect: Effect) -> set[int]:
    result = set(effect.gate.bits if effect.gate is not None else ())
    if effect.target is not None:
        result.update(effect.target.bits)
    if effect.value is not None:
        result.update(effect.value.reads)
    result.update(bit for guard, _ in effect.guards for bit in guard.reads)
    return result


def _physical_map(
    effects: list[Effect], public: tuple[int, ...]
) -> tuple[dict[int, int], int]:
    intervals: dict[int, tuple[int, int]] = {}
    for index, effect in enumerate(effects):
        for bit in all_wires(effect) - set(public):
            first, _ = intervals.get(bit, (index, index))
            intervals[bit] = first, index
    allocated = {bit: wire for wire, bit in enumerate(public)}
    live: dict[int, int] = {}
    position = len(public)
    for bit, (start, end) in sorted(intervals.items(), key=lambda item: item[1]):
        live = {wire: release for wire, release in live.items() if release >= start}
        wire = next(
            (w for w in range(len(public), position) if w not in live), position
        )
        position = max(position, wire + 1)
        live[wire] = end
        allocated[bit] = wire
    return allocated, position


def _cubes(table: tuple[bool, ...], inputs: tuple[int, ...]) -> tuple[_Mask, ...]:
    if not any(table):
        return ()
    if all(table):
        return ((),)
    half = len(table) // 2
    low, high = table[:half], table[half:]
    if low == high:
        return _cubes(low, inputs[:-1])
    return tuple(
        ((inputs[-1], required), *cube)
        for required, subset in enumerate((low, high))
        for cube in _cubes(subset, inputs[:-1])
    )


def _mask_value(
    scalar: Scalar, bit: int, mask: _Mask, mapping: Mapping[int, int]
) -> tuple[_Mask, ...]:
    inputs = tuple(sorted(scalar.reads))
    if len(inputs) > 12:
        raise ValueError("bounded symbolic Boolean table exceeds 12 bits")
    table = tuple(
        bool(
            scalar.evaluate(
                sum(1 << wire for i, wire in enumerate(inputs) if word >> i & 1)
            )
            >> bit
            & 1
        )
        for word in range(1 << len(inputs))
    )
    return tuple(
        (*mask, *((mapping[wire], state) for wire, state in cube))
        for cube in _cubes(table, inputs)
    )


class Tape:
    def __init__(self, mapping: dict[int, int], position: int) -> None:
        self.mapping = mapping
        self.position = position
        self.wires = position
        self.operations: list[Instruction] = []

    def gate(self, name: str, wires: tuple[int, ...] = (), **params: float) -> None:
        self.operations.append(Instruction(name, tuple(wires), params))

    def flip(self, target: int, mask: _Mask) -> None:
        merged = dict(mask)
        if target in merged:
            raise ValueError("scoped XOR target overlaps its guards")
        if any(state != merged[wire] for wire, state in mask):
            return
        wires = tuple(merged)
        negative = tuple(wire for wire, value in merged.items() if not value)
        for wire in negative:
            self.gate("x", (wire,))
        if len(wires) <= 2:
            self.gate(("x", "cx", "ccx")[len(wires)], (*wires, target))
        else:
            scratch = tuple(range(self.position, self.position + len(wires) - 2))
            self.wires = max(self.wires, scratch[-1] + 1)
            self.operations.extend(_clean_mcx_instructions(wires, target, scratch))
        for wire in reversed(negative):
            self.gate("x", (wire,))

    def guard(self, guards: _Guards, mask: _Mask) -> tuple[_Mask, ...]:
        inputs = frozenset(bit for scalar, _ in guards for bit in scalar.reads)
        scalar = Scalar(
            lambda word: int(
                all(bool(cond.evaluate(word)) == on for cond, on in guards)
            ),
            inputs,
        )
        return _mask_value(scalar, 0, mask, self.mapping)

    def walk(
        self, effects: tuple[Effect, ...], depth: int = 0, mask: _Mask = ()
    ) -> None:
        for guard, group in groupby(
            effects, key=lambda step: step.guards[depth : depth + 1]
        ):
            instructions = tuple(group)
            if not guard:
                for step in instructions:
                    self.emit(step, mask)
                continue
            covers = self.guard(guard, mask)
            if not covers:
                continue
            if covers == (mask,) or (
                len(covers) == 1 and len(covers[0]) <= 2 and len(instructions) <= 3
            ):
                self.walk(instructions, depth + 1, covers[0])
                continue
            flag = self.position
            self.position += 1
            self.wires = max(self.wires, self.position)
            for cube in covers:
                self.flip(flag, cube)
            self.walk(instructions, depth + 1, ((flag, 1),))
            for cube in reversed(covers):
                self.flip(flag, cube)
            self.position -= 1

    def emit(self, step: Effect, mask: _Mask) -> None:
        gate = step.gate
        if gate is None:
            if step.update is None:
                return
            target, source = step.target, step.value
            assert target is not None and source is not None
            wires = tuple(self.mapping[bit] for bit in target.bits)
            if step.update == "^=":
                for bit, wire in enumerate(wires):
                    for cube in _mask_value(source, bit, mask, self.mapping):
                        self.flip(wire, cube)
            else:
                delta = source.evaluate(0) * (-1 if step.update == "-=" else 1)
                if source.reads or delta not in {-1, 0, 1}:
                    raise ValueError(
                        "scoped arithmetic admits only constant unit increments"
                    )
                if delta:
                    for i in reversed(range(len(wires))):
                        self.flip(
                            wires[i],
                            (*mask, *((wire, int(delta == 1)) for wire in wires[:i])),
                        )
            return
        wires = tuple(self.mapping[bit] for bit in gate.bits)
        if gate.name == "X":
            self.flip(wires[0], mask)
        elif gate.name == "CX":
            self.flip(wires[1], (*mask, (wires[0], 1)))
        elif gate.name == "GLOBAL":
            if not mask:
                self.gate("phase", (0,), theta=gate.angle)
                self.gate("x", (0,))
                self.gate("phase", (0,), theta=gate.angle)
                self.gate("x", (0,))
                return
            negative = tuple(wire for wire, value in mask if not value)
            for wire in negative:
                self.gate("x", (wire,))
            op = Instruction("phase", (mask[-1][0],), {"theta": gate.angle})
            controls = tuple(wire for wire, _ in mask[:-1])
            self.operations.extend(
                controlled_instruction(op, controls) if controls else (op,)
            )
            for wire in reversed(negative):
                self.gate("x", (wire,))
        else:
            name = gate.name.lower()
            op = Instruction(
                name,
                wires,
                (
                    {"theta": gate.angle}
                    if name in {"ry", "rz", "phase", "cphase"}
                    else {}
                ),
            )
            if not mask:
                self.operations.append(op)
            else:
                flipped = [wire for wire, value in mask if not value]
                for wire in flipped:
                    self.gate("x", (wire,))
                self.operations.extend(
                    controlled_instruction(op, tuple(wire for wire, _ in mask))
                )
                for wire in reversed(flipped):
                    self.gate("x", (wire,))


def compact_region(
    functions: Mapping[str, Function], records: Mapping[str, Record]
) -> tuple[Tape, list[Effect]]:
    main = functions["main"]
    declarations = {port.name.text: port for port in main.parameters}
    for step in main.statements:
        if step.declaration is not None:
            declarations[step.declaration.name.text] = step.declaration
    frame = Frame(declarations, {}, {}, frozenset())
    evaluator = GateRegion(functions, records)
    effects = evaluator.block(main.statements, frame, (), ("main",))
    if set(frame.views) != {port.name.text for port in main.parameters}:
        reject(main.name, "proof", "scoped unitary must return every owned output once")
    outputs = tuple(
        bit
        for port in main.parameters
        for bit in reversed(frame.views[port.name.text].bits)
    )
    if len(outputs) > 16 or len(outputs) != len(set(outputs)):
        reject(
            main.name,
            "proof",
            "scoped unitary output wires must be disjoint and bounded",
        )
    physical, width = _physical_map(effects, outputs)
    tape = Tape(physical, width)
    tape.walk(tuple(effects))
    return tape, effects


def candidate_tape(
    functions: Mapping[str, Function], records: Mapping[str, Record]
) -> Tape | None:
    main = functions["main"]
    if not any(
        field.record is not None
        for record in records.values()
        for field in record.fields
    ):
        return None
    if not any(
        arg.token.text == "[]" and 128 <= len(arg.children) <= 256
        for action in main.statements
        for arg in action.arguments
    ):
        return None
    if main.permutation or any(
        p.role != "output" or p.width is None or p.record is not None
        for p in main.parameters
    ):
        return None
    calls = {name: _region_calls(function) for name, function in functions.items()}
    if set(functions) != _reachable("main", calls):
        return None

    def explicit_free(statements: tuple[Statement, ...]) -> bool:
        for stmt in statements:
            if stmt.token.text == "free":
                return True
            nested = (
                stmt.within or (),
                stmt.apply,
                stmt.control_body or (),
                stmt.control_else or (),
                stmt.repeat_body,
                stmt.inverse_body or (),
                stmt.static_if_body or (),
                stmt.skip_control or (),
            )
            if any(explicit_free(block) for block in nested):
                return True
        return False

    if any(explicit_free(function.statements) for function in functions.values()):
        return None
    return compact_region(functions, records)[0]


def emit_compact_region(owner: _Adapt, tape: Tape) -> None:
    from ...compiler._construction.gate_tape import pack_gate_templates

    token = owner.syntax.name
    ports = tuple(
        owner.register(port.name, allow_qubit_array=True)
        for port in owner.syntax.parameters
    )
    data = sum(port.width or 0 for port in owner.syntax.parameters)
    for port in ports:
        owner.effect_op("qreg.allocate", token, (port,))
    templates, schedule = pack_gate_templates(tape.operations)
    owner.effect_op(
        "qreg.gate_tape",
        token,
        ports,
        workspace=tape.wires - data,
        templates=templates,
        schedule=schedule,
    )
