"""Qmod arithmetic updates and phase lowering."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import (
    register_type,
    register_width,
)
from ...compiler._hybrid.model import (
    Operation,
    Value,
    ValueId,
)
from ...compiler._hybrid.verifier import HybridVerificationError
from .adapter_model import (
    _Expression,
)
from .affine import lower_affine_update
from .bindings import (
    publish_bound_array,
    record_phase_spans,
)
from .decisions import lower_decision_phase
from .masked_selection import (
    emit_predicate_xor,
)
from .phase import MAX_PHASE_INPUT_BITS, expand_phase
from .phase_factors import Cube, preserve_factors
from .syntax import (
    Expression,
    Token,
    reject,
)

if TYPE_CHECKING:
    from .context import _Adapt


def real_phase(
    self: _Adapt,
    token: Token,
    expression: Expression,
    coefficient: Expression,
    *,
    condition: Expression | None = None,
    disjoint_condition_target: tuple[Value, int] | None = None,
    conjugate_xor: bool = False,
    xor_cubes: bool = False,
    phase_sign: bool = False,
) -> None:
    saw_signed = False

    def signed_word(node: Expression) -> bool:
        nonlocal saw_signed
        if node.token.text == "." and node.children[1].token.text not in {
            "size",
            "len",
        }:
            selected = self.record_field(node)
            signed = selected is not None and selected[0].kind == "qnum_signed"
        else:
            signed = not node.children and node.token.text in self.signed
        saw_signed |= signed
        return signed

    def quantum_word(node: Expression) -> tuple[tuple[Value, int], ...] | None:
        if node.token.text == "$predicate_word":
            return self.array_view(node.children[0])
        cell = self.record_bit(node)
        if cell is not None:
            return (cell,)
        source = node.children[0] if node.token.text == "[" else node
        if source.token.text == ".":
            if source is node and source.children[1].token.text in {"size", "len"}:
                return None
            resolved = self.record_field(source)
            assert resolved is not None
            field, view = resolved
            if node.token.text == "[":
                if field.kind == "qbit[][]":
                    reject(
                        node.token,
                        "field",
                        "phase bit fields require row and column indices",
                    )
                if field.kind not in {"qbit[]", "qnum[]"}:
                    reject(node.token, "field", "indexing requires an array field")
                index = self.static_integer(node.children[1])
                if not 0 <= index < (field.length or len(view)):
                    reject(node.token, "width", "field index is out of bounds")
                if field.kind == "qnum[]":
                    assert field.width is not None
                    return view[index * field.width : (index + 1) * field.width]
                return (view[index],)
            if field.kind not in {"qbit", "qnum", "qnum_signed"}:
                reject(node.token, "field", "phase arrays require an element index")
            return view
        if source.children:
            return None
        parameter = self.parameters.get(source.token.text)
        if parameter is None or parameter.kind not in {
            "qbit",
            "qbit[]",
            "qnum",
            "qphase",
        }:
            return None
        if parameter.kind == "qbit[]" and node.token.text != "[":
            reject(
                node.token,
                "phase",
                "phase arithmetic requires indexed array elements",
            )
        view = self.array_view(
            source,
            allow_signed=signed_word(node),
            allow_phase=parameter.kind == "qphase",
        )
        if node.token.text == "[":
            index = self.static_integer(node.children[1])
            if not 0 <= index < len(view):
                reject(node.token, "width", "phase bit index is out of bounds")
            return (view[index],)
        return view

    def phase_scale(node: Expression) -> int:
        parameter = self.parameters.get(node.token.text) if not node.children else None
        return (
            (self.widths[node.token.text] or 0)
            if parameter is not None and parameter.kind == "qphase"
            else 0
        )

    factors: tuple[Cube, ...] = ()
    factor_values: tuple[Value, ...] = ()
    try:
        polynomial, locations = expand_phase(
            expression,
            quantum_word,
            self.classical_number,
            signed_word=signed_word,
            fraction_bits=phase_scale,
            condition=condition,
            disjoint_condition_target=disjoint_condition_target,
        )
    except HybridVerificationError as error:
        if (
            conjugate_xor
            and not saw_signed
            and all(d.code == "qmod.budget.phase" for d in error.diagnostics)
        ):
            assert disjoint_condition_target is not None
            owner, wire = disjoint_condition_target
            terms, reads = expand_phase(
                Expression(Token("1", token.location)),
                quantum_word,
                self.classical_number,
                condition=condition,
                disjoint_condition_target=disjoint_condition_target,
            )
            for value, bit in reads:
                self.effect_op(
                    "qreg.control_phase", token, (value,), indices=(bit,), angle=0.0
                )
            for mask, weight in sorted(terms.items()):
                if int(weight) % 2 == 0:
                    continue
                if mask:
                    controls = tuple(
                        location for i, location in enumerate(reads) if mask & (1 << i)
                    )
                    self.emit_indexed_x(token, (*controls, disjoint_condition_target))
                else:
                    self.effect_op("qreg.gate", token, (owner,), gate="x", index=wire)
            return
        if (
            condition is not None
            and disjoint_condition_target is None
            and not saw_signed
            and all(
                diagnostic.code == "qmod.budget.phase"
                for diagnostic in error.diagnostics
            )
        ):
            lower_decision_phase(
                self, token, expression, coefficient, condition, quantum_word
            )
            return
        # Preserve the established small-polynomial path and its resources.
        # Only expansion-budget refusals admit this exact alternative.
        if condition is not None or not all(
            diagnostic.code == "qmod.budget.phase" for diagnostic in error.diagnostics
        ):
            raise
        factored, factors = preserve_factors(
            expression,
            lambda node: None if signed_word(node) else quantum_word(node),
        )
        if not factors:
            raise
        factor_values = tuple(
            Value(ValueId(self.scope, self.counter + i), register_type(1))
            for i in range(len(factors))
        )

        def factored_word(node: Expression) -> tuple[tuple[Value, int], ...] | None:
            if node.token.text.startswith("$phase_factor"):
                return ((factor_values[int(node.token.text[13:])], 0),)
            return quantum_word(node)

        polynomial, locations = expand_phase(
            factored, factored_word, self.classical_number, signed_word=signed_word
        )
        physical_reads = {location for factor in factors for location, _ in factor} | {
            location for location in locations if location[0] not in factor_values
        }
        if len(physical_reads) > MAX_PHASE_INPUT_BITS:
            reject(token, "budget.phase", "phase exceeds 24 quantum input bits")
        self.counter += len(factors)
    if conjugate_xor and (saw_signed or xor_cubes):
        assert disjoint_condition_target is not None
        emit_predicate_xor(
            self, token, locations, polynomial, disjoint_condition_target
        )
        return
    angle = (
        math.pi
        if not coefficient.children
        and coefficient.token.text == "pi"
        and "pi" not in self.parameters
        else self.classical_number(coefficient)
    )
    if conjugate_xor:
        assert disjoint_condition_target is not None
        owner, wire = disjoint_condition_target
        self.effect_op("qreg.gate", token, (owner,), gate="h", index=wire)
    for factor, value in zip(factors, factor_values, strict=True):
        self.phase_factor(token, factor, value)
    # Check reads even when algebraic cancellation removes their phase terms.
    for value, wire in locations:
        self.effect_op(
            "qreg.control_phase", token, (value,), indices=(wire,), angle=0.0
        )
    for mask, coefficient_value in sorted(polynomial.items()):
        if phase_sign:
            # The folded bit contributes only (-1) to the selected basis states.
            # Even terms of this Boolean Möbius expansion are exact identities.
            assert coefficient_value == int(coefficient_value) and angle == math.pi
            if not int(coefficient_value) % 2:
                continue
            coefficient_value = 1
        radians = angle * coefficient_value
        if not math.isfinite(radians) or abs(radians) > 2**53:
            reject(token, "phase", "phase angle exceeds finite ±2**53 range")
        if not mask:
            self.effect_op("qreg.phase", token, (), angle=radians)
            continue
        selected = tuple(
            location for bit, location in enumerate(locations) if mask & (1 << bit)
        )
        self.effect_op(
            "qreg.control_phase",
            token,
            tuple(register for register, _ in selected),
            indices=tuple(wire for _, wire in selected),
            angle=radians,
        )
    for value in reversed(factor_values):
        self.effect_op("qreg.uncompute", token, (value,))
        self.effect_op("qreg.release", token, (value,))

    if conjugate_xor:
        assert disjoint_condition_target is not None
        owner, wire = disjoint_condition_target
        self.effect_op("qreg.gate", token, (owner,), gate="h", index=wire)


def phase_factor(self: _Adapt, token: Token, factor: Cube, value: Value) -> None:
    owners = tuple(dict.fromkeys(owner for (owner, _), _ in factor))
    widths = {f"owner{i}": register_width(owner) for i, owner in enumerate(owners)}

    def node(text: str, *children: Expression) -> Expression:
        return Expression(replace(token, text=text), children)

    predicates = []
    for i, owner in enumerate(owners):
        mask = sum(1 << bit for (source, bit), _ in factor if source == owner)
        expected = sum(
            1 << bit
            for (source, bit), positive in factor
            if source == owner and positive
        )
        predicates.append(
            node(
                "==",
                node("&", node(f"owner{i}"), node(str(mask))),
                node(str(expected)),
            )
        )
    predicate = predicates[0]
    for other in predicates[1:]:
        predicate = node("&", predicate, other)
    expression = _Expression(
        predicate, widths, f"{self.scope}.factor{value.id.index}", set()
    )
    graph = expression.function()
    self.functions.append(graph)
    self.append_op(
        token,
        Operation(
            "qreg.local",
            results=(value,),
            attributes={"name": graph.name},
            location=token.location,
        ),
    )
    self.effect_op("qreg.compute", token, (value, *owners), callee=graph.name)


def bind_registers(
    self: _Adapt,
    token: Token,
    source: Expression,
    destination: Expression,
    *,
    publish_output: bool = False,
) -> None:
    sources = source.children if source.token.text == "{" else (source,)
    destinations = (
        destination.children if destination.token.text == "{" else (destination,)
    )
    source_names = tuple(self.atom(item).text for item in sources)
    destination_names = tuple(self.atom(item).text for item in destinations)
    if len(set(source_names + destination_names)) != len(
        source_names + destination_names
    ):
        reject(token, "binding", "binding names must be distinct")
    if record_phase_spans(self, source_names) or record_phase_spans(
        self, destination_names
    ):
        reject(
            token,
            "precision",
            "fractional record binding requires checked output publication",
        )
    locations = tuple(
        location for item in sources for location in self.array_view(item)
    )
    if len(set(locations)) != len(locations):
        reject(token, "binding", "binding source locations must be disjoint")
    offset = 0
    assignments: list[tuple[str, tuple[tuple[Value, int], ...]]] = []
    for item in destinations:
        name = self.atom(item).text
        parameter = self.parameters.get(name)
        if parameter is None or parameter.role not in {"local", "output", "inout"}:
            reject(
                item.token,
                "binding",
                "binding destination must be a declared local or output",
            )
        if name in self.views and name not in self.suspended:
            reject(item.token, "binding", "binding destination already owns storage")
        width = self.widths[name]
        if width is None:
            if len(destinations) != 1:
                reject(
                    item.token,
                    "binding",
                    "split destinations require explicit widths",
                )
            width = len(locations)
        view = locations[offset : offset + width]
        offset += width
        if len(view) != width:
            reject(
                item.token,
                "binding",
                "binding widths must cover the source exactly",
            )
        if name in self.registers:
            value = self.registers[name]
            expected = tuple((value, index) for index in range(register_width(value)))
            if name not in self.suspended or view != expected:
                reject(
                    item.token,
                    "binding",
                    "restored storage must match the original wire order",
                )
        elif name in self.views:
            if name not in self.suspended or view != self.views[name]:
                reject(
                    item.token,
                    "binding",
                    "restored storage must match the original wire order",
                )
        elif parameter.role != "local":
            if parameter.role != "output":
                reject(
                    item.token,
                    "binding",
                    "borrowed storage must be restored, not rebound",
                )
            if not publish_output or len(destinations) != 1:
                reject(
                    item.token,
                    "binding",
                    "fresh output binding requires an allocated storage owner",
                )
            publish_bound_array(self, item.token, view)
        assignments.append((name, view))
    if offset != len(locations):
        reject(token, "binding", "binding widths must cover the source exactly")
    # Identity effects let the shared verifier check source initialization.
    checked: set[Value] = set()
    for value, wire in locations:
        if value not in checked:
            self.effect_op("qreg.gate", token, (value,), gate="i", index=wire)
            checked.add(value)
    self.suspended.update(source_names)
    for name, view in assignments:
        self.widths[name] = len(view)
        self.suspended.discard(name)
        if name in self.registers:
            self.views.pop(name, None)
        else:
            self.views[name] = view


def increment(
    self: _Adapt,
    token: Token,
    target: Expression,
    amount: Expression,
    *,
    controls: tuple[tuple[Value, int], ...] = (),
) -> None:
    field = self.record_field(target)
    name = target.token.text if field is not None else self.atom(target).text
    parameter = field[0] if field is not None else self.parameters.get(name)
    if parameter is None or parameter.kind != "qnum":
        reject(target.token, "type", "in-place increment requires an integer")
    if name in self.fractional_bits:
        reject(target.token, "type", "constant increment requires an integer")
    negative = amount.token.text == "-" and len(amount.children) == 1
    magnitude = amount.children[0] if negative else amount
    if magnitude.children or not magnitude.token.text.isdecimal():
        if field is not None or controls or name in self.signed:
            reject(target.token, "type", "signed increment requires a constant offset")
        lower_affine_update(self, token, target, amount)
        return
    if int(magnitude.token.text) > 256:
        reject(
            amount.token,
            "arithmetic",
            "constant increment requires a literal offset in -256..256",
        )
    view = self.array_view(target, allow_signed=True)
    modulus = 1 << len(view)
    offset = ((-1 if negative else 1) * int(magnitude.token.text)) % modulus
    if offset > modulus // 2:
        offset -= modulus
    if offset == 0:
        value, bit = view[0]
        self.effect_op("qreg.gate", token, (value,), gate="i", index=bit)
    for _ in range(abs(offset)):
        order = range(len(view)) if offset < 0 else reversed(range(len(view)))
        for bit in order:
            if bit or controls:
                self.emit_indexed_x(token, (*controls, *view[:bit], view[bit]))
            else:
                value, wire = view[0]
                self.effect_op("qreg.gate", token, (value,), gate="x", index=wire)


def emit_indexed_x(
    self: _Adapt, token: Token, locations: tuple[tuple[Value, int], ...]
) -> None:
    if len(locations) < 2 or len(set(locations)) != len(locations):
        reject(token, "control", "controls and target must be distinct qubit locations")
    self.effect_op(
        "qreg.control_x",
        token,
        tuple(value for value, _ in locations),
        indices=tuple(wire for _, wire in locations),
    )
