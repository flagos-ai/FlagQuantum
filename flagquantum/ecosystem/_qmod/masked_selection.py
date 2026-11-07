"""Coherent selection with negative controls represented as closed effects."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from .callback_arrays import capture_effects
from .syntax import Statement, Token, reject

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Operation
    from .context import _Adapt
    from .phase import Location


def lower_masked_selection(owner: _Adapt, statement: Statement) -> bool:
    guard = statement.arguments[0]
    body = statement.control_body
    if body is None or guard.token.text != "==" or len(guard.children) != 2:
        return False
    source, expected = guard.children
    # Static selectors and record flags are handled here only for compositions.
    port = owner.parameters.get(source.token.text)
    field = owner.record_field(source) if source.token.text == "." else None
    structured = field is not None and field[0].kind == "qbit"
    loop_selector = (
        not expected.children
        and expected.token.text in owner.loop_indices
        and any(child.static_if_body is not None for child in body)
    )
    callback_branch = len(body) == 1 and (
        body[0].token.text in owner.zero_callbacks
        or body[0].inverse_body is not None
        and len(body[0].inverse_body) == 1
        and body[0].inverse_body[0].token.text in owner.zero_callbacks
    )
    if not structured and not loop_selector and not callback_branch:
        return False
    controls: tuple[Location, ...]
    if structured or callback_branch:
        controls = (owner.gate_target(source),)
    elif port is not None and not source.children and port.kind == "qnum":
        controls = owner.array_view(source)
    else:
        return False
    if len(controls) > 7:
        reject(guard.token, "control", "selector needs at most seven locations")
    value = owner.static_integer(expected)
    if not 0 <= value < 1 << len(controls):
        reject(expected.token, "control", "selector comparison value must fit")
    branches = [(value, body)]
    if statement.control_else is not None:
        if len(controls) != 1:
            reject(guard.token, "control", "selected else requires one control")
        branches.append((1 - value, statement.control_else))
    for pattern, children in branches:
        effects = capture_effects(owner, statement.token, children)
        emit_selected(owner, guard.token, controls, pattern, effects)
    return True


def emit_selected(
    owner: _Adapt,
    token: Token,
    controls: tuple[Location, ...],
    pattern: int,
    effects: tuple[Operation, ...],
) -> None:
    def union(
        inner: tuple[Location, ...], state: int
    ) -> tuple[tuple[Location, ...], int] | None:
        locations = list(controls)
        selected = pattern
        for bit, location in enumerate(inner):
            positive = bool(state & (1 << bit))
            if location in locations:
                if positive != bool(selected & (1 << locations.index(location))):
                    return None
                continue
            if positive:
                selected |= 1 << len(locations)
            locations.append(location)
        return tuple(locations), selected

    def emit(
        name: str, locations: tuple[Location, ...], state: int, **attributes: object
    ) -> None:
        owner.effect_op(
            name,
            token,
            tuple(value for value, _ in locations),
            indices=tuple(bit for _, bit in locations),
            control_state=state,
            **attributes,
        )

    def phase(inner: tuple[Location, ...], state: int, angle: float) -> None:
        selected = union(inner, state)
        if selected is not None:
            locations, mask = selected
            emit("qreg.control_phase", locations, mask, angle=angle)

    def flip(inner: tuple[Location, ...], state: int, target: Location) -> None:
        selected = union(inner, state)
        if selected is not None:
            locations, mask = selected
            if target in locations:
                reject(
                    token, "alias", "selected target must be disjoint from its controls"
                )
            emit("qreg.control_x", (*locations, target), mask)

    for effect in effects:
        operands, attributes = effect.operands[1:], effect.attributes
        indices = attributes.get("indices")
        locations = (
            () if indices is None else tuple(zip(operands, indices, strict=True))
        )
        if effect.name == "qreg.control_x" and locations:
            inner, target = locations[:-1], locations[-1]
            flip(inner, attributes.get("control_state", (1 << len(inner)) - 1), target)
        elif effect.name == "qreg.phase" and not operands:
            phase((), 0, attributes["angle"])
        elif effect.name == "qreg.control_phase" and locations:
            phase(
                locations,
                attributes.get("control_state", (1 << len(locations)) - 1),
                attributes["angle"],
            )
        elif effect.name == "qreg.inplace_prepare_amplitudes":
            if not locations:
                locations = tuple(
                    (operands[0], bit)
                    for bit in reversed(range(register_width(operands[0])))
                )
            count = attributes.get("control_count", 0)
            inner, targets = locations[:count], locations[count:]
            selected = union(inner, attributes.get("control_state", (1 << count) - 1))
            if selected is not None:
                inputs, mask = selected
                if set(inputs).intersection(targets):
                    reject(
                        token, "alias", "selected preparation cannot write its control"
                    )
                retained = {
                    k: v
                    for k, v in attributes.items()
                    if k not in {"indices", "control_count", "control_state"}
                }
                emit(
                    effect.name,
                    (*inputs, *targets),
                    mask,
                    **retained,
                    control_count=len(inputs),
                )
        elif effect.name == "qreg.pair_gate" and attributes["gate"] in {"cx", "swap"}:
            left, right = tuple(zip(operands, indices or (0, 0), strict=True))
            if left in controls or right in controls:
                reject(
                    token, "alias", "selected pair must be disjoint from its controls"
                )
            if attributes["gate"] == "swap":
                owner.emit_indexed_x(token, (right, left))
            flip((left,), 1, right)
            if attributes["gate"] == "swap":
                owner.emit_indexed_x(token, (right, left))
        elif effect.name == "qreg.pair_gate" and attributes["gate"] == "cry":
            source, target = tuple(zip(operands, indices or (0, 0), strict=True))
            if target in controls or source == target:
                reject(token, "alias", "selected rotation cannot write its control")
            half = attributes["angle"] / 2
            for angle in (half, -half):
                owner.effect_op(
                    "qreg.gate",
                    token,
                    (target[0],),
                    gate="ry",
                    index=target[1],
                    angle=angle,
                )
                phase((source, target), 3, math.pi)
        elif effect.name == "qreg.gate":
            gate = attributes["gate"]
            target = operands[0], attributes["index"]
            if gate == "i":
                owner.effect_op(effect.name, token, operands, **attributes)
                continue
            if target in controls:
                reject(token, "alias", "selected gate cannot write its control")
            sign = -1 if attributes.get("inverse") else 1
            if gate == "x":
                flip((), 0, target)
            elif gate in {"z", "s", "sdg", "t", "tdg"}:
                angles = {
                    "z": math.pi,
                    "s": math.pi / 2,
                    "sdg": -math.pi / 2,
                    "t": math.pi / 4,
                    "tdg": -math.pi / 4,
                }
                phase((target,), 1, sign * angles[gate])
            elif gate in {"rz", "phase"}:
                angle = sign * attributes["angle"]
                if gate == "rz":
                    phase((), 0, -angle / 2)
                phase((target,), 1, angle)
            elif gate in {"h", "ry", "rx"}:
                half = -math.pi / 4 if gate == "h" else sign * attributes["angle"] / 2
                for direction in (1, -1):
                    owner.effect_op(
                        "qreg.gate",
                        token,
                        (target[0],),
                        gate="ry" if gate == "h" else gate,
                        index=target[1],
                        angle=direction * half,
                    )
                    if direction == 1 or gate != "h":
                        phase((target,), 1, math.pi)
            else:
                reject(token, "control", f"unsupported selected gate {gate}")
        else:
            reject(token, "control", f"unsupported selected effect {effect.name}")


def emit_predicate_xor(
    owner: _Adapt,
    token: Token,
    locations: tuple[Location, ...],
    polynomial: dict[int, float],
    target: Location,
) -> None:
    marker = 1 << locations.index(target)
    reads = tuple(item for item in locations if item != target)
    coefficients = [0] * (1 << len(reads))
    for mask, weight in polynomial.items():
        assert mask & marker and int(weight) == weight
        compressed = (mask & (marker - 1)) | ((mask & ~(2 * marker - 1)) >> 1)
        coefficients[compressed] ^= int(weight) & 1
    for bit in range(len(reads)):
        for basis in range(len(coefficients)):
            if basis & (1 << bit):
                coefficients[basis] ^= coefficients[basis ^ (1 << bit)]

    for value, wire in locations:
        owner.effect_op(
            "qreg.control_phase", token, (value,), indices=(wire,), angle=0.0
        )
    emit_truth_xor(owner, token, reads, target, coefficients)


def emit_truth_xor(
    owner: _Adapt,
    token: Token,
    reads: tuple[Location, ...],
    target: Location,
    coefficients: list[int],
) -> None:
    """Emit disjoint Boolean cofactor cubes into a known distinct target."""

    def walk(
        values: list[int], width: int, fixed: tuple[tuple[Location, int], ...]
    ) -> None:
        if not any(values):
            return
        if all(values):
            selected = tuple(location for location, _ in fixed)
            if selected:
                combined = (*selected, target)
                owner.effect_op(
                    "qreg.control_x",
                    token,
                    tuple(value for value, _ in combined),
                    indices=tuple(wire for _, wire in combined),
                    control_state=sum(bit << i for i, (_, bit) in enumerate(fixed)),
                )
            else:
                owner.effect_op(
                    "qreg.gate", token, (target[0],), gate="x", index=target[1]
                )
            return
        half = len(values) // 2
        zero, one = values[:half], values[half:]
        if zero == one:
            walk(zero, width - 1, fixed)
        else:
            walk(zero, width - 1, (*fixed, (reads[width - 1], 0)))
            walk(one, width - 1, (*fixed, (reads[width - 1], 1)))

    walk(coefficients, len(reads), ())
