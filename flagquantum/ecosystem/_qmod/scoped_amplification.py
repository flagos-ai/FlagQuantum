"""Checked exact amplification of a bounded borrowed preparation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction.clean_unitary_proof import verify_zero_preparation
from ...compiler._construction.register_profile import register_type
from ...compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from .amplification import exact_rotation
from .callback_arrays import capture_effects
from .masked_selection import emit_selected
from .quantum_callbacks import Lambda
from .syntax import Expression, Parameter, Token, reject
from .unitary_effects import inverse_effects

if TYPE_CHECKING:
    from .context import _Adapt
    from .phase import Location

_PREPARATION = frozenset(
    {
        "qreg.gate",
        "qreg.pair_gate",
        "qreg.control_x",
        "qreg.control_phase",
        "qreg.phase",
    }
)


def known_zero(owner: _Adapt, value: Value) -> bool:
    for op in reversed(owner.operations):
        if value in op.operands[1:]:
            return op.name == "qreg.allocate" or (
                op.name == "qreg.initialize" and op.attributes.get("value") == 0
            )
    return owner.zero_entry is not None and owner.zero_entry(value)


def declare_workspace(owner: _Adapt, token: Token) -> Value:
    index = owner.counter
    owner.counter += 1
    value = Value(ValueId(owner.scope, index), register_type(1))
    owner.certified_workspace.add(value)
    owner.append_op(
        token,
        Operation(
            "qreg.hidden",
            results=(value,),
            attributes={"name": f"$scoped_amplification.{index}", "workspace": True},
            location=token.location,
        ),
    )
    return value


def _capture(
    owner: _Adapt, ref: Expression, locations: tuple[Location, ...]
) -> tuple[Operation, ...]:
    if not isinstance(ref, Lambda) or len(ref.parameters) != 1 or not ref.body:
        reject(
            ref.token,
            "amplification.signature",
            "expected a lexical callback with one packed parameter",
        )
    formal = ref.parameters[0]
    name = formal.text
    if name in owner.parameters or name in owner.constants or name in owner.definitions:
        reject(
            formal,
            "amplification.alias",
            "packed callback parameter must not shadow a bound name",
        )
    owner.parameters[name] = Parameter(formal, "qbit[]", len(locations), "inout")
    owner.widths[name] = len(locations)
    owner.views[name] = locations
    try:
        return capture_effects(owner, ref.token, ref.body, allowed_effects=_PREPARATION)
    finally:
        del owner.views[name], owner.widths[name], owner.parameters[name]


def _locations(op: Operation, token: Token) -> tuple[Location, ...]:
    operands = op.operands[1:]
    if op.name == "qreg.phase" and not operands:
        return ()
    if op.name == "qreg.gate":
        return ((operands[0], op.attributes["index"]),)
    if op.name == "qreg.pair_gate":
        return tuple(zip(operands, op.attributes.get("indices", (0, 0)), strict=True))
    if "indices" in op.attributes:
        return tuple(zip(operands, op.attributes["indices"], strict=True))
    reject(token, "amplification.profile", "callback requires a bounded indexed effect")


def _remap_preparation(
    effects: tuple[Operation, ...], positions: dict[Location, int], token: Token
) -> HybridProgram:
    scope = "$scoped_amplification_proof"
    value = Value(ValueId(scope, 0), register_type(len(positions)))
    entry = Value(ValueId(scope, 1), QUANTUM_EFFECT)
    operations = [
        Operation("qreg.declare", results=(value,), attributes={"name": "packed"})
    ]
    incoming = entry
    operands: tuple[Value, ...]
    for op in (Operation("qreg.allocate"), *effects):
        attrs = dict(op.attributes)
        if op.name == "qreg.allocate":
            operands = (value,)
        else:
            selected = _locations(op, token)
            operands = (value,) * len(selected)
            indices = tuple(positions[location] for location in selected)
            if op.name == "qreg.gate":
                attrs["index"] = indices[0]
            elif indices:
                attrs["indices"] = indices
        outgoing = Value(ValueId(scope, len(operations) + 1), QUANTUM_EFFECT)
        operations.append(
            Operation(
                op.name,
                (incoming, *operands),
                (outgoing,),
                attrs,
                location=token.location,
            )
        )
        incoming = outgoing
    operations.append(
        Operation("qreg.return", (incoming, value), location=token.location)
    )
    return HybridProgram(
        scope, Region((Block((entry,), tuple(operations)),)), token.location
    )


def _marked_words(
    oracle: tuple[Operation, ...], positions: dict[Location, int], token: Token
) -> tuple[int, ...]:
    if len(oracle) > 256:
        reject(token, "amplification.budget", "oracle proof admits at most 256 effects")
    phases: list[tuple[tuple[int, ...], int, float]] = []
    for op in oracle:
        locations = _locations(op, token)
        if op.name == "qreg.gate" and op.attributes["gate"] in {"i", "z"}:
            angle = math.pi if op.attributes["gate"] == "z" else 0.0
        elif op.name in {"qreg.control_phase", "qreg.phase"} and len(locations) <= 7:
            angle = op.attributes["angle"]
        else:
            reject(
                token,
                "amplification.oracle",
                "oracle must implement indexed basis signs",
            )
        bits = tuple(positions[location] for location in locations)
        phases.append(
            (bits, op.attributes.get("control_state", (1 << len(bits)) - 1), angle)
        )

    marked = []
    for word in range(1 << len(positions)):
        phase = math.fsum(
            angle
            for bits, expected, angle in phases
            if sum(((word >> bit) & 1) << i for i, bit in enumerate(bits)) == expected
        )
        if abs(math.sin(phase)) > 1e-14:
            reject(
                token,
                "amplification.oracle",
                "oracle must apply signs +1 or -1 on every word",
            )
        if math.cos(phase) < 0:
            marked.append(word)
    return tuple(marked)


def lower_scoped_amplification(
    owner: _Adapt, token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 4:
        reject(
            token,
            "amplification.signature",
            "expected amplitude, oracle, preparation and packed storage",
        )
    amplitude_arg, oracle_ref, prep_ref, target = args
    arguments = target.children if target.token.text == "{" else (target,)
    locations = tuple(
        location
        for argument in arguments
        for location in (
            (owner.gate_target(argument),)
            if target.token.text == "{" and argument.token.text == "["
            else owner.array_view(argument, allow_phase=True)
        )
    )
    positions = {location: index for index, location in enumerate(locations)}
    if not 1 <= len(locations) <= 12 or len(locations) != len(positions):
        reject(
            target.token,
            "amplification.bounds",
            "packed storage needs one through twelve distinct bits",
        )
    if any(not known_zero(owner, value) for value in {value for value, _ in locations}):
        reject(
            target.token,
            "amplification.zero",
            "exact amplification requires proven zero entry storage",
        )

    forward = _capture(owner, prep_ref, locations)
    oracle = _capture(owner, oracle_ref, locations)
    for ref, effects in ((prep_ref, forward), (oracle_ref, oracle)):
        if any(
            loc not in positions for op in effects for loc in _locations(op, ref.token)
        ):
            reject(
                ref.token,
                "amplification.escape",
                "callback touches quantum storage outside the packed target",
            )
    marked = _marked_words(oracle, positions, oracle_ref.token)
    prepared = verify_zero_preparation(
        _remap_preparation(forward, positions, token), len(positions)
    )
    observed = math.sqrt(
        math.fsum(abs(complex(prepared[word])) ** 2 for word in marked)
    )
    rounds, angle = exact_rotation(
        amplitude_arg.token, owner.classical_number(amplitude_arg), observed
    )
    backward = inverse_effects(token, forward)

    def replay(effects: tuple[Operation, ...]) -> None:
        for op in effects:
            owner.effect_op(op.name, token, op.operands[1:], **op.attributes)

    replay(forward)
    if not rounds:
        return
    aux = declare_workspace(owner, token)
    owner.effect_op("qreg.allocate", token, (aux,))
    owner.effect_op("qreg.gate", token, (aux,), gate="ry", index=0, angle=angle)
    for _ in range(rounds):
        # Only the joint good subspace with the attenuation bit at zero is marked.
        emit_selected(owner, token, ((aux, 0),), 0, oracle)
        owner.effect_op("qreg.gate", token, (aux,), gate="ry", index=0, angle=-angle)
        replay(backward)
        for value, bit in (*locations, (aux, 0)):
            owner.effect_op("qreg.gate", token, (value,), gate="x", index=bit)
        owner.effect_op("qreg.gate", token, (aux,), gate="h", index=0)
        owner.emit_indexed_x(token, (*locations, (aux, 0)))
        owner.effect_op("qreg.gate", token, (aux,), gate="h", index=0)
        for value, bit in (*locations, (aux, 0)):
            owner.effect_op("qreg.gate", token, (value,), gate="x", index=bit)
        owner.effect_op("qreg.phase", token, (), angle=math.pi)
        replay(forward)
        owner.effect_op("qreg.gate", token, (aux,), gate="ry", index=0, angle=angle)
    # The checked Grover remainder bounds all amplitudes outside the joint good subspace.
    owner.effect_op("qreg.drop", token, (aux,))
