"""Prove and eliminate a local phase-kickback bit before Compiler lowering."""

from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Operation
from .syntax import Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def lower_free(owner: "_Adapt", name: Token) -> None:
    parameter = owner.parameters.get(name.text)
    value = owner.registers.get(name.text)
    if (
        parameter is None
        or parameter.role != "local"
        or parameter.kind != "qbit"
        or name.text in owner.dropped
        or value is None
    ):
        reject(name, "free", "free requires a live local qbit")

    def fail() -> None:
        reject(name, "free", "cannot prove local qbit returned to zero before free")

    touched = [
        op for op in owner.operations if value in op.operands or value in op.results
    ]
    signature = [(op.name, op.attributes.get("gate")) for op in touched]
    if (
        signature[:4]
        != [
            ("qreg.hidden", None),
            ("qreg.allocate", None),
            ("qreg.gate", "x"),
            ("qreg.gate", "h"),
        ]
        or signature[-2:] != [("qreg.gate", "h"), ("qreg.gate", "x")]
        or len(touched) < 6
    ):
        fail()

    # X;H prepares the -1 eigenstate of X. Each CX contributes a Z on its
    # control and leaves that eigenstate intact. H;X returns the bit to zero.
    replacements: dict[int, Operation] = {}
    for op in touched[4:-2]:
        if (
            op.name != "qreg.pair_gate"
            or op.attributes.get("gate") != "cx"
            or len(op.operands) != 3
            or op.operands[1] == value
            or op.operands[2] != value
        ):
            fail()
        indices = op.attributes.get("indices", (0, 0))
        replacements[id(op)] = replace(
            op,
            name="qreg.gate",
            operands=op.operands[:2],
            attributes={"gate": "z", "index": indices[0]},
        )
    removed = {id(op) for op in touched}
    rewritten: list[Operation] = []
    incoming = owner.entry
    for op in owner.operations:
        identity = id(op)
        if identity in removed and identity not in replacements:
            continue
        op = replacements.get(identity, op)
        if op.operands and op.operands[0].type == incoming.type:
            op = replace(op, operands=(incoming, *op.operands[1:]))
            incoming = op.results[0]
        rewritten.append(op)
    owner.operations[:] = rewritten
    owner.effect = incoming
    owner.dropped.add(name.text)
