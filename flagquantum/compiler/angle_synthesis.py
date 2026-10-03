"""Exact quarter-turn synthesis of a z-rotation into a Clifford+T word.

A Clifford+T target publishes a Hadamard, a phase gate, and the pi/8 gate, and
`basis_translation` reaches it through the `s`, `sdg`, and `z` identities alone.
Every parameterized rotation stays out of reach that way. `rz`, `phase`, and
`u1` are the same gate up to a global phase, their leaves are the same gate, and
the equivalence table deliberately holds no rule for any of them, because the
Euler synthesis needs a z-rotation *opcode* of its own to rewrite into and a
Clifford+T basis publishes none. A target with nothing but `h`, `s`, `t`, and
`cx` therefore cannot express a rotation at all.

`rz(k * pi / 4)` is a different matter: those angles are exact single-qubit
Clifford+T operators, and this module carries their words. `rz(n * pi / 4)` is
`t ** n` up to a power of `omega = exp(1j * pi / 4)` for every integer `n`, so
the family collapses onto eight residues. `QUARTER_TURN_WORDS` holds, per
residue, the shortest word over `t`, `tdg`, `s`, `sdg`, and `z` that reproduces
it, plus a second word that keeps the residue reachable when the target
publishes only one of an inverse pair. Both words of a residue were verified
against the runtime's own gate matrices, and the first word of every residue
carries the minimum number of `t` gates that any word over that vocabulary can
carry.

Nothing here approximates. An angle that is not an exact multiple of `pi/4` is
refused, and so is a trainable angle, for the reason `one_qubit_synthesis`
refuses one: a tolerance here would silently answer with a rotation nobody
asked for. Approximating an arbitrary angle is the Ross-Selinger problem, which
needs exact arithmetic in `Z[omega, 1/sqrt(2)]`, a Diophantine norm equation,
and integer factorization; FlagQuantum carries none of that, and the parity
contract records the gap rather than papering over it.

Global phase is dropped, as it is everywhere else in Compiler. `rz`, `phase`,
and `u1` already differ from one another by a global phase, and a word for
residue `n` reproduces `t ** n` up to one further power of `omega`. FlagQuantum
IR has no field to record either phase.

The module never touches a gate matrix. It reads the operator schema for arity
and the source instruction's own parameter; the table is literal data. That
keeps the Compiler boundary intact -- `basis_translation` may not import the
simulation layer at runtime -- and it is why the table is verified in the test
suite, against the matrices, rather than computed here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..core.ir import Instruction
from ..core.operator_schema import get_operator_schema
from .one_qubit_synthesis import Z_ROTATION_OPCODES

_QUARTER_PI = 0.25 * math.pi

#: The shortest word, and a basis fallback, for each residue of `n` in
#: `rz(n * pi / 4)`. A word is a product of `t`, `tdg`, `s`, `sdg`, and `z`
#: applied in the order written; the empty word is the identity, which is what
#: residue 0 is. The first word of a residue is the shortest over the whole
#: vocabulary and carries the minimum `t` count; the second exists only so that
#: a target publishing one of an inverse pair still reaches all eight residues.
#: Both are exact, so a caller may use either without checking which one it got.
QUARTER_TURN_WORDS: Mapping[int, tuple[tuple[str, ...], ...]] = MappingProxyType(
    {
        0: ((),),
        1: (("t",), ("s", "tdg")),
        2: (("s",), ("sdg", "z")),
        3: (("s", "t"), ("tdg", "z")),
        4: (("z",), ("s", "s")),
        5: (("sdg", "tdg"), ("t", "z")),
        6: (("sdg",), ("s", "z")),
        7: (("tdg",), ("sdg", "t")),
    }
)


def _real_angle(value: Any) -> float | None:
    """Return `value` as a float, or None when it is not a static real angle.

    A trainable value never selects a word, for the same reason
    `one_qubit_synthesis` refuses to let one select a short Euler form: the
    words drop the parameter entirely, which would silently detach a rotation
    from the autograd graph.
    """

    if bool(getattr(value, "requires_grad", False)):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (RuntimeError, ValueError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def quarter_turns(angle: Any) -> int | None:
    """Return `angle` in units of `pi/4` modulo 8, or None when it is not one.

    The test is exact. `n * (pi/4)` and `round(angle / (pi/4)) * (pi/4)` are
    bit-for-bit equal for every quarter turn this module can be handed, and they
    differ by a rounding for every angle that is not one, so a tolerance would
    buy nothing and would cost the guarantee: `pi/4 + 1e-9` is not a quarter
    turn and must not be answered with one. None is returned for a trainable
    angle, for a value that is not a real number, and for a real number outside
    the eight residues.

    The residue is returned rather than the signed turn count, because
    `rz(n * pi/4)` depends only on `n` modulo 8: `t ** 8` is the identity up to
    a global phase, and a negative `n` is `tdg ** -n` up to the same kind of
    phase.
    """

    value = _real_angle(angle)
    if value is None:
        return None
    turns = round(value / _QUARTER_PI)
    if value != turns * _QUARTER_PI:
        return None
    return turns % 8


def quarter_turn_words(
    instruction: Instruction,
) -> tuple[tuple[Instruction, ...], ...]:
    """Return the candidate Clifford+T words for one z-rotation instruction.

    The result is empty when nothing here applies: an opcode that is not `rz`,
    `phase`, or `u1`, one the operator schema does not describe, one that is not
    unitary, one of another arity, or an angle that `quarter_turns` refuses.
    Otherwise it holds one instruction tuple per word of
    `QUARTER_TURN_WORDS[residue]`, in table order, each leaf carrying the source
    instruction's wire and metadata.

    Whether a word can actually run is not decided here. A leaf such as `z` is a
    rule source in `basis_translation`, so a target without `z` may still reach
    the word through `s`, and only the caller's search knows which leaves the
    target publishes. Returning the table instead of a chosen word is what lets
    that search rank these candidates against every other rewrite.
    """

    if instruction.name not in Z_ROTATION_OPCODES:
        return ()
    schema = get_operator_schema(instruction.name)
    if schema is None or not schema.unitary or schema.arity != 1:
        return ()
    turns = quarter_turns(instruction.params.get("theta"))
    if turns is None:
        return ()
    return tuple(
        tuple(
            Instruction(opcode, instruction.wires, metadata=instruction.metadata)
            for opcode in word
        )
        for word in QUARTER_TURN_WORDS[turns]
    )


__all__ = (
    "QUARTER_TURN_WORDS",
    "quarter_turn_words",
    "quarter_turns",
)
