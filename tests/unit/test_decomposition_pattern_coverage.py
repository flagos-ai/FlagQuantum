"""What the built-in decomposition rules cover, and what they refuse.

`gate_decomposition_patterns` is a claim about a *rule library*: which opcodes
the compiler can rewrite when the basis it was handed is missing them, and which
it refuses. The library is `EQUIVALENCE_RULES`, and the claim is checkable, so
this module checks it rather than restating it.

Two facts about the table itself are pinned first, because both are properties
the table cannot have by accident: every rule names a declared unitary opcode
(a rule for a channel, a misspelling, or an opcode the schema does not know
would be a table entry nothing can ever read), and the table is narrower than
the opcode vocabulary -- 18 of the 31 declared unitary opcodes, leaving 13 that
no rule names.

The rest of the module is the partition those numbers produce, measured by
converting every declared unitary opcode into two named bases and sorting each
outcome into the refusal kind that produced it.

Into Clifford+T, ten of the 31 convert and 21 refuse. The 21 are not one kind of
failure. Eleven of them *have* a rule and still refuse, because the rule's own
leaves ask for a z-rotation the basis does not publish; the other ten have no
rule at all. That split is the substance of the row: a rule is necessary and not
sufficient, and a coverage number read off the table alone would overstate what
a tight basis can reach by eleven opcodes.

Into a basis that publishes a z-rotation and a two-wire entangler, all 31
convert, and each converted program is compared against its source through the
runtime's statevector kernel up to one global phase. The same 13 unnamed opcodes
that no rule names are reached here through the synthesis fallback, which is why
the table's own width is not the coverage boundary.

Each test states the number the parity row asserts. A rule that is added, a
synthesis path that stops reaching an opcode, or a refusal that changes kind
moves one of these numbers and fails the test, which is the point: prose cannot
keep them true.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.compiler.basis_conversion import (
    BasisConversionError,
    convert_basis,
)
from flagquantum.compiler.basis_translation import EQUIVALENCE_RULES
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

# Every declared opcode that denotes a unitary on at least one wire, sorted. A
# channel is not a gate and no named basis may hold one, so the conversion entry
# point refuses it before any rewrite, which is a different question from the
# one this module measures.
_UNITARY_OPCODES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and not schema.channel
    )
)

# One representative instruction per opcode. A parameter value is arbitrary but
# fixed, because the partition below is a statement about the rule table and the
# synthesis bases rather than about any particular angle.
_PARAMETER_VALUES = {"theta": 0.7, "phi": 0.3, "lbd": 1.1}

# The 13 unitary opcodes no rule in `EQUIVALENCE_RULES` names. They are listed
# rather than derived so that a rule added to the table has to be looked at
# against this claim instead of silently moving it.
_UNNAMED_BY_RULES = frozenset(
    {
        "h",
        "i",
        "phase",
        "rz",
        "s",
        "sx",
        "sxdg",
        "t",
        "tdg",
        "u1",
        "u2",
        "u3",
        "y",
    }
)

# A tight basis: the Clifford+T gate set. It publishes no z-rotation, which is
# what makes the escape branch below reachable at all.
_CLIFFORD_T = ("h", "s", "t", "cx")

# A permissive basis: one z-rotation and one supercontrolled entangler, the two
# names the Euler and KAK syntheses are driven with.
_ROTATION_AND_ENTANGLER = ("cz", "rx", "rz")


def _single_opcode_program(opcode: str) -> CircuitIR:
    """One instruction of `opcode`, on as many wires as its schema declares."""

    schema = OPERATOR_SCHEMAS[opcode]
    return CircuitIR(
        max(schema.arity, 1),
        (
            Instruction(
                opcode,
                tuple(range(schema.arity)),
                params={name: _PARAMETER_VALUES[name] for name in schema.parameters},
            ),
        ),
        dtype="complex128",
    )


def _partition(gates: tuple[str, ...]) -> tuple[set[str], set[str], set[str]]:
    """Sort every unitary opcode into converted, escaped-leaf, and no-rule.

    The three sets are disjoint and their union is the whole vocabulary, so a
    caller that only inspects one of them cannot report a coverage number the
    other two contradict.
    """

    converted: set[str] = set()
    escaped: set[str] = set()
    unnamed: set[str] = set()
    for opcode in _UNITARY_OPCODES:
        try:
            convert_basis(_single_opcode_program(opcode), gates=gates)
        except BasisConversionError as error:
            message = str(error)
            if message.startswith("decomposition of "):
                escaped.add(opcode)
            else:
                unnamed.add(opcode)
            continue
        converted.add(opcode)
    assert converted.isdisjoint(escaped)
    assert converted.isdisjoint(unnamed)
    assert escaped.isdisjoint(unnamed)
    assert converted | escaped | unnamed == set(_UNITARY_OPCODES)
    return converted, escaped, unnamed


def _up_to_one_phase(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether `left` is `right` times a single unit-modulus complex number.

    A synthesized rewrite is equal to its source only up to one global phase,
    which FlagQuantum IR cannot record, so the phase is divided out rather than
    ignored: the ratio has to be one number of modulus one everywhere, not
    merely somewhere.
    """

    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    support = torch.abs(flat_right) > 1.0e-9
    if not bool(support.any()):
        return bool(torch.allclose(left, right))
    ratios = flat_left[support] / flat_right[support]
    if not bool(torch.allclose(ratios, ratios[0], atol=1.0e-9)):
        return False
    return abs(abs(complex(ratios[0])) - 1.0) < 1.0e-9


# --- the table itself -------------------------------------------------------


def test_every_rule_names_a_declared_unitary_opcode() -> None:
    """A rule the schema does not know is a rule nothing can ever read.

    `basis_translation` consults the table by the opcode an instruction carries,
    and that opcode has already been canonicalised against the operator schema.
    A rule whose key is a channel, a misspelling, or a name outside the schema
    is therefore unreachable, and the conversion it was written for would refuse
    while the table appeared to answer for it.
    """

    assert EQUIVALENCE_RULES
    for opcode, rules in EQUIVALENCE_RULES.items():
        schema = OPERATOR_SCHEMAS.get(opcode)
        assert schema is not None, f"rule for unknown opcode {opcode!r}"
        assert schema.unitary, f"rule for {opcode!r}, which is not a gate"
        assert not schema.channel, f"rule for channel {opcode!r}"
        assert rules, f"rule list for {opcode!r} is empty"


def test_the_rule_table_is_narrower_than_the_opcode_vocabulary() -> None:
    """The row's central claim: the table names 18 of 31 unitary opcodes.

    A table that had grown to the whole vocabulary would make the coverage
    question moot, and one that had shrunk would make the escape branch below
    less reachable. The 13 unnamed names are listed explicitly rather than
    counted, so widening the table is a change to this claim and not a silent
    drift under it.
    """

    assert len(_UNITARY_OPCODES) == 31
    assert len(EQUIVALENCE_RULES) == 18
    assert set(_UNITARY_OPCODES) - set(EQUIVALENCE_RULES) == _UNNAMED_BY_RULES


# --- what the table reaches into a tight basis ------------------------------


def test_clifford_t_reaches_ten_opcodes_and_refuses_twenty_one() -> None:
    """The tight-basis partition, split by the kind of refusal that produced it.

    Eleven of the twenty-one refusals are opcodes the table *does* carry. Their
    rules decompose into gates the basis does not publish, so the loop stops on
    an escaped leaf and names it: `crz` decomposes through `rz`, and Clifford+T
    has no `rz`. Reading coverage off the table alone would credit those eleven
    as reachable and overstate the row by exactly that many opcodes.
    """

    converted, escaped, unnamed = _partition(_CLIFFORD_T)
    assert len(converted) == 10
    assert len(escaped) == 11
    assert len(unnamed) == 10

    assert converted == {
        "cx",
        "cy",
        "cz",
        "h",
        "s",
        "sdg",
        "swap",
        "t",
        "x",
        "z",
    }
    assert escaped == {
        "ccx",
        "cphase",
        "crx",
        "cry",
        "crz",
        "cswap",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rzz",
    }
    assert unnamed == {
        "i",
        "phase",
        "rz",
        "sx",
        "sxdg",
        "tdg",
        "u1",
        "u2",
        "u3",
        "y",
    }


def test_every_escape_is_an_opcode_the_table_carries() -> None:
    """The non-vacuity of the split above: escapes are rules, not missing rules.

    If the escape branch were reachable for an opcode with no rule, the two
    refusal kinds would be describing the same thing and the eleven-opcode
    correction this module exists to state would be meaningless.
    """

    _, escaped, _ = _partition(_CLIFFORD_T)
    assert escaped <= set(EQUIVALENCE_RULES)
    assert escaped  # a split with an empty half states nothing


def test_the_escaped_leaf_is_never_a_gate_the_basis_publishes() -> None:
    """The refusal names the gate the basis was missing, literally.

    The message is the reader's only route from "this program will not compile"
    to "add this gate", so the name it carries has to be a gate the named basis
    genuinely does not hold. `crz` into Clifford+T names `rz`, which is the
    measurement; a message naming `t` would send the caller to a gate they
    already have.
    """

    named: set[str] = set()
    for opcode in sorted(_UNITARY_OPCODES):
        try:
            convert_basis(_single_opcode_program(opcode), gates=_CLIFFORD_T)
        except BasisConversionError as error:
            message = str(error)
            if message.startswith("decomposition of "):
                leaf = message.rsplit(": ", 1)[1].strip().strip("'")
                assert leaf not in _CLIFFORD_T
                named.add(leaf)
            continue
    assert named == {"rz", "tdg"}


# --- what a permissive basis reaches ----------------------------------------


def test_a_rotation_and_an_entangler_reach_every_unitary_opcode() -> None:
    """The other half of the row: 31 of 31, including the 13 no rule names.

    The unnamed opcodes are reached here by the Euler and KAK syntheses rather
    than by a rule, which is why the table's width is not the coverage boundary
    and why the row's `reason` has to state both numbers to be true.
    """

    converted, escaped, unnamed = _partition(_ROTATION_AND_ENTANGLER)
    assert len(converted) == 31
    assert escaped == set()
    assert unnamed == set()
    assert converted == set(_UNITARY_OPCODES)


@pytest.mark.parametrize("opcode", _UNITARY_OPCODES)
def test_every_reached_conversion_is_exact_up_to_one_phase(opcode: str) -> None:
    """Reaching an opcode is not the same as rewriting it correctly.

    A rule or a synthesis path that returned *something* in the basis would pass
    the coverage test above and produce a program that computes a different
    unitary. Each conversion is therefore compared against its source through
    the runtime's own statevector kernel, with the global phase divided out
    because the IR cannot carry one.
    """

    source = _single_opcode_program(opcode)
    result = convert_basis(source, gates=_ROTATION_AND_ENTANGLER)
    assert _up_to_one_phase(get_unitary(source), get_unitary(result.program))
