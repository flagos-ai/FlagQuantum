"""Unit tests for the exact quarter-turn words and the basis search that uses them.

`angle_synthesis` closes one measured hole: a Clifford+T target could not
express a rotation at all, because `basis_translation` holds no identity for
`rz`, `phase`, or `u1`. Two obligations follow, and this module holds both. The
table has to be exact against the runtime's own gate matrices -- every word of
every residue, since the search may pick the fallback word instead of the first
one -- and the production consumer has to accept a rotation it refused before
without changing any answer it already gave.
"""

from __future__ import annotations

import ast
import itertools
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from flagquantum.compiler import angle_synthesis
from flagquantum.compiler.angle_synthesis import (
    QUARTER_TURN_WORDS,
    quarter_turn_words,
    quarter_turns,
)
from flagquantum.compiler.basis_translation import translate
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    legalize_native_gates,
)
from flagquantum.compiler.one_qubit_synthesis import Z_ROTATION_OPCODES
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.errors import CompilationError
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("native:0",))
_SOURCE = FactSource(kind="w506_test", ref="w506-evidence")

#: The whole vocabulary the table may use. Everything here is a named gate a
#: Clifford+T target publishes, and one of the five is a rule source rather than
#: a leaf, which is why a target missing `z` still reaches every residue.
_GENERATORS = ("t", "tdg", "s", "sdg", "z")

#: The stated minimum `t`-gate count of a residue. It is the property that makes
#: these words worth carrying: a target paying for magic states pays this.
_MINIMUM_T_COST = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}


def _rotation(
    opcode: str, theta: object, *, wires: tuple[int, ...] = (0,)
) -> Instruction:
    return Instruction(opcode, wires, params={"theta": theta})


def _matrix(instruction: Instruction) -> torch.Tensor:
    width = 2 ** len(instruction.wires)
    return gate_matrix(
        instruction, bsz=1, device="cpu", dtype=torch.complex128
    ).reshape(-1, width, width)[0]


def _up_to_phase(source: torch.Tensor, other: torch.Tensor) -> complex | None:
    """Return the phase relating two matrices, or None when none relates them.

    A zero entry cannot supply the phase, so the largest entry of the second
    matrix does; dividing by a zero would produce a nan and report every gate as
    a mismatch, which is the trap this helper exists to avoid.
    """

    flat_source, flat_other = source.reshape(-1), other.reshape(-1)
    index = int(torch.argmax(flat_other.abs()))
    if not torch.isclose(
        flat_other[index].abs(), torch.tensor(1.0, dtype=torch.float64)
    ):
        return None
    phase = flat_source[index] / flat_other[index]
    return (
        complex(phase) if torch.allclose(source, phase * other, atol=1.0e-12) else None
    )


def _word_matrix(word: tuple[str, ...]) -> torch.Tensor:
    total = torch.eye(2, dtype=torch.complex128)
    for opcode in word:
        total = _matrix(Instruction(opcode, (0,))) @ total
    return total


def _can_run(*opcodes: str):
    return lambda item: item.name in set(opcodes)


def _gate(name: str) -> dict[str, object]:
    schema = OPERATOR_SCHEMAS[name]
    return {"name": name, "parameters": tuple(schema.parameters)}


def _snapshot(native_gates: tuple[str, ...]) -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="w506-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="w506-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=tuple(_gate(name) for name in native_gates),
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=_SOURCE,
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="w506-evidence",
                sha256="d" * 64,
                level=EvidenceLevel.BASIC,
                scope=_SCOPE,
            ),
        ),
    )


def _legalize(program: CircuitIR, *opcodes: str):
    return legalize_native_gates(
        program, snapshot=_snapshot(opcodes), evaluated_at=_NOW
    )


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


def test_the_table_covers_the_eight_residues_and_nothing_else() -> None:
    # Non-vacuity: every check below iterates the table, so it has to have
    # exactly the residues `rz(n * pi/4)` can take and no residue that is not
    # one of the eight.
    assert sorted(QUARTER_TURN_WORDS) == list(range(8))
    assert all(QUARTER_TURN_WORDS[residue] for residue in QUARTER_TURN_WORDS)


def test_the_words_name_only_plain_named_gates() -> None:
    """The table may not name a rotation, a channel, or an unknown opcode."""

    for residue, words in QUARTER_TURN_WORDS.items():
        for word in words:
            assert set(word) <= set(_GENERATORS), (residue, word)
            for opcode in word:
                schema = OPERATOR_SCHEMAS[opcode]
                assert schema.unitary and schema.arity == 1
                assert not schema.channel
                assert not schema.parameters


def test_the_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        QUARTER_TURN_WORDS[0] = ()  # type: ignore[index]


@pytest.mark.parametrize("residue", range(8))
def test_every_word_reproduces_its_rotation_up_to_one_global_phase(
    residue: int,
) -> None:
    """Both words of a residue, because the search may pick either one."""

    target = _matrix(_rotation("rz", residue * math.pi / 4))
    assert _up_to_phase(target, torch.eye(2, dtype=torch.complex128)) is not None or (
        residue != 0
    )
    for word in QUARTER_TURN_WORDS[residue]:
        phase = _up_to_phase(target, _word_matrix(word))
        assert phase is not None, (residue, word)
        # The phase is a power of `omega = exp(1j * pi/4)`, which is the same
        # class of global phase the rest of Compiler already drops: every
        # single-qubit Clifford+T operator has a determinant that is one.
        assert abs(abs(phase) - 1.0) < 1.0e-12


@pytest.mark.parametrize("opcode", ["phase", "u1"])
def test_the_same_words_serve_phase_and_u1(opcode: str) -> None:
    """`phase`, `u1`, and `rz` are one gate up to a global phase.

    They are also one opcode family in `one_qubit_synthesis`, which delegates
    all three to the same Euler triple. Pinning that here is what keeps a
    target that publishes only `u1` from being told its rotations are
    unreachable.
    """

    for residue in range(8):
        target = _matrix(_rotation(opcode, residue * math.pi / 4))
        for word in QUARTER_TURN_WORDS[residue]:
            assert _up_to_phase(target, _word_matrix(word)) is not None, (
                opcode,
                residue,
                word,
            )


def test_the_first_word_of_a_residue_is_shortest_and_carries_the_fewest_t_gates() -> (
    None
):
    """Re-derive the table's ordering claim by exhaustive enumeration.

    The search ranks candidates by instruction count before it ranks them by
    anything else, so the first word of a residue has to be a shortest word over
    the vocabulary; and among the shortest words it has to be one of those
    carrying the minimum number of `t` gates, which is the property a
    fault-tolerant target cares about. Words longer than three gates are not
    enumerated: every residue has a word of length at most two, so a minimum
    over words of length at most three is a minimum over all words.
    """

    reachable: dict[int, list[tuple[str, ...]]] = {residue: [] for residue in range(8)}
    for length in range(4):
        for word in itertools.product(_GENERATORS, repeat=length):
            matrix = _word_matrix(word)
            for residue in range(8):
                target = _matrix(_rotation("rz", residue * math.pi / 4))
                if _up_to_phase(target, matrix) is not None:
                    reachable[residue].append(word)

    for residue, words in reachable.items():
        assert words, residue
        shortest = min(len(word) for word in words)
        primary = QUARTER_TURN_WORDS[residue][0]
        assert len(primary) == shortest, (residue, primary)
        fewest_t = min(
            sum(1 for opcode in word if opcode in ("t", "tdg"))
            for word in words
            if len(word) == shortest
        )
        assert sum(1 for opcode in primary if opcode in ("t", "tdg")) == fewest_t
        assert fewest_t == _MINIMUM_T_COST[residue], residue


@pytest.mark.parametrize(
    "basis", [None, *_GENERATORS], ids=lambda basis: basis or "full"
)
def test_every_residue_stays_reachable_when_one_generator_is_withdrawn(
    basis: str | None,
) -> None:
    """A target publishing one of an inverse pair still reaches all eight.

    Without the fallback word this is exactly what breaks: `{"t", "s"}` is not a
    group and misses residue seven, and a target that publishes `t` but not
    `tdg` cannot express `rz(-pi/4)` with the primary word alone.
    """

    published = set(_GENERATORS) if basis is None else set(_GENERATORS) - {basis}
    for residue, words in QUARTER_TURN_WORDS.items():
        assert any(set(word) <= published for word in words), (residue, published)


# --------------------------------------------------------------------------
# Recognizing a quarter turn
# --------------------------------------------------------------------------


@pytest.mark.parametrize("turns", range(-64, 65))
def test_a_quarter_turn_is_recognized_exactly(turns: int) -> None:
    assert quarter_turns(turns * math.pi / 4) == turns % 8


@pytest.mark.parametrize("angle", [math.pi, 2 * math.pi, -2 * math.pi, 0.0, -0.0])
def test_the_wrap_around_is_a_quarter_turn(angle: float) -> None:
    assert quarter_turns(angle) is not None


@pytest.mark.parametrize(
    "angle",
    [
        0.3,
        -0.3,
        math.pi / 8,
        math.pi / 3,
        1.0e-15,
        math.pi / 4 + 1.0e-9,
        math.pi / 4 * (1.0 + 1.0e-15),
        math.nextafter(math.pi / 4, math.inf),
        math.nextafter(math.pi / 4, -math.inf),
    ],
)
def test_a_near_miss_is_refused(angle: float) -> None:
    """An angle that is not an exact multiple of `pi/4` has no word here.

    A tolerance would answer every one of these with a rotation nobody asked
    for. Approximating an arbitrary angle is the Ross-Selinger problem, which
    needs exact arithmetic in `Z[omega, 1/sqrt(2)]` and integer factorization;
    the parity contract records that gap instead of smoothing it over.
    """

    assert quarter_turns(angle) is None
    assert quarter_turn_words(_rotation("rz", angle)) == ()
    assert translate(_rotation("rz", angle), can_run=_can_run("h", "s", "t")) is None


@pytest.mark.parametrize(
    "angle", [True, False, "pi/4", None, [math.pi / 4], complex(1.0, 0.0)]
)
def test_a_value_that_is_not_a_real_angle_is_refused(angle: object) -> None:
    """The check is defensive; the IR already refuses most of these.

    `Instruction` rejects a bool, a string, and a complex parameter outright, so
    the value never reaches this module through the front door. `quarter_turns`
    is still called by nothing that guarantees that, and answering a list or a
    complex with a rotation would be worse than refusing it.
    """

    assert quarter_turns(angle) is None


def test_a_trainable_angle_is_never_synthesized() -> None:
    """The words drop the parameter, so a trainable one must not select them.

    `theta` initialized at `pi/4` happens to be a quarter turn, and answering
    with `t` would detach the rotation from the autograd graph. `_real_angle`
    refuses for the same reason `one_qubit_synthesis` refuses to let a trainable
    value select a short Euler form.

    The tensor is `float64`, and that is load-bearing rather than incidental.
    `torch.tensor(pi/4)` is a `float32` and reads back as `0.7853981852531433`,
    which this module refuses on the exactness check alone -- a float32 tensor
    would pass this test with the trainable guard deleted, which is the opposite
    of what the test is for. The same value without `requires_grad` is asserted
    first, so a failure below is about the guard rather than about the angle.
    """

    constant = torch.tensor(math.pi / 4, dtype=torch.float64)
    assert quarter_turns(constant) == 1
    theta = constant.clone().requires_grad_(True)
    assert theta.item() == math.pi / 4
    assert quarter_turns(theta) is None
    assert quarter_turn_words(_rotation("rz", theta)) == ()
    assert translate(_rotation("rz", theta), can_run=_can_run("h", "s", "t")) is None


def test_the_z_rotation_vocabulary_the_producer_gates_on_is_one_wire_and_unitary() -> (
    None
):
    """The premise the producer's own schema re-check rests on, asserted.

    `quarter_turn_words` admits an opcode by membership in `Z_ROTATION_OPCODES`
    and then re-checks the schema for unitarity and arity one. That second check
    cannot change any answer while every member of the vocabulary already is a
    one-wire unitary, which is why no test can kill its removal: it is a guard
    against a later widening of the vocabulary rather than a live condition. The
    premise is what is asserted here, so widening the tuple without widening the
    table is a failure in this file rather than a wrong-arity word at runtime.
    """

    assert Z_ROTATION_OPCODES == ("rz", "phase", "u1")
    for opcode in Z_ROTATION_OPCODES:
        schema = OPERATOR_SCHEMAS[opcode]
        assert schema.unitary and schema.arity == 1, opcode
        assert not schema.channel, opcode


@pytest.mark.parametrize("opcode", ["x", "h", "t", "sdg", "i", "rx", "u3", "cx", "rzz"])
def test_an_opcode_that_is_not_a_z_rotation_has_no_words(opcode: str) -> None:
    schema = OPERATOR_SCHEMAS[opcode]
    params = dict.fromkeys(schema.parameters, math.pi / 4)
    instruction = Instruction(opcode, tuple(range(schema.arity)), params=params)
    assert quarter_turn_words(instruction) == ()


def test_the_words_carry_the_source_wire_and_metadata() -> None:
    source = Instruction(
        "rz", (2,), params={"theta": math.pi / 4}, metadata={"layer": 3}
    )
    assert quarter_turn_words(source) == (
        (Instruction("t", (2,), metadata={"layer": 3}),),
        (
            Instruction("s", (2,), metadata={"layer": 3}),
            Instruction("tdg", (2,), metadata={"layer": 3}),
        ),
    )


def test_the_module_publishes_exactly_three_names() -> None:
    assert angle_synthesis.__all__ == (
        "QUARTER_TURN_WORDS",
        "quarter_turn_words",
        "quarter_turns",
    )


def test_the_module_does_not_reach_for_the_simulation_layer() -> None:
    """The Compiler boundary, held by the module that could most easily break it.

    A gate matrix lives in `flagquantum.simulation`, which Compiler may not
    import at runtime. This module answers from a literal table instead, and the
    obligation to verify that table lives in this test file. An empty import
    list is not enough: `flagquantum.simulation` is reachable as `from ...
    simulation`, so every import statement's module path is checked.
    """

    source = Path(angle_synthesis.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("simulation" not in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "simulation" not in (node.module or "")


# --------------------------------------------------------------------------
# The search, and the target it now reaches
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        (0, []),
        (1, ["t"]),
        (2, ["s"]),
        (3, ["s", "t"]),
        (4, ["s", "s"]),
        (5, ["t", "s", "s"]),
        (6, ["s", "s", "s"]),
        (7, ["s", "s", "s", "t"]),
    ],
)
def test_a_clifford_t_target_reaches_a_quarter_turn_rotation(
    turns: int, expected: list[str]
) -> None:
    """The decisive scenario: this returned None before the table existed.

    The basis here is the one a real Clifford+T target publishes and no more:
    `h`, `s`, `t`, and `cx`, with no `sdg`, no `tdg`, and no `z`. So a residue
    whose shortest word names an inverse gate is reached through the `z` and
    `sdg` identities instead -- `rz(-pi/4)` is `s s s t`, not `tdg`. The search
    ranks a fully native rewrite above a shorter one that names a gate the
    target lacks, which is the same rule it already applied to `x` and `cphase`.
    """

    leaves = translate(
        _rotation("rz", turns * math.pi / 4), can_run=_can_run("h", "s", "t", "cx")
    )
    assert leaves is not None
    assert [leaf.name for leaf in leaves] == expected


@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        (0, []),
        (1, ["t"]),
        (2, ["s"]),
        (3, ["s", "t"]),
        (4, ["s", "s"]),
        (5, ["sdg", "tdg"]),
        (6, ["sdg"]),
        (7, ["tdg"]),
    ],
)
def test_a_target_with_the_inverse_gates_takes_the_shortest_word(
    turns: int, expected: list[str]
) -> None:
    """With `sdg` and `tdg` published, every residue is its shortest word."""

    leaves = translate(
        _rotation("rz", turns * math.pi / 4),
        can_run=_can_run("h", "s", "sdg", "t", "tdg", "cx"),
    )
    assert leaves is not None
    assert [leaf.name for leaf in leaves] == expected


def test_the_identity_rotation_is_removed_rather_than_left_in_place() -> None:
    """Residue zero is the identity, so the answer is the empty replacement."""

    leaves = translate(
        _rotation("rz", 2 * math.pi), can_run=_can_run("h", "s", "t", "cx")
    )
    assert leaves == ()


def test_a_target_without_z_reaches_the_word_through_the_z_identity() -> None:
    """`z` is not a leaf: the table's own rules rewrite it into `s` pairs."""

    without_z = _can_run("h", "s", "t", "sdg", "tdg", "cx")
    leaves = translate(_rotation("rz", math.pi), can_run=without_z)
    assert leaves is not None
    assert [leaf.name for leaf in leaves] == ["s", "s"]


@pytest.mark.parametrize("turns", range(8))
def test_a_native_z_rotation_is_never_displaced_by_the_table(turns: int) -> None:
    """The caller keeps a gate the target already has, at every residue."""

    source = _rotation("rz", turns * math.pi / 4)
    assert translate(source, can_run=_can_run("rz", "sx", "cx")) is None


def test_a_rule_that_reaches_a_quarter_turn_now_reaches_the_target() -> None:
    """`rzz(pi/4)` is `cx` around a `t`, which no rule could say before."""

    leaves = translate(
        Instruction("rzz", (0, 1), params={"theta": math.pi / 4}),
        can_run=_can_run("h", "s", "t", "cx"),
    )
    assert leaves is not None
    assert [leaf.name for leaf in leaves] == ["cx", "t", "cx"]


# --------------------------------------------------------------------------
# The production consumer
# --------------------------------------------------------------------------


def test_the_legalizer_decomposes_a_quarter_turn_onto_a_clifford_t_target() -> None:
    """`legalize_native_gates` refused this program before the table existed."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            Instruction("h", (0,)),
            _rotation("rz", math.pi / 4),
            Instruction("cx", (0, 1)),
            _rotation("rz", 3 * math.pi / 4, wires=(1,)),
            _rotation("rz", math.pi, wires=(1,)),
        ),
    )
    result = _legalize(program, "h", "s", "t", "sdg", "tdg", "cx")
    assert [leaf.name for leaf in result.program.instructions] == [
        "h",
        "t",
        "cx",
        "s",
        "t",
        "s",
        "s",
    ]
    assert [record.replacement_opcodes for record in result.decompositions] == [
        ("t",),
        ("s", "t"),
        ("s", "s"),
    ]
    assert [record.source_opcode for record in result.decompositions] == ["rz"] * 3
    assert result.changed


def test_the_legalizer_drops_an_identity_rotation_instead_of_keeping_it() -> None:
    program = CircuitIR(
        n_wires=1,
        instructions=(Instruction("h", (0,)), _rotation("rz", 2 * math.pi)),
    )
    result = _legalize(program, "h", "s", "t", "cx")
    assert [leaf.name for leaf in result.program.instructions] == ["h"]


def test_the_legalizer_still_refuses_a_rotation_that_is_not_a_quarter_turn() -> None:
    """Fail closed, and name the gate the target is actually missing."""

    program = CircuitIR(n_wires=1, instructions=(_rotation("rz", 0.3),))
    with pytest.raises(NativeGateLegalizationError, match="'rz'"):
        _legalize(program, "h", "s", "t", "cx")
    assert issubclass(NativeGateLegalizationError, CompilationError)
