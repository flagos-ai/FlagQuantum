"""Unit tests for the named-basis conversion path.

`basis_conversion` takes the per-instruction rewrite loop that used to live
inside `native_gate_legalization` and puts it behind a second caller that has no
device in it: convert this program from the basis it is written in into a basis
the caller names. Three obligations follow, and this module holds all three.
The loop has to behave identically for the device path, which the legalization
suite covers; the named basis has to be validated as a gate set before anything
is touched; and a conversion has to be a *proof* rather than a claim, so every
converted program is compared against its source through the runtime's own
statevector kernel up to a single global phase.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Collection

import pytest
import torch

from flagquantum.compiler.basis_conversion import (
    REFUSAL_KINDS,
    BasisConversionError,
    BasisConversionResult,
    convert_basis,
    convert_instructions,
    entangler_opcode,
    half_pi_pulse_opcode,
    require_basis,
    z_rotation_opcode,
)
from flagquantum.compiler.basis_translation import (
    EQUIVALENCE_RULES,
    with_equivalence_rule,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.errors import CompilationError
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

_CLIFFORD_T = ("h", "s", "t", "cx")
_VOCABULARY = frozenset({"t", "tdg", "s", "sdg", "z"})


def _program(*instructions: Instruction, n_wires: int = 2) -> CircuitIR:
    return CircuitIR(n_wires, instructions, dtype="complex128")


def _gate(opcode: str, wires: tuple[int, ...], **params: object) -> Instruction:
    return Instruction(opcode, wires, params=dict(params))


def _rotation(theta: float) -> Instruction:
    return Instruction("rz", (0,), params={"theta": theta})


def _names(program: CircuitIR) -> list[str]:
    return [instruction.name for instruction in program.instructions]


def _up_to_phase(left: torch.Tensor, right: torch.Tensor) -> bool:
    """True when `left` equals `right` times one unit-modulus complex number.

    A synthesized rewrite is equal to its source only up to one global phase,
    which FlagQuantum IR cannot record, so the phase is divided out rather than
    ignored: the ratio of the two matrices has to be a single number of modulus
    one everywhere, not merely somewhere.
    """

    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    magnitude = torch.abs(flat_right)
    support = magnitude > 1.0e-9
    if not bool(support.any()):
        return bool(torch.allclose(left, right))
    ratios = flat_left[support] / flat_right[support]
    scale = ratios[0]
    if abs(complex(scale)) < 1.0e-9:
        return False
    return bool(torch.allclose(left, right * scale, atol=1.0e-9, rtol=0.0))


def _assert_converts_exactly(
    source: CircuitIR, target: tuple[str, ...], rules=EQUIVALENCE_RULES
) -> None:
    """Convert `source`, then prove the result is the same gate up to a phase."""

    result = convert_basis(source, gates=target, rules=rules)
    for instruction in result.program.instructions:
        assert instruction.name in target, instruction.name
    assert _up_to_phase(
        get_unitary(result.program), get_unitary(result.source_program)
    ), _names(result.program)
    assert result.result_content_hash == result.program.content_hash
    assert result.source_content_hash == result.source_program.content_hash


# --- the named basis is validated as a gate set ------------------------------


def test_a_basis_is_canonicalised_deduplicated_and_sorted() -> None:
    """An alias and a spacing mistake are the caller's, and both are absorbed."""

    assert require_basis(["CX", " h ", "cnot", "H"]) == ("cx", "h")


def test_a_basis_of_many_spellings_comes_back_canonical_sorted_and_unique() -> None:
    """Sorted, not merely deduplicated: nine names, one canonical spelling each.

    The larger basis is deliberate. Deduplication and canonicalisation are
    observable from a two-name basis, but *sorting* is not: the answer has to be
    a fixed sequence and not the iteration order of whatever container the
    implementation collected the names in, and a two-element container can
    iterate in the sorted order by accident. Nine distinct names make the
    sequence itself the assertion.
    """

    assert require_basis(
        [
            "CX",
            "h",
            "cnot",
            "H",
            "sd",
            "td",
            "p",
            "u",
            "ccnot",
            "toffoli",
            "fredkin",
            "hadamard",
            "id",
        ]
    ) == ("ccx", "cswap", "cx", "h", "i", "phase", "sdg", "tdg", "u3")


def test_a_basis_is_sorted_regardless_of_the_order_it_was_given_in() -> None:
    assert require_basis(["rz", "cx", "h"]) == require_basis(["h", "rz", "cx"])


def test_a_bare_string_is_refused_rather_than_iterated() -> None:
    """`"hcx"` would otherwise become a three-gate basis built from letters."""

    with pytest.raises(BasisConversionError, match="not a string"):
        require_basis("hcx")


def test_bytes_are_refused_like_a_string() -> None:
    with pytest.raises(BasisConversionError, match="not a string"):
        require_basis(b"hcx")


@pytest.mark.parametrize("entry", ["", "   ", 7, None, ["h"]])
def test_a_non_gate_name_is_refused(entry: object) -> None:
    with pytest.raises(BasisConversionError, match="non-empty gate name"):
        require_basis([entry])


def test_an_unknown_opcode_is_named_in_the_refusal() -> None:
    with pytest.raises(BasisConversionError, match="'frobnicate'"):
        require_basis(["h", "frobnicate"])


def test_every_unknown_opcode_is_named_in_one_message() -> None:
    """A caller fixing a basis wants the whole list, not the first mistake."""

    with pytest.raises(BasisConversionError) as raised:
        require_basis(["beta", "h", "alpha"])
    text = str(raised.value)
    assert "'alpha'" in text and "'beta'" in text
    assert text.index("'alpha'") < text.index("'beta'")


@pytest.mark.parametrize(
    "opcode", ["bit_flip", "phase_flip", "depolarizing", "amplitude_damping"]
)
def test_a_channel_is_refused_because_a_basis_holds_gates(opcode: str) -> None:
    with pytest.raises(BasisConversionError, match="are not gates"):
        require_basis(["h", opcode])


@pytest.mark.parametrize("opcode", ["measure", "reset", "mcx"])
def test_an_opcode_the_ir_does_not_declare_is_unknown(opcode: str) -> None:
    with pytest.raises(BasisConversionError, match="unknown opcode"):
        require_basis([opcode])


def test_an_empty_basis_is_refused() -> None:
    with pytest.raises(BasisConversionError, match="at least one gate"):
        require_basis([])


def test_a_phase_gate_is_a_gate_even_though_it_is_not_clifford_t() -> None:
    assert require_basis(["phase"]) == ("phase",)


def test_the_schema_table_makes_the_three_refusal_clauses_one_clause() -> None:
    """Pins the premise that three separate disjuncts refuse the same opcodes.

    `require_basis` refuses an entry when it is a channel, or is not unitary, or
    spans no wire. In the table as it stands those three tests select the same
    entries -- no declared channel is unitary, and no declared opcode spans zero
    wires -- so each clause is individually redundant and only their union is
    observable. Asserting that here means a future schema that separates them
    fails this test instead of silently narrowing the refusal.
    """

    unitary_channels = [
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.channel and schema.unitary
    ]
    zero_wire = [name for name, schema in OPERATOR_SCHEMAS.items() if schema.arity < 1]
    assert unitary_channels == []
    assert zero_wire == []


# --- the three synthesis bases are a fixed function of the published set -----


def test_the_z_rotation_follows_the_vocabulary_order_not_the_published_set() -> None:
    """Three spellings of one gate are published, and the answer is pinned."""

    assert z_rotation_opcode({"phase", "rz", "u1"}) == "rz"
    assert z_rotation_opcode({"u1", "phase"}) == "phase"
    assert z_rotation_opcode({"u1"}) == "u1"


def test_a_basis_without_a_z_rotation_has_none() -> None:
    assert z_rotation_opcode({"h", "cx", "t"}) is None


def test_the_half_pi_pulse_prefers_sx_over_rx() -> None:
    assert half_pi_pulse_opcode({"rx", "sx"}) == "sx"
    assert half_pi_pulse_opcode({"rx"}) == "rx"


def test_a_basis_without_a_half_pi_pulse_has_none() -> None:
    assert half_pi_pulse_opcode({"h", "cx"}) is None


def test_the_entangler_prefers_a_parameter_free_spelling() -> None:
    """`cx` and `rzz` are both supercontrolled, and `cx` needs no angle."""

    assert entangler_opcode({"rzz", "cx"}) == "cx"
    assert entangler_opcode({"rzz", "ryy"}) == "rzz"
    assert entangler_opcode({"ryy"}) == "ryy"


def test_a_basis_without_an_entangler_has_none() -> None:
    assert entangler_opcode({"h", "t", "rz"}) is None


@pytest.mark.parametrize(
    "helper", [z_rotation_opcode, half_pi_pulse_opcode, entangler_opcode]
)
def test_the_three_bases_do_not_depend_on_iteration_order(
    helper: Callable[[Collection[str]], str | None],
) -> None:
    """A set and a reversed list publish the same gates and answer the same."""

    published = ["u1", "rz", "rx", "sx", "ryy", "rzz", "cx"]
    assert helper(set(published)) == helper(list(reversed(published)))


# --- a conversion is exact and reported --------------------------------------


def test_a_program_already_in_the_basis_is_returned_unrewritten() -> None:
    """No rewrite means the very same program object, not a copy of it."""

    source = _program(_gate("h", (0,)), _gate("cx", (0, 1)))
    result = convert_basis(source, gates=_CLIFFORD_T)
    assert result.changed is False
    assert result.decompositions == ()
    assert result.program is source
    assert result.conversion_identity


def test_a_two_wire_gate_is_rewritten_into_the_named_basis() -> None:
    result = convert_basis(_program(_gate("cx", (0, 1))), gates=("h", "cz"))
    assert result.changed is True
    assert _names(result.program) == ["h", "cz", "h"]
    assert len(result.decompositions) == 1
    record = result.decompositions[0]
    assert (record.instruction_index, record.source_opcode) == (0, "cx")
    assert record.replacement_opcodes == ("h", "cz", "h")


def test_the_report_carries_both_bases() -> None:
    result = convert_basis(
        _program(_gate("h", (0,)), _gate("cx", (0, 1))), gates=("cz", "h")
    )
    assert result.source_basis == ("cx", "h")
    assert result.target_basis == ("cz", "h")


def test_the_index_of_a_rewritten_instruction_is_the_source_position() -> None:
    result = convert_basis(
        _program(_gate("h", (0,)), _gate("swap", (0, 1)), _gate("x", (1,))),
        gates=("h", "cx", "s", "t", "tdg"),
    )
    assert [item.instruction_index for item in result.decompositions] == [1, 2]
    assert [item.source_opcode for item in result.decompositions] == ["swap", "x"]


def test_a_cx_into_clifford_t_is_the_identity_rewrite_of_one_gate() -> None:
    """`cx` is in the basis, so nothing is rewritten and nothing is claimed."""

    source = _program(_gate("h", (0,)), _gate("cx", (0, 1)))
    result = convert_basis(source, gates=_CLIFFORD_T)
    assert result.program is source
    assert result.decompositions == ()


def test_an_rzz_into_clifford_t_carries_no_rotation() -> None:
    """The rotation the equivalence table cannot reach is refused, not faked."""

    with pytest.raises(BasisConversionError, match="'rzz'"):
        convert_basis(_program(_gate("rzz", (0, 1), theta=0.7137)), gates=_CLIFFORD_T)


# --- the quarter-turn table is reachable from a named basis ------------------


@pytest.mark.parametrize("turns", [1, 2, 3, 4, 5, 6, 7])
def test_a_quarter_turn_rotation_reaches_a_clifford_t_basis(turns: int) -> None:
    """Kills the mutant that drops the words from the search.

    The parity gap this closes is not observable from inside
    `angle_synthesis`: what matters is that a caller who names a Clifford+T
    basis gets a program *in* that basis. Each residue is converted and compared
    against the source through the runtime's own kernel.
    """

    _assert_converts_exactly(_program(_rotation(turns * math.pi / 4)), _CLIFFORD_T)


def test_a_native_rotation_is_left_alone_when_the_basis_publishes_it() -> None:
    source = _program(_rotation(math.pi / 4))
    result = convert_basis(source, gates=("h", "s", "t", "rz"))
    assert result.program is source
    assert result.decompositions == ()


def test_a_rotation_that_is_not_a_quarter_turn_fails_closed() -> None:
    """Nothing is approximated on the way into a named basis either."""

    with pytest.raises(BasisConversionError, match="'rz'"):
        convert_basis(_program(_rotation(0.7137)), gates=_CLIFFORD_T)


def test_the_words_reach_a_basis_that_publishes_one_of_an_inverse_pair() -> None:
    """The fallback word is what makes this conversion possible at all."""

    _assert_converts_exactly(
        _program(_rotation(5 * math.pi / 4)), ("h", "s", "t", "cx")
    )


# --- a matrix instruction goes through the same syntheses as a device --------


def test_a_custom_matrix_becomes_the_basis_it_declares() -> None:
    # `i X`: unitary, and not a diagonal, so the Euler form is not trivially
    # the identity the way an `i Z` would be.
    matrix = torch.tensor([[0.0, 1.0j], [1.0j, 0.0]], dtype=torch.complex128)
    instruction = Instruction("custom", (0,), matrix=matrix)
    source = _program(instruction)
    result = convert_basis(source, gates=("rz", "sx", "h", "cx"))
    assert result.changed is True
    assert result.decompositions[0].source_opcode == "custom"
    assert _up_to_phase(get_unitary(result.program), get_unitary(source))


def test_a_two_wire_matrix_goes_through_kak_like_a_device_target() -> None:
    """The entangler branch of the synthesis is the same one a device uses."""

    swap = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=torch.complex128,
    )
    source = _program(Instruction("custom", (0, 1), matrix=swap))
    result = convert_basis(source, gates=("rz", "sx", "h", "cx"))
    assert result.changed is True
    assert result.decompositions[0].source_opcode == "custom"
    assert _up_to_phase(get_unitary(result.program), get_unitary(source))


def test_a_custom_matrix_with_no_half_pi_pulse_in_the_basis_is_refused() -> None:
    """A z-rotation alone is not an Euler basis, and the refusal says so."""

    matrix = torch.tensor([[0.0, 1.0j], [1.0j, 0.0]], dtype=torch.complex128)
    with pytest.raises(BasisConversionError, match="has no named-basis contract"):
        convert_basis(
            _program(Instruction("custom", (0,), matrix=matrix)),
            gates=("rz", "h", "cx"),
        )


def test_a_custom_matrix_with_no_z_rotation_in_the_basis_is_refused() -> None:
    matrix = torch.eye(2, dtype=torch.complex128)
    with pytest.raises(BasisConversionError, match="has no named-basis contract"):
        convert_basis(
            _program(Instruction("custom", (0,), matrix=matrix)), gates=("h", "cx")
        )


# --- the conversion report is tamper-evident ---------------------------------


def test_a_result_that_does_not_match_its_source_hash_is_refused() -> None:
    result = convert_basis(_program(_gate("cx", (0, 1))), gates=("h", "cz"))
    with pytest.raises(ValueError, match="does not match source_content_hash"):
        dataclasses.replace(result, source_content_hash="0" * 64)


def test_a_result_that_does_not_match_its_own_hash_is_refused() -> None:
    result = convert_basis(_program(_gate("cx", (0, 1))), gates=("h", "cz"))
    with pytest.raises(ValueError, match="does not match result_content_hash"):
        dataclasses.replace(result, result_content_hash="0" * 64)


def test_the_conversion_identity_is_the_recorded_golden_value() -> None:
    """A golden value, because the identity is a contract and not a convenience.

    The identity is what a caller stores to say which conversion produced a
    program, so its definition has to be pinned rather than merely exercised: a
    payload that quietly stopped covering the target basis, or the source, or
    the result, would still yield two different identities for two different
    inputs and would still pass a test that only compares them with each other.
    This value therefore has to be recomputed deliberately if the payload ever
    changes, which is the point.
    """

    source = _program(_gate("h", (0,)), _gate("cx", (0, 1)), _gate("cx", (0, 1)))
    result = convert_basis(source, gates=("h", "cz"))
    assert result.source_content_hash == (
        "7def2b4b589cc2bb36681c1ebbb1bbd63abc077b5f37da2c3a5537f3560d92b6"
    )
    assert result.result_content_hash == (
        "e817d595783ab0359075e29e9187859cfdf6a6a0e0712d4d7ea52a9c20c4ac3d"
    )
    assert result.conversion_identity == (
        "4ce350977df840cc787d83001c666e146d3e7f31fc602443a861bccef4caf494"
    )
    assert result.target_basis == ("cz", "h")
    assert [item.instruction_index for item in result.decompositions] == [1, 2]


def test_the_conversion_identity_moves_with_the_target_basis() -> None:
    """Two bases that rewrite the same way still convert differently.

    `('h', 'cz')` and `('h', 'cz', 't')` produce the *same* result program for
    this source, because `t` is never needed, so only the target basis
    distinguishes the two conversions.
    """

    source = _program(_gate("cx", (0, 1)))
    left = convert_basis(source, gates=("h", "cz"))
    right = convert_basis(source, gates=("h", "cz", "t"))
    assert [item.name for item in left.program.instructions] == [
        item.name for item in right.program.instructions
    ]
    assert left.result_content_hash == right.result_content_hash
    assert left.conversion_identity != right.conversion_identity


def test_the_conversion_identity_moves_with_the_program() -> None:
    one = convert_basis(_program(_gate("cx", (0, 1))), gates=("h", "cz"))
    two = convert_basis(
        _program(_gate("cx", (0, 1)), _gate("h", (0,))), gates=("h", "cz")
    )
    assert one.decompositions == two.decompositions
    assert one.conversion_identity != two.conversion_identity


def test_the_conversion_identity_is_deterministic() -> None:
    source = _program(_gate("cx", (0, 1)))
    assert (
        convert_basis(source, gates=("h", "cz")).conversion_identity
        == convert_basis(source, gates=("h", "cz")).conversion_identity
    )


# --- the bound on growth ------------------------------------------------------


def test_the_added_operation_bound_is_enforced() -> None:
    with pytest.raises(
        BasisConversionError,
        match="named-basis decomposition exceeds max_added_operations",
    ):
        convert_basis(
            _program(_gate("ccx", (0, 1, 2)), n_wires=3),
            gates=("h", "cx", "t", "tdg"),
            max_added_operations=1,
        )


def test_the_added_operation_bound_does_not_fire_when_it_is_not_exceeded() -> None:
    result = convert_basis(
        _program(_gate("cx", (0, 1))), gates=("h", "cz"), max_added_operations=2
    )
    assert result.changed is True


def test_a_bool_is_not_an_operation_bound() -> None:
    """`True` is an `int` in Python and would silently mean a bound of one."""

    with pytest.raises(TypeError, match="max_added_operations must be an integer"):
        convert_basis(
            _program(_gate("cx", (0, 1))), gates=("h", "cz"), max_added_operations=True
        )


def test_a_negative_operation_bound_is_refused() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        convert_basis(
            _program(_gate("cx", (0, 1))), gates=("h", "cz"), max_added_operations=-1
        )


# --- the error type and the shared vocabulary --------------------------------


def test_the_refusal_is_a_compilation_error() -> None:
    assert issubclass(BasisConversionError, CompilationError)
    assert issubclass(BasisConversionError, RuntimeError)


def test_the_result_type_is_frozen() -> None:
    result = convert_basis(_program(_gate("cx", (0, 1))), gates=("h", "cz"))
    assert isinstance(result, BasisConversionResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.program = result.source_program  # type: ignore[misc]


def test_the_refusal_vocabulary_has_exactly_the_four_kinds_the_loop_raises() -> None:
    """The loop validates its own table, and this pins the four names."""

    assert REFUSAL_KINDS == ("matrix", "no_rule", "escapes", "budget")


def test_a_refusal_table_missing_a_kind_is_refused_before_any_program_is_touched() -> (
    None
):
    """A half-filled table would mean a TypeError from inside the loop instead."""

    with pytest.raises(ValueError, match="refusals must have exactly the keys"):
        convert_instructions(
            _program(_gate("h", (0,))),
            published={"h"},
            can_run=lambda item: item.name in {"h"},
            refusals={"matrix": "no"},
            error=BasisConversionError,
        )


def test_a_program_written_in_the_target_basis_is_a_fixed_point() -> None:
    """Converting twice changes nothing the second time."""

    once = convert_basis(_program(_gate("swap", (0, 1))), gates=("h", "cx"))
    twice = convert_basis(once.program, gates=("h", "cx"))
    assert twice.changed is False
    assert twice.program is once.program


# --- a registered rule reaches a conversion, without widening the basis -----


def _tdg_to_x_t_x(instruction: Instruction) -> tuple[Instruction, ...]:
    """`Tdg` is `X T X` up to one global phase, the phase this IR cannot hold."""

    return tuple(
        Instruction(name, instruction.wires, metadata=dict(instruction.metadata))
        for name in ("x", "t", "x")
    )


def test_a_registered_rule_makes_an_unreachable_opcode_convertible() -> None:
    """Clifford+T cannot express `tdg`, until a caller verifies and registers one."""

    source = _program(_gate("tdg", (0,)))
    basis = ("h", "s", "t", "cx")
    with pytest.raises(BasisConversionError, match="no verified decomposition"):
        convert_basis(source, gates=basis)

    rules = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", _tdg_to_x_t_x)
    result = convert_basis(source, gates=basis, rules=rules)
    assert set(_names(result.program)) == {"h", "s", "t"}
    assert len(result.decompositions) == 1
    assert result.decompositions[0].source_opcode == "tdg"
    _assert_converts_exactly(
        source, tuple(sorted(set(_names(result.program)))), rules=rules
    )


def test_a_registered_rule_cannot_leave_the_named_basis() -> None:
    """Extending the search is not permission to widen what "in the basis" means.

    The rule names `phase`, the basis does not carry it, and `phase` has no rule
    of its own, so the loop stops on the leaf and names it. This is the `escapes`
    refusal, and a registered rule is how a caller reaches it.
    """

    def build(instruction: Instruction) -> tuple[Instruction, ...]:
        return (Instruction("phase", instruction.wires, params={"theta": 0.5}),)

    rules = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", build)
    with pytest.raises(
        BasisConversionError,
        match="decomposition of 'tdg' requires a gate outside the named basis: 'phase'",
    ):
        convert_basis(
            _program(_gate("tdg", (0,))), gates=("h", "s", "t", "cx"), rules=rules
        )


def test_a_registered_rule_does_not_change_a_conversion_it_is_not_needed_for() -> None:
    """The same program converts to the same identity with or without the extra rule."""

    source = _program(_gate("cx", (0, 1)))
    plain = convert_basis(source, gates=("h", "cz"))
    extended = convert_basis(
        source,
        gates=("h", "cz"),
        rules=with_equivalence_rule(EQUIVALENCE_RULES, "tdg", _tdg_to_x_t_x),
    )
    assert plain.conversion_identity == extended.conversion_identity
