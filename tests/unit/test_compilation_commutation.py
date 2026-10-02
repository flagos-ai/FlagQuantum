"""Pin the commutation rule source against the runtime's own gate matrices.

`flagquantum.compiler.commutation` decides that two instructions commute without
reading a parameter, because every rule it holds is a statement about opcodes and
wires rather than about values. That is a strong claim, and the module cannot
check it against `flagquantum.simulation`, which the compiler layer may not
import. This module is where the claim is checked instead: it builds each pair's
two operators from `gate_matrix` and compares `left @ right` with `right @ left`.

Only the "yes" direction is a claim the analysis makes. A "no" is the fail-closed
answer and needs no defence, but it is counted here, so the tests can state how
much of the true commuting set the rule source declines rather than leaving it
unmeasured.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import random

import pytest
import torch

from flagquantum.compiler.commutation import (
    _RULE_OPCODES,
    CommutationAnalysis,
    analyze_commutation,
    commute,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: Every declared unitary the rule source could meet. The non-unitary channels are
#: absent deliberately: they are commutation barriers by construction and
#: `test_every_non_unitary_channel_is_a_commutation_barrier` states that.
_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity <= 2
    )
)

#: Three qubits, so a pair can be disjoint, share exactly one qubit, or share both.
_REGISTER_WIDTH = 3

#: The exact comparison uses one tolerance and no relative slack: two products that
#: differ are a rule that is wrong, not a rule that is close.
_ATOL = 1.0e-10


def _matrix(instruction: Instruction, width: int) -> torch.Tensor:
    """One instruction's operator lifted into a `width`-qubit register."""

    local = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    arity = len(instruction.wires)
    local = local.reshape(-1, 2**arity, 2**arity)[0]
    rest = [qubit for qubit in range(width) if qubit not in instruction.wires]
    order = list(instruction.wires) + rest
    lifted = local
    for _ in rest:
        lifted = torch.kron(lifted, torch.eye(2, dtype=torch.complex128))
    # `kron` reads the left operand as the more significant factor, which is where
    # `instruction.wires[0]` belongs; `order` records what that produced so the
    # register can be relabelled to plain ascending wire order.
    permutation = [order.index(qubit) for qubit in range(width)]
    size = 2**width
    return (
        lifted.reshape([2] * width + [2] * width)
        .permute(permutation + [index + width for index in permutation])
        .reshape(size, size)
    )


def _instruction(name: str, wires: tuple[int, ...], seed: random.Random) -> Instruction:
    schema = OPERATOR_SCHEMAS[name]
    return Instruction(
        name,
        wires,
        params={parameter: seed.uniform(-3.0, 3.0) for parameter in schema.parameters},
    )


def _commutes_exactly(left: Instruction, right: Instruction) -> bool:
    width = max(max(left.wires), max(right.wires)) + 1
    width = max(width, _REGISTER_WIDTH)
    left_matrix = _matrix(left, width)
    right_matrix = _matrix(right, width)
    return bool(
        torch.allclose(
            left_matrix @ right_matrix, right_matrix @ left_matrix, atol=_ATOL, rtol=0
        )
    )


def _pairs() -> list[tuple[Instruction, Instruction]]:
    """Every opcode pair over the placements the tests below sweep.

    Each arity gets its own placements, so an arity-one gate really does meet
    another gate disjointly, on a shared qubit, and on the target of a two-qubit
    neighbour. A placement list that only offered two- and three-wire tuples would
    silently drop every single-qubit opcode from the sweep, which is exactly the
    kind of vacuous pass these tests exist to prevent.
    """

    seed = random.Random(20261006)
    placements = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]
    pairs = []
    for left_name, right_name in itertools.product(_UNITARIES, repeat=2):
        for left_wires in placements:
            if len(left_wires) != OPERATOR_SCHEMAS[left_name].arity:
                continue
            for right_wires in placements:
                if len(right_wires) != OPERATOR_SCHEMAS[right_name].arity:
                    continue
                pairs.append(
                    (
                        _instruction(left_name, left_wires, seed),
                        _instruction(right_name, right_wires, seed),
                    )
                )
    return pairs


def test_every_rule_that_says_yes_is_true_against_the_runtime_matrices() -> None:
    """No answer of "yes" may be wrong, over every opcode pair and placement.

    This is the test the module's docstring promises. It sweeps all 31 declared
    arity-one and arity-two unitaries in both roles across disjoint, single-shared,
    and identical placements, which is every way two of them can meet in a
    three-qubit register. A false "yes" here is a wrong program, not a missed
    optimization.
    """

    pairs = _pairs()
    assert len(pairs) == 5776
    claimed = [
        (left, right)
        for left, right in pairs
        if commute(left, right) or commute(right, left)
    ]
    assert claimed, "the rule source answered yes for nothing; the sweep is vacuous"
    wrong = [
        (left.name, left.wires, right.name, right.wires)
        for left, right in claimed
        if not _commutes_exactly(left, right)
    ]
    assert wrong == []
    # The sweep has to reach the interesting shape, not only the trivial one.
    assert any(left.wires == right.wires for left, right in claimed)
    assert any(set(left.wires) & set(right.wires) for left, right in claimed)
    assert any(not set(left.wires) & set(right.wires) for left, right in claimed)


def test_the_rule_source_is_worth_shipping_over_the_true_commuting_set() -> None:
    """Measure the fail-closed direction instead of leaving it unstated.

    A rule source that answered "no" to everything would pass the test above, so
    this one measures what it actually recovers. Of the 3826 pairs of declared
    arity-one and arity-two unitaries that really do commute in the placements
    swept above, 3602 are answered and 224 are declined. Every decline is listed
    below with the number of (placement, role) combinations it accounts for, so a
    regression that widens or narrows the gap fails here instead of passing
    silently.

    The declines are three families, none of them surprising:

    * the identity gate against a rotation or a controlled operator -- it commutes
      with everything, and nothing in the rule source says so;
    * a single-qubit Pauli or Pauli-span gate against a two-qubit Pauli-span
      entangler on an overlapping wire, such as `rx` against `rxx`, which commute
      because `X (x) I` and `X (x) X` commute;
    * a controlled operator against a diagonal or exchange-symmetric two-qubit
      operator, such as `cx` against `cz`, which commute because `Z` on the
      control pulls out of every term of `cx`.

    None of these is needed by the cancellation pass this round ships, whose gap
    is always bridged by a parameterized rotation on a control. They are recorded
    as an owned gap rather than papered over with a sampled table, because a table
    sampled from matrices would have to guess at angles and a rule that reads no
    angle cannot.
    """

    pairs = _pairs()
    truth = [(left, right) for left, right in pairs if _commutes_exactly(left, right)]
    answered = [(left, right) for left, right in truth if commute(left, right)]
    declined = [(left, right) for left, right in truth if not commute(left, right)]
    assert len(truth) == 3826
    assert len(answered) == 3602
    assert len(declined) == 224
    assert len(answered) + len(declined) == len(truth)

    counts = collections.Counter(
        tuple(sorted((left.name, right.name))) for left, right in declined
    )
    assert dict(sorted(counts.items())) == _DECLINED_GAP

    # A decline is reach left on the table, not a wrong answer: every pair listed
    # as a gap really does commute, which is what makes it a gap.
    assert all(_commutes_exactly(left, right) for left, right in declined)


#: Every pair the rule source declines although the runtime's matrices say the two
#: operators commute, with the number of placements that reach it. The three
#: families are described in the test docstring above.
_DECLINED_GAP: dict[tuple[str, str], int] = {
    # A controlled operator against a diagonal or exchange-symmetric operator: the
    # diagonal factors out of the control's `|a><a|` terms.
    ("cx", "cz"): 2,
    ("cx", "rxx"): 2,
    ("cx", "rzz"): 2,
    ("cy", "cz"): 2,
    ("cy", "ryy"): 2,
    ("cy", "rzz"): 2,
    ("crx", "crz"): 2,
    ("crx", "cz"): 2,
    ("crx", "rxx"): 2,
    ("crx", "rzz"): 2,
    ("cry", "crz"): 2,
    ("cry", "cz"): 2,
    ("cry", "ryy"): 2,
    ("cry", "rzz"): 2,
    ("crz", "cx"): 2,
    ("crz", "cy"): 2,
    ("cphase", "cx"): 2,
    ("cphase", "cy"): 2,
    ("cphase", "crx"): 2,
    ("cphase", "cry"): 2,
    # A single-qubit Pauli or Pauli-span gate against a two-qubit Pauli-span
    # entangler with an overlapping wire.
    ("rx", "rxx"): 8,
    ("rx", "sx"): 6,
    ("rx", "sxdg"): 6,
    ("rx", "x"): 6,
    ("rxx", "sx"): 8,
    ("rxx", "sxdg"): 8,
    ("rxx", "x"): 8,
    ("ry", "ryy"): 8,
    ("ry", "y"): 6,
    ("ryy", "y"): 8,
    ("sx", "sxdg"): 6,
    ("sx", "x"): 6,
    ("sxdg", "x"): 6,
    # The identity gate against anything at all.
    ("crx", "i"): 4,
    ("cry", "i"): 4,
    ("cx", "i"): 4,
    ("cy", "i"): 4,
    ("h", "i"): 6,
    ("i", "rx"): 6,
    ("i", "rxx"): 8,
    ("i", "ry"): 6,
    ("i", "ryy"): 8,
    ("i", "swap"): 8,
    ("i", "sx"): 6,
    ("i", "sxdg"): 6,
    ("i", "u2"): 6,
    ("i", "u3"): 6,
    ("i", "x"): 6,
    ("i", "y"): 6,
}


def test_the_verdict_is_symmetric_in_its_arguments() -> None:
    pairs = _pairs()
    asymmetric = [
        (left.name, right.name)
        for left, right in pairs
        if commute(left, right) != commute(right, left)
    ]
    assert asymmetric == []


def test_a_rule_is_read_as_an_opcode_class_not_as_a_sample_of_angles() -> None:
    """Every rule is exact for any angle, including a trainable or batched one.

    A table sampled from matrices would have to decline these, because a sampled
    verdict is only as good as the angles it was sampled at. `commute` reads no
    angle, so these pairs are decided exactly as the constant ones are.
    """

    trainable = torch.nn.Parameter(torch.tensor(0.4))
    batched = torch.arange(3, dtype=torch.float64) * 0.25
    for angle in (trainable, batched):
        left = Instruction("rz", (0,), params={"theta": angle})
        right = Instruction("cx", (0, 1))
        assert commute(left, right) is True
        assert commute(right, left) is True
        # A parameter that makes the pair unequal is answered, not raised on.
        assert commute(left, Instruction("rz", (0,), params={"theta": angle})) is True
        assert commute(left, Instruction("cx", (0, 1))) is True


def test_the_diagonal_class_is_the_diagonal_class_the_runtime_ships() -> None:
    """The diagonal rule's premise is checked, not asserted by opcode name.

    The rule says a diagonal single-qubit gate commutes with a diagonal
    two-qubit gate. That is only true if both runtime matrices really are diagonal,
    so this reads them and not the names.
    """

    diagonal = [
        Instruction("rz", (0,), params={"theta": 0.7}),
        Instruction("phase", (0,), params={"theta": 0.7}),
        Instruction("u1", (0,), params={"theta": 0.7}),
        Instruction("z", (0,)),
        Instruction("s", (0,)),
        Instruction("t", (0,)),
        Instruction("cz", (0, 1)),
        Instruction("cphase", (0, 1), params={"theta": 0.7}),
        Instruction("crz", (0, 1), params={"theta": 0.7}),
        Instruction("rzz", (0, 1), params={"theta": 0.7}),
    ]
    for instruction in diagonal:
        matrix = _matrix(instruction, 2)
        off_diagonal = matrix - torch.diag(torch.diagonal(matrix))
        assert bool(torch.all(off_diagonal.abs() < _ATOL)), instruction.name
        assert commute(instruction, Instruction("rz", (0,), params={"theta": 1.1}))


def test_the_control_rule_matches_the_operators_runtime_matrix() -> None:
    """A diagonal gate on a control commutes; the same gate on the target does not.

    This is the rule the round exists for: it is what lets a `cx` pair annihilate
    across an `rz` sitting on the control. The negative half is asserted here too,
    because a rule that said "yes" for every wire position would pass the positive
    half alone.

    `cz`, `crz`, and `cphase` are not in this loop: they are diagonal outright, so
    a diagonal gate commutes with them on *both* wires and their answer comes from
    the diagonal rule rather than from the control rule. `test_the_diagonal_class_
    is_the_diagonal_class_the_runtime_ships` covers that shape.
    """

    seed = random.Random(20261007)
    for controlled, controls, target in (
        ("cx", (0,), 1),
        ("cy", (0,), 1),
        ("crx", (0,), 1),
        ("cry", (0,), 1),
        ("cswap", (0,), 1),
        ("ccx", (0, 1), 2),
    ):
        multi = _instruction(
            controlled, tuple(range(OPERATOR_SCHEMAS[controlled].arity)), seed
        )
        for qubit in controls:
            on_control = Instruction("rz", (qubit,), params={"theta": 0.9})
            assert commute(on_control, multi) is True, (controlled, qubit)
            assert _commutes_exactly(on_control, multi) is True
        on_target = Instruction("rz", (target,), params={"theta": 0.9})
        assert commute(on_target, multi) is False, controlled
        assert _commutes_exactly(on_target, multi) is False


def test_the_target_pauli_rule_matches_the_operators_runtime_matrix() -> None:
    """The gate on a target has to sit in the Pauli family the control applies.

    `cx` needs `span{I, X}`, which is `x` and every `rx`, and also `sx` and `sxdg`
    -- they are `e^{+-i pi/4} RX(+-pi/2)`, so they are in the same span -- while
    `rz`, `h`, and `sx` respectively are not. `cy` needs `span{I, Y}`, which `sx`
    is not in, and the test asserts that asymmetry rather than assuming it.
    """

    seed = random.Random(20261008)
    for controlled, target, allowed, refused in (
        ("cx", 1, ("x", "rx", "sx", "sxdg"), ("y", "ry", "z", "h", "rz")),
        ("cy", 1, ("y", "ry"), ("x", "rx", "sx", "z", "h", "rz")),
        ("ccx", 2, ("x", "rx", "sx", "sxdg"), ("y", "ry", "z", "h", "rz")),
    ):
        multi = _instruction(
            controlled, tuple(range(OPERATOR_SCHEMAS[controlled].arity)), seed
        )
        for name in allowed + refused:
            gate = _instruction(name, (target,), seed)
            expected = name in allowed
            assert commute(gate, multi) is expected, (controlled, name)
            assert _commutes_exactly(gate, multi) is expected


def test_the_ising_entanglers_commute_on_one_opcode_anywhere_and_across_wires() -> None:
    """`XX` with `XX` commutes on any wires; `XX` with `YY` only on the same wires.

    Neither half is guessable from the opcode class alone, so both are read off
    the runtime matrices. `X_1 X_2` and `X_0 X_1` multiply to `X_0 X_2` in either
    order, so one opcode twice commutes even when the wires only overlap --
    whereas `X_1 X_2` and `Y_0 Y_1` share qubit 1 with factors `X` and `Y`, which
    anticommute, so those two do not.
    """

    for left, right in itertools.product(("rxx", "ryy", "rzz"), repeat=2):
        on_same = (
            Instruction(left, (0, 1), params={"theta": 0.3}),
            Instruction(right, (0, 1), params={"theta": 0.4}),
        )
        assert commute(*on_same) is True, (left, right)
        assert _commutes_exactly(*on_same) is True
        on_shared_qubit = (
            Instruction(left, (1, 2), params={"theta": 0.3}),
            Instruction(right, (0, 1), params={"theta": 0.4}),
        )
        expected = left == right
        assert commute(*on_shared_qubit) is expected, (left, right)
        assert _commutes_exactly(*on_shared_qubit) is expected


def test_every_non_unitary_channel_is_a_commutation_barrier() -> None:
    for name, schema in sorted(OPERATOR_SCHEMAS.items()):
        if schema.unitary:
            continue
        channel = Instruction(
            name,
            (0,),
            params=dict.fromkeys(schema.parameters, 0.3),
        )
        for other in (
            Instruction("rz", (0,), params={"theta": 0.4}),
            Instruction("x", (0,)),
        ):
            assert commute(channel, other) is False, name
            assert commute(other, channel) is False, name


def test_a_matrix_override_is_declined_whatever_its_opcode_name_says() -> None:
    """A `matrix` field replaces what runs, so no rule may read the opcode name."""

    override = Instruction("rz", (0,), params={"theta": 0.4}, matrix=[[1, 0], [0, 1]])
    assert commute(override, Instruction("cx", (0, 1))) is False
    assert commute(override, Instruction("rz", (0,), params={"theta": 1.0})) is False
    # Disjointness is decided before the opcode is read, and it holds for an
    # override too: two operators on different wires never touch.
    assert commute(override, Instruction("cx", (1, 2))) is True


def test_the_rule_source_carries_no_import_of_the_simulation_layer() -> None:
    """The rules are proven against the runtime, not copied from it.

    The check reads the module's imports rather than its text, because the
    docstring names the layer on purpose: the point of this test file is that the
    rules are *proven against* `flagquantum.simulation` from here, where that
    import is allowed, rather than taken from it there, where it is not.
    """

    import ast
    import importlib

    module = importlib.import_module("flagquantum.compiler.commutation")
    assert module.__file__ is not None
    with open(module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
    assert imported == [
        "__future__",
        "collections.abc",
        "dataclasses",
        "types",
        "typing",
        "..core.ir",
        "..core.operator_schema",
    ]
    # The reach of the rule source, as a fact rather than as a guess.
    assert len(_RULE_OPCODES) == 28


def _circuit(*instructions: Instruction) -> CircuitIR:
    return CircuitIR(n_wires=3, instructions=instructions)


def test_the_partition_keeps_only_mutually_commuting_members_together() -> None:
    """The blocks are checked against `commute`, not against their own construction."""

    seed = random.Random(11)
    opcodes = ["rz", "phase", "cx", "cz", "h", "x", "rxx", "ccx", "u3", "sx"]
    for _ in range(60):
        instructions = []
        for _ in range(8):
            name = seed.choice(opcodes)
            arity = OPERATOR_SCHEMAS[name].arity
            wires = tuple(sorted(seed.sample(range(3), arity)))
            instructions.append(_instruction(name, wires, seed))
        ir = _circuit(*instructions)
        analysis = analyze_commutation(ir)
        tuple_ir = tuple(ir)
        for qubit, blocks in analysis.blocks_by_qubit.items():
            for block in blocks:
                members = [tuple_ir[position] for position in block]
                assert all(instruction for instruction in members)
                for left, right in itertools.combinations(members, 2):
                    assert commute(left, right) is True, (left.name, right.name)
                # A block is a consecutive run of that qubit's instructions.
                for position in block:
                    assert qubit in tuple_ir[position].wires
                    assert analysis.block_of(position, qubit) == blocks.index(block)
            # Every instruction on the qubit is in exactly one block.
            covered = [position for block in blocks for position in block]
            assert covered == sorted(covered)
            assert len(covered) == len(set(covered))
        assert analysis.queried_pairs >= analysis.commuting_pairs


def test_the_analysis_reports_a_shape_that_can_be_told_from_a_degenerate_one() -> None:
    """The counters must distinguish a real partition from "nothing commutes"."""

    commuting = analyze_commutation(
        _circuit(
            Instruction("rz", (0,), params={"theta": 0.1}),
            Instruction("cx", (0, 1)),
            Instruction("rz", (0,), params={"theta": 0.2}),
        )
    )
    assert commuting.summary() == {
        "qubit_count": 2,
        "block_count": 2,
        "widest_block": 3,
        "queried_pairs": 3,
        "commuting_pairs": 3,
    }
    separating = analyze_commutation(
        _circuit(
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("h", (1,)),
        )
    )
    # `h` on a control, `h` on a target, and `cx` itself are all barriers: no two
    # of the three commute, so every block is one instruction wide.
    assert separating.summary() == {
        "qubit_count": 2,
        "block_count": 4,
        "widest_block": 1,
        "queried_pairs": 2,
        "commuting_pairs": 0,
    }


def test_the_analysis_is_a_value_and_not_a_mutation_of_the_property_bag() -> None:
    """Qiskit's pass writes `property_set`; this one returns the structure.

    FlagQuantum has no pass manager and no property bag, so the port has to answer
    the same question in the shape this layer can carry. The frozen dataclass and
    the mapping proxies are what make that a fact rather than a statement.
    """

    ir = _circuit(
        Instruction("cx", (0, 1)), Instruction("rz", (0,), params={"theta": 0.3})
    )
    analysis = analyze_commutation(ir)
    assert isinstance(analysis, CommutationAnalysis)
    assert len(ir) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        analysis.queried_pairs = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        analysis.blocks_by_qubit[0] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        analysis.block_index[(0, 0)] = 5  # type: ignore[index]


def test_a_position_outside_the_analyzed_circuit_declines_rather_than_misreads() -> (
    None
):
    ir = _circuit(Instruction("cx", (0, 1)))
    analysis = analyze_commutation(ir)
    assert analysis.block_of(0, 0) == 0
    assert analysis.block_of(99, 0) is None
    assert analysis.block_of(0, 99) is None
    assert analysis.signature(99, (0, 1)) is None
    assert analysis.signature(0, (0, 1)) == (0, 0)


def test_an_empty_circuit_analyzes_to_an_empty_partition() -> None:
    analysis = analyze_commutation(CircuitIR(n_wires=2, instructions=()))
    assert analysis.blocks_by_qubit == {}
    assert analysis.summary() == {
        "qubit_count": 0,
        "block_count": 0,
        "widest_block": 0,
        "queried_pairs": 0,
        "commuting_pairs": 0,
    }
