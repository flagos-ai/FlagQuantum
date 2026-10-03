"""Unit tests for the exact two-wire identities and the composed basis search.

`basis_translation` replaces the four hand-written rewrite rules that used to sit
in `native_gate_legalization`. Two obligations follow from that, and this module
holds both: every entry of the table has to reproduce its source through the
runtime's own gate matrices, and the production consumer has to keep returning
exactly the programs it returned before.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.basis_translation import (
    EQUIVALENCE_RULES,
    EquivalenceRule,
    EquivalenceRuleError,
    translate,
    validate_equivalence_rule,
    with_equivalence_rule,
)
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    _native_descriptors,
    _supports,
    legalize_native_gates,
)
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

_ANGLES = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}
_NOW = datetime(2026, 10, 3, 8, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("native:0",))
_SOURCE = FactSource(kind="w804_test", ref="w804-evidence")

_TWO_QUBIT_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 2
    )
)

#: Every unitary opcode this IR declares on more than two wires. It is the whole
#: of the multi-controlled surface: above arity three the IR has no opcode to
#: name, so a Qiskit construction that only differs at five controls and up has
#: nothing here to differ on.
_THREE_WIRE_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 3
    )
)

#: `cphase` is the only entry of the table that is not exact. Its rule needs a
#: third `rz` on the control wire to turn the controlled rotation into a
#: controlled phase, and that rotation leaves a global phase behind. Every other
#: entry is built only from conjugations and `sdg ... s` pairs, which compose to
#: a unit phase, so pinning the exact set keeps a rule that silently started
#: carrying a phase from passing as one that never did.
_PHASE_CARRYING_RULES = frozenset({"cphase"})


def _snapshot(native_gates: tuple[object, ...]) -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="w804-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="w804-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=native_gates,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=_SOURCE,
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="w804-evidence",
                sha256="c" * 64,
                level=EvidenceLevel.BASIC,
                scope=_SCOPE,
            ),
        ),
    )


def _gate(name: str, parameters: tuple[str, ...] | None = None) -> dict[str, object]:
    schema = OPERATOR_SCHEMAS[name]
    return {
        "name": name,
        "parameters": tuple(schema.parameters if parameters is None else parameters),
    }


def _descriptors(*opcodes: str):
    return _native_descriptors(
        _snapshot(tuple(_gate(opcode) for opcode in opcodes)), evaluated_at=_NOW
    )


def _can_run(*opcodes: str):
    descriptors = _descriptors(*opcodes)
    return lambda item: _supports(descriptors, item)


def _source(opcode: str) -> Instruction:
    schema = OPERATOR_SCHEMAS[opcode]
    angles = {name: _ANGLES[name] for name in schema.parameters}
    return Instruction(opcode, tuple(range(schema.arity)), params=angles)


def _rotation(theta: float) -> Instruction:
    """One source rotation, at an angle the caller chooses.

    `_source` reads a fixed angle out of `_ANGLES`, which is deliberately not a
    multiple of `pi/4`, so the quarter-turn cases below build their own.
    """

    return Instruction("rz", (0,), params={"theta": theta})


def _unitary(instruction: Instruction) -> torch.Tensor:
    width = 2 ** len(instruction.wires)
    matrix = gate_matrix(instruction, bsz=1, device="cpu", dtype=torch.complex128)
    return matrix.reshape(-1, width, width)[0]


def _register_matrix(instruction: Instruction, n_wires: int) -> torch.Tensor:
    """Lift one leaf into the whole register, as the statevector engine does.

    `gate_matrix` returns a gate's local matrix and does not reorder its wires:
    `cx` on `(1, 0)` comes back with the same block as `cx` on `(0, 1)`. The
    engine applies the wire order by laying the state out with the gate's wires
    in front, and a test that skipped that step would compare the rules in a
    different basis from the one a consumer runs them in. Wire 0 is the most
    significant bit, so wire `w` owns bit `n_wires - 1 - w`.
    """

    local = _unitary(instruction)
    if len(instruction.wires) < n_wires:
        # In the wires-first layout the gate acts on the high-order bits.
        rest = torch.eye(2 ** (n_wires - len(instruction.wires)), dtype=local.dtype)
        local = torch.kron(local, rest)
    order = list(instruction.wires) + [
        wire for wire in range(n_wires) if wire not in instruction.wires
    ]
    permutation = []
    for moved_index in range(2**n_wires):
        register_index = 0
        for position, wire in enumerate(order):
            bit = (moved_index >> (n_wires - 1 - position)) & 1
            register_index |= bit << (n_wires - 1 - wire)
        permutation.append(register_index)
    index = torch.tensor(permutation)
    matrix = torch.zeros((2**n_wires, 2**n_wires), dtype=local.dtype)
    matrix[index[:, None], index[None, :]] = local
    return matrix


def _product(leaves: tuple[Instruction, ...], n_wires: int) -> torch.Tensor:
    total = torch.eye(2**n_wires, dtype=torch.complex128)
    for leaf in leaves:
        total = _register_matrix(leaf, n_wires) @ total
    return total


def _legalize(program: CircuitIR, *opcodes: str):
    return legalize_native_gates(
        program,
        snapshot=_snapshot(tuple(_gate(name) for name in opcodes)),
        evaluated_at=_NOW,
    )


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


def test_the_table_covers_every_opcode_it_claims_and_no_non_unitary_one() -> None:
    # Non-vacuity: the checks below iterate the table, so it has to be the size
    # it claims, and a channel must not have crept into it.
    assert len(EQUIVALENCE_RULES) == 18
    for opcode, rules in EQUIVALENCE_RULES.items():
        assert rules, opcode
        schema = OPERATOR_SCHEMAS[opcode]
        assert schema.unitary, opcode
        for rule in rules:
            assert isinstance(rule, EquivalenceRule)
            assert rule.opcode == opcode
            assert rule.build(_source(opcode)), opcode
    for opcode in ("depolarizing", "bit_flip", "amplitude_damping"):
        assert opcode not in EQUIVALENCE_RULES


def test_every_rule_reproduces_its_source_through_the_runtime() -> None:
    """The gate matrices are the only authority a rule can be checked against.

    Seventeen of the eighteen entries are exact and `cphase` is not, which is
    asserted in both directions: a rule that started carrying a phase fails the
    exact branch, and `cphase` losing its phase fails the inexact branch. The two
    arity-3 entries are covered here like any other, which is the point of
    checking a rule against the runtime matrix rather than against a source.
    """

    exact = sorted(set(EQUIVALENCE_RULES) - _PHASE_CARRYING_RULES)
    # Non-vacuity: both branches below have to have something in them.
    assert len(exact) == 17
    assert len(_PHASE_CARRYING_RULES) == 1
    assert exact == sorted(set(EQUIVALENCE_RULES) - {"cphase"})
    for opcode in sorted(EQUIVALENCE_RULES):
        source = _source(opcode)
        n_wires = len(source.wires)
        leaves = EQUIVALENCE_RULES[opcode][0].build(source)
        product = _product(leaves, n_wires) @ _unitary(source).conj().mT
        identity = torch.eye(product.shape[0], dtype=torch.complex128)
        if opcode not in _PHASE_CARRYING_RULES:
            torch.testing.assert_close(product, identity, rtol=0.0, atol=1e-12)
            continue
        torch.testing.assert_close(
            product, product[0, 0] * identity, rtol=0.0, atol=1e-12
        )
        # A phase-carrying rule is listed as one exactly because its phase is
        # not one; a change that made it exact must fail here, not pass.
        assert abs(product[0, 0] - 1.0) > 1e-9, opcode


def test_the_table_carries_the_parameters_the_caller_owns() -> None:
    """A rule forwards the source's own objects, so autograd can follow them."""

    theta = torch.tensor(0.7137, requires_grad=True)
    source = Instruction("rzz", (0, 1), params={"theta": theta}, metadata={"tag": 1})
    leaves = EQUIVALENCE_RULES["rzz"][0].build(source)
    rotations = [leaf for leaf in leaves if leaf.name == "rz"]
    assert len(rotations) == 1
    assert rotations[0].params["theta"] is theta
    assert all(dict(leaf.metadata) == {"tag": 1} for leaf in leaves)
    # `rzz` is `cx` around an `rz` on the *target* wire; putting the rotation on
    # the control wire is the mistake this assertion exists to catch.
    assert [leaf.wires for leaf in leaves] == [(0, 1), (1,), (0, 1)]


def test_the_swap_rule_keeps_the_middle_entangler_reversed() -> None:
    """The three-`cx` form is a swap only with the middle one reversed.

    This is the entry a reader is most likely to "tidy up", and tidying it up
    leaves a legal-looking program that is the identity instead of a swap, so
    the wire tuples are asserted rather than only the opcode sequence.
    """

    leaves = EQUIVALENCE_RULES["swap"][0].build(Instruction("swap", (0, 1)))
    assert [leaf.wires for leaf in leaves] == [(0, 1), (1, 0), (0, 1)]


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def test_a_native_instruction_is_not_a_translation() -> None:
    """The caller keeps a gate the target already has; nothing is returned."""

    source = Instruction("rz", (0,), params={"theta": 0.7137})
    assert translate(source, can_run=_can_run("rz")) is None


def test_the_search_prefers_the_shortest_fully_native_rewrite() -> None:
    leaves = translate(Instruction("x", (0,)), can_run=_can_run("h", "z"))
    assert leaves is not None
    assert [leaf.name for leaf in leaves] == ["h", "z", "h"]


def test_the_search_composes_rules_instead_of_stopping_at_one() -> None:
    """`cphase` is a `cz` plus a phase, and `cz` is `cx` under two `h`s."""

    leaves = translate(
        Instruction("cphase", (0, 1), params={"theta": math.pi / 2}),
        can_run=_can_run("h", "rz", "cx"),
    )
    assert leaves is not None
    assert set(leaf.name for leaf in leaves) <= {"h", "rz", "cx"}
    assert sum(1 for leaf in leaves if leaf.name == "cx") == 2


def test_the_search_branches_on_opcode_names_only() -> None:
    """A different angle may not change the chosen path, trainable or not."""

    can_run = _can_run("rz", "sx", "cx")
    for theta in (0.0, 1.0e-12, 0.7137, math.pi, -2.5):
        leaves = translate(
            Instruction("rzz", (0, 1), params={"theta": theta}), can_run=can_run
        )
        assert leaves is not None, theta
        assert [leaf.name for leaf in leaves] == ["cx", "rz", "cx"], theta


def test_a_trainable_angle_stays_in_the_autograd_graph() -> None:
    theta = torch.tensor(0.7137, requires_grad=True)
    leaves = translate(
        Instruction("crz", (0, 1), params={"theta": theta}),
        can_run=_can_run("rz", "sx", "cx"),
    )
    assert leaves is not None
    rotations = [leaf for leaf in leaves if leaf.name == "rz"]
    assert rotations
    assert all(leaf.params["theta"].requires_grad for leaf in rotations)


def test_an_opcode_with_no_rule_and_no_basis_is_refused() -> None:
    """`y` has no identity into `h` and `t`, so nothing can be said about it."""

    assert translate(Instruction("y", (0,)), can_run=_can_run("h", "t")) is None


def test_a_clifford_t_target_reaches_a_quarter_turn_rotation() -> None:
    """The measured hole this file's table could not close: a rotation.

    No identity in `EQUIVALENCE_RULES` names `rz`, `phase`, or `u1`, so a target
    publishing only `h`, `s`, `t`, and `cx` used to refuse every rotation in the
    program. The exact quarter-turn words are the additional candidate the search
    now consults. The consumer is what is asserted here rather than the table:
    the composition is committed by `test_compilation_angle_synthesis.py`, and
    this test pins that the production path reaches it.
    """

    program = CircuitIR(
        2, (Instruction("h", (0,)), _rotation(math.pi / 4)), dtype="complex128"
    )
    result = _legalize(program, "h", "s", "t", "cx")
    assert result.changed is True
    assert [item.name for item in result.program.instructions] == ["h", "t"]
    assert result.decompositions[0].source_opcode == "rz"
    assert result.decompositions[0].replacement_opcodes == ("t",)
    # Non-vacuity: the same rotation on a target that publishes `rz` is native,
    # so the answer above is the search working rather than a refusal.
    native = _legalize(
        CircuitIR(1, (_rotation(math.pi / 4),), dtype="complex128"), "rz", "sx", "cx"
    )
    assert native.changed is False
    assert native.decompositions == ()


def test_a_rotation_that_is_not_a_quarter_turn_still_fails_closed() -> None:
    """The word table answers eight residues, and the ninth has no answer.

    `0.7137` is this file's own angle and is not a multiple of `pi/4`, so the
    refusal that existed before the words existed has to survive them. A
    tolerance here would answer with a rotation nobody asked for, which is why
    the classification is an exact equality.
    """

    program = CircuitIR(1, (_rotation(0.7137),), dtype="complex128")
    with pytest.raises(NativeGateLegalizationError, match="'rz'"):
        _legalize(program, "h", "s", "t", "cx", "sdg", "tdg")


def test_an_unreachable_rewrite_is_returned_so_the_caller_names_the_gate() -> None:
    """A basis without `z` or `s` must be told *which* gate it is missing.

    None would report the source opcode, and the source opcode is not the gate
    the target lacks. `x` is `h z h` and `z` is `s s`, so the leaves that reach
    the target are `h s s h` and the missing gate is `s`; the caller re-checks
    the leaves and says so. The expansion runs to the table's leaves rather than
    stopping at `z`, which is what makes the answer name a gate the target could
    actually add.
    """

    leaves = translate(Instruction("x", (0,)), can_run=_can_run("h"))
    assert leaves is not None
    assert [leaf.name for leaf in leaves] == ["h", "s", "s", "h"]
    descriptors = _descriptors("h")
    assert any(not _supports(descriptors, leaf) for leaf in leaves)
    # Non-vacuity: there really is a runnable rewrite of the same source, so the
    # assertion above is about the search falling short rather than about there
    # being no rewrite at all.
    assert translate(Instruction("x", (0,)), can_run=_can_run("h", "z")) is not None


def test_the_recursion_stops_on_a_cycle() -> None:
    """`cx` and `cz` rewrite into each other, and a rule may not re-enter itself.

    This basis publishes neither, nor the `h` both rules need, so the search has
    to terminate on the wire identities rather than on the entangler pair.
    """

    leaves = translate(Instruction("cz", (0, 1)), can_run=_can_run("rz"))
    assert leaves is not None
    assert len(leaves) <= 3
    assert all(leaf.name in {"h", "cx"} for leaf in leaves)


def test_every_declared_two_qubit_unitary_has_a_rule_or_is_named_native() -> None:
    """The table has to account for the whole arity-2 group, not most of it.

    A claim that a target reaches the group is only meaningful if the group is
    every declared two-qubit unitary, so the size is asserted rather than
    assumed, and each member is checked to be either native or carried by a rule.
    The table covers all eleven, `cx` included: a `cz` basis needs the `cx` rule
    even though a `cx` basis does not.
    """

    assert len(_TWO_QUBIT_UNITARIES) == 11
    assert set(EQUIVALENCE_RULES) >= set(_TWO_QUBIT_UNITARIES)
    can_run = _can_run("rz", "sx", "cx")
    for name in _TWO_QUBIT_UNITARIES:
        source = _source(name)
        if can_run(source):
            # The one opcode the basis publishes needs no translation at all.
            assert name == "cx"
            assert translate(source, can_run=can_run) is None
            continue
        assert translate(source, can_run=can_run) is not None, name


def test_every_declared_three_wire_unitary_reaches_a_cx_basis() -> None:
    """The multi-controlled group is closed by the table, not by the search.

    `ccx` and `cswap` are the whole arity-3 surface. A basis publishing `rz`,
    `sx` and `cx` reaches both, and the recursion is real for `cswap`: its rule
    emits a `ccx`, which the search expands again, so the leaves returned are
    fifteen-gate groups rather than three.
    """

    assert _THREE_WIRE_UNITARIES == ("ccx", "cswap")
    for name in _THREE_WIRE_UNITARIES:
        assert name in EQUIVALENCE_RULES, name
    basis = ("rz", "sx", "cx")
    can_run = _can_run(*basis)
    for name in _THREE_WIRE_UNITARIES:
        # Non-vacuity: the source is not native, so a result is a rewrite.
        assert not can_run(_source(name)), name
        result = _legalize(CircuitIR(3, (_source(name),), dtype="complex128"), *basis)
        names = tuple(item.name for item in result.program.instructions)
        assert set(names) <= set(basis), (name, sorted(set(names) - set(basis)))
        assert result.changed is True
        assert result.decompositions, name
        # The two `tdg` leaves of the `ccx` rule are Euler-synthesized, so the
        # report names the source opcode and a rewrite that left the basis.
        assert result.decompositions[0].source_opcode == name
    # `cswap` recurses through its `ccx` leaf, so it is strictly longer than the
    # three leaves its own rule states, and it is the eight-entangler shape.
    direct = EQUIVALENCE_RULES["cswap"][0].build(_source("cswap"))
    assert [leaf.name for leaf in direct] == ["cx", "ccx", "cx"]
    expanded = translate(
        _source("cswap"), can_run=can_run, z_rotation="rz", pulse_opcode="sx"
    )
    assert expanded is not None
    assert [leaf.name for leaf in expanded].count("cx") == 8


def test_the_ccx_rule_is_the_fifteen_gate_form_qiskit_decomposes_to() -> None:
    """`ccx` is `CCXGate._define`, leaf for leaf and in order.

    Qiskit's `MCXGrayCode` returns `CCXGate` at two controls, so the standard
    fifteen-gate definition is the reference here rather than the Gray-code
    statement. The order is pinned because it is what makes the six `cx` count
    observable: a different but equally valid ordering is a different number of
    entanglers on some bases.
    """

    source = Instruction("ccx", (0, 1, 2))
    leaves = EQUIVALENCE_RULES["ccx"][0].build(source)
    assert [leaf.name for leaf in leaves] == [
        "h",
        "cx",
        "tdg",
        "cx",
        "t",
        "cx",
        "tdg",
        "cx",
        "t",
        "t",
        "h",
        "cx",
        "t",
        "tdg",
        "cx",
    ]
    assert [leaf.wires for leaf in leaves[:4]] == [
        (2,),
        (1, 2),
        (2,),
        (0, 2),
    ]
    # Six entanglers, which is the count Qiskit's `BasisTranslator` also pays.
    assert [leaf.name for leaf in leaves].count("cx") == 6
    # Every leaf is a declared opcode, not a name the IR has no schema for.
    for leaf in leaves:
        assert OPERATOR_SCHEMAS[leaf.name].unitary, leaf.name


def test_the_three_wire_rules_fail_closed_without_an_entangler_sink() -> None:
    """No sink, no reach -- and the refusal names the gate that is missing.

    The search may not borrow an entangler it was not given. It still returns the
    shortest rewrite it found, as documented, but every leaf of that rewrite is
    re-checked by the consumer, and a rewrite containing an unsupported gate is a
    refusal rather than a result. Both halves are asserted, so the refusal cannot
    pass by the private helper returning None for an unrelated reason.
    """

    can_run = _can_run("h", "s", "t")
    for name in _THREE_WIRE_UNITARIES:
        leaves = translate(_source(name), can_run=can_run)
        assert leaves is not None, name
        assert any(leaf.name not in {"h", "s", "t"} for leaf in leaves), name
        with pytest.raises(NativeGateLegalizationError) as raised:
            _legalize(CircuitIR(3, (_source(name),), dtype="complex128"), "h", "s", "t")
        assert "requires unsupported native gate" in str(raised.value), name


# --------------------------------------------------------------------------
# The replacement: the production consumer must not move
# --------------------------------------------------------------------------


def test_the_sdg_rule_lets_a_clifford_t_target_carry_the_controlled_gates() -> None:
    """`sdg = s s s`, and `cy` is what needs it.

    `clifford-t` publishes `h`, `s` and `t` but not `sdg`, and three of the rules
    emit an `sdg`. Without this entry the table stops one gate short of a basis
    that can express them, which is a gap in the table rather than a limitation
    of the search. Qiskit's standard equivalence library carries the same
    expansion, and reaches the same four names into the same basis.
    """

    basis = ("h", "s", "t", "cx")
    result = _legalize(
        CircuitIR(2, (Instruction("cy", (0, 1)),), dtype="complex128"), *basis
    )
    names = tuple(item.name for item in result.program.instructions)
    assert names == ("s", "s", "s", "cx", "s"), names
    assert set(names) <= set(basis)


def test_the_removed_hand_written_rules_are_reproduced_leaf_for_leaf() -> None:
    """The exact obligations the four deleted branches carried, unchanged.

    Each case is asserted through the public legalization entry point rather
    than through the private helper, so this is a test of what consumers see.
    """

    basis = ("h", "z", "s", "sdg", "rz", "cx")
    cases = (
        (Instruction("x", (0,)), ("h", "z", "h")),
        (Instruction("rx", (0,), params={"theta": 0.7137}), ("h", "rz", "h")),
        (
            Instruction("ry", (0,), params={"theta": 0.7137}),
            ("sdg", "h", "rz", "h", "s"),
        ),
        (Instruction("swap", (0, 1)), ("cx", "cx", "cx")),
    )
    for source, expected in cases:
        program = CircuitIR(2, (source,), dtype="complex128")
        result = _legalize(program, *basis)
        assert (
            tuple(item.name for item in result.program.instructions) == expected
        ), source.name
        # Non-vacuity: the rewrite really happened, and the record says so.
        assert result.changed is True
        assert len(result.decompositions) == 1
        assert result.decompositions[0].source_opcode == source.name
        assert result.decompositions[0].replacement_opcodes == expected


def test_the_replacement_leaves_a_consumer_program_running_and_equal() -> None:
    """A mixed program keeps its own native instructions and gains only rewrites."""

    source = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("swap", (0, 1)),
            Instruction("rx", (1,), params={"theta": 0.7137}),
        ),
        dtype="complex128",
    )
    result = _legalize(source, "h", "z", "s", "sdg", "rz", "cx")
    assert result.changed is True
    names = [item.name for item in result.program.instructions]
    assert names.count("cx") == 1 + 3
    assert set(names) <= {"h", "z", "s", "sdg", "rz", "cx"}
    assert len(result.decompositions) == 3
    assert result.source_content_hash == source.content_hash


# --- the table is extensible, one validated rule at a time ------------------


def _tdg_to_x_t_x(instruction: Instruction) -> tuple[Instruction, ...]:
    """`Tdg` is `X T X` up to one global phase, the phase this IR cannot hold."""

    wire = instruction.wires
    metadata = dict(instruction.metadata)
    return tuple(Instruction(name, wire, metadata=metadata) for name in ("x", "t", "x"))


def test_a_registered_rule_does_not_touch_the_built_in_table() -> None:
    """The argument is a value, so no other caller sees the addition."""

    before = dict(EQUIVALENCE_RULES)
    extended = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", _tdg_to_x_t_x)
    assert "tdg" not in EQUIVALENCE_RULES
    assert {name: items for name, items in EQUIVALENCE_RULES.items()} == before
    assert extended["tdg"][0].opcode == "tdg"
    assert set(extended) == set(before) | {"tdg"}


def test_a_registered_rule_reaches_the_search() -> None:
    """`tdg` has no built-in rewrite, and one registration gives it one.

    The leaves are the registered rule's `x` and `t` expanded through the table
    again -- `x` becomes `h z h` -- which is the same treatment a built-in rule
    receives.
    """

    extended = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", _tdg_to_x_t_x)
    basis = ("h", "s", "t", "cx")
    assert translate(Instruction("tdg", (0,)), can_run=_can_run(*basis)) is None
    leaves = translate(
        Instruction("tdg", (0,)), can_run=_can_run(*basis), rules=extended
    )
    assert leaves is not None
    names = [item.name for item in leaves]
    assert names == ["h", "s", "s", "h", "t", "h", "s", "s", "h"]
    # The `t` is the rule's own middle leaf, not a quarter-turn word.
    assert names[4] == "t"
    assert set(names) <= set(basis)


def test_a_registered_rule_is_expanded_through_the_table_like_any_other() -> None:
    """A rule may name a gate the target lacks, and the search still reaches it."""

    extended = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", _tdg_to_x_t_x)
    leaves = translate(
        Instruction("tdg", (0,)),
        can_run=_can_run("h", "s", "sdg", "t", "z"),
        rules=extended,
    )
    assert leaves is not None
    assert [item.name for item in leaves] == ["h", "z", "h", "t", "h", "z", "h"]


def test_a_registered_rule_forwards_the_callers_own_objects() -> None:
    """A trainable angle stays in the graph through a registered rule too."""

    theta = torch.tensor(0.7137, requires_grad=True)

    def build(instruction: Instruction) -> tuple[Instruction, ...]:
        return (
            Instruction(
                "rz",
                instruction.wires,
                params={"theta": theta},
                metadata=dict(instruction.metadata),
            ),
        )

    extended = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", build)
    leaves = translate(
        Instruction("tdg", (0,), metadata={"tag": 1}),
        can_run=_can_run("rz"),
        rules=extended,
    )
    assert leaves is not None
    assert leaves[0].params["theta"] is theta
    assert dict(leaves[0].metadata) == {"tag": 1}


def test_a_second_rule_is_appended_rather_than_put_first() -> None:
    """An earlier rule keeps the tie, so a later one cannot displace it."""

    def longer(instruction: Instruction) -> tuple[Instruction, ...]:
        return (
            Instruction("x", instruction.wires),
            Instruction("x", instruction.wires),
            Instruction("t", instruction.wires),
            Instruction("x", instruction.wires),
        )

    first = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", longer)
    second = with_equivalence_rule(first, "tdg", _tdg_to_x_t_x)
    assert [rule.build for rule in second["tdg"]] == [longer, _tdg_to_x_t_x]
    leaves = translate(
        Instruction("tdg", (0,)), can_run=_can_run("h", "s", "t", "cx"), rules=second
    )
    assert leaves is not None
    # The shorter registered rule wins, and the longer one is still there.
    assert [item.name for item in leaves] == [
        "h",
        "s",
        "s",
        "h",
        "t",
        "h",
        "s",
        "s",
        "h",
    ]


def test_a_built_in_rule_is_not_displaced_without_asking() -> None:
    """Registering for an opcode that already has a rule appends, it does not swap."""

    appended = with_equivalence_rule(EQUIVALENCE_RULES, "x", _tdg_to_x_t_x)
    assert len(appended["x"]) == 2
    assert appended["x"][0].build is EQUIVALENCE_RULES["x"][0].build
    leaves = translate(
        Instruction("x", (0,)), can_run=_can_run("h", "z"), rules=appended
    )
    assert leaves is not None
    assert [item.name for item in leaves] == ["h", "z", "h"]

    replaced = with_equivalence_rule(
        EQUIVALENCE_RULES, "x", _tdg_to_x_t_x, replace=True
    )
    assert len(replaced["x"]) == 1
    assert replaced["x"][0].build is _tdg_to_x_t_x
    # Non-vacuity: the two really are different rules.
    assert EQUIVALENCE_RULES["x"][0].build is not _tdg_to_x_t_x


def test_replacing_a_rule_that_does_not_exist_is_refused() -> None:
    """`replace=True` is not a way to say "add this one", and saying it is an error."""

    with pytest.raises(EquivalenceRuleError, match="no rule to replace"):
        with_equivalence_rule(EQUIVALENCE_RULES, "tdg", _tdg_to_x_t_x, replace=True)


def test_registration_validates_before_it_returns_a_rule_set() -> None:
    """A rule that cannot be a rule never reaches a mapping, let alone the search."""

    with pytest.raises(EquivalenceRuleError, match="replaces it with nothing"):
        with_equivalence_rule(EQUIVALENCE_RULES, "tdg", lambda instruction: ())
    with pytest.raises(EquivalenceRuleError, match="unknown opcode"):
        with_equivalence_rule(EQUIVALENCE_RULES, "frobnicate", _tdg_to_x_t_x)


def test_a_rule_set_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="a rule set is a mapping"):
        with_equivalence_rule(
            [EquivalenceRule("h", _tdg_to_x_t_x)], "tdg", _tdg_to_x_t_x
        )


def test_the_key_is_canonicalised_before_it_is_checked() -> None:
    """A caller writing `CNOT` registers a rule under the name the search looks up."""

    def build(instruction: Instruction) -> tuple[Instruction, ...]:
        return (
            Instruction("h", (instruction.wires[1],)),
            Instruction("cz", instruction.wires),
            Instruction("h", (instruction.wires[1],)),
        )

    extended = with_equivalence_rule(EQUIVALENCE_RULES, "CNOT", build, replace=True)
    assert "CNOT" not in extended
    assert extended["cx"][0].build is build


# --- a rule that cannot be a rule is refused, clause by clause --------------


def test_a_non_canonical_key_is_refused_by_the_validator() -> None:
    """Only reachable directly: `with_equivalence_rule` canonicalises first."""

    with pytest.raises(EquivalenceRuleError, match="is not canonical"):
        validate_equivalence_rule(EquivalenceRule("CNOT", _tdg_to_x_t_x))


def test_a_rule_for_an_unknown_opcode_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="unknown opcode"):
        validate_equivalence_rule(EquivalenceRule("frobnicate", _tdg_to_x_t_x))


def test_a_rule_for_a_channel_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="non-unitary opcode"):
        validate_equivalence_rule(EquivalenceRule("bit_flip", _tdg_to_x_t_x))


def test_a_rule_with_no_build_callable_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="no build callable"):
        validate_equivalence_rule(EquivalenceRule("h", None))  # type: ignore[arg-type]


def test_a_rule_that_builds_a_list_is_refused() -> None:
    """A sequence and a tuple are not the same promise, and the loop indexes one."""

    with pytest.raises(EquivalenceRuleError, match="not a tuple"):
        validate_equivalence_rule(
            EquivalenceRule("h", lambda instruction: [instruction])  # type: ignore[arg-type,return-value]
        )


def test_a_rule_that_replaces_nothing_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="replaces it with nothing"):
        validate_equivalence_rule(EquivalenceRule("h", lambda instruction: ()))


def test_a_rule_that_builds_something_that_is_not_an_instruction_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="not an Instruction"):
        validate_equivalence_rule(
            EquivalenceRule("h", lambda instruction: ("h",))  # type: ignore[arg-type,return-value]
        )


def test_a_rule_that_rewrites_an_opcode_only_to_itself_is_refused() -> None:
    """Such a rule is a loop, and the search would re-enter it forever."""

    with pytest.raises(EquivalenceRuleError, match="only to itself"):
        validate_equivalence_rule(
            EquivalenceRule("h", lambda instruction: (instruction,))
        )


def test_a_rule_that_is_not_a_rule_is_refused() -> None:
    with pytest.raises(EquivalenceRuleError, match="is an EquivalenceRule, not str"):
        validate_equivalence_rule("h")  # type: ignore[arg-type]


def test_the_built_in_table_passes_its_own_validator() -> None:
    """The validator and the table agree, entry for entry."""

    for opcode, rules in EQUIVALENCE_RULES.items():
        assert rules, opcode
        for rule in rules:
            validate_equivalence_rule(rule)


def test_the_registration_surface_is_what_the_module_exports() -> None:
    import flagquantum.compiler.basis_translation as module

    assert "with_equivalence_rule" in module.__all__
    assert "validate_equivalence_rule" in module.__all__
    assert "EquivalenceRuleError" in module.__all__
    assert issubclass(EquivalenceRuleError, CompilationError)
