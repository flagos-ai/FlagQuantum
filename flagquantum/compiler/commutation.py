"""Decide whether two instructions commute, and partition a circuit by that.

A commuting-block partition is the precondition almost every reduction needs.
`merge_self_inverse` and `merge_adjacent_rotations` in `pipeline` only look at the
latest writer of a qubit, so a single gate standing between a pair hides it:
`cx(0, 1) rz(0) cx(0, 1)` is one `rz(0)`, and neither pass sees it, because `rz`
on the *control* of a `cx` commutes with it and the pair annihilates across the
gap. This module answers the two questions that reduction needs: do these two
instructions commute, and which instructions form one commuting block on a qubit.

Qiskit splits this over `CommutationChecker`, a shipped 3264-line rule table, and
`CommutationAnalysis`, an `AnalysisPass` that writes `property_set`. Neither shape
transfers. FlagQuantum has no pass manager and no property bag, so the analysis
here is a function of the IR that returns a value; and a generated table is a rule
source of record, which would have to be trusted, where the rules below are a rule
source of proof.

Four properties make a verdict admissible.

**Two instructions on disjoint qubits commute.** This is unconditional and holds
for every instruction the IR can carry, including `measure`, `reset`, and
`barrier`, so it is answered before any opcode is inspected. It is also the rule
that carries most of the reach, because a circuit is wide and most pairs never
touch.

**No rule reads a parameter value.** Every rule below is a statement about
opcodes, qubits, and two declared opcode classes, and each of them holds for *every*
value of every parameter. So a trainable rotation and a batch of angles are
decided exactly like a compile-time constant, and nothing here can detach a
gradient or pin a program to one batch entry. That is the property a table of
sampled matrices could not have, and it is why this module needs no tolerance:
the only comparison it makes is exact equality of a small integer and a string.

**Every rule is checked against the runtime's matrices, not assumed.**
`tests/unit/test_compilation_commutation.py` builds each claimed pair's operators
from `flagquantum.simulation.gate_matrix` -- which the compiler layer may not
import -- and compares `left @ right` with `right @ left` entry for entry, so an
entry that is wrong is a failing test rather than a wrong program.

**Everything else is declined.** A false "no" costs an optimization the caller
would have to earn; a false "yes" changes the program. An opcode this module has
no rule for, a caller-supplied `matrix` override whose contents cannot be read as
a declared opcode, the non-unitary channels, and `measure`, `reset`, and `barrier`
all answer "no", which makes them commutation barriers. That is the fail-closed
reading, and for a `measure` it is also the only correct one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import canonical_opcode, get_operator_schema

#: Single-qubit opcodes whose runtime matrix is diagonal in the computational
#: basis. Every angle is acceptable, which is what makes the rule exact without
#: reading one: `U3` is excluded, because `U3(theta, 0, 0)` is `RY(theta)`.
_DIAGONAL_ONE_QUBIT = frozenset({"i", "z", "s", "sdg", "t", "tdg", "rz", "phase", "u1"})

#: Two-qubit opcodes whose runtime matrix is diagonal in the computational basis.
_DIAGONAL_ENTANGLER = frozenset({"cz", "cphase", "crz", "rzz"})

#: The Ising entanglers, each `exp(-i theta/2 P)` for a Pauli product `P` on the two
#: qubits -- `XX`, `YY`, and `ZZ`. Two properties decide the pair, and the rule
#: below reads them rather than approximating them. Two of one opcode commute
#: whatever their qubits are, because every factor is the same Pauli and Paulis
#: commute with themselves: `X_1 X_2` and `X_0 X_1` multiply to `X_0 X_2` in either
#: order. Two of *different* opcodes commute only on the same qubits, where
#: `(X_0 Y_0)(X_1 Y_1)` is `-Z_0 Z_1` in either order; on qubits that merely
#: overlap, `X_1 X_2` and `Y_0 Y_1` share qubit 1 with factors `X` and `Y`, which
#: anticommute, so the exponentials do too.
_ISING_ENTANGLER = frozenset({"rxx", "ryy", "rzz"})

#: Two-qubit opcodes that are unchanged by exchanging their two qubits, so `swap`
#: commutes with each of them on the same ordered qubits: `swap` *is* that exchange.
#: This is the whole content of the class -- exchange symmetry alone does not make
#: its members commute with each other, and `cz` against `rxx` is the counterexample
#: that says so: `Z (x) Z` and `X (x) X` anticommute, and `cz` is a scalar multiple
#: of `exp(-i pi/4 ZZ)`. `cz` and `cphase` commute with each other by being diagonal
#: and `rxx`, `ryy`, and `rzz` by being Ising entanglers, both of which are separate
#: rules above. `crx`, `cry`, and `crz` are absent, because a controlled rotation is
#: not exchange-symmetric and `swap` does not commute with one.
_WIRE_EXCHANGE_SYMMETRIC = frozenset({"cz", "cphase", *sorted(_ISING_ENTANGLER)})

#: Two families of controlled opcodes that each act as the identity on the
#: control's `|0>` branch and as a Pauli-family operator on its `|1>` branch, so
#: two members of one family on the same qubits commute for any angles. `cx` and
#: `cy` are the parameter-free ends of their families and `crx` and `cry` the
#: rotations, which is why `cx` against `crx` is a pair this rule has to state:
#: `crx` alone is in `_SINGLE_GENERATOR`, but the pair is two different opcodes
#: and no branch above covers a pair of different opcodes.
_PAULI_FAMILIES = (
    frozenset({"cx", "crx"}),
    frozenset({"cy", "cry"}),
)

#: Opcodes of the form `exp(-i theta/2 G)` for one Hermitian generator `G`, so any
#: two of them sharing an opcode and their qubits commute for any angles.
_SINGLE_GENERATOR = frozenset(
    {
        "rx",
        "ry",
        "rz",
        "phase",
        "u1",
        "crx",
        "cry",
        "crz",
        "cphase",
        "rxx",
        "ryy",
        "rzz",
    }
)

#: For each controlled opcode, the qubit positions that act as controls. A diagonal
#: single-qubit gate on a control commutes with the whole controlled operator,
#: which is `sum_a |a><a|_control (x) U_a`: a diagonal gate scales `|a>` by a
#: scalar, and that scalar pulls out of every term. `cz`, `crz`, and `cphase` are
#: diagonal outright and are answered by `_DIAGONAL_ENTANGLER` before this is
#: consulted. `swap` has no control and is deliberately absent.
_CONTROL_POSITIONS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "cx": (0,),
        "cy": (0,),
        "crx": (0,),
        "cry": (0,),
        "cswap": (0,),
        "ccx": (0, 1),
    }
)

#: For each controlled opcode, the target position and the single-qubit opcodes
#: that commute with it there. A gate commutes with the target position when it
#: commutes with the operator applied to the target: `cx` needs a gate in
#: `span{I, X}`, which is `x` and every `rx`, and also `sx` and `sxdg`, which are
#: `e^{+-i pi/4} RX(+-pi/2)`; `cy` needs `span{I, Y}`, which is `y` and every `ry`;
#: `ccx` applies `X` to its target only under both controls, so it needs
#: `span{I, X}` as well. `cswap` applies `swap`, which no single-qubit gate
#: commutes with.
_TARGET_PAULI: Mapping[str, tuple[int, frozenset[str]]] = MappingProxyType(
    {
        "cx": (1, frozenset({"x", "rx", "sx", "sxdg"})),
        "crx": (1, frozenset({"x", "rx", "sx", "sxdg"})),
        "cy": (1, frozenset({"y", "ry"})),
        "cry": (1, frozenset({"y", "ry"})),
        "ccx": (2, frozenset({"x", "rx", "sx", "sxdg"})),
    }
)

#: Every opcode the structural rules below can answer "yes" for. An opcode outside
#: this set still commutes by disjointness, and still commutes with the same
#: opcode on the same qubits through `_is_same_operator_family`; what it cannot do
#: is commute with a *different* operator it touches. Stating the set here keeps
#: the reach of the rule source a readable fact rather than something to be
#: reconstructed from the branches of `commute`.
_RULE_OPCODES = frozenset(
    _DIAGONAL_ONE_QUBIT
    | _DIAGONAL_ENTANGLER
    | _ISING_ENTANGLER
    | _WIRE_EXCHANGE_SYMMETRIC
    | {"swap"}
    | frozenset(_CONTROL_POSITIONS)
    | frozenset(_TARGET_PAULI)
    | frozenset().union(
        *(names for _, names in _TARGET_PAULI.values()),
        *(family for family in _PAULI_FAMILIES),
    )
)


def _parameters(opcode: str) -> tuple[str, ...] | None:
    """The declared parameter names of ``opcode``, or None when it has no schema.

    None covers the three instruction names that are not operators at all --
    `measure`, `reset`, and `barrier` -- and every opcode that is not declared.
    Both are declined rather than read as an empty parameter list, which would
    make an unknown instruction look like a fixed gate.
    """

    schema = get_operator_schema(opcode)
    return None if schema is None else schema.parameters


def _is_declared(instruction: Instruction) -> bool:
    """Whether ``instruction`` is an operator this module is allowed to read.

    A `matrix` override replaces the opcode's matrix with caller-supplied content,
    so the opcode name stops describing what runs and no rule may be applied to
    it. This is the same refusal `one_qubit_optimization` makes, for the same
    reason.
    """

    return (
        instruction.matrix is None
        and _parameters(canonical_opcode(instruction.name)) is not None
    )


def _is_same_operator_family(left: Instruction, right: Instruction) -> bool:
    """Whether both are one opcode on the same qubits that commutes with itself.

    Two instructions of one opcode on one set of qubits commute when the opcode is
    fixed -- an operator always commutes with itself -- or when it is
    `exp(-i theta/2 G)`, because two such exponentials add their angles whichever
    order they are applied in. Nothing else qualifies: two different `u3` triples
    share an opcode and generally do not commute, so `u3` is absent from
    `_SINGLE_GENERATOR` and a pair of them is a barrier.
    """

    opcode = canonical_opcode(left.name)
    if opcode != canonical_opcode(right.name) or left.wires != right.wires:
        return False
    if opcode in _SINGLE_GENERATOR:
        return True
    declared = _parameters(opcode)
    return (
        declared is not None and not declared and not left.params and not right.params
    )


def _one_against_controlled(single: Instruction, controlled: Instruction) -> bool:
    """Whether the single-qubit ``single`` commutes with ``controlled``.

    ``controlled`` has to be the multi-qubit operator and ``single`` the gate
    placed on one of its qubits; the caller tries both assignments because the two
    instructions arrive in program order rather than in role order.
    """

    if len(single.wires) != 1 or len(controlled.wires) < 2:
        return False
    if single.wires[0] not in controlled.wires:
        return False

    single_opcode = canonical_opcode(single.name)
    controlled_opcode = canonical_opcode(controlled.name)
    position = controlled.wires.index(single.wires[0])

    if single_opcode in _DIAGONAL_ONE_QUBIT:
        return position in _CONTROL_POSITIONS.get(controlled_opcode, ())
    target = _TARGET_PAULI.get(controlled_opcode)
    if target is None or target[0] != position:
        return False
    return single_opcode in target[1]


def commute(left: Instruction, right: Instruction) -> bool:
    """Whether ``left`` and ``right`` commute exactly.

    Returns True only for a pair proven to commute for every value of every
    parameter. False means "not proven", which a caller must read as "may not be
    reordered": the rule source fails closed, so `measure`, `reset`, `barrier`,
    the non-unitary channels, and any opcode outside `_RULE_OPCODES` are all
    commutation barriers.
    """

    if set(left.wires).isdisjoint(right.wires):
        return True
    if not _is_declared(left) or not _is_declared(right):
        return False
    if _is_same_operator_family(left, right):
        return True

    left_opcode = canonical_opcode(left.name)
    right_opcode = canonical_opcode(right.name)

    # A pair no rule can answer "yes" for is answered here, which keeps the reach
    # of the rule source the readable fact `_RULE_OPCODES` states rather than
    # something a reader has to reconstruct from the branches below.
    if left_opcode not in _RULE_OPCODES or right_opcode not in _RULE_OPCODES:
        return False

    # Every rule below states that two operators are simultaneously diagonal in
    # the computational basis of the whole register. Diagonal matrices commute
    # whatever qubits they sit on and whichever order the qubits are named in, so
    # this one branch covers a single-qubit `rz` against another, a single-qubit
    # `rz` against a `cz`, and a `cz` against a `cphase`.
    if left_opcode in _DIAGONAL_ONE_QUBIT | _DIAGONAL_ENTANGLER and (
        right_opcode in _DIAGONAL_ONE_QUBIT | _DIAGONAL_ENTANGLER
    ):
        return True
    if left_opcode in _ISING_ENTANGLER and right_opcode in _ISING_ENTANGLER:
        # One opcode twice commutes on any qubits, and two different opcodes commute
        # only when they sit on the same qubits. See `_ISING_ENTANGLER`.
        return left_opcode == right_opcode or left.wires == right.wires

    # The remaining two rules need the two operators to act on the same ordered
    # qubits: `swap` against an exchange-symmetric operator, and the `cx`/`crx`
    # family pair, which no branch above reaches because the two are different
    # opcodes.
    if left.wires == right.wires:
        if left_opcode == "swap" and right_opcode in _WIRE_EXCHANGE_SYMMETRIC:
            return True
        if right_opcode == "swap" and left_opcode in _WIRE_EXCHANGE_SYMMETRIC:
            return True
        if any(
            left_opcode in family and right_opcode in family
            for family in _PAULI_FAMILIES
        ):
            return True

    return _one_against_controlled(left, right) or _one_against_controlled(right, left)


@dataclass(frozen=True)
class CommutationAnalysis:
    """The commuting blocks of a circuit, one ordered partition per qubit.

    Qiskit's `CommutationAnalysis` is an `AnalysisPass` that writes
    `property_set["commutation_set"]`, keyed by DAG node and qubit. This carries
    the same structure as a value keyed by program position, which is what the IR
    has: ``blocks_by_qubit[qubit]`` is an ordered tuple of blocks, each a tuple of
    instruction positions in program order, and ``block_index`` answers which
    block of a qubit a position fell into.

    The invariant a consumer may rely on is the one `commute` supports: every
    block is a set of instructions that pairwise commute, and each qubit's blocks
    partition that qubit's instructions into consecutive runs. ``queried_pairs``
    and ``commuting_pairs`` are the counters behind that invariant, so a caller
    can tell a real partition from a degenerate one where nothing commutes.
    """

    blocks_by_qubit: Mapping[int, tuple[tuple[int, ...], ...]]
    block_index: Mapping[tuple[int, int], int]
    queried_pairs: int
    commuting_pairs: int

    def block_of(self, position: int, qubit: int) -> int | None:
        """The index of the block of ``qubit`` holding ``position``, or None."""

        return self.block_index.get((position, qubit))

    def signature(
        self, position: int, qubits: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """The block index of ``position`` on each of ``qubits``, or None.

        None means the position is not in a block on one of the qubits it was
        asked about, which no instruction of the analyzed circuit can be. It is
        returned rather than raised so that a caller holding an instruction from
        another program declines instead of misreading a block index.
        """

        indices = []
        for qubit in qubits:
            index = self.block_index.get((position, qubit))
            if index is None:
                return None
            indices.append(index)
        return tuple(indices)

    def summary(self) -> dict[str, Any]:
        """The partition's shape and the work it took, for a compilation record."""

        block_count = sum(len(blocks) for blocks in self.blocks_by_qubit.values())
        widest = max(
            (
                len(block)
                for blocks in self.blocks_by_qubit.values()
                for block in blocks
            ),
            default=0,
        )
        return {
            "qubit_count": len(self.blocks_by_qubit),
            "block_count": block_count,
            "widest_block": widest,
            "queried_pairs": self.queried_pairs,
            "commuting_pairs": self.commuting_pairs,
        }


def analyze_commutation(ir: CircuitIR) -> CommutationAnalysis:
    """Partition ``ir`` into commuting blocks, one ordered partition per qubit.

    The sweep is the one Qiskit's `CommutationAnalysis` performs: walk a qubit's
    instructions in program order and grow the open block while the candidate
    commutes with every instruction already in it, otherwise close the block and
    open a new one with the candidate. The result is a valid partition rather than
    the coarsest one -- a candidate that fails is not retried against the block
    opened after it -- and a valid one is all a consumer needs, because the
    guarantee it uses is that a block's members pairwise commute.
    """

    positions_by_qubit: dict[int, list[int]] = {}
    for position, instruction in enumerate(ir):
        for qubit in instruction.wires:
            positions_by_qubit.setdefault(qubit, []).append(position)

    instructions = tuple(ir)
    blocks_by_qubit: dict[int, tuple[tuple[int, ...], ...]] = {}
    block_index: dict[tuple[int, int], int] = {}
    queried = 0
    commuting = 0
    for qubit, positions in positions_by_qubit.items():
        blocks: list[tuple[int, ...]] = []
        open_block: list[int] = []
        for position in positions:
            candidate = instructions[position]
            accepted = True
            for member in open_block:
                queried += 1
                if commute(instructions[member], candidate):
                    commuting += 1
                else:
                    accepted = False
                    break
            if not accepted:
                blocks.append(tuple(open_block))
                open_block = []
            open_block.append(position)
            block_index[(position, qubit)] = len(blocks)
        blocks.append(tuple(open_block))
        blocks_by_qubit[qubit] = tuple(blocks)

    return CommutationAnalysis(
        blocks_by_qubit=MappingProxyType(blocks_by_qubit),
        block_index=MappingProxyType(block_index),
        queried_pairs=queried,
        commuting_pairs=commuting,
    )


__all__ = ["CommutationAnalysis", "analyze_commutation", "commute"]
