"""Measure how far two-qubit KAK synthesis extends native-gate legalization.

W8-02 of the Qiskit parity backlog asks for the two-qubit decomposition Qiskit
keeps in ``synthesis/two_qubit``. Before it, native-gate legalization refused
every matrix-carrying instruction outright, so a target that publishes ``cx``
plus a z-rotation and a pi/2 x-rotation — which is the whole of the two-qubit
group a real QPU can execute — could not legalize ``swap``, ``cz``, ``rxx``, or
any other two-qubit unitary a caller handed it as a matrix.

``two_qubit_synthesis`` closes that with the KAK identity

    ``U == exp(i*phase) * (K1l x K1r) * exp(i*(a XX + b YY + c ZZ)) * (K2l x K2r)``

up to a global phase, plus the local factors emitted by the ``one_qubit`` Euler
synthesis W8-01 added. Qiskit's ``TwoQubitBasisDecomposer`` does the same thing
with a different Euler basis and with specialization heuristics the port
deliberately leaves out; see that module for why.

This module measures the reach that buys, the cost in entanglers, and the one
property a consumer has to know about:

* A ``cz`` or ``cx`` basis with a z-rotation and a pulse legalizes **11 of 11**
  declared two-qubit unitary opcodes, and every one of them through an explicit
  matrix as well as by name. The entangler cost is **1** for the controlled
  gates, **2** for the controllized rotations, and **3** for ``swap``, which is
  the far corner of the Weyl chamber.
* The synthesized program is equal to its source up to **one global phase**, one
  per decomposed instruction plus one per local Euler factor. The overlap
  magnitude is 1.0 and the raw statevector difference is not zero.
* Qiskit's own ``TwoQubitBasisDecomposer`` chooses the **same entangler count**
  for the same physical operator on every case measured here, which is the one
  cross-framework number this benchmark asserts. Raw instruction counts differ
  because the two sides use a different Euler basis (``ZSX`` against a different
  leaf order), which is a presentation difference, not a reach or cost one.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_two_qubit_synthesis.py --json-output /tmp/w802.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import torch

from flagquantum.compiler.basis_translation import translate
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    _native_descriptors,
    _supports,
    legalize_native_gates,
)
from flagquantum.compiler.two_qubit_optimization import split_two_qubit_blocks
from flagquantum.compiler.two_qubit_synthesis import (
    SUPERCONTROLLED_ENTANGLERS,
    synthesize_two_qubit,
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
from flagquantum.simulation.gate_matrix import gate_matrix
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_two_qubit_synthesis_benchmark_v1"

_NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
_ANGLES: dict[str, float] = {"theta": 0.7137}
COMPLEX = torch.complex128

#: The seeded population the reach and cost tables are measured on, so the
#: numbers are reproducible without shipping a matrix fixture.
RANDOM_CASE_COUNT = 32
_RANDOM_SEED = 20261002


@dataclass(frozen=True)
class Basis:
    """One target basis, named by the gates its evidence publishes."""

    label: str
    gates: tuple[dict[str, Any], ...]
    entangler: str | None
    z_rotation: str | None
    pulse_opcode: str


def _gate(name: str) -> dict[str, Any]:
    return {"name": name, "parameters": tuple(OPERATOR_SCHEMAS[name].parameters)}


#: The two IBM-style shapes a QPU basis publishes, the rotation-only basis this
#: repository already records elsewhere, and a trapped-ion shape whose only
#: two-qubit gate is a rotation. ``clifford-t`` is the fail-closed control: it
#: publishes neither a z-rotation nor a pulse.
DEFAULT_BASES: tuple[Basis, ...] = (
    Basis(
        "ibm-rz-sx-cx",
        (_gate("rz"), _gate("sx"), _gate("x"), _gate("cx")),
        "cx",
        "rz",
        "sx",
    ),
    # Heron-class hardware publishes ECR rather than CX, but ECR is not a
    # declared FlagQuantum opcode, so the closest real basis is `cz`.
    Basis(
        "ibm-heron-cz",
        (_gate("rz"), _gate("sx"), _gate("x"), _gate("cz")),
        "cz",
        "rz",
        "sx",
    ),
    Basis("rotational", (_gate("rz"), _gate("rx"), _gate("cz")), "cz", "rz", "rx"),
    # Ion-trap and flux-tunable-coupler targets publish an interaction rotation
    # and no `cx` at all; this is the basis W8-03 opens.
    Basis(
        "ion-trap-rz-rx-rzz",
        (_gate("rz"), _gate("rx"), _gate("rzz")),
        "rzz",
        "rz",
        "rx",
    ),
    Basis(
        "clifford-t",
        (_gate("h"), _gate("s"), _gate("t"), _gate("cx")),
        "cx",
        None,
        "sx",
    ),
)

#: One instruction of every declared arity-2 unitary opcode. The reach below is
#: only a statement about the group if this really is the whole group.
TWO_QUBIT_OPCODES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 2
    )
)


def two_qubit_instruction(opcode: str, *, matrix: bool) -> Instruction:
    """One arity-2 instruction of ``opcode``, with or without its matrix."""

    schema = OPERATOR_SCHEMAS[opcode]
    params = {name: _ANGLES[name] for name in schema.parameters}
    if not matrix:
        return Instruction(opcode, (0, 1), params=params)
    values = gate_matrix(
        Instruction(opcode, (0, 1), params=params),
        bsz=1,
        device=torch.device("cpu"),
        dtype=COMPLEX,
    ).reshape(4, 4)
    return Instruction(opcode, (0, 1), params=params, matrix=values.tolist())


def _random_su4(seed: int) -> list[list[complex]]:
    """A seeded SU(4) built from a complex Gram-Schmidt, in pure Python."""

    rng = random.Random(seed)
    rows = [
        [complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(4)]
        for _ in range(4)
    ]
    columns: list[list[complex]] = []
    for index in range(4):
        vector = [rows[row][index] for row in range(4)]
        for previous in columns:
            overlap = sum(previous[row].conjugate() * vector[row] for row in range(4))
            vector = [vector[row] - overlap * previous[row] for row in range(4)]
        norm = math.sqrt(sum(abs(item) ** 2 for item in vector))
        columns.append([item / norm for item in vector])
    matrix = [[columns[column][row] for column in range(4)] for row in range(4)]
    determinant = _determinant(matrix)
    phase = math.atan2(determinant.imag, determinant.real) / 4.0
    rotation = complex(math.cos(phase), -math.sin(phase))
    return [[item * rotation for item in row] for row in matrix]


def _determinant(matrix: list[list[complex]]) -> complex:
    total = 0j
    for permutation in _permutations((0, 1, 2, 3)):
        sign = 1.0
        for position, value in enumerate(permutation):
            for other in range(position + 1, 4):
                if permutation[other] < value:
                    sign = -sign
        term = complex(sign)
        for row, column in enumerate(permutation):
            term *= matrix[row][column]
        total += term
    return total


def _permutations(items: tuple[int, ...]):
    if len(items) <= 1:
        yield items
        return
    for index, item in enumerate(items):
        rest = items[:index] + items[index + 1 :]
        for tail in _permutations(rest):
            yield (item, *tail)


def random_case(index: int) -> list[list[complex]]:
    return _random_su4(_RANDOM_SEED + index)


def snapshot(basis: Basis) -> TargetCapabilitySnapshot:
    """A verified, in-window capability snapshot declaring ``basis``'s gates."""

    scope = CapabilityScope(device_ids=("w802:0",))
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=f"w802-{basis.label}",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="w802-environment",
        ),
        scope=scope,
        captured_at=_NOW.isoformat(),
        valid_until=_NOW.replace(hour=9).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=basis.gates,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=FactSource(kind="w802_probe", ref="w802-evidence"),
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="w802-evidence",
                sha256="b" * 64,
                level=EvidenceLevel.BASIC,
                scope=scope,
            ),
        ),
    )


def legalize(program: CircuitIR, basis: Basis):
    """Legalize ``program``, returning the error text instead of raising."""

    try:
        return legalize_native_gates(
            program, snapshot=snapshot(basis), evaluated_at=_NOW
        )
    except NativeGateLegalizationError as error:
        return str(error)


#: The only arity-2 opcode the legalizer had a rule for before the equivalence
#: table replaced those rules at W8-04. That rule is still in the table verbatim,
#: so replaying the named path with the table restricted to this opcode measures
#: the pre-W8-04 reach from the same descriptors instead of quoting it.
_PRE_TABLE_NAMED_OPCODES = frozenset({"swap"})


def baseline_reach(
    basis: Basis, *, opcodes: frozenset[str] | None = None
) -> dict[str, Any]:
    """How far the named path reaches through the equivalence table, no synthesis.

    With ``opcodes`` the table is restricted to that set, which replays an
    earlier version of the table rather than the current one; the default is the
    whole table. Called once with ``_PRE_TABLE_NAMED_OPCODES`` for the before
    count and once unrestricted for what the table reaches on its own. A
    matrix-carrying instruction has no path here at all, which
    ``legalized_by_matrix_count`` records separately.
    """

    descriptors = _native_descriptors(snapshot(basis), evaluated_at=_NOW)
    legalized: list[str] = []
    for opcode in TWO_QUBIT_OPCODES:
        instruction = two_qubit_instruction(opcode, matrix=False)
        if _supports(descriptors, instruction):
            legalized.append(opcode)
            continue
        if opcodes is not None and opcode not in opcodes:
            continue
        replacement = translate(instruction, can_run=partial(_supports, descriptors))
        if replacement is not None and all(
            _supports(descriptors, leaf) for leaf in replacement
        ):
            legalized.append(opcode)
    return {
        "label": basis.label,
        "legalized_opcode_count": len(legalized),
        "legalized_opcodes": legalized,
    }


def reach(basis: Basis) -> dict[str, Any]:
    """How much of the declared two-qubit group ``basis`` can legalize.

    Both entry points are measured: a caller that names a two-qubit gate, and a
    caller that holds its matrix, which is what a two-qubit synthesis pass
    produces. The second is the one W8-02 adds.
    """

    by_name: list[str] = []
    by_matrix: list[str] = []
    unresolved: dict[str, str] = {}
    entanglers: dict[str, int] = {}
    for opcode in TWO_QUBIT_OPCODES:
        named = legalize(
            CircuitIR(
                2, (two_qubit_instruction(opcode, matrix=False),), dtype="complex128"
            ),
            basis,
        )
        if not isinstance(named, str):
            by_name.append(opcode)
        carrying = legalize(
            CircuitIR(
                2, (two_qubit_instruction(opcode, matrix=True),), dtype="complex128"
            ),
            basis,
        )
        if isinstance(carrying, str):
            unresolved[opcode] = carrying
            continue
        by_matrix.append(opcode)
        # Counted on the emitted program rather than on the decomposition record:
        # an opcode the basis already carries natively, such as `cx` in a `cx`
        # basis, is kept as it is and produces no record at all.
        entanglers[opcode] = sum(
            1 for item in carrying.program.instructions if item.name == basis.entangler
        )
    before = baseline_reach(basis, opcodes=_PRE_TABLE_NAMED_OPCODES)
    table = baseline_reach(basis)
    return {
        "label": basis.label,
        "declared_opcode_count": len(TWO_QUBIT_OPCODES),
        "legalized_by_name_count": len(by_name),
        "legalized_by_name_opcodes": by_name,
        "legalized_by_matrix_count": len(by_matrix),
        "unresolved_opcode_count": len(unresolved),
        "unresolved_errors": unresolved,
        "entangler_count_by_opcode": entanglers,
        "total_entangler_count": sum(entanglers.values()),
        "entangler_opcode": basis.entangler,
        "z_rotation_opcode": basis.z_rotation,
        "pulse_opcode": basis.pulse_opcode,
        "native_opcodes": sorted(
            item.name if isinstance(item, str) else str(item["name"])
            for item in basis.gates
        ),
        "hand_written_reach_count": before["legalized_opcode_count"],
        "hand_written_reach_opcodes": before["legalized_opcodes"],
        "identity_table_reach_count": table["legalized_opcode_count"],
        "identity_table_reach_opcodes": table["legalized_opcodes"],
    }


def random_reach(basis: Basis) -> dict[str, Any]:
    """The same measurement on a seeded population that is not a named gate."""

    worst_gap = 0.0
    entanglers: dict[int, int] = {}
    reached = 0
    for index in range(RANDOM_CASE_COUNT):
        matrix = random_case(index)
        instruction = Instruction("blob", (0, 1), matrix=matrix)
        result = legalize(CircuitIR(2, (instruction,), dtype="complex128"), basis)
        if isinstance(result, str):
            continue
        reached += 1
        count = sum(
            1 for item in result.program.instructions if item.name == basis.entangler
        )
        entanglers[count] = entanglers.get(count, 0) + 1
        worst_gap = max(worst_gap, _reconstruction_gap(matrix, result.program))
    return {
        "label": basis.label,
        "case_count": RANDOM_CASE_COUNT,
        "reached_count": reached,
        "worst_reconstruction_gap": worst_gap,
        "entangler_count_distribution": dict(sorted(entanglers.items())),
    }


def _embed(instruction: Instruction) -> torch.Tensor:
    width = 2 ** len(instruction.wires)
    matrix = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1, width, width)[0]
    if len(instruction.wires) == 2:
        return matrix
    identity = torch.eye(2, dtype=COMPLEX)
    if instruction.wires[0] == 0:
        return torch.kron(matrix, identity)
    return torch.kron(identity, matrix)


def _product(instructions: tuple[Instruction, ...]) -> torch.Tensor:
    total = torch.eye(4, dtype=COMPLEX)
    for instruction in instructions:
        total = _embed(instruction) @ total
    return total


def _reconstruction_gap(matrix: list[list[complex]], program: CircuitIR) -> float:
    """Max entry of the residual once the single global phase is divided out."""

    source = torch.tensor(matrix, dtype=COMPLEX)
    product = _product(program.instructions)
    overlap = torch.trace(source.conj().T @ product) / 4
    assert abs(overlap) > 0.5, "the synthesized product is not close to the source"
    return float(torch.max(torch.abs(source - product / overlap)))


def mixed_program(basis: Basis) -> tuple[CircuitIR, CircuitIR]:
    """A named reference program and the same program carrying matrices.

    The reference is what the circuit *is*; the second is what a caller hands the
    legalizer when it holds a unitary rather than a gate name, which is the path
    W8-02 adds. The statevector runtime can execute the reference directly; it
    cannot execute a matrix-carrying instruction, because a matrix instruction is
    a compiler input rather than a runtime operation.
    """

    entangler = basis.entangler or "cx"
    second = "cz" if entangler == "cx" else "cx"
    names: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("swap", (0, 1)),
        ("h", (0,)),
        (second, (1, 0)),
        ("u3", (1,)),
        ("sdg", (0,)),
    )
    reference = CircuitIR(
        2,
        tuple(
            Instruction(
                name,
                wires,
                params=(
                    {"theta": -1.2, "phi": 0.4, "lbd": 2.1} if name == "u3" else {}
                ),
            )
            for name, wires in names
        ),
        dtype="complex128",
    )
    carrying = CircuitIR(
        2,
        tuple(
            Instruction(
                name,
                wires,
                params=(
                    {"theta": -1.2, "phi": 0.4, "lbd": 2.1} if name == "u3" else {}
                ),
                matrix=(
                    gate_matrix(
                        Instruction(name, wires),
                        bsz=1,
                        device=torch.device("cpu"),
                        dtype=COMPLEX,
                    )
                    .reshape(2 ** len(wires), 2 ** len(wires))
                    .tolist()
                    if name in ("swap", second)
                    else None
                ),
            )
            for name, wires in names
        ),
        dtype="complex128",
    )
    return reference, carrying


def phase_evidence(basis: Basis) -> dict[str, Any]:
    """Measure what legalizing an entangled mixed program does to the state.

    The comparison is against the *named* program the matrices came from, so the
    measurement covers the whole path: matrix in, native gates out, state equal
    up to one global phase.
    """

    reference, source = mixed_program(basis)
    result = legalize(source, basis)
    if isinstance(result, str):
        return {"label": basis.label, "legalized": False, "error": result}
    original = run_local_statevector(
        reference, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1)
    legalized = run_local_statevector(
        result.program, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1)
    overlap = torch.vdot(original, legalized)
    return {
        "label": basis.label,
        "legalized": True,
        "entangler": basis.entangler,
        "source_gate_count": len(reference.instructions),
        "legalized_gate_count": len(result.program.instructions),
        "decomposition_count": len(result.decompositions),
        "matrix_instruction_count": sum(
            1 for item in result.program.instructions if item.matrix is not None
        ),
        "state_overlap_magnitude": float(torch.abs(overlap)),
        "global_phase_radians": float(torch.angle(overlap)),
        "max_raw_state_difference": float(torch.max(torch.abs(legalized - original))),
    }


#: The population the local-gate split is measured on. A product of two declared
#: single-qubit unitaries is the one input class `split_two_qubit_blocks` admits,
#: and every factor is a declared opcode rather than a random matrix, so the
#: population is one the schema table already names and re-derives from the index
#: alone.
_SPLIT_SEED = 20261024
SPLIT_CASE_COUNT = 96
SPLIT_FACTORS: tuple[str, ...] = (
    "i",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "t",
    "tdg",
)

#: A target that publishes a z-rotation and a pulse and no entangler at all. The
#: entangler route has no answer on it, so the reach the split route adds there is
#: a count rather than a claim.
ROTATION_ONLY_BASIS = Basis(
    "rz-sx-rotation-only", (_gate("rz"), _gate("sx")), None, "rz", "sx"
)


def _factor_matrix(opcode: str) -> list[list[complex]]:
    values = gate_matrix(
        Instruction(opcode, (0,)), bsz=1, device=torch.device("cpu"), dtype=COMPLEX
    )
    return values.reshape(2, 2).tolist()


def product_case(index: int) -> tuple[str, str, list[list[complex]]]:
    """One product of two declared single-qubit unitaries, addressed by case index.

    The two opcodes are returned beside the product so that the hand-split arm of
    `product_split` can be built from the factors themselves rather than from the
    pass's own reading of the product, which is what keeps that arm independent.
    """

    rng = random.Random(_SPLIT_SEED + index)
    left_opcode = rng.choice(SPLIT_FACTORS)
    right_opcode = rng.choice(SPLIT_FACTORS)
    left = _factor_matrix(left_opcode)
    right = _factor_matrix(right_opcode)
    return (
        left_opcode,
        right_opcode,
        [
            [
                left[row // 2][column // 2] * right[row % 2][column % 2]
                for column in range(4)
            ]
            for row in range(4)
        ],
    )


def _matrix_residual(
    matrix: list[list[complex]], program: CircuitIR
) -> tuple[float, float]:
    """Max entry and overlap angle of `matrix` against a program's own product.

    Nothing is divided out, so the entry residual answers "does this program
    compute the operator itself" rather than "does it compute it up to a phase".
    The second number is the angle that would have to be recorded to make the two
    agree, which is the quantity CircuitIR has no field for.
    """

    source = torch.tensor(matrix, dtype=COMPLEX)
    product = _product(program.instructions)
    overlap = torch.trace(source.conj().T @ product) / 4
    return (
        float(torch.max(torch.abs(source - product))),
        abs(float(torch.angle(overlap))),
    )


def product_split(basis: Basis) -> dict[str, Any]:
    """Measure the local-gate split on the input class it admits.

    Three arms over one seeded population, and each is asked only what it can
    answer:

    * ``shipped`` -- the product handed over as one two-wire matrix instruction,
      which is what `legalize_native_gates` sees from an adapter or a caller;
    * ``hand_split`` -- the same operator written as its two declared factors on
      their own wires, which is the only thing a caller could write before the
      split existed;
    * ``entangler`` -- the route the two-wire matrix branch took instead,
      `synthesize_two_qubit` against the basis's own entangler.

    The claim is **reach**: every case the hand-written route serves, the shipped
    route serves too, and it serves them as matrices rather than as the declared
    names the hand-written route could lean on. Its length is deliberately *not*
    compared against the hand-written route case for case, because the two
    programs are not the same input -- the hand-written one names ``x`` where the
    shipped one holds an opaque matrix, and a basis that publishes ``x`` keeps the
    name. That difference is name preservation, not reach, and counting it would
    report an artifact of the comparison as a property of the pass.

    The entangler arm is compared by length honestly: both arms are handed the
    same matrix. Where a basis publishes no entangler at all, that arm reaches
    nothing and the count is a failure rather than a tie.

    A basis that serves no arm -- ``clifford-t`` publishes no z-rotation, so a
    one-wire matrix has no route either -- is recorded with zero comparable cases
    rather than folded into a comparison that cannot be made.
    """

    entangler_longer = 0
    entangler_shorter = 0
    shipped_unreachable = 0
    hand_split_unreachable = 0
    entangler_unreachable = 0
    comparable = 0
    worst_split_residual = 0.0
    worst_split_phase = 0.0
    worst_entangler_phase = 0.0
    descriptors = _native_descriptors(snapshot(basis), evaluated_at=_NOW)
    for index in range(SPLIT_CASE_COUNT):
        left_opcode, right_opcode, matrix = product_case(index)
        # The pass's own round trip, measured on the matrices it emits rather than
        # on a statevector: a matrix-carrying instruction is a compiler input, so
        # the runtime cannot execute either side of this comparison. Nothing is
        # divided out, so the residual is against the operator itself.
        emitted = split_two_qubit_blocks(
            CircuitIR(
                2, (Instruction("blob", (0, 1), matrix=matrix),), dtype="complex128"
            )
        )
        residual, phase = _matrix_residual(matrix, emitted)
        worst_split_residual = max(worst_split_residual, residual)
        worst_split_phase = max(worst_split_phase, phase)
        shipped = legalize(
            CircuitIR(
                2, (Instruction("blob", (0, 1), matrix=matrix),), dtype="complex128"
            ),
            basis,
        )
        if isinstance(shipped, str):
            shipped_unreachable += 1
        # The hand-split arm is written from the two factors' own declared
        # matrices, not from the pass's reading of the product, so the agreement
        # between the arms is a statement about the pass rather than about one
        # arithmetic used on both sides.
        hand = legalize(
            CircuitIR(
                2,
                (
                    Instruction(left_opcode, (0,), matrix=_factor_matrix(left_opcode)),
                    Instruction(
                        right_opcode, (1,), matrix=_factor_matrix(right_opcode)
                    ),
                ),
                dtype="complex128",
            ),
            basis,
        )
        if isinstance(hand, str):
            hand_split_unreachable += 1
        if isinstance(shipped, str) or isinstance(hand, str):
            continue
        comparable += 1
        leaves = (
            None
            if basis.entangler is None or basis.z_rotation is None
            else synthesize_two_qubit(
                matrix,
                qubits=(0, 1),
                entangler=basis.entangler,
                z_rotation=basis.z_rotation,
                pulse_opcode=basis.pulse_opcode,
            )
        )
        if leaves is None:
            entangler_unreachable += 1
            continue
        # The leaves are the opcodes the basis itself publishes, so the entangler
        # arm needs no translation and its leaf count is the count the basis sees.
        if not all(_supports(descriptors, leaf) for leaf in leaves):
            entangler_unreachable += 1
            continue
        if len(leaves) > len(shipped.program.instructions):
            entangler_longer += 1
        elif len(leaves) < len(shipped.program.instructions):
            entangler_shorter += 1
        worst_entangler_phase = max(
            worst_entangler_phase,
            _matrix_residual(matrix, CircuitIR(2, leaves, dtype="complex128"))[1],
        )
    return {
        "label": basis.label,
        "entangler": basis.entangler,
        "case_count": SPLIT_CASE_COUNT,
        "comparable_case_count": comparable,
        "shipped_unreachable_case_count": shipped_unreachable,
        "hand_split_unreachable_case_count": hand_split_unreachable,
        "entangler_unreachable_case_count": entangler_unreachable,
        "entangler_longer_than_shipped_count": entangler_longer,
        "entangler_shorter_than_shipped_count": entangler_shorter,
        "worst_split_entry_residual": worst_split_residual,
        "worst_split_phase_radians": worst_split_phase,
        "worst_entangler_phase_radians": worst_entangler_phase,
    }


def entangler_table() -> list[dict[str, Any]]:
    """The entangler cost of every declared opcode over ``cx``, before the
    legalizer sees it, so the cost is a property of the decomposition."""

    rows: list[dict[str, Any]] = []
    for opcode in TWO_QUBIT_OPCODES:
        instruction = two_qubit_instruction(opcode, matrix=True)
        assert instruction.matrix is not None
        for entangler in SUPERCONTROLLED_ENTANGLERS:
            leaves = synthesize_two_qubit(
                instruction.matrix,
                qubits=(0, 1),
                entangler=entangler,
                z_rotation="rz",
                pulse_opcode="sx",
            )
            assert leaves is not None, (opcode, entangler)
            rows.append(
                {
                    "opcode": opcode,
                    "entangler": entangler,
                    "entangler_count": sum(
                        1 for leaf in leaves if leaf.name == entangler
                    ),
                    "leaf_count": len(leaves),
                    "leaf_opcode_set": sorted({leaf.name for leaf in leaves}),
                    "leaf_opcodes": [leaf.name for leaf in leaves],
                }
            )
    return rows


def run_benchmark(*, bases: tuple[Basis, ...] = DEFAULT_BASES) -> dict[str, Any]:
    """Measure reach, cost, and phase on every basis of ``bases``."""

    rows = [reach(basis) for basis in bases]
    return {
        "schema": SCHEMA,
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_two_qubit_basis_decomposer",
        "reference_revision": "qiskit 1.2.4 TwoQubitBasisDecomposer(euler_basis='ZSX')",
        "synthesis_identity": "U == exp(i*phase) * (K1l x K1r) * exp(i*(a XX + b YY + c ZZ)) * (K2l x K2r)",
        "phase_contract": "equal up to one global phase, which CircuitIR cannot record",
        "split_identity": "U == left (x) right, with no global phase to record",
        "split_case_seed": _SPLIT_SEED,
        "split_factors": list(SPLIT_FACTORS),
        "declared_two_qubit_unitary_opcodes": list(TWO_QUBIT_OPCODES),
        "supercontrolled_entanglers": list(SUPERCONTROLLED_ENTANGLERS),
        "random_case_seed": _RANDOM_SEED,
        "basis_count": len(rows),
        "reach": rows,
        "random_reach": [random_reach(basis) for basis in bases],
        "phase": [phase_evidence(basis) for basis in bases],
        "product_split": [product_split(basis) for basis in bases],
        "product_split_rotation_only": product_split(ROTATION_ONLY_BASIS),
        "entangler_cost": entangler_table(),
        "reference_anchor": _qiskit_anchor(),
    }


def _reference_entanglers() -> tuple[tuple[str, Any], ...]:
    """The port's entanglers as Qiskit gate objects, in the port's own order.

    Qiskit's `KAK_GATE_NAMES` lists only `cx`, `cz`, `iswap`, `rxx`, `ecr` and
    `rzx`, so this table is built here rather than looked up: `cy`, `ryy` and
    `rzz` are accepted by `TwoQubitBasisDecomposer` and are supercontrolled, they
    are simply not the spellings Qiskit's basis *chooser* knows. `rxx`, `ryy` and
    `rzz` are pinned to the angle the port applies them at.
    """

    import math

    from qiskit.circuit.library import (  # type: ignore[import-not-found]
        CXGate,
        CYGate,
        CZGate,
        RXXGate,
        RYYGate,
        RZZGate,
    )

    builders = {
        "cx": CXGate,
        "cz": CZGate,
        "cy": CYGate,
        "rzz": lambda: RZZGate(math.pi / 2),
        "rxx": lambda: RXXGate(math.pi / 2),
        "ryy": lambda: RYYGate(math.pi / 2),
    }
    return tuple((name, builders[name]()) for name in SUPERCONTROLLED_ENTANGLERS)


def _qiskit_anchor() -> dict[str, Any]:
    """The same reach and cost from Qiskit, when Qiskit is importable.

    The anchor is a cross-check, not a dependency: this module is a local
    benchmark and must run on a machine with no Qiskit at all. Both sides are
    fed the same physical operator -- the port reads its matrix with wire 0 on
    the most significant index bit, Qiskit reads its own with qubit 0 on the
    least significant one -- so the Qiskit input is the register-swap conjugate
    of the port input and the Qiskit output is converted back the same way.

    Only the entangler count is asserted. The two sides choose different Euler
    bases, so their raw instruction lists are different presentations of the same
    cost, and claiming otherwise would be a presentation claim dressed up as an
    algorithmic one.
    """

    try:
        import numpy as np
        import qiskit  # type: ignore[import-not-found]
        from qiskit.synthesis.two_qubit import (  # type: ignore[import-not-found]
            TwoQubitBasisDecomposer,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        return {"available": False, "reason": str(error)}

    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for entangler, gate in _reference_entanglers():
        decomposer = TwoQubitBasisDecomposer(
            gate, euler_basis="ZSX", pulse_optimize=False
        )
        for opcode in TWO_QUBIT_OPCODES:
            instruction = two_qubit_instruction(opcode, matrix=True)
            assert instruction.matrix is not None
            target = np.array(instruction.matrix, dtype=complex)
            leaves = synthesize_two_qubit(
                instruction.matrix,
                qubits=(0, 1),
                entangler=entangler,
                z_rotation="rz",
                pulse_opcode="sx",
            )
            assert leaves is not None, (opcode, entangler)
            port_count = sum(1 for leaf in leaves if leaf.name == entangler)
            try:
                circuit = decomposer(swap @ target @ swap)
            except Exception as error:  # pragma: no cover - environment dependent
                rows.append(
                    {
                        "opcode": opcode,
                        "entangler": entangler,
                        "reference_failed": str(error),
                    }
                )
                continue
            reference_count = sum(
                1 for item in circuit.data if item.operation.name == gate.name.lower()
            )
            row = {
                "opcode": opcode,
                "entangler": entangler,
                "port_entangler_count": port_count,
                "reference_entangler_count": reference_count,
                "port_leaf_count": len(leaves),
                "reference_gate_count": circuit.size(),
                "entangler_count_agrees": port_count == reference_count,
            }
            rows.append(row)
            if port_count != reference_count:
                mismatches.append(row)
    return {
        "available": True,
        # The library reading, not the pinned `reference_revision` above: the
        # per-row `reference_gate_count` below moves between installed Qiskit
        # versions while the entangler counts the test compares do not, so which
        # of the two a reader is looking at is decided by this field.
        "qiskit_version": qiskit.__version__,
        "decomposer": "TwoQubitBasisDecomposer(gate, euler_basis='ZSX')",
        "reference_entanglers": [name for name, _ in _reference_entanglers()],
        "compared_case_count": len(rows),
        "entangler_count_agreement_count": sum(
            1 for row in rows if row.get("entangler_count_agrees")
        ),
        "mismatches": mismatches,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    declared = len(payload["declared_two_qubit_unitary_opcodes"])
    print(
        f"{declared} declared two-qubit unitary opcodes on {payload['basis_count']} bases"
    )
    print(
        f"{'basis':28s} {'named':>7s} {'matrix':>7s} {'table':>7s} "
        f"{'entanglers':>11s} {'unresolved':>11s}"
    )
    for row in payload["reach"]:
        print(
            f"{row['label']:28s} "
            f"{row['legalized_by_name_count']:>3d}/{declared} "
            f"{row['legalized_by_matrix_count']:>3d}/{declared} "
            f"{row['identity_table_reach_count']:>3d}/{declared} "
            f"{row['total_entangler_count']:>11d} "
            f"{row['unresolved_opcode_count']:>11d}"
        )
    print()
    print(f"{'basis':28s} {'reached':>9s} {'worst gap':>12s} entangler distribution")
    for row in payload["random_reach"]:
        print(
            f"{row['label']:28s} "
            f"{row['reached_count']:>4d}/{row['case_count']:<4d} "
            f"{row['worst_reconstruction_gap']:>12.3e} "
            f"{row['entangler_count_distribution']}"
        )
    print()
    print(
        f"{'basis':28s} {'gates':>13s} {'overlap':>18s} {'phase':>9s} "
        f"{'raw diff':>9s}"
    )
    for row in payload["phase"]:
        if not row["legalized"]:
            print(f"{row['label']:28s} not legalized: {row['error']}")
            continue
        print(
            f"{row['label']:28s} {row['source_gate_count']:>4d}->"
            f"{row['legalized_gate_count']:<8d} "
            f"{row['state_overlap_magnitude']:>18.15f} "
            f"{row['global_phase_radians']:>+9.6f} "
            f"{row['max_raw_state_difference']:>9.6f}"
        )
    anchor = payload["reference_anchor"]
    if anchor["available"]:
        print()
        print(
            f"Qiskit anchor: {anchor['entangler_count_agreement_count']}/"
            f"{anchor['compared_case_count']} cases agree on the entangler count"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
