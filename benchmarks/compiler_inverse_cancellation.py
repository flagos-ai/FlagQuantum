"""Measure what the schema's inverse declaration buys the optimization pipeline.

W9-09 of the Qiskit parity backlog asks for the pass behind
``transpiler/passes/optimization/inverse_cancellation.py``. Before it,
``compiler.optimize`` cancelled a self-inverse pair but never a pair of *different*
opcodes that invert each other, so

    ``s(0)  sdg(0)``

survived as two instructions even though it is the identity, and so did
``t(1) tdg(1)`` and ``sx(2) sxdg(2)``.

This module measures four things and refuses to measure a fifth.

* **Where the rule comes from.** The pass holds no opcode table. It reads
  ``OperatorSchema.adjoint`` through ``operator_schema.inverse_operator``, the same
  declaration ``Circuit.adjoint`` already consumes. The declaration table below
  lists every declared unitary, its adjoint rule, and whether the pass acts on it,
  which is what makes "no second source of truth" checkable rather than asserted.
* **That the declaration really is a rule.** For every declared unitary, the
  declared inverse is multiplied by the gate itself on the runtime's own matrices,
  over seeded angle draws, and the residual is reported. A declaration that only
  held on average would be a rule source of last resort; this one is exact, and the
  table states by how much.
* **The pipeline delta.** On seeded populations the source circuit is measured
  through the whole pipeline with and without the new pass, and every removed
  instruction is checked against an execution of the circuit.
* **Where the pass stops, and why.** A gap that is empty and a gap whose operator
  merely *commutes* with the pair are different problems. This pass solves only the
  first. The shape table drives every pair through ten gaps and measures two things
  per row rather than assuming either: whether the gap's operator commutes with the
  pair member, and whether the pair was in fact removable. That splits the refusals
  in two. Every row whose gap does not commute is a shape where removing anything
  would have changed the program, so declining it was correctness. Every row whose
  gap does commute and was not crossed is reach left on the table, counted as a
  number rather than promised. The one gap whose operator is not a declared gate
  reports `None` in both columns and is counted separately, because a question that
  cannot be asked is not an answer of "no".

* **What it does not measure:** a gate-count parity claim against Qiskit. The shape
  table feeds both implementations the identical circuit and reports both counts.
  It asserts agreement only on the rows it actually drove, and the closing section
  states which those are.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_inverse_cancellation.py --json-output /tmp/w909.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from flagquantum.compiler.inverse_cancellation import inverse_pairs
from flagquantum.compiler.pipeline import (
    _SELF_INVERSE,
    _optimize_to_fixed_point,
    merge_adjacent_rotations,
    merge_self_inverse,
    remove_identity_gates,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import (
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
    inverse_operator,
)
from flagquantum.simulation.gate_matrix import gate_matrix
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_inverse_cancellation_benchmark_v1"

_NOW = datetime(2026, 10, 8, 8, 0, tzinfo=timezone.utc)
COMPLEX = torch.complex128

#: Seeded so the counts below are reproducible, which is what lets them be pinned in
#: a benchmark contract.
_SWEEP_SEED = 20261008
_CIRCUIT_SEED = 20261013
_REGISTER_WIDTH = 3
_ANGLE = 0.7137

#: The operator comparison is exact, so it uses one absolute tolerance and no
#: relative slack. Two operators that differ are a shape that was not redundant.
_UNITARY_ATOL = 1.0e-12

#: Angle draws per opcode in the identity proof. The residual is exact, so this
#: number controls coverage rather than precision.
_DRAW_COUNT = 20

#: The opcodes the pass can act on, read from the pass rather than restated.
_PAIRS = dict(inverse_pairs())


def _instruction(opcode: str, wires: tuple[int, ...], angle: float) -> Instruction:
    schema = OPERATOR_SCHEMAS[opcode]
    return Instruction(opcode, wires, params=dict.fromkeys(schema.parameters, angle))


def _dynamic(opcode: str, qubit: int) -> Instruction:
    """An instruction outside the operator schemas, which is how a barrier acts."""

    return Instruction(opcode, (qubit,), metadata={"is_dynamic": True})


def _embed(instruction: Instruction, width: int) -> torch.Tensor:
    """The instruction as a whole-register operator, qubit 0 most significant."""

    local = _local_matrix(instruction)
    rest = [qubit for qubit in range(width) if qubit not in instruction.wires]
    order = list(instruction.wires) + rest
    total = local
    for _ in rest:
        total = torch.kron(total, torch.eye(2, dtype=COMPLEX))
    permutation = [order.index(qubit) for qubit in range(width)]
    size = 2**width
    return (
        total.reshape([2] * width + [2] * width)
        .permute(permutation + [index + width for index in permutation])
        .reshape(size, size)
    )


def _unitary(ir: CircuitIR, width: int) -> torch.Tensor | None:
    """The operator ``ir`` computes, or ``None`` when it is not a gate-only program.

    The comparison in `shape_table` has to be against the operator and not against
    one statevector. Several of the gates in scope act trivially on ``|0...0>``, so
    a state comparison would report a pair as redundant whenever the input happened
    to be an eigenstate -- a statement about the sample rather than about the shape.
    The operator comparison has no input to be lucky about.
    """

    if not _executable(ir):
        return None
    total = torch.eye(2**width, dtype=COMPLEX)
    for instruction in ir:
        total = _embed(instruction, width) @ total
    return total


def _local_matrix(instruction: Instruction) -> torch.Tensor:
    """The instruction's own matrix, on the wires it names, in the order it names."""

    arity = len(instruction.wires)
    return gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1, 2**arity, 2**arity)[0]


def _identity_residual(opcode: str, partner: str, angles: list[float]) -> float:
    """The largest ``|inverse(gate) @ gate - I|`` over the drawn angles.

    Both operators are taken on the wires the opcode declares, in the order the
    schema declares them, which is the same convention the runtime uses. No
    embedding is needed: the comparison is between two matrices of one size.
    """

    schema = OPERATOR_SCHEMAS[opcode]
    wires = tuple(range(schema.arity))
    identity = torch.eye(2**schema.arity, dtype=COMPLEX)
    worst = 0.0
    for angle in angles:
        params = dict.fromkeys(schema.parameters, angle)
        gate = _local_matrix(Instruction(opcode, wires, params=params))
        if schema.parameters:
            declared = inverse_operator(schema, params)
            assert declared is not None
            name, inverse_params = declared
            inverse = _local_matrix(Instruction(name, wires, params=inverse_params))
        else:
            inverse = _local_matrix(Instruction(partner, wires))
        worst = max(worst, float((inverse @ gate - identity).abs().max()))
    return worst


def declaration_table() -> dict[str, Any]:
    """Every declared unitary, its adjoint rule, and whether this pass acts on it.

    The pass derives its reach from this declaration, so the table is the rule
    source stated explicitly. ``in_pass_scope`` is true only for an opcode whose
    declared inverse is a *different* parameter-free opcode: the self-inverse
    answers belong to ``merge_self_inverse`` and the angle rules belong to
    ``merge_adjacent_rotations``, and the three sets are disjoint.
    """

    seed = random.Random(_SWEEP_SEED)
    rows = []
    for opcode in sorted(OPERATOR_SCHEMAS):
        schema = OPERATOR_SCHEMAS[opcode]
        if not schema.unitary:
            continue
        angles = [seed.uniform(-3.0, 3.0) for _ in range(_DRAW_COUNT)]
        declared = inverse_operator(schema, dict.fromkeys(schema.parameters, _ANGLE))
        partner = None if declared is None else declared[0]
        in_scope = opcode in _PAIRS
        rows.append(
            {
                "opcode": opcode,
                "arity": schema.arity,
                "adjoint": schema.adjoint,
                "parameter_count": len(schema.parameters),
                "declared_partner": partner,
                "inverse_is_a_different_opcode": partner is not None
                and partner != opcode,
                "in_self_inverse_pass": opcode in _SELF_INVERSE,
                "in_this_pass_scope": in_scope,
                "identity_residual": _identity_residual(
                    opcode, partner if partner is not None else opcode, angles
                ),
            }
        )

    scope = [row for row in rows if row["in_this_pass_scope"]]
    parameter_free = [
        row
        for row in rows
        if row["declared_partner"] == row["opcode"] and row["parameter_count"] == 0
    ]
    partner_rows = [row for row in rows if row["inverse_is_a_different_opcode"]]
    return {
        "unitary_opcode_count": len(rows),
        "declared_inverse_count": sum(
            1 for row in rows if row["declared_partner"] is not None
        ),
        "partner_opcode_count": len(partner_rows),
        "parameter_free_self_inverse_count": len(parameter_free),
        "angle_rule_opcode_count": len(rows) - len(partner_rows) - len(parameter_free),
        "in_pipeline_self_inverse_count": sum(
            1 for row in rows if row["in_self_inverse_pass"]
        ),
        "pipeline_self_inverse_gap": sorted(
            row["opcode"] for row in parameter_free if not row["in_self_inverse_pass"]
        ),
        "scope_opcode_count": len(scope),
        "scope_pair_count": len(scope) // 2,
        "max_identity_residual": max(row["identity_residual"] for row in rows),
        "disjoint_from_self_inverse_pass": not (set(_PAIRS) & _SELF_INVERSE),
        "rows": rows,
    }


def _wrong_partner(opcode: str) -> str | None:
    """A parameter-free unitary of the same arity that is not the declared inverse."""

    schema = OPERATOR_SCHEMAS[opcode]
    for other in sorted(OPERATOR_SCHEMAS):
        candidate = OPERATOR_SCHEMAS[other]
        if (
            candidate.unitary
            and not candidate.parameters
            and candidate.arity == schema.arity
            and other != opcode
            and other != _PAIRS.get(opcode)
        ):
            return other
    return None


def identity_proof() -> dict[str, Any]:
    """Show that the declaration is exact, and that being exact is not trivial.

    Every declared inverse is checked on the runtime's matrices. The control is the
    same measurement with the partner replaced by an arbitrary gate of the same
    arity: if that control were also near zero, the residuals would be evidence
    about the tolerance rather than about the pairing.
    """

    seed = random.Random(_SWEEP_SEED + 1)
    rows = []
    wrong_rows = []
    for opcode in sorted(_PAIRS):
        angles = [seed.uniform(-3.0, 3.0) for _ in range(_DRAW_COUNT)]
        rows.append(
            {
                "opcode": opcode,
                "partner": _PAIRS[opcode],
                "residual": _identity_residual(opcode, _PAIRS[opcode], angles),
            }
        )
        other = _wrong_partner(opcode)
        if other is not None:
            wrong_rows.append(
                {
                    "opcode": opcode,
                    "partner": other,
                    "residual": _identity_residual(opcode, other, angles),
                }
            )
    for opcode in sorted(_SELF_INVERSE & set(OPERATOR_SCHEMAS)):
        angles = [seed.uniform(-3.0, 3.0) for _ in range(_DRAW_COUNT)]
        rows.append(
            {
                "opcode": opcode,
                "partner": opcode,
                "residual": _identity_residual(opcode, opcode, angles),
            }
        )
    return {
        "draw_count_per_opcode": _DRAW_COUNT,
        "checked_opcode_count": len(rows),
        "max_residual": max(row["residual"] for row in rows),
        "residual_rows": rows,
        "control_opcode_count": len(wrong_rows),
        "control_min_residual": min(row["residual"] for row in wrong_rows),
        "control_rows": wrong_rows,
    }


def _legacy_fixed_point(ir: CircuitIR) -> CircuitIR:
    """The pipeline with this pass removed, and nothing else changed."""

    for _ in range(len(ir) + 1):
        previous = len(ir)
        ir = remove_identity_gates(ir)
        ir = merge_self_inverse(ir)
        ir = merge_adjacent_rotations(ir)
        ir = remove_identity_gates(ir)
        if len(ir) == previous:
            return ir
    raise AssertionError("the legacy pipeline did not reach a fixed point")


def _executable(ir: CircuitIR) -> bool:
    """Whether every instruction in ``ir`` is a declared gate with a matrix.

    The operator schema is where the product declares that an opcode is a gate, so
    it is also the honest test for whether a circuit can be executed. A `measure` is
    declared nowhere, which is what makes it a barrier rather than a participant.
    """

    return all(get_operator_schema(instruction.name) is not None for instruction in ir)


def _state(ir: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        ir, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    )


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _prepared(instructions: tuple[Instruction, ...]) -> CircuitIR:
    """``instructions`` behind a layer that leaves no qubit in an eigenstate.

    Without it the state comparison in `pipeline_delta` can hold because the whole
    circuit happens to be trivial on ``|0...0>``, which would make the soundness
    check vacuous rather than strict.
    """

    return CircuitIR(
        _REGISTER_WIDTH,
        tuple(
            [Instruction("h", (qubit,)) for qubit in range(_REGISTER_WIDTH)]
            + list(instructions)
        ),
    )


def _pair_sequence(seed: random.Random, length: int) -> list[Instruction]:
    """``length`` adjacent pair members on a single randomly chosen qubit."""

    instructions: list[Instruction] = []
    for _ in range(length):
        opcode = seed.choice(sorted(_PAIRS))
        qubit = seed.randrange(_REGISTER_WIDTH)
        instructions.append(Instruction(opcode, (qubit,)))
        instructions.append(Instruction(_PAIRS[opcode], (qubit,)))
    return instructions


def _straddled_pair(seed: random.Random) -> CircuitIR:
    """A pair with a single-qubit gate on a *different* qubit between the two."""

    opcode = seed.choice(sorted(_PAIRS))
    qubit = seed.randrange(_REGISTER_WIDTH)
    other = (qubit + 1 + seed.randrange(_REGISTER_WIDTH - 1)) % _REGISTER_WIDTH
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(opcode, (qubit,)),
            _instruction("x", (other,), _ANGLE),
            Instruction(_PAIRS[opcode], (qubit,)),
        ),
    )


def _interleaved_pairs(seed: random.Random) -> CircuitIR:
    """Two pair threads on two qubits, straddling each other in program order."""

    first, second = seed.sample(range(_REGISTER_WIDTH), 2)
    left = seed.choice(sorted(_PAIRS))
    right = seed.choice(sorted(_PAIRS))
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(left, (first,)),
            Instruction(right, (second,)),
            Instruction(_PAIRS[left], (first,)),
            Instruction(_PAIRS[right], (second,)),
        ),
    )


def _barrier_pair(seed: random.Random) -> CircuitIR:
    """A pair separated by an instruction the operator schema does not declare."""

    opcode = seed.choice(sorted(_PAIRS))
    qubit = seed.randrange(_REGISTER_WIDTH)
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(opcode, (qubit,)),
            _dynamic("measure", qubit),
            Instruction(_PAIRS[opcode], (qubit,)),
        ),
    )


def pipeline_delta() -> list[dict[str, Any]]:
    """The pipeline's own reach on the seeded populations, with and without the pass."""

    seed = random.Random(_CIRCUIT_SEED)
    populations: list[tuple[str, list[CircuitIR]]] = [
        (
            "adjacent_pair_members",
            [
                CircuitIR(_REGISTER_WIDTH, tuple(_pair_sequence(seed, 4)))
                for _ in range(60)
            ],
        ),
        (
            "pair_across_another_qubit",
            [_prepared(_straddled_pair(seed).instructions) for _ in range(60)],
        ),
        (
            "interleaved_pairs",
            [_prepared(_interleaved_pairs(seed).instructions) for _ in range(60)],
        ),
        (
            "pair_across_a_declared_barrier",
            [_prepared(_barrier_pair(seed).instructions) for _ in range(60)],
        ),
    ]

    rows = []
    for label, circuits in populations:
        source = legacy = optimized = 0
        changed = 0
        executed = 0
        worst = 0.0
        for ir in circuits:
            source += len(ir)
            legacy_ir = _legacy_fixed_point(ir)
            optimized_ir = _optimize_to_fixed_point(ir)
            legacy += len(legacy_ir)
            optimized += len(optimized_ir)
            # The soundness check runs on every circuit and not only the changed
            # ones: a pass that removed nothing cannot be caught by comparing states.
            # It runs only where the program is gate-only, and the count of those is
            # reported, so a population that silently stopped being executable would
            # show up as a shrinking denominator rather than as a quiet pass.
            if _executable(ir) and _executable(optimized_ir):
                executed += 1
                worst = max(
                    worst, float((_state(optimized_ir) - _state(ir)).abs().max())
                )
            if len(optimized_ir) != len(legacy_ir):
                changed += 1
                if label == "pair_across_a_declared_barrier":
                    raise AssertionError("a declared barrier must stop the pass")
        rows.append(
            {
                "label": label,
                "circuit_count": len(circuits),
                "source_instruction_count": source,
                "legacy_instruction_count": legacy,
                "optimized_instruction_count": optimized,
                "removed_by_legacy": source - legacy,
                "removed_by_declared_inverse_pass": legacy - optimized,
                "changed_circuit_count": changed,
                "executed_circuit_count": executed,
                "max_state_difference": worst,
            }
        )
    return rows


#: The gaps the shape table drives a pair through. These are labels for what is
#: between the two members, not classifications: whether a gap commutes with the pair
#: is measured per row and reported in ``gap_commutes_with_member``. Naming a gap
#: "commuting" or "blocking" up front would be exactly the unverified claim the rest
#: of this round is written to avoid.
_GAPS: tuple[str, ...] = (
    "nothing_between",
    "a_gate_on_another_qubit",
    "rz_on_the_pairs_own_qubit",
    "rx_on_the_pairs_own_qubit",
    "x_on_the_pairs_own_qubit",
    "cz_on_the_pair_qubit_and_another",
    "cx_control_on_the_pair_qubit",
    "cx_target_on_the_pair_qubit",
    "measure_on_the_pairs_own_qubit",
    "a_member_of_the_other_pair",
)


def _gap_instructions(label: str, opcode: str, qubit: int) -> tuple[Instruction, ...]:
    """The instructions that sit between the two members of ``opcode``'s pair."""

    other = (qubit + 1) % _REGISTER_WIDTH
    if label == "nothing_between":
        return ()
    if label == "a_gate_on_another_qubit":
        return (Instruction("x", (other,)),)
    if label == "rz_on_the_pairs_own_qubit":
        return (_instruction("rz", (qubit,), _ANGLE),)
    if label == "rx_on_the_pairs_own_qubit":
        return (_instruction("rx", (qubit,), _ANGLE),)
    if label == "x_on_the_pairs_own_qubit":
        return (Instruction("x", (qubit,)),)
    if label == "cz_on_the_pair_qubit_and_another":
        return (Instruction("cz", (qubit, other)),)
    if label == "cx_control_on_the_pair_qubit":
        return (Instruction("cx", (qubit, other)),)
    if label == "cx_target_on_the_pair_qubit":
        return (Instruction("cx", (other, qubit)),)
    if label == "measure_on_the_pairs_own_qubit":
        return (_dynamic("measure", qubit),)
    if label == "a_member_of_the_other_pair":
        partner = _PAIRS[opcode]
        others = [op for op in sorted(_PAIRS) if op != opcode and op != partner]
        return (Instruction(others[0], (qubit,)),)
    raise AssertionError(f"unknown gap kind {label!r}")


def _shape_circuit(label: str, opcode: str, qubit: int) -> CircuitIR:
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(opcode, (qubit,)),
            *_gap_instructions(label, opcode, qubit),
            Instruction(_PAIRS[opcode], (qubit,)),
        ),
    )


def _qiskit_shape_anchor(circuits: dict[str, CircuitIR]) -> dict[str, Any]:
    """Qiskit's own ``InverseCancellation`` driven over the identical circuits.

    The anchor is a cross-check, not a dependency: this module has to run on a
    machine with no Qiskit at all. ``InverseCancellation`` takes the inverse pairs
    as an argument and rejects a bare self-inverse gate in the same list, which is
    the same split this port makes -- hence the two pairs-of-pairs and the third
    pair for ``sx``.
    """

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.circuit.library import (  # type: ignore[import-not-found]
            SdgGate,
            SGate,
            SXdgGate,
            SXGate,
            TdgGate,
            TGate,
        )
        from qiskit.transpiler import PassManager  # type: ignore[import-not-found]
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            InverseCancellation,
        )
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    import qiskit  # type: ignore[import-not-found]

    passes = PassManager(
        [
            InverseCancellation(
                [
                    (SGate(), SdgGate()),
                    (TGate(), TdgGate()),
                    (SXGate(), SXdgGate()),
                ]
            )
        ]
    )
    counts: dict[str, int] = {}
    for key, ir in circuits.items():
        circuit = QuantumCircuit(_REGISTER_WIDTH, 1)
        for instruction in ir:
            if instruction.wires and canonical_opcode(instruction.name) == "measure":
                circuit.measure(instruction.wires[0], 0)
            elif instruction.params:
                getattr(circuit, instruction.name)(
                    instruction.params["theta"], *instruction.wires
                )
            else:
                getattr(circuit, instruction.name)(*instruction.wires)
        counts[key] = circuit.size() - passes.run(circuit).size()
    return {
        "available": True,
        "pass_name": "InverseCancellation",
        "qiskit_version": qiskit.__version__,
        "source_gate_count": sum(len(ir) for ir in circuits.values()),
        "removed_count": sum(counts.values()),
        "removed_by_shape": counts,
    }


def _gap_commutes_with_member(ir: CircuitIR, opcode: str, qubit: int) -> bool | None:
    """Whether every gap instruction commutes with the pair's first member.

    This is the fact the pass would need in order to reach the pair across the gap,
    and it is measured here rather than asserted about the gap's name. A gap that is
    not a declared gate makes the question unanswerable with the matrices at hand,
    which is reported as ``None`` rather than as a "no".
    """

    gap = ir.instructions[1:-1]
    if not gap:
        return True
    member = _unitary(
        CircuitIR(_REGISTER_WIDTH, (Instruction(opcode, (qubit,)),)), _REGISTER_WIDTH
    )
    if member is None:
        return None
    for instruction in gap:
        matrix = _unitary(CircuitIR(_REGISTER_WIDTH, (instruction,)), _REGISTER_WIDTH)
        if matrix is None:
            return None
        if not torch.allclose(
            matrix @ member, member @ matrix, atol=_UNITARY_ATOL, rtol=0
        ):
            return False
    return True


def shape_table() -> dict[str, Any]:
    """One row per pair and gap, with both implementations and the measured truth.

    Two columns are measured rather than inferred. ``removable_in_fact`` strips the
    pair and compares the two whole-register operators -- an operator comparison and
    not a statevector comparison, because several of these gates act trivially on
    ``|0...0>`` and a state comparison would report a row as redundant for a reason
    that has nothing to do with the shape. ``gap_commutes_with_member`` measures
    whether the pass *could* have reached the pair by proving commutation.

    Together the columns separate the two kinds of refusal. A row that is removable
    in fact and whose gap commutes is reach this pass leaves on the table: deferred
    work, counted. A row that is not removable in fact is a shape where removing
    anything would have changed the program, so declining it was correctness rather
    than conservatism. ``non_commuting_gap_removed_count`` is asserted to be zero,
    which is what turns the pass's own claim into something checked.
    """

    rows = []
    circuits: dict[str, CircuitIR] = {}
    for opcode in sorted(_PAIRS):
        qubit = 0
        for label in _GAPS:
            ir = _shape_circuit(label, opcode, qubit)
            circuits[f"{opcode}|{label}"] = ir
            optimized = _optimize_to_fixed_point(ir)
            stripped = CircuitIR(_REGISTER_WIDTH, ir.instructions[1:-1])
            removable = None
            if optimized == ir:
                whole = _unitary(ir, _REGISTER_WIDTH)
                without = _unitary(stripped, _REGISTER_WIDTH)
                if whole is not None and without is not None:
                    removable = bool(
                        torch.allclose(whole, without, atol=_UNITARY_ATOL, rtol=0)
                    )
            rows.append(
                {
                    "opcode": opcode,
                    "partner": _PAIRS[opcode],
                    "gap": label,
                    "gap_instruction_count": len(ir) - 2,
                    "gap_commutes_with_member": _gap_commutes_with_member(
                        ir, opcode, qubit
                    ),
                    "source_instruction_count": len(ir),
                    "optimized_instruction_count": len(optimized),
                    "removed": len(ir) - len(optimized),
                    "removable_in_fact": removable,
                }
            )

    anchored = _qiskit_shape_anchor(circuits)
    if anchored["available"]:
        for row in rows:
            row["qiskit_removed"] = anchored["removed_by_shape"][
                f"{row['opcode']}|{row['gap']}"
            ]
            row["agrees_with_qiskit"] = row["removed"] == row["qiskit_removed"]

    empty = [row for row in rows if row["gap_instruction_count"] == 0]
    commuting = [row for row in rows if row["gap_commutes_with_member"] is True]
    non_commuting = [row for row in rows if row["gap_commutes_with_member"] is False]
    unknowable = [row for row in rows if row["gap_commutes_with_member"] is None]
    crossed = [row for row in non_commuting if row["removed"]]
    if crossed:
        raise AssertionError(
            "a pair was removed across a gap that does not commute: "
            f"{[row['opcode'] + '|' + row['gap'] for row in crossed]}"
        )
    return {
        "gap_count": len(_GAPS),
        "row_count": len(rows),
        "empty_gap_row_count": len(empty),
        "empty_gap_removed_count": sum(row["removed"] for row in empty),
        "commuting_gap_row_count": len(commuting),
        "commuting_gap_removed_count": sum(row["removed"] for row in commuting),
        "commuting_gap_removable_in_fact_count": sum(
            1 for row in commuting if row["removable_in_fact"]
        ),
        "non_commuting_gap_row_count": len(non_commuting),
        "non_commuting_gap_removed_count": len(crossed),
        "non_commuting_gap_removable_in_fact_count": sum(
            1 for row in non_commuting if row["removable_in_fact"]
        ),
        "unknowable_gap_row_count": len(unknowable),
        "unmeasurable_row_count": sum(
            1 for row in rows if row["removable_in_fact"] is None
        ),
        "declined_but_removable_count": sum(
            1 for row in rows if row["removable_in_fact"] and row["removed"] == 0
        ),
        "reached_but_not_removable_in_fact_count": sum(
            1 for row in rows if row["removable_in_fact"] is False and row["removed"]
        ),
        "rows": rows,
        "reference_anchor": anchored,
    }


def run_benchmark() -> dict[str, Any]:
    shapes = shape_table()
    anchor = shapes["reference_anchor"]
    disagreements = [
        row
        for row in shapes["rows"]
        if "agrees_with_qiskit" in row and not row["agrees_with_qiskit"]
    ]
    return {
        "schema": SCHEMA,
        "generated_at": _NOW.isoformat(),
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "seed": {"sweep": _SWEEP_SEED, "circuit": _CIRCUIT_SEED},
        "reference_algorithm": "qiskit_inverse_cancellation",
        "reference_revision": "qiskit 1.2.4 InverseCancellation",
        "declaration_table": declaration_table(),
        "identity_proof": identity_proof(),
        "pipeline_delta": pipeline_delta(),
        "shape_table": shapes,
        "anchor_disagreement_count": len(disagreements),
        "anchor_agreement_count": (
            0 if not anchor["available"] else len(shapes["rows"]) - len(disagreements)
        ),
        "anchor_agreement_scope": (
            f"{len(shapes['rows'])} pair/gap rows driven through both ports; "
            "agreement is claimed only over rows actually driven"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    declaration = payload["declaration_table"]
    proof = payload["identity_proof"]
    print(
        f"declaration: {declaration['unitary_opcode_count']} unitary opcodes, "
        f"{declaration['declared_inverse_count']} with a declared inverse, "
        f"{declaration['partner_opcode_count']} name a partner opcode, "
        f"{declaration['parameter_free_self_inverse_count']} are parameter-free "
        f"self-inverse, {declaration['angle_rule_opcode_count']} are angle rules"
    )
    print(
        f"  this pass's scope: {declaration['scope_opcode_count']} opcodes "
        f"({declaration['scope_pair_count']} pairs); disjoint from the "
        f"self-inverse pass: {declaration['disjoint_from_self_inverse_pass']}"
    )
    print(
        f"  the pipeline's own self-inverse set omits: "
        f"{declaration['pipeline_self_inverse_gap']}"
    )
    print(
        f"  identity residual over {proof['checked_opcode_count']} opcodes x "
        f"{proof['draw_count_per_opcode']} draws: {proof['max_residual']:.3e}"
    )
    print(
        f"  control, same measurement against a wrong partner: "
        f"{proof['control_min_residual']:.3e} at best"
    )
    print()
    print(
        f"{'population':32s} {'circuits':>8s} {'source':>7s} {'legacy':>7s} "
        f"{'now':>7s} {'via pass':>9s} {'changed':>8s} {'exec':>5s} {'max diff':>10s}"
    )
    for row in payload["pipeline_delta"]:
        print(
            f"{row['label']:32s} {row['circuit_count']:>8d} "
            f"{row['source_instruction_count']:>7d} "
            f"{row['legacy_instruction_count']:>7d} "
            f"{row['optimized_instruction_count']:>7d} "
            f"{row['removed_by_declared_inverse_pass']:>9d} "
            f"{row['changed_circuit_count']:>8d} "
            f"{row['executed_circuit_count']:>5d} "
            f"{row['max_state_difference']:>10.3e}"
        )
    shapes = payload["shape_table"]
    print()
    print(f"shape table: {shapes['row_count']} rows over {shapes['gap_count']} gaps")
    print(
        f"  {shapes['empty_gap_row_count']} empty-gap rows: removed "
        f"{shapes['empty_gap_removed_count']}"
    )
    print(
        f"  {shapes['commuting_gap_row_count']} rows whose gap commutes: removed "
        f"{shapes['commuting_gap_removed_count']}, removable in fact "
        f"{shapes['commuting_gap_removable_in_fact_count']}"
    )
    print(
        f"  {shapes['non_commuting_gap_row_count']} rows whose gap does not commute: "
        f"removed {shapes['non_commuting_gap_removed_count']}, removable in fact "
        f"{shapes['non_commuting_gap_removable_in_fact_count']}"
    )
    print(
        f"  {shapes['unknowable_gap_row_count']} rows whose gap decides nothing; "
        f"declined but removable in fact overall: "
        f"{shapes['declined_but_removable_count']}"
    )
    anchor = shapes["reference_anchor"]
    print()
    if anchor["available"]:
        print(
            f"Qiskit {anchor['qiskit_version']} {anchor['pass_name']}: "
            f"{payload['anchor_agreement_count']}/{shapes['row_count']} rows agree, "
            f"{payload['anchor_disagreement_count']} differ"
        )
    else:
        print(f"Qiskit anchor unavailable: {anchor['reason']}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
