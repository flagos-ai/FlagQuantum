"""Measure what proving commutation buys the optimization pipeline.

W9-01 and W9-02 of the Qiskit parity backlog ask for the analysis Qiskit keeps in
``transpiler/passes/optimization/commutation_analysis.py`` and for the gate
cancellation that consumes it. Before them, ``compiler.optimize`` cancelled a
self-inverse pair only when nothing that touched the pair's wires stood between
the two occurrences, so

    ``cx(0, 1)  rz(0)  cx(0, 1)``

survived as three instructions even though it is a bare ``rz(0)``.

This module measures three things and refuses to measure a fourth.

* **The rule source's own reach.** Over every ordered pair of the declared
  arity-one and arity-two unitaries in every placement where the two can meet in
  a three-qubit register, the rule source is compared against the runtime's
  gate matrices. It answers **3602** of the **3826** pairs that really do commute
  and never says "yes" to a pair that does not. The residue is listed by shape.
* **The pipeline delta.** On seeded circuits whose controlled gates are separated
  by a rotation on a control, on a target, or on an unrelated qubit, the source
  circuit is measured through the whole pipeline with and without the new pass,
  and every removed instruction is checked against an execution of the circuit.
* **The cost of the proof.** The analysis walks each qubit's instruction list
  once and grows a commuting block while the candidate commutes with every block
  member, so the work is quadratic in the widest block and linear in the number of
  blocks. The widest block over the measured population is reported.

* **What it does not measure:** a gate-count parity claim against Qiskit. Qiskit's
  ``CommutativeCancellation`` also merges runs of z-rotations, so the two sides
  count different things; the anchor below feeds both sides the same circuit and
  reports both counts without asserting them equal.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_commutation_cancellation.py --json-output /tmp/w901.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from flagquantum.compiler.commutation import (
    _RULE_OPCODES,
    analyze_commutation,
    commute,
)
from flagquantum.compiler.commutation_cancellation import (
    cancellable_positions,
    rotation_groups,
)
from flagquantum.compiler.pipeline import (
    _optimize_to_fixed_point,
    merge_adjacent_rotations,
    merge_self_inverse,
    remove_identity_gates,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.simulation.gate_matrix import gate_matrix
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_commutation_cancellation_benchmark_v1"

_NOW = datetime(2026, 10, 7, 8, 0, tzinfo=timezone.utc)
COMPLEX = torch.complex128

#: The sweep and the circuit populations are seeded so the counts below are
#: reproducible, which is what lets them be pinned in a benchmark contract.
_SWEEP_SEED = 20261006
_CIRCUIT_SEED = 20261011
_REGISTER_WIDTH = 3
_ANGLE = 0.7137

#: Placements two instructions can occupy in a three-qubit register: disjoint,
#: sharing one qubit, and identical. A single-qubit gate is placed on every qubit
#: so its placement is exercised rather than assumed.
_PLACEMENTS: tuple[tuple[int, ...], ...] = (
    (0,),
    (1,),
    (2,),
    (0, 1),
    (1, 2),
    (0, 1, 2),
)

#: The cancellation a controlled gate admits: a diagonal single-qubit gate on a
#: control has no effect on the operator, so the pair around it annihilates.
_CONTROLLED = ("cx", "cy", "cz", "swap", "ccx", "cswap")


def _instruction(opcode: str, wires: tuple[int, ...], angle: float) -> Instruction:
    schema = OPERATOR_SCHEMAS[opcode]
    return Instruction(
        opcode,
        wires,
        params=dict.fromkeys(schema.parameters, angle),
    )


def _unitaries() -> list[str]:
    return sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity <= 2
    )


def _embed(instruction: Instruction, width: int) -> torch.Tensor:
    """The instruction as a ``2**width`` matrix, with qubit 0 most significant."""

    arity = len(instruction.wires)
    local = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1, 2**arity, 2**arity)[0]
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


def rule_source_sweep() -> dict[str, Any]:
    """Compare the rule source against the runtime's matrices, pair by pair.

    A false "yes" changes a program; a "no" costs an optimization. The two are
    counted separately and the declines are reported by opcode pair, because a
    rule source is worth shipping only if the first count is zero and the second
    is small enough to explain.
    """

    seed = random.Random(_SWEEP_SEED)
    unitaries = _unitaries()
    pair_count = 0
    true_commuting = 0
    answered = 0
    false_yes: list[tuple[str, tuple[int, ...], str, tuple[int, ...]]] = []
    declined: dict[tuple[str, str], int] = {}
    same_wires = 0

    for left_name, right_name in itertools.product(unitaries, repeat=2):
        for left_wires in _PLACEMENTS:
            if len(left_wires) != OPERATOR_SCHEMAS[left_name].arity:
                continue
            for right_wires in _PLACEMENTS:
                if len(right_wires) != OPERATOR_SCHEMAS[right_name].arity:
                    continue
                left = _instruction(left_name, left_wires, seed.uniform(-3.0, 3.0))
                right = _instruction(right_name, right_wires, seed.uniform(-3.0, 3.0))
                pair_count += 1
                width = max(max(left.wires), max(right.wires)) + 1
                left_matrix = _embed(left, max(width, _REGISTER_WIDTH))
                right_matrix = _embed(right, max(width, _REGISTER_WIDTH))
                truth = bool(
                    torch.allclose(
                        left_matrix @ right_matrix,
                        right_matrix @ left_matrix,
                        atol=1.0e-10,
                        rtol=0,
                    )
                )
                verdict = commute(left, right)
                if truth:
                    true_commuting += 1
                    if verdict:
                        answered += 1
                    else:
                        key = tuple(sorted((left_name, right_name)))
                        declined[key] = declined.get(key, 0) + 1
                elif verdict:
                    false_yes.append((left_name, left_wires, right_name, right_wires))
                if left.wires == right.wires and verdict:
                    same_wires += 1

    return {
        "pair_count": pair_count,
        "true_commuting_pair_count": true_commuting,
        "answered_pair_count": answered,
        "declined_pair_count": true_commuting - answered,
        "answered_share": answered / true_commuting,
        "false_yes_count": len(false_yes),
        "false_yes": false_yes[:8],
        "same_wire_yes_count": same_wires,
        "rule_opcode_count": len(_RULE_OPCODES),
        "declined_by_opcode_pair": [
            {"opcode_pair": list(key), "count": value}
            for key, value in sorted(declined.items())
        ],
    }


def _legacy_fixed_point(ir: CircuitIR) -> CircuitIR:
    """The pipeline as it stood before the commutation pass, to a fixed point."""

    for _ in range(len(ir) + 1):
        previous = len(ir)
        ir = remove_identity_gates(ir)
        ir = merge_self_inverse(ir)
        ir = merge_adjacent_rotations(ir)
        ir = remove_identity_gates(ir)
        if len(ir) == previous:
            return ir
    raise AssertionError("the legacy pipeline did not reach a fixed point")


def _commuting_gap_circuit(seed: random.Random, wire_index: int) -> CircuitIR:
    """A controlled gate pair separated by a rotation on a chosen wire.

    ``wire_index`` picks the position in the gate's own wire tuple. Position 0 is
    a control for every gate in the table, which is the case the new pass exists
    for; the higher positions are the target of a controlled gate or a swapped
    wire, where the rules mostly refuse, and a position past the end of the tuple
    is a qubit the gate does not touch, which ``merge_self_inverse`` already
    steps over. Reporting those three rows side by side is what shows that the
    pass adds reach *and* where it stops.
    """

    controlled = seed.choice(_CONTROLLED)
    wires = tuple(
        seed.sample(range(_REGISTER_WIDTH), OPERATOR_SCHEMAS[controlled].arity)
    )
    if wire_index < len(wires):
        gap_qubit = wires[wire_index]
    else:
        free = [qubit for qubit in range(_REGISTER_WIDTH) if qubit not in wires]
        gap_qubit = free[0] if free else wires[-1]
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(controlled, wires),
            _instruction("rz", (gap_qubit,), _ANGLE),
            Instruction(controlled, wires),
        ),
    )


def gap_position_table() -> list[dict[str, Any]]:
    """Where each controlled gate tolerates a diagonal rotation, and where it does not.

    This is the rule source's own answer read back as a table, and it is what
    makes the ``gap_on_wire_1`` row of the delta below explainable rather than
    surprising: ``cz`` is diagonal, so a rotation on either of its wires is
    invisible; ``cx`` and ``cy`` tolerate one only on the control; ``swap``
    tolerates neither, because it exchanges them.
    """

    rows = []
    for controlled in _CONTROLLED:
        arity = OPERATOR_SCHEMAS[controlled].arity
        wires = tuple(range(arity))
        answers = [
            commute(
                _instruction("rz", (qubit,), _ANGLE),
                Instruction(controlled, wires),
            )
            for qubit in wires
        ]
        rows.append(
            {
                "opcode": controlled,
                "arity": arity,
                "rotation_commutes_on_wire": answers,
                "tolerated_wire_count": sum(1 for answer in answers if answer),
            }
        )
    return rows


def _chained_gap_circuit(seed: random.Random, length: int) -> CircuitIR:
    """A chain of controlled gates interleaved with rotations on their controls."""

    instructions: list[Instruction] = []
    for _ in range(length):
        controlled = seed.choice(_CONTROLLED)
        arity = OPERATOR_SCHEMAS[controlled].arity
        wires = tuple(seed.sample(range(_REGISTER_WIDTH), arity))
        instructions.append(Instruction(controlled, wires))
        instructions.append(_instruction("rz", (wires[0],), _ANGLE))
        instructions.append(Instruction(controlled, wires))
    return CircuitIR(_REGISTER_WIDTH, tuple(instructions))


def _state(ir: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        ir, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    )


def _measure_population(label: str, circuits: list[CircuitIR]) -> dict[str, Any]:
    """Run one circuit family through both pipelines and check every removal."""

    source_instructions = 0
    legacy_instructions = 0
    optimized_instructions = 0
    changed = 0
    worst_difference = 0.0
    widest_block = 0
    groups_found = 0
    cancellable_candidates = 0
    rotation_groups_found = 0
    rotation_candidates = 0

    for ir in circuits:
        source_instructions += len(ir)
        legacy = _legacy_fixed_point(ir)
        optimized = _optimize_to_fixed_point(ir)
        legacy_instructions += len(legacy)
        optimized_instructions += len(optimized)
        if len(optimized) != len(legacy):
            changed += 1
            difference = float((_state(optimized) - _state(ir)).abs().max())
            worst_difference = max(worst_difference, difference)
        summary = analyze_commutation(ir).summary()
        widest_block = max(widest_block, int(summary["widest_block"]))
        analysis = analyze_commutation(ir)
        groups = cancellable_positions(tuple(ir), analysis)
        groups_found += sum(1 for positions in groups.values() if len(positions) > 1)
        cancellable_candidates += sum(len(positions) for positions in groups.values())
        rotations = rotation_groups(tuple(ir), analysis)
        rotation_groups_found += sum(
            1 for positions in rotations.values() if len(positions) > 1
        )
        rotation_candidates += sum(
            len(positions) for positions in rotations.values() if len(positions) > 1
        )

    return {
        "label": label,
        "circuit_count": len(circuits),
        "source_instruction_count": source_instructions,
        "legacy_instruction_count": legacy_instructions,
        "optimized_instruction_count": optimized_instructions,
        "removed_by_legacy": source_instructions - legacy_instructions,
        "removed_by_commutation": legacy_instructions - optimized_instructions,
        "changed_circuit_count": changed,
        "changed_circuit_share": changed / len(circuits),
        "max_state_difference": worst_difference,
        "widest_commuting_block": widest_block,
        "cancellation_group_count": groups_found,
        "cancellable_position_count": cancellable_candidates,
        "rotation_group_count": rotation_groups_found,
        "rotation_position_count": rotation_candidates,
    }


def _rotation_chain_circuit(
    seed: random.Random, length: int, control_share: float
) -> CircuitIR:
    """A chain of z-rotations on one wire interleaved with entanglers touching it.

    ``control_share`` is the share of entanglers placed so that the wire is their
    first position, which is where a diagonal rotation is proven to commute. The
    rest put the wire on a non-first position, which for most of the table is a
    refusal. Both directions are in the population on purpose: a chain of controls
    only would measure the reach and say nothing about the boundary, and the counts
    are pinned so a change to where the rule stops shows up as a moved number.
    """

    others = list(range(1, _REGISTER_WIDTH))
    instructions: list[Instruction] = []
    for index in range(length):
        instructions.append(_instruction("rz", (0,), _ANGLE + 0.001 * index))
        controlled = seed.choice(_CONTROLLED)
        arity = OPERATOR_SCHEMAS[controlled].arity
        if seed.random() < control_share:
            wires = (0, *seed.sample(others, arity - 1))
        else:
            wires = (others[0], 0, *seed.sample(others[1:], arity - 2))
        instructions.append(Instruction(controlled, wires))
    instructions.append(_instruction("rz", (0,), _ANGLE))
    return CircuitIR(_REGISTER_WIDTH, tuple(instructions))


def pipeline_delta() -> list[dict[str, Any]]:
    """The pipeline's own reach on the seeded populations."""

    seed = random.Random(_CIRCUIT_SEED)
    populations: list[tuple[str, list[CircuitIR]]] = []
    for wire_index, name in (
        (0, "gap_on_wire_0"),
        (1, "gap_on_wire_1"),
        (3, "gap_on_unrelated_qubit"),
    ):
        populations.append(
            (
                name,
                [_commuting_gap_circuit(seed, wire_index) for _ in range(60)],
            )
        )
    populations.append(
        ("chained_control_gaps", [_chained_gap_circuit(seed, 6) for _ in range(40)])
    )
    # The rotation merge has no reach on any population above, which is a fact
    # worth stating rather than leaving implicit: `_commuting_gap_circuit` carries a
    # single rotation and `_chained_gap_circuit` puts two identical entanglers next
    # to each other, so the adjacent passes have emptied the gap before either
    # rotation rule is asked. Without this population the new rule would have no
    # aggregate measurement at all.
    populations.append(
        (
            "rotation_chain_on_controls",
            [_rotation_chain_circuit(seed, 8, 1.0) for _ in range(40)],
        )
    )
    populations.append(
        (
            "rotation_chain_mixed_placement",
            [_rotation_chain_circuit(seed, 8, 0.5) for _ in range(40)],
        )
    )
    return [_measure_population(label, circuits) for label, circuits in populations]


#: The controlled opcodes Qiskit's `CommutativeCancellation` reasons about at
#: all. Its `_gates` set holds `cx`, `cy`, `cz`, `h` and `y`; `swap`, `ccx` and
#: `cswap` are not in it, so a per-opcode comparison is only meaningful where the
#: two passes are attempting the same removal.
_QISKIT_SCOPE = ("cx", "cy", "cz")


def _anchor_circuit(opcode: str, index: int) -> CircuitIR:
    """A controlled-gate pair around a rotation on its first wire, per opcode."""

    wires = tuple(range(OPERATOR_SCHEMAS[opcode].arity))
    angle = _ANGLE + 0.01 * index
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(opcode, wires),
            _instruction("rz", (0,), angle),
            Instruction(opcode, wires),
        ),
    )


def _anchor_rotation_circuit(opcode: str, wire_index: int) -> CircuitIR:
    """Two z-rotations around an entangler, on the entangler's own wires.

    The two angles differ, so a pass that merged them has to have added them rather
    than cancelled them; a pair of equal angles would be reachable by the
    cancellation rule alone and the row could not tell the two rules apart.
    """

    wires = tuple(range(OPERATOR_SCHEMAS[opcode].arity))
    qubit = wires[wire_index]
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            _instruction("rz", (qubit,), _ANGLE),
            Instruction(opcode, wires),
            _instruction("rz", (qubit,), _ANGLE + 0.01),
        ),
    )


def _to_qiskit(ir: CircuitIR, circuit_type: Any) -> Any:
    """The same program as a Qiskit circuit, for the optional anchor only."""

    circuit = circuit_type(_REGISTER_WIDTH)
    for instruction in ir:
        if instruction.name == "rz":
            circuit.rz(instruction.params["theta"], instruction.wires[0])
        else:
            getattr(circuit, instruction.name)(*instruction.wires)
    return circuit


def _qiskit_anchor() -> dict[str, Any]:
    """Qiskit's own cancellation on the same circuits, when Qiskit is importable.

    The anchor is a cross-check, not a dependency: this module has to run on a
    machine with no Qiskit at all. Qiskit's ``CommutativeCancellation`` reads the
    ``commutation_set`` property that ``CommutationAnalysis`` writes, so it is a
    two-pass sequence there against one analysis call here.

    Three differences are declared rather than smoothed over, and the table is
    split so a reader can see them:

    * Qiskit's cancellation set is single-qubit rotations only (``_x_rotations``
      and ``_z_rotations``), so it never tries the ``ccx``, ``cswap``, ``swap`` or
      ``cz`` pairs this pass removes. Rows outside ``_QISKIT_SCOPE`` measure a
      difference in scope, not in reasoning.
    * Its pass also merges runs of z-rotations in the same call, so a lower count
      on its side would not mean better commutation.
    * Its ``_gates`` set excludes Y rotations on purpose, because they do not
      commute with ``CNOT``; this pass reaches the same conclusion from the
      matrices rather than from a hand-maintained list.
    """

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.transpiler import PassManager  # type: ignore[import-not-found]
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            CommutationAnalysis,
            CommutativeCancellation,
        )
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    passes = PassManager([CommutationAnalysis(), CommutativeCancellation()])
    case_count = 20
    rows = []
    for opcode in _CONTROLLED:
        ours_removed = 0
        theirs_removed = 0
        source_gates = 0
        for index in range(case_count):
            ir = _anchor_circuit(opcode, index)
            source_gates += len(ir)
            ours_removed += len(ir) - len(_optimize_to_fixed_point(ir))
            qiskit_circuit = _to_qiskit(ir, QuantumCircuit)
            reduced = passes.run(qiskit_circuit)
            theirs_removed += qiskit_circuit.size() - reduced.size()
        rows.append(
            {
                "opcode": opcode,
                "in_qiskit_scope": opcode in _QISKIT_SCOPE,
                "case_count": case_count,
                "source_gate_count": source_gates,
                "port_removed_count": ours_removed,
                "qiskit_removed_count": theirs_removed,
            }
        )
    in_scope = [row for row in rows if row["in_qiskit_scope"]]

    # The second shape: two different z-rotations around an entangler. This is the
    # rule this round adds, and Qiskit's pass merges z-rotation runs in the same
    # call, so this is where the two implementations are actually attempting the
    # same rewrite. It is restricted to `_QISKIT_SCOPE` for the same reason the
    # table above is split: outside it Qiskit is not attempting the question.
    rotation_rows = []
    for opcode in _QISKIT_SCOPE:
        for wire_index in range(OPERATOR_SCHEMAS[opcode].arity):
            ir = _anchor_rotation_circuit(opcode, wire_index)
            qiskit_circuit = _to_qiskit(ir, QuantumCircuit)
            reduced = passes.run(qiskit_circuit)
            rotation_rows.append(
                {
                    "opcode": opcode,
                    "wire_index": wire_index,
                    "source_gate_count": len(ir),
                    "port_optimized_gate_count": len(_optimize_to_fixed_point(ir)),
                    "qiskit_optimized_gate_count": reduced.size(),
                }
            )
    return {
        "available": True,
        "pass_name": "CommutationAnalysis + CommutativeCancellation",
        "qiskit_version": __import__("qiskit").__version__,
        "rows": rows,
        "rotation_rows": rotation_rows,
        "in_scope_opcodes": list(_QISKIT_SCOPE),
        "in_scope_source_gate_count": sum(row["source_gate_count"] for row in in_scope),
        "in_scope_port_removed_count": sum(
            row["port_removed_count"] for row in in_scope
        ),
        "in_scope_qiskit_removed_count": sum(
            row["qiskit_removed_count"] for row in in_scope
        ),
        "out_of_scope_port_removed_count": sum(
            row["port_removed_count"] for row in rows if not row["in_qiskit_scope"]
        ),
        "rotation_row_count": len(rotation_rows),
        "rotation_source_gate_count": sum(
            row["source_gate_count"] for row in rotation_rows
        ),
        "rotation_port_optimized_gate_count": sum(
            row["port_optimized_gate_count"] for row in rotation_rows
        ),
        "rotation_qiskit_optimized_gate_count": sum(
            row["qiskit_optimized_gate_count"] for row in rotation_rows
        ),
        "note": (
            "Compared per opcode, because Qiskit's cancellation set does not "
            "include swap, ccx or cswap; counts are reported, never asserted equal."
        ),
    }


#: The single-qubit rotations a commuting gap can hide. `pipeline._ROTATION_PARAM`
#: also names seven two-qubit rotations; those are a different claim with their own
#: wire placement, so they are out of this table rather than silently folded into it.
_REDUCIBLE_ROTATIONS = ("rx", "ry", "rz", "phase", "u1")


def rotation_merge_table() -> dict[str, Any]:
    """Where two rotations of one opcode become one across a proven commuting gap.

    Each row places one rotation on both sides of a controlled gate, on one of the
    gate's own wires, and reports three things side by side: what the rule source
    says about that placement, whether the pipeline actually rewrote the pair, and
    -- when it did -- how far the compiled statevector moved from the source's.

    Two directions matter and they are counted separately. A row that merged where
    the rule source declines to prove commutation would be a wrong rewrite, so
    `merged_without_a_proof_count` has to stay zero; a row the rule source proves
    but the pipeline leaves standing is reach this pass does not yet take, reported
    as `proven_but_unmerged_count` rather than smoothed over. The control, the
    target of a controlled gate and a `swap`'s second wire all appear, because a
    table that listed only the placements that work could not show the boundary.
    """

    rows: list[dict[str, Any]] = []
    for rotation in _REDUCIBLE_ROTATIONS:
        for controlled in _CONTROLLED:
            arity = OPERATOR_SCHEMAS[controlled].arity
            wires = tuple(range(arity))
            for position in range(arity):
                qubit = wires[position]
                first = _instruction(rotation, (qubit,), _ANGLE)
                source = CircuitIR(
                    _REGISTER_WIDTH,
                    (
                        first,
                        Instruction(controlled, wires),
                        _instruction(rotation, (qubit,), _ANGLE),
                    ),
                )
                optimized = _optimize_to_fixed_point(source)
                merged = len(optimized) < len(source)
                difference = 0.0
                if merged:
                    difference = float((_state(optimized) - _state(source)).abs().max())
                rows.append(
                    {
                        "rotation_opcode": rotation,
                        "entangler_opcode": controlled,
                        "rotation_wire_index": position,
                        "commutes": commute(first, Instruction(controlled, wires)),
                        "merged": merged,
                        "source_instruction_count": len(source),
                        "optimized_instruction_count": len(optimized),
                        "max_state_difference": difference,
                    }
                )

    merged_rows = [row for row in rows if row["merged"]]
    return {
        "rotation_opcodes": list(_REDUCIBLE_ROTATIONS),
        "entangler_opcodes": list(_CONTROLLED),
        "row_count": len(rows),
        "merged_count": len(merged_rows),
        "proven_commuting_count": sum(1 for row in rows if row["commutes"]),
        "merged_without_a_proof_count": sum(
            1 for row in rows if row["merged"] and not row["commutes"]
        ),
        "proven_but_unmerged_count": sum(
            1 for row in rows if row["commutes"] and not row["merged"]
        ),
        "merged_by_opcode": {
            rotation: sum(
                1 for row in merged_rows if row["rotation_opcode"] == rotation
            )
            for rotation in _REDUCIBLE_ROTATIONS
        },
        "worst_merged_state_difference": max(
            (row["max_state_difference"] for row in merged_rows), default=0.0
        ),
        "rows": rows,
    }


def _identity_program(program: CircuitIR) -> CircuitIR:
    """The substitution used to attribute the rotation merge's own reach."""

    return program


def rotation_merge_reach() -> list[dict[str, Any]]:
    """What the rotation merge adds on its own, population by population.

    `removed_by_commutation` in `pipeline_delta` is the sum of both commuting rules,
    so it cannot attribute anything to the newer one. The attribution is measured by
    substitution instead: the shipped fixed point is compared against one in which
    only `merge_commuting_rotations` is replaced by the identity. Substituting one
    rule and re-running the loop is the only way to separate the two rules' reach
    here, and it is the same shape an earlier round used for the one-qubit fold.

    The substitution is visible because `_optimize_to_fixed_point` imports the rule
    inside its body, so it reads the module attribute on every call rather than
    capturing it at import time.
    """

    from flagquantum.compiler.commutation_cancellation import (
        merge_commuting_rotations,
    )

    seed = random.Random(_CIRCUIT_SEED)
    populations: list[tuple[str, list[CircuitIR]]] = []
    for wire_index, name in (
        (0, "gap_on_wire_0"),
        (1, "gap_on_wire_1"),
        (3, "gap_on_unrelated_qubit"),
    ):
        populations.append(
            (name, [_commuting_gap_circuit(seed, wire_index) for _ in range(60)])
        )
    populations.append(
        ("chained_control_gaps", [_chained_gap_circuit(seed, 6) for _ in range(40)])
    )
    populations.append(
        (
            "rotation_chain_on_controls",
            [_rotation_chain_circuit(seed, 8, 1.0) for _ in range(40)],
        )
    )
    populations.append(
        (
            "rotation_chain_mixed_placement",
            [_rotation_chain_circuit(seed, 8, 0.5) for _ in range(40)],
        )
    )

    module = sys.modules[merge_commuting_rotations.__module__]
    rows: list[dict[str, Any]] = []
    for label, circuits in populations:
        shipped_count = 0
        without_count = 0
        worst_difference = 0.0
        for ir in circuits:
            shipped = _optimize_to_fixed_point(ir)
            shipped_count += len(shipped)
            with patch.object(module, "merge_commuting_rotations", _identity_program):
                without = _optimize_to_fixed_point(ir)
            without_count += len(without)
            if len(without) != len(shipped):
                difference = float((_state(without) - _state(shipped)).abs().max())
                worst_difference = max(worst_difference, difference)
        rows.append(
            {
                "label": label,
                "circuit_count": len(circuits),
                "shipped_instruction_count": shipped_count,
                "without_rotation_merge_instruction_count": without_count,
                "removed_by_rotation_merge": without_count - shipped_count,
                "max_state_difference": worst_difference,
            }
        )
    return rows


def run_benchmark() -> dict[str, Any]:
    sweep = rule_source_sweep()
    delta = pipeline_delta()
    reach = rotation_merge_reach()
    return {
        "schema": SCHEMA,
        "generated_at": _NOW.isoformat(),
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "seed": {"sweep": _SWEEP_SEED, "circuit": _CIRCUIT_SEED},
        "reference_algorithm": "qiskit_commutation_analysis_and_commutative_cancellation",
        "reference_revision": "qiskit 1.2.4 CommutationAnalysis + CommutativeCancellation",
        "rule_source_sweep": sweep,
        "gap_position_table": gap_position_table(),
        "rotation_merge_table": rotation_merge_table(),
        "pipeline_delta": delta,
        "rotation_merge_reach": reach,
        "reference_anchor": _qiskit_anchor(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    sweep = payload["rule_source_sweep"]
    print(
        f"rule source over {sweep['pair_count']} opcode/placement pairs "
        f"({sweep['rule_opcode_count']} opcodes carry a rule)"
    )
    print(
        f"  truly commuting {sweep['true_commuting_pair_count']}, "
        f"answered {sweep['answered_pair_count']}, "
        f"declined {sweep['declined_pair_count']}, "
        f"false yes {sweep['false_yes_count']}"
    )
    print(f"  answered share {sweep['answered_share']:.4f}")
    print()
    print("a diagonal rotation on each wire of a controlled gate:")
    for row in payload["gap_position_table"]:
        marks = " ".join(
            "yes" if answer else " no" for answer in row["rotation_commutes_on_wire"]
        )
        print(f"  {row['opcode']:6s} arity {row['arity']}  {marks}")
    print()
    merge = payload["rotation_merge_table"]
    print(
        f"rotations merged across a proven gap: {merge['merged_count']} of "
        f"{merge['row_count']} placements "
        f"({merge['proven_commuting_count']} proven, "
        f"{merge['merged_without_a_proof_count']} merged without a proof, "
        f"{merge['proven_but_unmerged_count']} proven but left standing)"
    )
    print(
        "  merged by opcode: "
        + ", ".join(
            f"{opcode} {count}" for opcode, count in merge["merged_by_opcode"].items()
        )
        + f"; worst movement {merge['worst_merged_state_difference']:.3e}"
    )
    print()
    print(
        f"{'population':28s} {'circuits':>8s} {'source':>7s} {'legacy':>7s} "
        f"{'W9':>7s} {'via pass':>9s} {'changed':>8s} {'max diff':>10s}"
    )
    for row in payload["pipeline_delta"]:
        print(
            f"{row['label']:28s} {row['circuit_count']:>8d} "
            f"{row['source_instruction_count']:>7d} "
            f"{row['legacy_instruction_count']:>7d} "
            f"{row['optimized_instruction_count']:>7d} "
            f"{row['removed_by_commutation']:>9d} "
            f"{row['changed_circuit_count']:>8d} "
            f"{row['max_state_difference']:>10.3e}"
        )
    anchor = payload["reference_anchor"]
    print()
    if anchor["available"]:
        print(f"Qiskit {anchor['qiskit_version']} {anchor['pass_name']}:")
        print(
            f"  {'opcode':8s} {'in scope':>9s} {'removed here':>13s} {'removed there':>14s}"
        )
        for row in anchor["rows"]:
            print(
                f"  {row['opcode']:8s} {row['in_qiskit_scope']!s:>9s} "
                f"{row['port_removed_count']:>13d} "
                f"{row['qiskit_removed_count']:>14d}"
            )
        print(
            f"  in-scope total: {anchor['in_scope_port_removed_count']} here vs "
            f"{anchor['in_scope_qiskit_removed_count']} there over "
            f"{anchor['in_scope_source_gate_count']} source gates"
        )
        print(
            f"  two z-rotations around an entangler: "
            f"{anchor['rotation_source_gate_count']} source gates -> "
            f"{anchor['rotation_port_optimized_gate_count']} here vs "
            f"{anchor['rotation_qiskit_optimized_gate_count']} there "
            f"over {anchor['rotation_row_count']} placements"
        )
        for row in anchor["rotation_rows"]:
            print(
                f"    {row['opcode']:8s} wire {row['wire_index']} "
                f"{row['source_gate_count']:>3d} -> "
                f"{row['port_optimized_gate_count']:>3d} here, "
                f"{row['qiskit_optimized_gate_count']:>3d} there"
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
