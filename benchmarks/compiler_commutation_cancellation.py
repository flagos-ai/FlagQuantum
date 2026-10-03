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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from flagquantum.compiler.commutation import (
    _RULE_OPCODES,
    analyze_commutation,
    commute,
)
from flagquantum.compiler.commutation_cancellation import cancellable_positions
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
    }


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
            qiskit_circuit = QuantumCircuit(_REGISTER_WIDTH)
            for instruction in ir:
                if instruction.name == "rz":
                    qiskit_circuit.rz(instruction.params["theta"], instruction.wires[0])
                else:
                    getattr(qiskit_circuit, instruction.name)(*instruction.wires)
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
    return {
        "available": True,
        "pass_name": "CommutationAnalysis + CommutativeCancellation",
        "qiskit_version": __import__("qiskit").__version__,
        "rows": rows,
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
        "note": (
            "Compared per opcode, because Qiskit's cancellation set does not "
            "include swap, ccx or cswap; counts are reported, never asserted equal."
        ),
    }


def run_benchmark() -> dict[str, Any]:
    sweep = rule_source_sweep()
    delta = pipeline_delta()
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
        "pipeline_delta": delta,
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
    else:
        print(f"Qiskit anchor unavailable: {anchor['reason']}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
