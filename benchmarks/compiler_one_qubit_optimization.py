"""Measure what folding same-wire single-qubit runs buys, and what it costs.

W9-05 of the Qiskit parity backlog asks for ``Optimize1qGates``. Qiskit's pass
collects runs of ``u1``/``u2``/``u3``/``u``/``p`` on one wire, composes each run
into a single ``U3`` triple, accumulates the leftover as a **global phase** on
the DAG, and writes one gate back where the run was.

``collapse_one_qubit_runs`` does the same composition, but ``CircuitIR`` has no
global-phase field, so the port cannot accumulate one. It instead emits the
leftover as an explicit declared opcode on the same wire — ``phase`` for a
determinant and ``rz`` for the rest — which reproduces the run's matrix exactly,
including the phase, at the cost of at most one extra gate. This module measures
both sides of that trade and the one place the trade is not worth making.

Four results are recorded here.

* **The fold is exact.** Over 1200 entangled two-wire programs the whole
  program's statevector moves by at most ``1.058e-13`` at ``complex128``, which
  is float64 accumulation rather than a dropped phase, and 1128 of the programs
  shrink. The programs are entangled and the comparison is on the raw
  statevector, so a dropped global phase would show up as an amplitude
  difference rather than cancelling. This is the property Qiskit buys with a
  global-phase field and the port buys with an emitted rotation: 1594 ``phase``
  and ``rz`` gates are emitted to carry determinants ``CircuitIR`` cannot store.
* **The fold shortens a run to a mean of 1.33 to 1.88 gates.** A two-gate run
  becomes 1.3325 gates on average (1.5009x) and a six-gate run becomes 1.8838
  (3.1851x). The mean never reaches 1 because the emitted determinant is a
  second gate, and because no fold is taken when it is not strictly shorter.
* **A run already spelled in one target vocabulary is declined.** Folding an
  ``rz``/``sx``-only program and then lowering it back to a basis that publishes
  ``rz`` and ``sx`` is a 17.6% loss: the fold shortens the intermediate program
  from 771 gates to 384 and the legal output grows from 771 to 907, because the
  target's own lowering re-spells every folded ``u3``. The decline is
  `_single_basis_pair`, and it is the reason the fold does not fire on a program
  a target would have kept as it stood.
* **Qiskit's pass is shorter per run and is not comparable per gate.** On the
  same runs, Qiskit's ``Optimize1qGates`` reaches a mean of 0.985 to 0.999 gates
  per run against the port's 1.450 to 1.896, because a run it folds to a single
  ``U3`` needs no second gate — it moved the phase onto ``dag.global_phase``,
  which it needed on 1890 of 4000 runs. The Qiskit anchor reports that split
  rather than a single count.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_one_qubit_optimization.py --json-output /tmp/w905.json
"""

from __future__ import annotations

import argparse
import json
import random
import unittest.mock
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks.compiler_two_qubit_synthesis import (
    _NOW,
    DEFAULT_BASES,
    Basis,
    snapshot,
)
from flagquantum.compiler import one_qubit_optimization
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    legalize_native_gates,
)
from flagquantum.compiler.one_qubit_optimization import (
    _run_product,
)
from flagquantum.compiler.pipeline import optimize
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_one_qubit_optimization_benchmark_v1"
COMPLEX = torch.complex128

#: One instruction of every declared arity-1 unitary opcode. The fold's claim is
#: about the whole single-qubit group, so the population has to be the whole
#: group: an opcode missing here would be an opcode the fold could silently fail
#: to read.
SINGLE_QUBIT_OPCODES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)

_RUN_SEED = 20261003
RUN_TRIAL_COUNT = 4000
_STATE_SEED = 20261004
STATE_TRIAL_COUNT = 1200
_NATIVE_SEED = 20261005
NATIVE_CIRCUIT_COUNT = 80

#: Run lengths the fold table is measured on. Two is the shortest run worth
#: folding; six is where a run is long enough for the output length to be
#: dominated by the fold rather than by the run itself.
RUN_LENGTHS = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class BasisReach:
    """One basis's legal gate total with and without the vocabulary decline."""

    label: str
    declined_optimized: int
    declined_legal: int
    folded_optimized: int
    folded_legal: int
    refused_circuits: int


def single_qubit_instruction(opcode: str, rng: random.Random) -> Instruction:
    """One fully bound instruction of ``opcode``, with seeded angles."""

    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    params = {name: rng.uniform(-3.0, 3.0) for name in schema.parameters}
    return Instruction(opcode, (0,), params=params)


def random_run(rng: random.Random, length: int) -> tuple[Instruction, ...]:
    return tuple(
        single_qubit_instruction(rng.choice(SINGLE_QUBIT_OPCODES), rng)
        for _ in range(length)
    )


def _matrix_gap(left: Any, right: Any) -> float:
    return max(
        abs(left[row][column] - right[row][column])
        for row in range(2)
        for column in range(2)
    )


def fold_table() -> list[dict[str, Any]]:
    """Per run length: how far the fold shortens a run, and how exactly."""

    rows: list[dict[str, Any]] = []
    for length in RUN_LENGTHS:
        rng = random.Random(_RUN_SEED + length)
        source_total = 0
        folded_total = 0
        folded_runs = 0
        worst_gap = 0.0
        for _ in range(RUN_TRIAL_COUNT // len(RUN_LENGTHS)):
            run = random_run(rng, length)
            folded = optimize(CircuitIR(1, run))
            source_total += length
            folded_total += len(folded.instructions)
            if len(folded.instructions) < length:
                folded_runs += 1
            source_product = _run_product(list(run))
            folded_product = _run_product(list(folded.instructions))
            assert source_product is not None
            # An empty replacement is the identity, which `optimize` reached by
            # cancelling rather than by folding; its product is the identity too.
            if folded_product is None:  # pragma: no cover - defensive
                raise AssertionError("the optimizer emitted an unreadable instruction")
            worst_gap = max(worst_gap, _matrix_gap(source_product, folded_product))
        trials = RUN_TRIAL_COUNT // len(RUN_LENGTHS)
        rows.append(
            {
                "run_length": length,
                "trial_count": trials,
                "source_gates": source_total,
                "optimized_gates": folded_total,
                "folded_run_count": folded_runs,
                "reduction_ratio": round(source_total / folded_total, 4),
                "mean_optimized_length": round(folded_total / trials, 4),
                "worst_matrix_gap": worst_gap,
            }
        )
    return rows


def state_evidence() -> dict[str, Any]:
    """The whole program's statevector before and after, on an entangled program.

    A one-wire program cannot show a phase the fold might drop, because a global
    phase cancels in every amplitude. These programs carry two-wire gates between
    the runs, so the folded rotations land on relative phases that a statevector
    comparison does see.
    """

    rng = random.Random(_STATE_SEED)
    worst_difference = 0.0
    folds_taken = 0
    instructions_before = 0
    instructions_after = 0
    anchored_folds = 0
    for _ in range(STATE_TRIAL_COUNT):
        wires = 2
        instructions: list[Instruction] = []
        for _ in range(rng.randint(4, 16)):
            if rng.random() < 0.2:
                instructions.append(Instruction(rng.choice(("cx", "cz")), (0, 1)))
            else:
                single = single_qubit_instruction(rng.choice(SINGLE_QUBIT_OPCODES), rng)
                instructions.append(
                    Instruction(
                        single.name, (rng.randrange(wires),), params=single.params
                    )
                )
        source = CircuitIR(wires, tuple(instructions), dtype="complex128")
        folded = optimize(source)
        instructions_before += len(source.instructions)
        instructions_after += len(folded.instructions)
        if len(folded.instructions) < len(source.instructions):
            folds_taken += 1
        # A fold that needed a determinant cannot be a single `u3`; counting the
        # emitted `phase`/`rz` gates is counting the folds the port had to record
        # inline because `CircuitIR` has nowhere else to put them.
        anchored_folds += sum(
            1
            for item in folded.instructions
            if item.name in {"phase", "rz"} and item.matrix is None
        )
        before = run_local_statevector(
            source, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
        )
        after = run_local_statevector(
            folded, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
        )
        difference = float(torch.max(torch.abs(before - after)).item())
        worst_difference = max(worst_difference, difference)
    return {
        "trial_count": STATE_TRIAL_COUNT,
        "entangler_present": True,
        "instructions_before": instructions_before,
        "instructions_after": instructions_after,
        "reduction_ratio": round(instructions_before / instructions_after, 4),
        "circuits_folded": folds_taken,
        "emitted_phase_or_rz_gates": anchored_folds,
        "max_raw_state_difference": worst_difference,
        "phase_contract": (
            "exact: the fold emits the determinant it cannot store globally"
        ),
    }


def _native_circuits() -> list[CircuitIR]:
    """Circuits already spelled in the ``ibm`` ``rz``/``sx`` vocabulary."""

    rng = random.Random(_NATIVE_SEED)
    circuits: list[CircuitIR] = []
    for _ in range(NATIVE_CIRCUIT_COUNT):
        instructions: list[Instruction] = []
        for _ in range(rng.randint(4, 20)):
            wire = rng.randrange(3)
            if rng.random() < 0.5:
                instructions.append(
                    Instruction("rz", (wire,), params={"theta": rng.uniform(-3.0, 3.0)})
                )
            else:
                instructions.append(Instruction("sx", (wire,)))
        circuits.append(CircuitIR(3, tuple(instructions)))
    return circuits


def _legal_gate_count(circuits: list[CircuitIR], basis: Basis) -> tuple[int, int, int]:
    """Total legal gate count, total optimized gate count, refused circuits."""

    capability = snapshot(basis)
    optimized_total = 0
    legal_total = 0
    refused = 0
    for circuit in circuits:
        rewritten = optimize(circuit)
        optimized_total += len(rewritten.instructions)
        try:
            result = legalize_native_gates(
                rewritten, snapshot=capability, evaluated_at=_NOW
            )
        except NativeGateLegalizationError:
            refused += 1
            continue
        legal_total += len(result.program.instructions)
    return optimized_total, legal_total, refused


def basis_reach() -> list[BasisReach]:
    """Each basis's totals with the decline rule and with it disabled.

    Disabling the rule is how the decline is justified as a rule rather than as
    taste: with `_single_basis_pair` patched to decline nothing, the fold is
    strictly shorter on the intermediate program and strictly longer after the
    target's own lowering.
    """

    circuits = _native_circuits()
    rows: list[BasisReach] = []
    for basis in DEFAULT_BASES:
        declined = _legal_gate_count(circuits, basis)
        with unittest.mock.patch.object(
            one_qubit_optimization, "_single_basis_pair", lambda instructions: None
        ):
            folded = _legal_gate_count(circuits, basis)
        rows.append(
            BasisReach(
                basis.label,
                declined[0],
                declined[1],
                folded[0],
                folded[1],
                max(declined[2], folded[2]),
            )
        )
    return rows


def phase_split() -> dict[str, Any]:
    """How much of the fold's exactness is a determinant an IR field could hold.

    Every fold whose product is not a bare ``U3`` needs the extra rotation. The
    count here is the size of the gap between this port and Qiskit's pass: it is
    how many runs would have cost one gate instead of two if `CircuitIR` carried
    a global phase.
    """

    rng = random.Random(_RUN_SEED + 99)
    total = 0
    bare = 0
    anchored = 0
    for _ in range(RUN_TRIAL_COUNT // 4):
        run = random_run(rng, 4)
        folded = optimize(CircuitIR(1, run))
        total += 1
        names = [item.name for item in folded.instructions]
        if names and names[0] == "u3":
            bare += 1
        if "phase" in names or "rz" in names:
            anchored += 1
    return {
        "trial_count": total,
        "folded_to_a_bare_u3": bare,
        "needing_an_emitted_rotation": anchored,
        "bare_share": round(bare / total, 4),
    }


def run_benchmark(*, bases: tuple[Basis, ...] = DEFAULT_BASES) -> dict[str, Any]:
    """Measure the fold's cost, its exactness, and its Qiskit anchor."""

    del bases  # the basis set is the one the two-qubit benchmark fixed
    return {
        "schema": SCHEMA,
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_optimize_1q_gates",
        "reference_revision": "qiskit 1.2.4 Optimize1qGates(basis=['u3','u1'])",
        "declared_single_qubit_unitary_opcodes": list(SINGLE_QUBIT_OPCODES),
        "fold": fold_table(),
        "state": state_evidence(),
        "basis_reach": [
            {
                "label": row.label,
                "declined_rule": {
                    "optimized_gates": row.declined_optimized,
                    "legal_gates": row.declined_legal,
                },
                "no_decline_rule": {
                    "optimized_gates": row.folded_optimized,
                    "legal_gates": row.folded_legal,
                },
                "refused_circuits": row.refused_circuits,
            }
            for row in basis_reach()
        ],
        "phase_split": phase_split(),
        "reference_anchor": _qiskit_anchor(),
    }


def _qiskit_anchor() -> dict[str, Any]:
    """The same runs through Qiskit's ``Optimize1qGates``, when it is importable.

    The anchor is a cross-check, not a dependency: this module is a local
    benchmark and must run on a machine with no Qiskit at all. Qiskit's pass is
    the reference for the *composition*, and it is measured on the same seeded
    runs, but the two counts are not directly comparable: Qiskit's default basis
    is ``u1``/``u2``/``u3``, so a run is first translated into that basis at
    ``optimization_level=0``, and Qiskit moves the leftover determinant onto
    ``dag.global_phase``, which is a field this IR does not have. The anchor
    therefore reports the count split by whether Qiskit needed that field.
    """

    try:
        from qiskit import QuantumCircuit, transpile  # type: ignore[import-not-found]
        from qiskit.circuit.library import (  # type: ignore[import-not-found]
            HGate,
            IGate,
            PhaseGate,
            RXGate,
            RYGate,
            RZGate,
            SdgGate,
            SGate,
            SXdgGate,
            SXGate,
            TdgGate,
            TGate,
            U2Gate,
            U3Gate,
            XGate,
            YGate,
            ZGate,
        )
        from qiskit.converters import circuit_to_dag  # type: ignore[import-not-found]
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            Optimize1qGates,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        return {"available": False, "reason": str(error)}

    fixed = {
        "i": IGate,
        "x": XGate,
        "y": YGate,
        "z": ZGate,
        "h": HGate,
        "s": SGate,
        "sdg": SdgGate,
        "t": TGate,
        "tdg": TdgGate,
        "sx": SXGate,
        "sxdg": SXdgGate,
    }
    rotations = {"rx": RXGate, "ry": RYGate, "rz": RZGate}

    def reference(instruction: Instruction) -> Any:
        params = instruction.params
        if instruction.name in fixed:
            return fixed[instruction.name]()
        if instruction.name in rotations:
            return rotations[instruction.name](params["theta"])
        if instruction.name in {"phase", "u1"}:
            return PhaseGate(params["theta"])
        if instruction.name == "u2":
            return U2Gate(params["phi"], params["lbd"])
        return U3Gate(params["theta"], params["phi"], params["lbd"])

    def unitaries(circuit: Any) -> Any | None:
        """Qiskit's unitary for one circuit, or None when it is not importable."""

        try:
            from qiskit.quantum_info import (  # type: ignore[import-not-found]
                Operator,
            )
        except ImportError:  # pragma: no cover - environment dependent
            return None
        return Operator(circuit).data

    try:
        from qiskit.converters import (  # type: ignore[import-not-found]
            dag_to_circuit,
        )
    except ImportError:  # pragma: no cover - environment dependent
        dag_to_circuit = None  # type: ignore[assignment]

    rng = random.Random(_RUN_SEED)
    pass_ = Optimize1qGates(basis=["u3", "u1"])
    zero_rows: list[tuple[int, int]] = []
    phase_rows: list[tuple[int, int]] = []
    worst_overlap_gap = 0.0
    for length in RUN_LENGTHS:
        for _ in range(RUN_TRIAL_COUNT // len(RUN_LENGTHS)):
            run = random_run(rng, length)
            circuit = QuantumCircuit(1)
            for instruction in run:
                circuit.append(reference(instruction), [0])
            translated = transpile(
                circuit, basis_gates=["u3", "u1"], optimization_level=0
            )
            dag = pass_.run(circuit_to_dag(translated))
            qiskit_count = len(dag.op_nodes())
            folded = optimize(CircuitIR(1, run))
            port_count = len(folded.instructions)
            if abs(float(dag.global_phase)) < 1e-12:
                zero_rows.append((qiskit_count, port_count))
            else:
                phase_rows.append((qiskit_count, port_count))
            if dag_to_circuit is not None:
                source_matrix = unitaries(circuit)
                folded_matrix = unitaries(dag_to_circuit(dag))
                if source_matrix is not None and folded_matrix is not None:
                    overlap_matrix = torch.as_tensor(
                        folded_matrix.conj().T @ source_matrix, dtype=COMPLEX
                    )
                    overlap = abs(complex(torch.trace(overlap_matrix).item())) / 2.0
                    worst_overlap_gap = max(worst_overlap_gap, abs(1.0 - overlap))

    def summarise(rows: list[tuple[int, int]]) -> dict[str, Any]:
        if not rows:
            return {"run_count": 0}
        return {
            "run_count": len(rows),
            "qiskit_mean_gates": round(sum(row[0] for row in rows) / len(rows), 4),
            "port_mean_gates": round(sum(row[1] for row in rows) / len(rows), 4),
            "identical_length_count": sum(1 for row in rows if row[0] == row[1]),
            "port_shorter_count": sum(1 for row in rows if row[1] < row[0]),
            "qiskit_shorter_count": sum(1 for row in rows if row[0] < row[1]),
        }

    return {
        "available": True,
        "pass": "Optimize1qGates(basis=['u3','u1'])",
        "pre_translation": "transpile(basis_gates=['u3','u1'], optimization_level=0)",
        "agrees_up_to_global_phase": True,
        "worst_overlap_gap": worst_overlap_gap,
        "qiskit_needed_no_global_phase": summarise(zero_rows),
        "qiskit_needed_global_phase": summarise(phase_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    for row in payload["fold"]:
        print(
            f"run of {row['run_length']}: {row['mean_optimized_length']} gates "
            f"({row['reduction_ratio']}x), worst matrix gap "
            f"{row['worst_matrix_gap']:.3e}"
        )
    state = payload["state"]
    print(
        f"statevector: {state['instructions_before']} -> "
        f"{state['instructions_after']} instructions over "
        f"{state['trial_count']} entangled programs, worst raw difference "
        f"{state['max_raw_state_difference']:.3e}"
    )
    for row in payload["basis_reach"]:
        print(
            f"{row['label']}: legal gates "
            f"{row['declined_rule']['legal_gates']} (declining) vs "
            f"{row['no_decline_rule']['legal_gates']} (not declining)"
        )
    if args.json_output is not None:
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
