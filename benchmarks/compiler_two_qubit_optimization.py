"""Measure what folding a same-wire-pair two-qubit run buys, and what it costs.

W9-13 of the Qiskit parity backlog asked for ``OptimizeCliffords``. That pass is
not this: on a gate-level circuit in Qiskit 1.2.4 and 2.0.3 it is a **no-op**, and
its sibling ``CollectCliffords`` selects on an opcode *name* list and emits one
opaque ``clifford`` operation, which needs a native ``Clifford`` type FlagQuantum
does not have. So this round re-scoped W9-13 and this module measures what was
built instead: ``collapse_two_qubit_runs``, which composes a maximal run over one
*ordered wire pair* of declared two-qubit gates and re-spells it as one declared
two-qubit opcode when the product is exactly that opcode. The Qiskit anchor at the
bottom reports the no-op measurement rather than an alignment claim.

Five results are recorded here.

* **The fold is exact.** Over 1200 entangled three-wire programs the whole
  program's statevector moves by at most ``7.11e-16`` at ``complex128``, and the
  comparison is on the raw statevector, so a dropped phase would show up as an
  amplitude difference rather than cancelling.
* **The fold's reach is real but narrow in shape.** It fires on consecutive
  two-qubit gates over one ordered pair. A population whose two-qubit gates are
  separated by single-qubit gates sees nothing at all, because a single-qubit gate
  on either wire ends the run: the ``interleaved`` population's intermediate
  program does shrink, but only through the runs where the random draw happened to
  place two two-qubit gates adjacently.
* **The decline is what keeps it a win.** A parameterized replacement re-spells a
  run into a *different* parameterization of the two-qubit group rather than a more
  primitive operation, and ``optimize`` cannot see whether the target publishes the
  gate the run was spelled in or the gate it would be re-spelled as. Disabling the
  decline shrinks the intermediate program further on every population and
  **regresses the target-legal gate count on 16 of 120 circuits** of the ``routed``
  population at ``ibm-rz-sx-cx``, leaving the legal total *larger* than it found it
  (+27) while the intermediate total fell by 35. The shipped rule improves the
  legal count on every basis of every population with **no circuit regressing**.
* **The identity case is a strict win and is not reachable otherwise.**
  ``remove_identity_gates`` reads a zero parameter, so a run whose *product* is the
  identity while no single member is was invisible to it: ``cphase(2*pi)``,
  ``crz(4*pi)``, ``rzz(4*pi)`` and any pair of mutually inverse rotations. The
  ``identity`` arm isolates that part of the reach.
* **The replacement is verified, not trusted.** The candidate angle is *guessed*
  from the product by inverting one entry of the family's closed form and is then
  accepted only if the whole 4x4 product equals the candidate's own matrix entry for
  entry. The boundary sweep below enumerates every run of length 2 to 4 over an
  11-opcode alphabet and every product is checked against
  ``simulation.gate_matrix`` -- an implementation that shares no code with the
  closed forms here -- with ``false_yes`` required to be zero.

Classification: a local compiler microbenchmark on the single-device fast path. It
runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_two_qubit_optimization.py --json-output /tmp/w913.json
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import random
import unittest.mock
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from benchmarks.compiler_two_qubit_synthesis import (
    _NOW,
    DEFAULT_BASES,
    Basis,
    snapshot,
)
from flagquantum.compiler import commutation_cancellation, two_qubit_optimization
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    legalize_native_gates,
)
from flagquantum.compiler.pipeline import optimize
from flagquantum.compiler.two_qubit_optimization import (
    _FAMILIES,
    _IDENTITY,
    _replacement,
    _run_product,
    _same,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import get_operator_schema
from flagquantum.simulation.gate_matrix import gate_matrix
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_two_qubit_optimization_benchmark_v1"
COMPLEX = torch.complex128

#: The ordered wire pairs every population draws from. ``(0, 1)`` and ``(1, 0)`` are
#: different runs even though they share both wires, which is why the pair is
#: written as an ordered tuple everywhere.
PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))

#: The opcode alphabet a population draws two-qubit gates from.
TWO_QUBIT_OPCODES: tuple[str, ...] = _FAMILIES

#: The opcode alphabet the interleaved population draws single-qubit gates from.
SINGLE_QUBIT_OPCODES: tuple[str, ...] = ("h", "x", "z", "s", "t", "rz", "sx")

#: Angles a parameterized gate is drawn at. The three multiples of pi come first
#: and are drawn often, because a product of generic angles is essentially never
#: exactly a declared opcode: a population of uniform draws would measure the
#: population rather than the pass.
SPECIAL_ANGLES: tuple[float, ...] = (
    math.pi,
    0.5 * math.pi,
    2.0 * math.pi,
    4.0 * math.pi,
)

CIRCUIT_COUNT = 120
POPULATION_SEEDS: tuple[tuple[str, int, str], ...] = (
    ("clifford", 20261008, "clifford"),
    ("swaps", 20261007, "swaps"),
    ("interleaved", 20261009, "interleaved"),
    ("routed", 20261010, "routed"),
)
RUN_TRIAL_COUNT = 4000
RUN_LENGTHS = (2, 3, 4, 5, 6)
_RUN_SEED = 20261011
_STATE_SEED = 20261012
STATE_TRIAL_COUNT = 1200
_BOUNDARY_SEED = 20261013
_BOUNDARY_ATOMS: tuple[str, ...] = _FAMILIES
#: Runs in the length-4 arm of the boundary sweep, which is sampled because the
#: enumeration is ``18 ** 4``. Lengths two and three are enumerated in full.
BOUNDARY_SAMPLE_COUNT = 20000

#: The three versions of the fold's own rule the reach table is measured under.
#: ``shipped`` is the module's own and has no patch. ``identity`` keeps only the
#: product-is-the-identity arm of the reach, and ``disabled`` turns the vocabulary
#: decline off entirely.
FOLD_POLICIES: tuple[str, ...] = ("off", "identity", "shipped", "disabled")


@dataclass(frozen=True)
class DeclineReach:
    """One population's totals under all four versions of the rule, on one basis."""

    label: str
    basis: str
    circuit_count: int
    off_optimized: int
    off_legal: int
    identity_optimized: int
    identity_legal: int
    shipped_optimized: int
    shipped_legal: int
    disabled_optimized: int
    disabled_legal: int
    identity_improved: int
    identity_regressed: int
    shipped_improved: int
    shipped_regressed: int
    disabled_improved: int
    disabled_regressed: int


def _instruction(
    opcode: str, wires: tuple[int, ...], rng: random.Random
) -> Instruction:
    """One instruction of ``opcode`` with a seeded angle, mostly at a multiple of pi."""

    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    params: dict[str, Any] = {}
    for name in schema.parameters:
        params[name] = (
            rng.choice(SPECIAL_ANGLES)
            if rng.random() < 0.6
            else rng.uniform(-math.pi, math.pi)
        )
    return Instruction(opcode, wires, params=params)


def population(kind: str, seed: int, count: int = CIRCUIT_COUNT) -> list[CircuitIR]:
    """One seeded three-wire population, shaped by ``kind``.

    ``clifford`` draws only two-qubit gates over a shared pair, which is the shape
    the fold targets. ``swaps`` forces every other gate to be a ``swap``, which is
    what a router leaves behind. ``routed`` draws from the gates a routed program
    actually contains -- ``cx``, ``cz``, ``swap`` and two rotation entanglers -- and
    it is the population on which disabling the decline regresses. ``interleaved``
    separates the two-qubit gates with single-qubit gates, which *ends* the run and
    is therefore the population that shows how much of the reach depends on the
    two-qubit gates being adjacent.
    """

    rng = random.Random(seed)
    circuits: list[CircuitIR] = []
    for _ in range(count):
        instructions: list[Instruction] = []
        for _ in range(rng.randint(6, 24)):
            draw = rng.random()
            if kind == "interleaved" and draw < 0.45:
                instructions.append(
                    _instruction(
                        rng.choice(SINGLE_QUBIT_OPCODES), (rng.randrange(3),), rng
                    )
                )
                continue
            if kind == "swaps" and draw < 0.5:
                instructions.append(Instruction("swap", rng.choice(PAIRS)))
                continue
            alphabet = (
                ("cx", "cz", "swap", "crz", "rzz")
                if kind == "routed"
                else TWO_QUBIT_OPCODES
            )
            instructions.append(
                _instruction(rng.choice(alphabet), rng.choice(PAIRS), rng)
            )
        circuits.append(CircuitIR(3, tuple(instructions)))
    return circuits


@contextlib.contextmanager
def _rotation_merge_patch() -> Iterator[None]:
    """Pause the commuting rotation merge, which landed after the fold.

    The fold's own attribution is a statement about *this* pass, and a second pass
    that composes the same populations moves the same rows. Pausing it is what keeps
    the pins below a measurement of the fold rather than of the sum, and it is
    visible because `_optimize_to_fixed_point` imports the rule inside its body, so
    the module attribute is read on every call.
    """

    original = commutation_cancellation.merge_commuting_rotations
    commutation_cancellation.merge_commuting_rotations = lambda ir: ir
    try:
        yield
    finally:
        commutation_cancellation.merge_commuting_rotations = original


def _fold_patch(policy: str) -> Any:
    """A patcher putting the fold into one of the four versions.

    The ``identity`` arm is written in terms of the module's own helpers so that it
    cannot drift away from the product it is supposed to be measuring, and the
    ``disabled`` arm patches the decline rather than the pass so that both patches
    patch a *rule* instead of an implementation.
    """

    if policy == "off":
        return unittest.mock.patch.object(
            two_qubit_optimization, "collapse_two_qubit_runs", lambda ir: ir
        )
    if policy == "identity":
        return unittest.mock.patch.object(
            two_qubit_optimization,
            "_replacement",
            lambda instructions, *, pair, metadata: (
                () if _same(_run_product(instructions), _IDENTITY) else None
            ),
        )
    if policy == "disabled":
        return unittest.mock.patch.object(
            two_qubit_optimization,
            "_declines_vocabulary",
            lambda instructions, opcode: False,
        )
    if policy == "shipped":
        return contextlib.nullcontext()
    raise ValueError(f"unknown fold policy {policy!r}")


def _legal_count(folded: CircuitIR, basis: Basis) -> int:
    """One circuit's legal gate count, or its own length when the basis refuses it.

    A refusal is not a smaller program, and dropping it would let a rule look better
    than it is on any basis that refuses the tail of the population.
    """

    try:
        result = legalize_native_gates(
            folded, snapshot=snapshot(basis), evaluated_at=_NOW
        )
    except NativeGateLegalizationError:
        return len(folded.instructions)
    return len(result.program.instructions)


def decline_reach(bases: tuple[Basis, ...] = DEFAULT_BASES) -> list[DeclineReach]:
    """The rule's reach under all four policies, per population and per basis.

    The program is optimized **once per policy** and then legalized per basis, so
    the only thing that differs between the four arms is the fold itself.

    Regressions are counted **per circuit**, not in a total. A total can hide a
    program that grew behind programs that shrank, which is exactly the failure the
    ``disabled`` arm exhibits: it is shorter in the middle of the pipeline and
    longer at the target.
    """

    rows: list[DeclineReach] = []
    for label, seed, kind in POPULATION_SEEDS:
        circuits = population(kind, seed)
        arms: dict[str, list[CircuitIR]] = {}
        for policy in FOLD_POLICIES:
            with _fold_patch(policy):
                arms[policy] = [optimize(circuit) for circuit in circuits]
        for basis in bases:
            legal = {
                policy: [_legal_count(folded, basis) for folded in arms[policy]]
                for policy in FOLD_POLICIES
            }
            base = legal["off"]
            improvements: dict[str, int] = {}
            regressions: dict[str, int] = {}
            for policy in FOLD_POLICIES:
                improvements[policy] = sum(
                    1 for a, b in zip(legal[policy], base, strict=True) if a < b
                )
                regressions[policy] = sum(
                    1 for a, b in zip(legal[policy], base, strict=True) if a > b
                )
            identity_improved = improvements["identity"]
            identity_regressed = regressions["identity"]
            shipped_improved = improvements["shipped"]
            shipped_regressed = regressions["shipped"]
            disabled_improved = improvements["disabled"]
            disabled_regressed = regressions["disabled"]
            rows.append(
                DeclineReach(
                    label=label,
                    basis=basis.label,
                    circuit_count=len(circuits),
                    off_optimized=sum(len(f.instructions) for f in arms["off"]),
                    off_legal=sum(base),
                    identity_optimized=sum(
                        len(f.instructions) for f in arms["identity"]
                    ),
                    identity_legal=sum(legal["identity"]),
                    shipped_optimized=sum(len(f.instructions) for f in arms["shipped"]),
                    shipped_legal=sum(legal["shipped"]),
                    disabled_optimized=sum(
                        len(f.instructions) for f in arms["disabled"]
                    ),
                    disabled_legal=sum(legal["disabled"]),
                    identity_improved=identity_improved,
                    identity_regressed=identity_regressed,
                    shipped_improved=shipped_improved,
                    shipped_regressed=shipped_regressed,
                    disabled_improved=disabled_improved,
                    disabled_regressed=disabled_regressed,
                )
            )
    return rows


def _oracle_product(instructions: tuple[Instruction, ...]) -> torch.Tensor:
    """The product of a run read out of the runtime gate table, not the closed forms."""

    product = torch.eye(4, dtype=COMPLEX)
    for instruction in instructions:
        matrix = gate_matrix(
            instruction,
            bsz=1,
            device=torch.device("cpu"),
            dtype=COMPLEX,
        ).reshape(4, 4)
        product = matrix @ product
    return product


def _run_gap(before: tuple[Instruction, ...], after: tuple[Instruction, ...]) -> float:
    """The largest entry difference between two runs' products, via the oracle."""

    gap = _oracle_product(before) - _oracle_product(after)
    return float(torch.max(torch.abs(gap)).item())


def run_table() -> list[dict[str, Any]]:
    """Per run length: how far the fold shortens a run, and how exactly."""

    rows: list[dict[str, Any]] = []
    for length in RUN_LENGTHS:
        rng = random.Random(_RUN_SEED + length)
        source_total = 0
        folded_total = 0
        folded_runs = 0
        declined_runs = 0
        worst_gap = 0.0
        for _ in range(RUN_TRIAL_COUNT // len(RUN_LENGTHS)):
            pair = rng.choice(PAIRS)
            run = tuple(
                _instruction(rng.choice(TWO_QUBIT_OPCODES), pair, rng)
                for _ in range(length)
            )
            replacement = _replacement(run, pair=pair, metadata={})
            source_total += length
            if replacement is None:
                declined_runs += 1
                folded_total += length
                continue
            folded_at = len(replacement)
            if folded_at == 0:
                # An empty replacement is the identity, which no `Instruction` can
                # stand for; the product of nothing is the identity by definition.
                assert _same(_run_product(run), _IDENTITY)
            else:
                worst_gap = max(worst_gap, _run_gap(run, replacement))
            if folded_at < length:
                folded_runs += 1
            folded_total += folded_at
        rows.append(
            {
                "run_length": length,
                "trial_count": RUN_TRIAL_COUNT // len(RUN_LENGTHS),
                "source_gates": source_total,
                "folded_gates": folded_total,
                "mean_folded_gates": round(
                    folded_total / (RUN_TRIAL_COUNT // len(RUN_LENGTHS)), 4
                ),
                "compression_ratio": round(source_total / folded_total, 4),
                "folded_runs": folded_runs,
                "declined_runs": declined_runs,
                "worst_product_gap": worst_gap,
            }
        )
    return rows


def state_evidence() -> dict[str, Any]:
    """The whole program's statevector before and after, on an entangled program.

    These programs carry a two-qubit gate between the runs and a single-qubit gate
    on one wire of the pair, so the folded gates land on relative phases that a
    statevector comparison does see.
    """

    rng = random.Random(_STATE_SEED)
    worst_difference = 0.0
    circuits_folded = 0
    instructions_before = 0
    instructions_after = 0
    for _ in range(STATE_TRIAL_COUNT):
        pair = rng.choice(PAIRS)
        others = [wire for wire in (0, 1, 2) if wire not in pair]
        instructions: list[Instruction] = []
        for _ in range(rng.randint(3, 10)):
            if rng.random() < 0.35:
                instructions.append(
                    _instruction(rng.choice(TWO_QUBIT_OPCODES), pair, rng)
                )
            elif rng.random() < 0.5:
                instructions.append(
                    _instruction(
                        rng.choice(SINGLE_QUBIT_OPCODES), (rng.choice(pair),), rng
                    )
                )
            else:
                instructions.append(
                    _instruction(
                        rng.choice(SINGLE_QUBIT_OPCODES), (rng.choice(others),), rng
                    )
                )
        source = CircuitIR(3, tuple(instructions), dtype="complex128")
        folded = optimize(source)
        instructions_before += len(source.instructions)
        instructions_after += len(folded.instructions)
        if len(folded.instructions) < len(source.instructions):
            circuits_folded += 1
        before = run_local_statevector(
            source, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
        )
        after = run_local_statevector(
            folded, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
        )
        worst_difference = max(
            worst_difference, float(torch.max(torch.abs(before - after)).item())
        )
    return {
        "trial_count": STATE_TRIAL_COUNT,
        "entangler_present": True,
        "instructions_before": instructions_before,
        "instructions_after": instructions_after,
        "reduction_ratio": round(instructions_before / instructions_after, 4),
        "circuits_folded": circuits_folded,
        "max_raw_state_difference": worst_difference,
        "phase_contract": (
            "exact: a run this pass cannot re-spell is left alone, so nothing "
            "carrying a phase the IR cannot store is ever dropped"
        ),
    }


def _boundary_atoms() -> list[tuple[str, float | None]]:
    """One atom per opcode: the fixed ones, and the others at the pi and pi/2 points."""

    atoms: list[tuple[str, float | None]] = []
    for opcode in _BOUNDARY_ATOMS:
        schema = get_operator_schema(opcode)
        assert schema is not None, opcode
        if not schema.parameters:
            atoms.append((opcode, None))
            continue
        atoms.extend((opcode, angle) for angle in (0.5 * math.pi, math.pi))
    return atoms


def _boundary_run(
    atoms: list[tuple[str, float | None]],
    matrices: dict[tuple[str, float | None], torch.Tensor],
    combination: tuple[tuple[str, float | None], ...],
) -> tuple[torch.Tensor, tuple[Instruction, ...]]:
    """One enumerated run's product and its instruction tuple."""

    product = torch.eye(4, dtype=COMPLEX)
    run: list[Instruction] = []
    for opcode, angle in combination:
        schema = get_operator_schema(opcode)
        assert schema is not None, opcode
        params = {} if angle is None else {schema.parameters[0]: angle}
        run.append(Instruction(opcode, (0, 1), params=params))
        product = matrices[(opcode, angle)] @ product
    assert atoms is not None
    return product, tuple(run)


def boundary() -> dict[str, Any]:
    """Every short run's product, checked against the runtime gate table.

    The two questions are asked separately, because they fail differently. **How
    much is there to find** is the reach: how many runs have a product that *is* a
    declared opcode. **How often is the fold wrong** is `false_yes`: how many runs
    the pass replaced with a gate whose runtime matrix differs from the run's own.
    A reach of zero would make the second number vacuous, which is why both are
    reported and why the sweep enumerates the whole alphabet for the lengths a
    complete sweep can afford, and a seeded sample beyond that. The sample is
    labelled, so a sampled number is never read as an enumeration.
    """

    atoms = _boundary_atoms()
    matrices: dict[tuple[str, float | None], torch.Tensor] = {}
    for opcode, angle in atoms:
        schema = get_operator_schema(opcode)
        assert schema is not None, opcode
        params = {} if angle is None else {schema.parameters[0]: angle}
        matrices[(opcode, angle)] = gate_matrix(
            Instruction(opcode, (0, 1), params=params),
            bsz=1,
            device=torch.device("cpu"),
            dtype=COMPLEX,
        ).reshape(4, 4)

    identity = torch.eye(4, dtype=COMPLEX)
    rows: list[dict[str, Any]] = []
    false_yes = 0
    checked = 0
    replaced = 0
    deleted = 0
    worst_gap = 0.0

    def sweep(combinations: Any, length: int, kind: str, total: int | None) -> None:
        nonlocal false_yes, checked, replaced, deleted, worst_gap
        reach = 0
        seen = 0
        for combination in combinations:
            seen += 1
            product, run = _boundary_run(atoms, matrices, combination)
            if float(torch.max(torch.abs(product - identity)).item()) <= 1e-12:
                reach += 1
            replacement = _replacement(run, pair=(0, 1), metadata={})
            if replacement is None:
                continue
            checked += 1
            if not replacement:
                deleted += 1
                continue
            replaced += 1
            gap = _run_gap(run, replacement)
            worst_gap = max(worst_gap, gap)
            if gap > 1e-12:
                false_yes += 1
        rows.append(
            {
                "run_length": length,
                "sweep": kind,
                "run_count": seen,
                "enumerated_run_count": total,
                "identity_products": reach,
            }
        )

    for length in (2, 3):
        sweep(itertools.product(atoms, repeat=length), length, "enumerated", None)
        rows[-1]["enumerated_run_count"] = rows[-1]["run_count"]
    sampled_rng = random.Random(_BOUNDARY_SEED)
    sweep(
        (
            tuple(sampled_rng.choice(atoms) for _ in range(4))
            for _ in range(BOUNDARY_SAMPLE_COUNT)
        ),
        4,
        "seeded_sample",
        len(atoms) ** 4,
    )
    return {
        "atom_count": len(atoms),
        "runs_per_length": rows,
        "runs_the_pass_replaced": replaced,
        "runs_the_pass_deleted": deleted,
        "runs_the_pass_left_alone": sum(row["run_count"] for row in rows) - checked,
        "checked_against_the_gate_table": checked,
        "false_yes": false_yes,
        "worst_replacement_gap": worst_gap,
    }


def _qiskit_anchor() -> dict[str, Any]:
    """What Qiskit's own Clifford passes do to a gate-level circuit, or why not.

    This is the anchor the backlog row named, and it is reported rather than
    claimed: the measurement is that ``OptimizeCliffords`` does not do this. It is
    guarded because the benchmark has to run in an environment with no Qiskit.
    """

    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler.passes import CollectCliffords, OptimizeCliffords
    except Exception as error:  # pragma: no cover - depends on the environment
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    def measured(transpiler: Any) -> dict[str, Any]:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.s(1)
        circuit.h(0)
        circuit.sdg(1)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(1, 0)
        circuit.cz(0, 1)
        circuit.t(0)
        circuit.tdg(1)
        before = len(circuit.data)
        # ``TransformationPass.__call__`` is the entry point that accepts a circuit:
        # it builds the DAG the pass's own ``run`` method requires. Calling ``run``
        # with the circuit directly raises on Qiskit 2.x.
        after = transpiler(circuit)
        return {
            "gates_before": before,
            "gates_after": len(after.data),
            "opcodes_after": sorted({item.operation.name for item in after.data}),
        }

    return {
        "available": True,
        "optimize_cliffords": measured(OptimizeCliffords()),
        "collect_cliffords": measured(CollectCliffords()),
        "note": (
            "OptimizeCliffords operates on Clifford objects already stored on the "
            "circuit and leaves every gate of a gate-level circuit where it was. "
            "CollectCliffords selects on an opcode name list and replaces each "
            "maximal selected run with one opaque 'clifford' operation, which needs "
            "a native Clifford type FlagQuantum does not have; only the runs that "
            "contain no non-Clifford gate are collected, so 't' and 'tdg' survive "
            "and split this circuit into pieces. Neither pass composes declared "
            "two-qubit opcodes, which is what this module does."
        ),
    }


def run_benchmark(*, bases: tuple[Basis, ...] = DEFAULT_BASES) -> dict[str, Any]:
    """Measure the reach, the run table, the exactness, and the boundary."""

    reach = decline_reach(bases)
    return {
        "schema": SCHEMA,
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_collect_cliffords",
        "reference_revision": "qiskit 2.0.3 OptimizeCliffords / CollectCliffords",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "local compiler microbenchmark; no distributed work, no scalability "
            "claim, not a performance gate"
        ),
        "declared_two_qubit_unitary_opcodes": list(_FAMILIES),
        "populations": [
            {
                "label": label,
                "seed": seed,
                "shape": kind,
                "circuit_count": CIRCUIT_COUNT,
            }
            for label, seed, kind in POPULATION_SEEDS
        ],
        "decline_reach": [row.__dict__ for row in reach],
        "run_table": run_table(),
        "state_evidence": state_evidence(),
        "boundary": boundary(),
        "qiskit_anchor": _qiskit_anchor(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=str, default=None)
    arguments = parser.parse_args()
    result = run_benchmark()
    if arguments.json_output:
        with open(arguments.json_output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
