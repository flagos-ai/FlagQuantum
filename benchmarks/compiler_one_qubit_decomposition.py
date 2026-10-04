"""Measure whether a single-qubit run can be re-spelled in a declared target basis.

`Optimize1qGatesDecomposition` is Qiskit's pass for the shape this repository does
not have: it collects the single-qubit runs and re-synthesizes each one **directly
into the target's own basis**. The obstacle to porting it is not the collection --
`one_qubit_optimization.collapse_one_qubit_runs` already collects a maximal
same-wire run -- and it is not a missing algorithm either. It is that a one-qubit
decomposition into a phase-free basis exists only up to a global phase, Qiskit
records that phase in `DAGCircuit.global_phase`, and FlagQuantum IR has no field
to record it in.

This module is the falsifiability instrument for that statement. It measures four
things and pins none of them to a particular implementation:

* ``determinants`` -- the determinant argument every declared arity-1 gate
  carries, and whether it moves with the gate's parameter. A word's determinant is
  the product of its factors' determinants, so a basis whose arity-1 gates have
  parameter-independent determinants can only spell operators whose determinant
  lies in the subgroup those determinants generate. That is an obstruction no
  algorithm can remove.
* ``reachability`` -- which declared arity-1 opcodes fall outside that subgroup for
  each target basis, at a probed angle. An opcode listed there is one whose
  presence in a run puts the run beyond every word over the basis, at any length,
  up to a global phase.
* ``declared_basis_reach`` -- over seeded mixed runs, two columns side by side: the
  words that agree with the run entry for entry, and the words that agree only up
  to a global phase. Where the basis is phase-free -- its z-rotation carries a
  fixed determinant -- the first column is *bounded* by the subgroup above and the
  gap between the columns is exactly the phase the IR has nowhere to put. Where
  the z-rotation's determinant is its parameter, the determinant blocks nothing
  and the first column instead measures how far this repository's own five-gate
  template reaches; a solver could move that column, and the row says so.

  The bound is reported separately from the column, as ``beyond_the_subgroup``,
  because the column is not itself a mathematical quantity: which runs this
  repository's template reproduces exactly is decided by exact-equality branch
  tests on `cmath` results, so it moves with the platform's C library, while the
  bound is a subgroup membership test whose two cases are 45 degrees apart and
  does not. A reader needs the bound to know what the obstruction forbids and the
  column to know what the template achieves; conflating the two is how a rounding
  coincidence becomes a claim.
* ``legalization_phase`` -- the same obstruction where it is already shipped.
  `native_gate_legalization` reaches a target basis through `one_qubit_synthesis`,
  which documents its result as equal to its source up to one global phase, so a
  legalized program's statevector is multiplied by a phase the IR cannot record.

The obstruction is a property of the *basis*, not of a run: a target that
publishes a z-rotation whose determinant argument is its parameter (`phase`, `u1`)
can spell every determinant, and one that publishes only `rz`/`sx` or `rz`/`rx`
cannot spell any but those in `{0, pi/2, pi, 3pi/2}`. This module reports both
cases rather than only the negative one, and each measured pair records which of
the two it is, so a row cannot be read as a bound where it is only a measurement.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import random
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
from flagquantum.compiler.native_gate_legalization import legalize_native_gates
from flagquantum.compiler.one_qubit_optimization import (
    _instruction_matrix,
    _run_product,
    _split_anchor,
    _u3_angles,
)
from flagquantum.compiler.one_qubit_synthesis import _emit_leaves, _leaves
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_one_qubit_decomposition_benchmark_v1"
COMPLEX = torch.complex128

#: Every declared arity-1 unitary opcode, which is the population a run can be
#: drawn from: the question is whether *any* run can be re-spelled, so an opcode
#: missing here would be a run the measurement cannot see.
SINGLE_QUBIT_OPCODES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)

_RUN_SEED = 20261020
#: Mixed runs per pair, over every declared arity-1 opcode, so a run's determinant
#: argument is generically outside any proper subgroup. Four thousand is the
#: population `compiler_one_qubit_optimization` uses for its own reach tables.
RUN_TRIAL_COUNT = 4000
_RUN_LENGTHS = (2, 8)

_LEGALIZATION_SEED = 20261021
#: Entangled programs per basis. The programs carry a two-wire gate between the
#: runs so the phase a run loses does not cancel: a one-wire program cannot show
#: it, because a global phase cancels in the single non-zero amplitude.
LEGALIZATION_CIRCUIT_COUNT = 80

#: The angle a parameterized gate is given when its determinant is probed, and the
#: second angle used to decide whether that determinant moves with the parameter.
#: Two values suffice: a determinant that is a function of the parameter differs
#: at them, and one that is not does not.
_DETERMINANT_ANGLES = (1.1, 2.3)

#: A word is called exact only when it agrees with the run entry for entry to this
#: tolerance. The pass this measures folds at `1.0e-12`; the tolerance is looser
#: here on purpose, because a false *yes* -- calling a word exact that is not --
#: is the error that would matter, and `1e-9` cannot produce one from float noise
#: three orders smaller.
_TOL = 1.0e-9

#: The tolerance, in degrees, on the subgroup membership test that produces
#: `beyond_the_subgroup`. Two subgroup values of a declared basis are 45 degrees
#: apart at the closest and a run's own determinant argument is computed in
#: `float64`, so this is enormous relative to the arithmetic and tiny relative to
#: the gap: the classification cannot be moved by a different C library.
_SUBGROUP_TOL = 1.0e-6


def _runtime_matrix(opcode: str, angle: float) -> Any:
    """The runtime matrix of ``opcode`` with every parameter set to ``angle``."""

    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    return _instruction_matrix(
        Instruction(opcode, (0,), params=dict.fromkeys(schema.parameters, angle))
    )


def _determinant_argument(opcode: str, angle: float) -> float:
    """The argument of ``opcode``'s determinant, in degrees."""

    matrix = _runtime_matrix(opcode, angle)
    return math.degrees(
        cmath.phase(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    )


def _probe_arguments(opcode: str) -> list[float]:
    return [_determinant_argument(opcode, angle) for angle in _DETERMINANT_ANGLES]


def _moves_with_parameter(opcode: str) -> bool:
    first, second = _probe_arguments(opcode)
    return abs(first - second) > _TOL


def _single_qubit_instruction(opcode: str, rng: random.Random) -> Instruction:
    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    return Instruction(
        opcode,
        (0,),
        params={name: rng.uniform(-3.0, 3.0) for name in schema.parameters},
    )


def _mixed_run(rng: random.Random) -> tuple[Instruction, ...]:
    return tuple(
        _single_qubit_instruction(rng.choice(SINGLE_QUBIT_OPCODES), rng)
        for _ in range(rng.randint(*_RUN_LENGTHS))
    )


def _circle_distance(first: float, second: float) -> float:
    return min(abs(first - second), 360.0 - abs(first - second))


def _generated_subgroup(arguments: list[float]) -> list[float]:
    """The subgroup of the circle the given determinant arguments generate.

    Every declared vocabulary contributes an integer multiple of 45 degrees, so
    the subgroup is the multiples of their greatest common divisor. The arithmetic
    runs on eighths of a turn to keep it exact; ``_TOL`` is far under half an
    eighth, so float noise rounds to the same integer.
    """

    eighths = sorted({round(value / 45.0) for value in arguments})
    step = 0
    for value in eighths:
        step = math.gcd(step, abs(value))
    if step == 0:
        return [0.0]
    return [round(index * step * 45.0, 6) for index in range(360 // (step * 45))]


def determinants(basis: Basis) -> dict[str, Any]:
    """The declared arity-1 gates of ``basis`` and the determinant each carries."""

    rows: list[dict[str, Any]] = []
    for gate in basis.gates:
        schema = get_operator_schema(gate["name"])
        if schema is None or not schema.unitary or schema.arity != 1:
            continue
        rows.append(
            {
                "opcode": gate["name"],
                "parameters": list(schema.parameters),
                "determinant_argument_degrees": [
                    round(value, 6) for value in _probe_arguments(gate["name"])
                ],
                "moves_with_parameter": _moves_with_parameter(gate["name"]),
            }
        )
    return {
        "label": basis.label,
        "arity_one_gates": rows,
        "determinant_is_free": any(row["moves_with_parameter"] for row in rows),
    }


def reachability(basis: Basis) -> dict[str, Any]:
    """The determinants a word over ``basis`` can carry, and what they block."""

    table = determinants(basis)
    if table["determinant_is_free"]:
        return {
            **table,
            "reachable_arguments_degrees": None,
            "blocked_opcodes": [],
            "argument": (
                "an arity-1 gate whose determinant argument is its parameter makes "
                "the generated subgroup dense, so no operator is blocked by its "
                "determinant"
            ),
        }
    declared = {row["opcode"] for row in table["arity_one_gates"]}
    reachable = _generated_subgroup(
        [row["determinant_argument_degrees"][0] for row in table["arity_one_gates"]]
    )
    blocked = [
        opcode
        for opcode in SINGLE_QUBIT_OPCODES
        if opcode not in declared
        and all(
            _circle_distance(_probe_arguments(opcode)[0], value) > _TOL
            for value in reachable
        )
    ]
    return {
        **table,
        "reachable_arguments_degrees": reachable,
        "blocked_opcodes": blocked,
        "argument": (
            f"a word over this basis has a determinant argument in {reachable} "
            f"degrees, so a run carrying any of the {len(blocked)} opcodes listed "
            "outside it has no word over this basis at any length, up to a global "
            "phase"
        ),
    }


def _candidate_words(
    matrix: Any, *, z_rotation: str, pulse_opcode: str
) -> list[tuple[Instruction, ...]]:
    """The words this repository can build in ``z_rotation``/``pulse_opcode``.

    The construction is `one_qubit_optimization._emit`'s, one arity down: split
    the trailing z-rotation off with `_split_anchor`, read the `U3` triple of the
    remainder, write the leaves `one_qubit_synthesis._leaves` tabulates, and put
    the anchor back. The only free knob is the sign: shifting the triple's `lam` by
    one turn multiplies the word by `-1` without adding a gate to a word whose
    first leaf is already a z-rotation, so both signs are offered. This is the
    whole of what the repository knows how to build here; a port would substitute
    a real solver, and a solver cannot move the columns below, because a word over
    a phase-free basis cannot carry a determinant the basis does not have.
    """

    anchor, remainder = _split_anchor(matrix)
    theta, phi, lam = _u3_angles(remainder)
    words = []
    for shift in (0.0, 2.0 * math.pi, -2.0 * math.pi):
        word = _emit_leaves(
            _leaves(
                theta,
                phi,
                lam + shift,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
            ),
            wires=(0,),
            metadata={},
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
        )
        if abs(anchor) > 1.0e-15:
            word = word + (Instruction(z_rotation, (0,), params={"theta": anchor}),)
        words.append(word)
    return words


def _quotient(left: Any, right: Any) -> complex | None:
    """The scalar ``c`` with ``left == c * right``, or None when there is none.

    ``left`` and ``right`` are the same shape: two 2x2 products when the question
    is whether a word reproduces a run, or two statevectors when it is whether a
    legalized program computes the same thing. An entry of ``right`` that is zero
    while ``left`` is not rules the scalar out; every other entry must agree on
    the same quotient, which is exactly the claim that the two differ by one
    global phase and by nothing else.
    """

    flat_left = _flatten(left)
    flat_right = _flatten(right)
    ratios: list[complex] = []
    for numerator, denominator in zip(flat_left, flat_right, strict=True):
        if abs(denominator) > _TOL:
            ratios.append(numerator / denominator)
        elif abs(numerator) > _TOL:
            return None
    if not ratios:
        return 0.0 + 0.0j
    if any(abs(value - ratios[0]) > _TOL for value in ratios):
        return None
    return ratios[0]


def _flatten(value: Any) -> list[complex]:
    """A 2x2 product, a statevector, or a tensor as one list of amplitudes."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        return [entry for row in value for entry in row]
    if value and isinstance(value[0], tuple):
        return [entry for row in value for entry in row]
    return list(value)


def _entry_gap(left: Any, right: Any) -> float:
    return float(
        max(
            abs(left[row][column] - right[row][column])
            for row in range(2)
            for column in range(2)
        )
    )


def _determinant(matrix: Any) -> complex:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def declared_basis_reach(pair: tuple[str, str]) -> dict[str, Any]:
    """Both columns of the re-spelling question for one declared pair.

    The entry-for-entry column is a count of the runs this repository's own
    five-gate template happens to hit exactly, and which runs those are is a
    rounding question (see `_candidate_words` and the module docstring). What is
    not a rounding question is the *bound*: if the pair's z-rotation has a fixed
    determinant, then the argument of a run's own product determinant either lies
    in the subgroup the pair can generate -- and then a word may exist -- or it
    does not, and then no word over the pair reproduces the run at any length. The
    two cases are at least 45 degrees apart, so a tolerance of `1.0e-6` degrees
    separates them on any platform, and `beyond_the_subgroup` is an upper bound a
    different C library cannot move.
    """

    z_rotation, pulse_opcode = pair
    obstruction = not _moves_with_parameter(z_rotation)
    reachable = (
        _generated_subgroup(
            [
                _determinant_argument(z_rotation, _DETERMINANT_ANGLES[0]),
                _determinant_argument(pulse_opcode, _DETERMINANT_ANGLES[0]),
            ]
        )
        if obstruction
        else None
    )
    rng = random.Random(_RUN_SEED)
    source_gates = 0
    found = 0
    shorter = 0
    saved = 0
    exact = 0
    exact_shorter = 0
    exact_saved = 0
    beyond_the_subgroup = 0
    worst_phase_degrees = 0.0
    for _ in range(RUN_TRIAL_COUNT):
        run = _mixed_run(rng)
        source_gates += len(run)
        product = _run_product(list(run))
        assert product is not None
        if reachable is not None:
            argument = math.degrees(cmath.phase(_determinant(product)))
            if all(
                _circle_distance(argument, value) > _SUBGROUP_TOL for value in reachable
            ):
                beyond_the_subgroup += 1
        matched: tuple[tuple[Instruction, ...], Any, complex] | None = None
        for word in _candidate_words(
            product, z_rotation=z_rotation, pulse_opcode=pulse_opcode
        ):
            spelled = _run_product(list(word))
            if spelled is None:
                continue
            quotient = _quotient(product, spelled)
            if quotient is None:
                continue
            if matched is None:
                matched = (word, spelled, quotient)
            if _entry_gap(product, spelled) <= _TOL:
                # The sign is free, so an exact word beats a phase-blind one and
                # is taken when the construction happens to reach it. The two
                # columns below are therefore both the best of the three signs.
                matched = (word, spelled, quotient)
                break
        if matched is None:
            continue
        word, spelled, quotient = matched
        found += 1
        worst_phase_degrees = max(
            worst_phase_degrees, abs(math.degrees(cmath.phase(quotient)))
        )
        if len(word) < len(run):
            shorter += 1
            saved += len(run) - len(word)
        if _entry_gap(product, spelled) <= _TOL:
            exact += 1
            if len(word) < len(run):
                exact_shorter += 1
                exact_saved += len(run) - len(word)
    obstruction = not _moves_with_parameter(z_rotation)
    return {
        "z_rotation": z_rotation,
        "pulse_opcode": pulse_opcode,
        "determinant_obstruction_applies": obstruction,
        "note": (
            "the z-rotation's determinant is fixed, so the entry-for-entry column "
            "is bounded by the subgroup a word over the basis can carry"
            if obstruction
            else "the z-rotation's determinant is its parameter, so the "
            "entry-for-entry column measures this repository's template and a "
            "solver could exceed it"
        ),
        "trial_count": RUN_TRIAL_COUNT,
        "run_lengths": list(_RUN_LENGTHS),
        "source_gates": source_gates,
        "up_to_a_global_phase": {
            "words_found": found,
            "words_shorter": shorter,
            "gates_saved": saved,
            "worst_leftover_phase_degrees": round(worst_phase_degrees, 6),
        },
        "entry_for_entry": {
            "words_found": exact,
            "words_shorter": exact_shorter,
            "gates_saved": exact_saved,
        },
        # `None` where the pair's z-rotation has a free determinant: the subgroup
        # is then dense and nothing is out of reach, so the entry-for-entry column
        # measures the template rather than an obstruction. Where it is a number it
        # is a bound on that column which no platform can move.
        "beyond_the_subgroup": (
            {
                "run_count": beyond_the_subgroup,
                "argument": (
                    "the argument of these runs' own product determinant lies "
                    f"outside {reachable} degrees, so no word over this pair "
                    "reproduces them at any length, up to a global phase"
                ),
            }
            if reachable is not None
            else None
        ),
    }


def _entangler(basis: Basis, left: int, right: int) -> Instruction:
    """A two-wire gate of ``basis``, so the program is not a one-wire program.

    The basis's own entangler is used rather than a fixed `cx`: a target that does
    not publish `cx` refuses a `cx`, and a refusal here would be a gap in the
    measurement rather than a property of the basis.
    """

    opcode = basis.entangler
    assert opcode is not None, basis.label
    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    return Instruction(
        opcode,
        (left, right),
        params=dict.fromkeys(schema.parameters, 0.5),
    )


def _statevector(program: CircuitIR) -> Any:
    return (
        run_local_statevector(
            program, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
        )
        .reshape(-1)
        .to(COMPLEX)
    )


def _phase_of_legalization(
    source: CircuitIR, basis: Basis
) -> tuple[complex | None, float]:
    """The phase legalization *introduced*, and the largest amplitude gap.

    The quotient is `after / before`, so the returned scalar is the factor the
    legalized program multiplies the source by, and not its reciprocal. The
    direction is part of the measurement: both quotients are proportional and both
    are non-trivial, so a report that carried the other one would look just as
    plausible while stating the opposite rotation. The sign is pinned in the
    contract test for that reason.
    """

    legalized = legalize_native_gates(
        source, snapshot=snapshot(basis), evaluated_at=_NOW
    )
    before = _statevector(source)
    after = _statevector(legalized.program)
    return _quotient(after, before), float((before - after).abs().max())


def legalization_phase(basis: Basis) -> dict[str, Any]:
    """The phase `native_gate_legalization` drops, measured on the statevector."""

    rng = random.Random(_LEGALIZATION_SEED)
    circuits = 0
    refused = 0
    unchanged = 0
    phased = 0
    beyond_a_sign = 0
    phases: list[float] = []
    worst_gap = 0.0
    for _ in range(LEGALIZATION_CIRCUIT_COUNT):
        instructions: list[Instruction] = []
        for _ in range(rng.randint(4, 14)):
            if rng.random() < 0.25:
                left, right = rng.sample(range(2), 2)
                instructions.append(_entangler(basis, left, right))
                continue
            drawn = _single_qubit_instruction(rng.choice(SINGLE_QUBIT_OPCODES), rng)
            instructions.append(
                Instruction(drawn.name, (rng.randrange(2),), params=dict(drawn.params))
            )
        source = CircuitIR(n_wires=2, instructions=tuple(instructions))
        try:
            quotient, gap = _phase_of_legalization(source, basis)
        except Exception:
            refused += 1
            continue
        circuits += 1
        if quotient is None:
            worst_gap = max(worst_gap, gap)
            continue
        degrees = math.degrees(cmath.phase(quotient)) % 360.0
        if min(degrees, 360.0 - degrees) < 1.0e-6:
            # A phase a hair under a full turn is float noise on a phase of zero,
            # not a rotation: it must be folded to zero before it is classified, or
            # the reported range reads `360.0` on a program that lost nothing.
            unchanged += 1
            continue
        phased += 1
        phases.append(degrees)
        if abs(degrees - 180.0) > 1.0e-6:
            beyond_a_sign += 1
    return {
        "label": basis.label,
        "circuit_count": circuits,
        "circuit_count_refused": refused,
        "statevector_unchanged": unchanged,
        "statevector_phased": phased,
        "phase_is_not_a_sign": beyond_a_sign,
        "phase_degrees_min": round(min(phases), 6) if phases else 0.0,
        "phase_degrees_max": round(max(phases), 6) if phases else 0.0,
        "worst_non_proportional_gap": round(worst_gap, 12),
    }


def phase_witness(basis: Basis) -> dict[str, Any]:
    """The smallest program whose legalized statevector is multiplied by a phase.

    `h` followed by `cx` is two instructions, so the phase cannot be attributed to
    a fold of many gates: it is the single `h` the target has no `h` for.

    `global_phase_degrees` is the signed phase legalization introduced, in
    `(-180, 180]`, so a program whose legalized state is `exp(-i*pi/4)` times the
    source reads `-45.0` and not `315.0`. It is the sign the contract test pins.
    """

    source = CircuitIR(
        n_wires=2,
        instructions=(Instruction("h", (0,)), _entangler(basis, 0, 1)),
    )
    quotient, gap = _phase_of_legalization(source, basis)
    legalized = legalize_native_gates(
        source, snapshot=snapshot(basis), evaluated_at=_NOW
    )
    return {
        "label": basis.label,
        "source_opcodes": [item.name for item in source.instructions],
        "legalized_opcodes": [item.name for item in legalized.program.instructions],
        "amplitude_ratios_are_constant": quotient is not None,
        "global_phase_degrees": (
            round(math.degrees(cmath.phase(quotient)), 6)
            if quotient is not None
            else None
        ),
        "max_amplitude_gap": round(gap, 12),
    }


def run_benchmark(*, bases: tuple[Basis, ...] = DEFAULT_BASES) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "single_qubit_opcodes": list(SINGLE_QUBIT_OPCODES),
        "run_seed": _RUN_SEED,
        "legalization_seed": _LEGALIZATION_SEED,
        "determinants": [determinants(basis) for basis in bases],
        "reachability": [reachability(basis) for basis in bases],
        # The first two pairs are the z-rotation and pulse of the shipped target
        # bases, where the determinant obstruction applies. The second two swap in
        # a z-rotation whose determinant is its parameter, which is what a target
        # would publish to escape the obstruction; no shipped basis declares one,
        # so the pair is a bracket on the obstruction rather than a target.
        "declared_basis_reach": [
            declared_basis_reach(pair)
            for pair in (("rz", "sx"), ("rz", "rx"), ("phase", "sx"), ("u1", "rx"))
        ],
        "legalization_phase": [legalization_phase(basis) for basis in bases],
        "phase_witness": [phase_witness(basis) for basis in bases],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    for row in payload["reachability"]:
        if row["determinant_is_free"]:
            print(
                f"{row['label']:>20}: an arity-1 gate's determinant argument is its "
                "parameter, so no operator is blocked by its determinant"
            )
        else:
            print(
                f"{row['label']:>20}: a word's determinant argument lies in "
                f"{row['reachable_arguments_degrees']} degrees; "
                f"{len(row['blocked_opcodes'])} of "
                f"{len(payload['single_qubit_opcodes'])} declared arity-1 opcodes "
                "fall outside it"
            )
    for row in payload["declared_basis_reach"]:
        print(
            f"{row['z_rotation']}/{row['pulse_opcode']}: "
            f"{row['up_to_a_global_phase']['words_found']}/{row['trial_count']} runs "
            f"have a word up to a phase "
            f"({row['up_to_a_global_phase']['gates_saved']} gates saved), but only "
            f"{row['entry_for_entry']['words_found']} agree entry for entry "
            f"({row['entry_for_entry']['gates_saved']} gates saved)"
        )
    for row in payload["legalization_phase"]:
        print(
            f"legalize {row['label']:>20}: {row['statevector_unchanged']}/"
            f"{row['circuit_count']} statevectors unchanged, "
            f"{row['statevector_phased']} phased, of which "
            f"{row['phase_is_not_a_sign']} are not a sign "
            f"({row['circuit_count_refused']} programs refused by the target)"
        )
    for row in payload["phase_witness"]:
        print(
            f"witness {row['label']:>20}: {' '.join(row['source_opcodes'])} -> "
            f"{' '.join(row['legalized_opcodes'])}, global phase "
            f"{row['global_phase_degrees']} degrees, amplitude gap "
            f"{row['max_amplitude_gap']:.3e}"
        )
    if args.json_output is not None:
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
