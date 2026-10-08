"""Measure the whole target-independent ladder's optimized depth and gate count.

W9-40 of the Qiskit parity backlog asked for this and it had never been measured.
Every compiler benchmark in this repository measures **one pass** or one rule, so
the two numbers a reader actually wants after ``optimize`` -- *how big* and *how
deep* is the program, next to what Qiskit's ``transpile`` gives at the same
``optimization_level`` -- did not exist anywhere. This module adds them.

Two things had to be established before a comparison could be stated at all, and
both are measurements rather than conventions.

**The depth metric.** FlagQuantum's IR has no depth function -- searching
``flagquantum/`` for one finds a Protocol in the runtime planner and nothing else
-- so the module has to define one, and comparing against a differently-defined
depth would compare two instruments rather than two compilers. ``program_depth`` is
therefore written to mirror ``QuantumCircuit.depth`` and is then *calibrated
against it*: the payload reports the count of programs where the two agree, and
``depth_conventions`` exercises each convention with a case whose value is
recorded, because a convention nothing in the corpus triggers is an assumption
about the metric rather than a property of it.

**The counting unit.** ``measure`` is not a declared opcode in FlagQuantum's
operator schema, so it enters the IR as a dynamic instruction, and
``ir.measurements`` is a *different* representation that ``len(ir)`` cannot count
while Qiskit's ``size()`` counts ``measure`` ops. A corpus that mixed the two would
produce a gate-count difference made entirely of the representation choice. Every
measurement here is therefore an instruction in both ports, and the node form's
cost is reported as its own row so the size of the mismatch is measured instead of
hidden.

What the measurement says, on the seeded corpus below. The source programs total
**177 instructions over depth 114**, and both ports read exactly that at level 0 --
which is the control, because Qiskit's level 0 with no basis and no coupling map
runs no optimization pass and the port's level 0 is the empty stage tuple.

============================  ==================  ==================
 ``optimization_level``        port (gates/depth)  Qiskit (gates/depth)
============================  ==================  ==================
 0                             177 / 114           177 / 114
 1                              70 / 53             62 / 47
 2                              60 / 43             57 / 42
 3                             refused              57 / 42
============================  ==================  ==================

* **Six of the eight populations read identically in both ports at every level.**
  Adjacent self-inverse cancellation (``h h``, ``cx cx``), adjacent same-opcode
  rotation merge (``rz .3 rz .4``), the commuting-gap merge at level 2, the parallel
  layer population, the irreducible control, and the ``rz`` read out only by a
  measurement. Those are agreements on mechanism, not coincidences: the trace in
  ``mechanism_table`` names the pass that produced each one.
* **The port's whole deficit at level 2 is three gates and one time step**, and the
  attribution table says where. ``square_is_another_gate`` is **+5 gates / +3
  depth**, and the two reasons are measured separately rather than merged: ``sx sx``
  is left standing although the port's fold *can* write it as one ``u3``, because the
  run is already spelled as one z-rotation/pulse pair and the vocabulary decline
  refuses it; while ``s sx`` and ``rz(0.3) z`` are refused because the replacement is
  not *strictly* shorter -- ``s sx`` emits ``u3, rz`` and ``rz(0.3) z`` emits
  ``phase, rz``, two instructions for two. ``_emit`` returns "the shortest **exact**
  sequence", so it carries a ``phase`` for the global phase that Qiskit's
  ``Optimize1qGatesDecomposition`` is free to drop. The remaining gate is in
  ``irreducible``, where Qiskit's level 2 merges a ``crz`` into a ``rzz`` and the
  port's ladder has no such rule: **+1 gate, 0 depth**.
* **The port's lead is two mechanisms and it is smaller.** ``h(1) cz(0,1) h(1)`` is
  ``cx(0,1)`` and ``h(1) cx(0,1) h(1)`` is ``cz(0,1)``; ``collapse_two_qubit_blocks``
  folds them at level 2 and Qiskit's level 2, with no basis, keeps all three
  operations -- worth −4 gates / −4 depth on that population. The two-wire diagonal
  ahead of a measurement is worth −1 / −1.
* **``swap cz swap`` is folded by neither port**, at either level. Its row is in the
  table with both readings at 3, measured rather than assumed.
* **Level 3 is the one level the ports differ on structurally, and the corpus cannot
  price it.** The port refuses it with a message naming the target-aware stage its
  ladder cannot contain; Qiskit accepts it and, with no basis and no coupling map,
  produces exactly its own level-2 reading. So "level 3 would not help here" is not a
  finding of this benchmark -- it is what a corpus with no target can show.

**What this is not.** A basis-legality claim. Both ports are driven with no
``basis_gates``, no ``coupling_map``, no layout and no routing, so *every* number here
is about the target-independent ladder and none of them is about what a device can
run. Qiskit's own levels unroll into a basis when one is named and the port's ladder
never does, so the two would stop being comparable the moment a basis was named --
which is also why ``compile(..., optimize=False)`` and
``native_gate_legalization.legalize_native_gates`` are outside this measurement
rather than folded into it.

Classification: a local compiler microbenchmark on the single-device fast path. It
runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_optimization_levels.py --json-output /tmp/w940.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from flagquantum.compiler import optimize  # noqa: E402
from flagquantum.compiler.optimization_levels import (  # noqa: E402
    IMPLEMENTED_OPTIMIZATION_LEVELS,
    OPTIMIZATION_LEVEL_STAGES,
    RESERVED_OPTIMIZATION_LEVELS,
)
from flagquantum.compiler.pipeline import _round_passes  # noqa: E402
from flagquantum.core.ir import CircuitIR, Instruction  # noqa: E402

SCHEMA = "flagquantum_compiler_optimization_levels_benchmark_v1"

_NOW = datetime.now(timezone.utc)

#: Seeded, so the counts below are reproducible and can be pinned in a benchmark
#: contract.
_SWEEP_SEED = 20261007

#: Four wires. Every population needs a wire a gate does *not* touch, the ``parallel``
#: population needs three for a depth-1 row, and the convention case
#: ``two_gates_on_disjoint_wires_are_not`` needs four.
_REGISTER_WIDTH = 4

#: ``complex128``, because the execution control compares statevectors exactly rather
#: than sampling.
_DTYPE = "complex128"

#: The levels the port implements, compared head to head with the second port's levels
#: of the same number.
_LEVELS: tuple[int, ...] = IMPLEMENTED_OPTIMIZATION_LEVELS

#: The level the ladder declares and this release refuses.
_RESERVED_LEVEL = 3

#: Instructions that occupy no wire time. This is not a guess about what a depth metric
#: *should* do: it is what ``QuantumCircuit.depth`` does, and ``depth_conventions``
#: records the reading. A ``barrier`` enters FlagQuantum's IR as a dynamic instruction,
#: so without this set the two metrics would differ on every program containing one.
_NO_TIME_OPCODES = frozenset({"barrier"})

#: Every measurement here is written in this form, because ``ir.measurements`` is a
#: representation ``len(ir)`` cannot count and the two must not be mixed.
_DYNAMIC_METADATA = {"is_dynamic": True}

#: The comparison floors. The statevector floor is ``complex128`` noise; the control
#: floor is what a rewrite that *changed* the program would have to clear to be visible,
#: so the control discriminates rather than merely passing.
_STATEVECTOR_FLOOR = 1e-12
_DISTRIBUTION_FLOOR = 1e-12
_CONTROL_FLOOR = 1e-6

#: The rotation family the corpus uses. Every member must have a Qiskit lowering,
#: because the second port is asked the same question about the same program --
#: ``conversion_coverage`` measures the one declared rotation opcode that does *not*, and
#: the corpus leaves it out for that reason rather than by preference.
_ROTATIONS = ("rx", "ry", "rz", "phase")

#: Every declared single-qubit rotation opcode, including the ones the exporter cannot
#: lower. Listed so the gap is a measurement rather than an avoided name.
_DECLARED_SINGLE_QUBIT_ROTATIONS = ("rx", "ry", "rz", "u1", "phase")

_SELF_INVERSE = ("h", "x", "y", "z", "cx", "cz", "swap")
#: Opcode pairs whose product is a *different* declared opcode rather than the identity.
#: ``sx sx`` is ``x``, ``s sx`` is ``sx``, ``s s`` is ``z``, so none of the three may be
#: removed and each is one operation after a resynthesis.
_SQUARES = (("sx", "sx"), ("s", "sx"), ("s", "s"))
_CONTROLLED = ("cx", "cy", "cz")
#: Diagonal for *every* angle. ``rz`` and ``phase`` are too, and their opcode is also in
#: ``_ROTATIONS``.
_DIAGONAL_ONE_WIRE = ("z", "s", "t", "rz", "phase")


def program_depth(ir: CircuitIR) -> int:
    """The length of the critical path, defined to mirror ``QuantumCircuit.depth``.

    Each instruction occupies a single time step on every wire it names and may begin as
    soon as the last of those wires is free, so the depth is the longest chain of
    instructions that are *not* mutually parallel. ``barrier`` occupies no step, which is
    what Qiskit does -- see ``depth_conventions``, where the reading is recorded rather
    than quoted.

    ``measure`` occupies a step, also matching Qiskit. That is what makes the depth of a
    program with measurements comparable at all: if measurements cost a step in one port
    and not the other, every row of the level table would carry a constant offset that
    has nothing to do with optimization.
    """

    wire_time = [0] * ir.n_wires
    for instruction in ir.instructions:
        if instruction.name in _NO_TIME_OPCODES:
            continue
        start = max((wire_time[wire] for wire in instruction.wires), default=0)
        for wire in instruction.wires:
            wire_time[wire] = start + 1
    return max(wire_time, default=0)


def _instruction(opcode: str, wires: tuple[int, ...], **params: Any) -> Instruction:
    return Instruction(opcode, wires, params or {})


def _measure(wire: int) -> Instruction:
    return Instruction(
        "measure", (wire,), metadata={**_DYNAMIC_METADATA, "classical_bit": wire}
    )


def _two_wires(seed: random.Random) -> tuple[int, int]:
    return tuple(seed.sample(range(_REGISTER_WIDTH), 2))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The depth metric's own conventions, each exercised by a case
# ---------------------------------------------------------------------------

#: One case per convention ``program_depth`` implements, each with the value recorded. A
#: convention the corpus does not reach is an assumption about the metric, so these are
#: here rather than left to the sweep: the sweep contains no barrier at all and no
#: three-wire gate.
_DEPTH_CONVENTION_CASES: dict[str, tuple[tuple[str, ...], int]] = {
    "three_parallel_single_qubit_gates_have_depth_1": (("h:0", "h:1", "h:2"), 1),
    "two_gates_sharing_a_wire_are_sequential": (("cx:0,1", "cx:0,2"), 2),
    "two_gates_on_disjoint_wires_are_not": (("cx:0,1", "cx:2,3"), 1),
    "a_barrier_occupies_no_time_step": (("h:0", "barrier:0,1", "cx:0,1"), 2),
    "a_measurement_occupies_a_time_step": (("h:0", "measure:0"), 2),
    "measurements_on_one_wire_are_sequential": (("measure:0", "measure:0"), 2),
    "measurements_on_different_wires_are_not": (("measure:0", "measure:1"), 1),
    "a_three_wire_gate_blocks_every_wire_it_names": (("ccx:0,1,2", "h:2"), 2),
    "an_empty_program_has_depth_0": ((), 0),
}


def _case_program(spec: tuple[str, ...]) -> CircuitIR:
    """Build the program one convention case names, at the width it needs.

    The width is the largest wire any element names plus one, so
    ``two_gates_on_disjoint_wires_are_not`` reaches wire 3 and the case is honest about
    needing four wires rather than silently wrapping onto a third.
    """

    instructions: list[Instruction] = []
    width = 1
    for element in spec:
        opcode, _, wire_text = element.partition(":")
        wires = tuple(int(part) for part in wire_text.split(",")) if wire_text else ()
        width = max(width, (max(wires) + 1) if wires else 1)
        if opcode == "measure":
            instructions.append(_measure(wires[0]))
        elif opcode == "barrier":
            instructions.append(
                Instruction("barrier", wires, metadata=dict(_DYNAMIC_METADATA))
            )
        else:
            instructions.append(Instruction(opcode, wires))
    return CircuitIR(width, tuple(instructions))


def depth_conventions() -> dict[str, Any]:
    """The metric's conventions, each one a case with its measured value.

    A metric is a set of decisions, and a decision nothing exercises is a claim rather
    than a property. Every entry below is the depth this module's own function returns
    for a program that isolates one decision, and the contract test re-derives each from
    ``program_depth`` rather than trusting the recorded number.
    """

    rows: list[dict[str, Any]] = []
    for label, (spec, expected) in _DEPTH_CONVENTION_CASES.items():
        measured = program_depth(_case_program(spec))
        rows.append(
            {
                "label": label,
                "program": list(spec),
                "expected_depth": expected,
                "measured_depth": measured,
                "agrees": measured == expected,
            }
        )
    return {
        "definition": (
            "the longest chain of instructions that are not mutually parallel: each "
            "instruction occupies one time step on every wire it names and begins as "
            "soon as the last of those wires is free"
        ),
        "no_time_opcodes": sorted(_NO_TIME_OPCODES),
        "counts_measurements": True,
        "rows": rows,
        "row_count": len(rows),
        "disagreement_count": sum(1 for row in rows if not row["agrees"]),
    }


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

#: The populations, each isolating one mechanism so a difference in the level table can
#: be attributed to it. A population is not a workload and the sizes are small on
#: purpose: the whole sweep has to stay enumerable in a contract test.
_POPULATIONS: tuple[str, ...] = (
    "adjacent_self_inverse",
    "square_is_another_gate",
    "rotation_runs",
    "commuting_gap",
    "two_qubit_blocks",
    "diagonal_before_measure",
    "parallel",
    "irreducible",
)

_PROGRAMS_PER_POPULATION = 4


def _adjacent_self_inverse(seed: random.Random) -> CircuitIR:
    """Pairs of one self-inverse opcode, adjacent, on one wire.

    ``InverseCancellation``'s subject matter and the port's ``merge_self_inverse``. Every
    pair here *is* the identity, so a port that keeps one has failed rather than merely
    optimized less.
    """

    instructions: list[Instruction] = []
    for _ in range(seed.randint(3, 6)):
        opcode = seed.choice(_SELF_INVERSE)
        wires = (
            _two_wires(seed)
            if opcode in {"cx", "cz", "swap"}
            else (seed.randrange(_REGISTER_WIDTH),)
        )
        instructions.append(_instruction(opcode, wires))
        instructions.append(_instruction(opcode, wires))
    return CircuitIR(_REGISTER_WIDTH, tuple(instructions))


def _square_is_another_gate(seed: random.Random) -> CircuitIR:
    """Pairs of one opcode whose product is a declared gate but is not the identity.

    ``sx sx`` is ``x``, ``s sx`` is ``sx``, ``s s`` is ``z``. None may be *removed* and
    each is one operation after a resynthesis, so this population is where a resynthesis
    that is exact-including-phase differs from one that is up-to-phase.
    """

    instructions: list[Instruction] = []
    for _ in range(seed.randint(2, 4)):
        wire = seed.randrange(_REGISTER_WIDTH)
        for opcode in seed.choice(_SQUARES):
            instructions.append(_instruction(opcode, (wire,)))
    return CircuitIR(_REGISTER_WIDTH, tuple(instructions))


def _rotation_runs(seed: random.Random) -> CircuitIR:
    """A run of one rotation opcode on one wire, which merges into one rotation."""

    opcode = seed.choice(_ROTATIONS)
    wire = seed.randrange(_REGISTER_WIDTH)
    return CircuitIR(
        _REGISTER_WIDTH,
        tuple(
            _instruction(opcode, (wire,), theta=seed.uniform(0.05, 1.0))
            for _ in range(seed.randint(2, 5))
        ),
    )


def _commuting_gap(seed: random.Random) -> CircuitIR:
    """Two rotations of one opcode with a controlled gate between them.

    The gap merge is level 2's subject matter and the only mechanism here that moves a
    gate *past* another, so this population is where level 1 and level 2 must differ.
    """

    opcode = seed.choice(_ROTATIONS)
    wires = _two_wires(seed)
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            _instruction(opcode, (wires[0],), theta=0.3),
            _instruction(seed.choice(_CONTROLLED), wires),
            _instruction(opcode, (wires[0],), theta=0.4),
        ),
    )


def _two_qubit_blocks(seed: random.Random) -> CircuitIR:
    """A single-qubit gate, a two-qubit gate, and that gate again.

    ``h(1) cz(0,1) h(1)`` is ``cx(0,1)``. The three operations are sequential on wire 1,
    so a fold moves the *depth* by two and not only the count -- which is why this
    population is where the port's lead is visible in both columns.
    """

    wires = _two_wires(seed)
    if seed.random() < 0.5:
        return CircuitIR(
            _REGISTER_WIDTH,
            (
                _instruction("rz", (wires[1],), theta=0.35),
                _instruction(seed.choice(("cz", "cx")), wires),
                _instruction("rz", (wires[1],), theta=-0.35),
            ),
        )
    conjugation = seed.choice(("h", "sx"))
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            _instruction(conjugation, (wires[1],)),
            _instruction(seed.choice(("cz", "cx")), wires),
            _instruction(conjugation, (wires[1],)),
        ),
    )


def _diagonal_before_measure(seed: random.Random) -> CircuitIR:
    """A diagonal gate whose only consumer is a measurement.

    Level 2 removes it and level 1 does not, which is the ladder's one recorded
    divergence from Qiskit 1.2.4 -- and this population is where that divergence becomes
    a reading rather than a docstring.
    """

    if seed.random() < 0.25:
        # The two-wire branch. A `cz` is diagonal at every angle and its second wire is
        # left free on purpose, so "the next instruction on *every* one of its wires is a
        # measurement" is exercised in both directions.
        wires = _two_wires(seed)
        instructions: list[Instruction] = [
            _instruction("cz", wires),
            _instruction(seed.choice(("h", "z", "s")), (wires[0],)),
            _measure(wires[0]),
        ]
        if seed.random() < 0.5:
            instructions.append(_measure(wires[1]))
        return CircuitIR(_REGISTER_WIDTH, tuple(instructions))

    wire = seed.randrange(_REGISTER_WIDTH)
    opcode = seed.choice(_DIAGONAL_ONE_WIRE)
    params = {"theta": 0.7} if opcode in {"rz", "phase"} else {}
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            _instruction(opcode, (wire,), **params),
            _instruction(seed.choice(("h", "z", "s")), (wire,)),
            _measure(wire),
        ),
    )


def _parallel(seed: random.Random) -> CircuitIR:
    """Layers of a self-inverse-single-qubit gate on every wire at once.

    Depth and count cannot move together here: ``n`` parallel layers on three wires is a
    known count and depth ``n``, so this is the population that shows the two columns are
    different instruments rather than one reported twice.
    """

    instructions: list[Instruction] = []
    for _ in range(seed.randint(2, 5)):
        for wire in range(3):
            instructions.append(_instruction("h", (wire,)))
    return CircuitIR(_REGISTER_WIDTH, tuple(instructions))


#: The atoms the irreducible population is built from, one list of instructions each.
#: Every one is a pair the ladder provably cannot rewrite, and the *proof* is that the
#: population is checked to be a fixed point at every level in the payload rather than
#: asserted here: if a future ladder gains a rule that folds one of these pairs, the
#: check fails and the population has to be rebuilt instead of the claim quietly becoming
#: false.
_IRREDUCIBLE_ATOMS: tuple[tuple[tuple[str, bool], ...], ...] = (
    (("cphase", True), ("cx", False)),
    (("rzz", True), ("crz", True)),
    (("swap", False), ("cx", False)),
    (("rxx", True), ("ryy", True)),
)


def _irreducible(seed: random.Random) -> CircuitIR:
    """A program the ladder leaves alone, as the control.

    A ladder that "improved" this population would be changing the program rather than
    optimizing it. Without a population nothing can be removed from, a ratio against the
    source count is unfalsifiable: every other population can only go down, and "the
    ladder always shrinks a program" cannot be tested against anything.
    """

    instructions: list[Instruction] = []
    for _ in range(seed.randint(2, 3)):
        for opcode, needs_angle in seed.choice(_IRREDUCIBLE_ATOMS):
            params = {"theta": seed.uniform(0.2, 1.0)} if needs_angle else {}
            instructions.append(_instruction(opcode, _two_wires(seed), **params))
    return CircuitIR(_REGISTER_WIDTH, tuple(instructions))


_CORPUS_BUILDERS = {
    "adjacent_self_inverse": _adjacent_self_inverse,
    "square_is_another_gate": _square_is_another_gate,
    "rotation_runs": _rotation_runs,
    "commuting_gap": _commuting_gap,
    "two_qubit_blocks": _two_qubit_blocks,
    "diagonal_before_measure": _diagonal_before_measure,
    "parallel": _parallel,
    "irreducible": _irreducible,
}


def corpus() -> dict[str, list[CircuitIR]]:
    """The seeded populations, in a fixed order, one generator per population.

    Each population gets its own ``Random`` seeded from the sweep seed plus its index, so
    adding a population cannot move another one's programs and the recorded counts stay
    comparable across a change to this module.
    """

    programs: dict[str, list[CircuitIR]] = {}
    for index, name in enumerate(_POPULATIONS):
        seed = random.Random(_SWEEP_SEED + index)
        programs[name] = [
            _CORPUS_BUILDERS[name](seed) for _ in range(_PROGRAMS_PER_POPULATION)
        ]
    return programs


# ---------------------------------------------------------------------------
# The second port
# ---------------------------------------------------------------------------


def _transpile_to_compare(circuit: Any, level: int) -> Any:
    """Qiskit's reading at one level, with everything that could differ switched off.

    ``basis_gates``, ``coupling_map`` and ``layout_method`` are all left unset on purpose:
    naming any of them makes Qiskit unroll into a basis or insert routing, and the port's
    ladder is target-independent by contract. A reading taken with a basis would be a
    reading of a different question.

    ``routing_method="none"`` is the one thing that *is* set, because with no coupling map
    Qiskit's default still looks for work to do; pinning it is what makes level 0 the
    identity, which is what makes level 0 usable as the control.
    """

    from qiskit import transpile  # type: ignore[import-not-found]

    return transpile(
        circuit,
        optimization_level=level,
        basis_gates=None,
        coupling_map=None,
        layout_method="trivial",
        routing_method="none",
    )


def _to_qiskit(ir: CircuitIR) -> Any:
    from flagquantum.ecosystem.qiskit.conversion import to_qiskit

    return to_qiskit(ir)


def reference_anchor() -> dict[str, Any]:
    """Qiskit's own readings, or the reason they are absent.

    The anchor is a cross-check and not a dependency: every number the port contributes is
    measured without Qiskit at all, and this function is the only place the second port is
    imported. When it is missing, the payload says so with the exception, so a host
    without Qiskit reports silence rather than agreement.
    """

    try:
        import qiskit  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    programs = corpus()

    level_totals: dict[str, dict[str, int]] = {}
    for level in (*_LEVELS, _RESERVED_LEVEL):
        size = 0
        depth = 0
        for name in _POPULATIONS:
            for ir in programs[name]:
                transpiled = _transpile_to_compare(_to_qiskit(ir), level)
                size += transpiled.size()
                depth += transpiled.depth()
        level_totals[str(level)] = {"gate_count": size, "depth": depth}

    # The calibration. A corpus program the second port cannot be asked about would turn
    # every comparison into a comparison over a different corpus, so it is recorded and
    # reported rather than skipped.
    calibrations: list[dict[str, Any]] = []
    unconvertible: list[dict[str, Any]] = []
    for name in _POPULATIONS:
        for index, ir in enumerate(programs[name]):
            try:
                circuit = _to_qiskit(ir)
            except Exception as error:
                unconvertible.append(
                    {
                        "population": name,
                        "index": index,
                        "opcodes": sorted({item.name for item in ir.instructions}),
                        "error_type": type(error).__name__,
                    }
                )
                continue
            calibrations.append(
                {
                    "population": name,
                    "port_depth": program_depth(ir),
                    "qiskit_depth": circuit.depth(),
                    "port_size": len(ir),
                    "qiskit_size": circuit.size(),
                }
            )
    depth_mismatches = [
        row for row in calibrations if row["port_depth"] != row["qiskit_depth"]
    ]
    size_mismatches = [
        row for row in calibrations if row["port_size"] != row["qiskit_size"]
    ]

    # Which convention cases the second port can be asked about at all. Derived rather
    # than listed: a case whose program the exporter cannot lower, and the empty case
    # which Qiskit cannot represent (a circuit with no wires), both come out as unaskable,
    # because a convention silently skipped is a convention assumed.
    convention_mismatches: list[str] = []
    unaskable: list[str] = []
    for label, (spec, _expected) in _DEPTH_CONVENTION_CASES.items():
        program = _case_program(spec)
        try:
            circuit = _to_qiskit(program)
        except Exception:
            unaskable.append(label)
            continue
        if circuit.num_qubits == 0:  # pragma: no cover - only the empty case
            unaskable.append(label)
            continue
        if program_depth(program) != circuit.depth():
            convention_mismatches.append(label)

    return {
        "available": True,
        "qiskit_version": qiskit.__version__,
        "conversion_module": "flagquantum.ecosystem.qiskit.conversion.to_qiskit",
        "transpile_arguments": {
            "basis_gates": None,
            "coupling_map": None,
            "layout_method": "trivial",
            "routing_method": "none",
        },
        "level_totals": level_totals,
        "calibration_row_count": len(calibrations),
        "unconvertible_program_count": len(unconvertible),
        "unconvertible_programs": unconvertible[:8],
        "depth_mismatch_count": len(depth_mismatches),
        "depth_mismatches": depth_mismatches[:8],
        "gate_count_mismatch_count": len(size_mismatches),
        "gate_count_mismatches": size_mismatches[:8],
        "convention_mismatch_labels": convention_mismatches,
        "unaskable_convention_labels": unaskable,
        "convention_case_count": len(_DEPTH_CONVENTION_CASES),
    }


# ---------------------------------------------------------------------------
# The level table, for the port
# ---------------------------------------------------------------------------


def level_table() -> dict[str, Any]:
    """Gate count and depth per population per level, for the port.

    Both columns are reported because they are different instruments: the ``parallel``
    population has three times the count of its depth by construction, and a fold can move
    one column without the other.
    """

    programs = corpus()
    rows: list[dict[str, Any]] = []
    for name in _POPULATIONS:
        for level in _LEVELS:
            source_gates = source_depth = optimized_gates = optimized_depth = 0
            for ir in programs[name]:
                optimized = optimize(ir, optimization_level=level)
                source_gates += len(ir)
                source_depth += program_depth(ir)
                optimized_gates += len(optimized)
                optimized_depth += program_depth(optimized)
            rows.append(
                {
                    "population": name,
                    "optimization_level": level,
                    "program_count": len(programs[name]),
                    "source_gate_count": source_gates,
                    "port_gate_count": optimized_gates,
                    "port_gate_delta": optimized_gates - source_gates,
                    "source_depth": source_depth,
                    "port_depth": optimized_depth,
                    "port_depth_delta": optimized_depth - source_depth,
                }
            )
    return {"rows": rows, "row_count": len(rows)}


def level_totals(table: dict[str, Any]) -> dict[str, Any]:
    """The per-level totals, which are the two numbers the backlog row asked for."""

    totals: dict[str, dict[str, int]] = {}
    for level in _LEVELS:
        rows = [row for row in table["rows"] if row["optimization_level"] == level]
        totals[str(level)] = {
            "source_gate_count": sum(row["source_gate_count"] for row in rows),
            "source_depth": sum(row["source_depth"] for row in rows),
            "port_gate_count": sum(row["port_gate_count"] for row in rows),
            "port_depth": sum(row["port_depth"] for row in rows),
            "port_gate_delta": sum(row["port_gate_delta"] for row in rows),
            "port_depth_delta": sum(row["port_depth_delta"] for row in rows),
        }
    irreducible = [row for row in table["rows"] if row["population"] == "irreducible"]
    return {
        "level_totals": totals,
        "source_totals": {
            "gate_count": totals[str(_LEVELS[0])]["source_gate_count"],
            "depth": totals[str(_LEVELS[0])]["source_depth"],
        },
        "irreducible_population": "irreducible",
        # The control, checked from the level table's own rows: the population with
        # nothing to remove must read identically at every level. A ladder that shrank it
        # would be changing the program, and every other population's ratio against its
        # source count would be unfalsifiable without this.
        "irreducible_unchanged_at_every_level": bool(irreducible)
        and all(
            row["port_gate_delta"] == 0 and row["port_depth_delta"] == 0
            for row in irreducible
        ),
        "irreducible_readings": [
            {
                "optimization_level": row["optimization_level"],
                "gate_count": row["port_gate_count"],
                "depth": row["port_depth"],
            }
            for row in irreducible
        ],
    }


# ---------------------------------------------------------------------------
# The two ports side by side
# ---------------------------------------------------------------------------


def _population_totals(
    programs: dict[str, list[CircuitIR]],
) -> dict[tuple[str, int], tuple[int, int]]:
    """Qiskit's per-population readings, measured once and reused by every caller.

    Without this each of ``comparison`` and ``mechanism_table`` would convert and
    transpile the whole corpus again, and two callers measuring the same thing separately
    is how two reported numbers stop agreeing.
    """

    totals: dict[tuple[str, int], tuple[int, int]] = {}
    for name in _POPULATIONS:
        for level in _LEVELS:
            size = depth = 0
            for ir in programs[name]:
                transpiled = _transpile_to_compare(_to_qiskit(ir), level)
                size += transpiled.size()
                depth += transpiled.depth()
            totals[(name, level)] = (size, depth)
    return totals


def comparison(
    table: dict[str, Any],
    anchor: dict[str, Any],
    population_totals: dict[tuple[str, int], tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """The port's and Qiskit's readings on the same populations, with their deltas.

    This is the section the backlog row asked for. Its per-level half is built from the
    port's own level table and the anchor's per-level totals rather than by re-running
    either, so the comparison cannot disagree with the tables it summarizes.

    The deltas are signed ``port minus Qiskit``, so a negative number is the port reaching
    a smaller program and a positive number is Qiskit reaching one. Neither sign is a
    verdict: a smaller program that is a *different* program is not an improvement, which
    is what ``execution_control`` and ``phase_control`` exist to rule out.
    """

    if not anchor.get("available"):
        return {
            "available": False,
            "reason": anchor.get("reason"),
            "per_level": [],
            "per_population": [],
        }
    if population_totals is None:  # pragma: no cover - the payload always passes it
        population_totals = _population_totals(corpus())

    per_level: list[dict[str, Any]] = []
    for level in _LEVELS:
        port = next(
            row for row in _totals_by_level(table) if row["optimization_level"] == level
        )
        theirs = anchor["level_totals"][str(level)]
        per_level.append(
            {
                "optimization_level": level,
                "source_gate_count": port["source_gate_count"],
                "source_depth": port["source_depth"],
                "port_gate_count": port["port_gate_count"],
                "port_depth": port["port_depth"],
                "qiskit_gate_count": theirs["gate_count"],
                "qiskit_depth": theirs["depth"],
                "gate_count_delta": port["port_gate_count"] - theirs["gate_count"],
                "depth_delta": port["port_depth"] - theirs["depth"],
            }
        )

    per_population: list[dict[str, Any]] = []
    for level in _LEVELS:
        for row in table["rows"]:
            if row["optimization_level"] != level:
                continue
            theirs = population_totals[(row["population"], level)]
            per_population.append(
                {
                    "population": row["population"],
                    "optimization_level": level,
                    "port_gate_count": row["port_gate_count"],
                    "port_depth": row["port_depth"],
                    "qiskit_gate_count": theirs[0],
                    "qiskit_depth": theirs[1],
                    "gate_count_delta": row["port_gate_count"] - theirs[0],
                    "depth_delta": row["port_depth"] - theirs[1],
                }
            )

    agreeing = [
        row
        for row in per_population
        if row["gate_count_delta"] == 0 and row["depth_delta"] == 0
    ]
    return {
        "available": True,
        "per_level": per_level,
        "per_population": per_population,
        "population_level_readings_identical": len(agreeing),
        "population_level_readings": len(per_population),
        "port_smaller_count": sum(
            1 for row in per_population if row["gate_count_delta"] < 0
        ),
        "qiskit_smaller_count": sum(
            1 for row in per_population if row["gate_count_delta"] > 0
        ),
        "populations_identical_at_every_level": sorted(
            name
            for name in _POPULATIONS
            if all(
                row["gate_count_delta"] == 0 and row["depth_delta"] == 0
                for row in per_population
                if row["population"] == name
            )
        ),
        "port_deficit_at_level_2": next(
            row["gate_count_delta"]
            for row in per_level
            if row["optimization_level"] == 2
        ),
        "port_depth_deficit_at_level_2": next(
            row["depth_delta"] for row in per_level if row["optimization_level"] == 2
        ),
        "note": (
            "signed port minus Qiskit at each optimization level, over the same seeded "
            "corpus, with no basis, coupling map, layout or routing in either port"
        ),
    }


def _totals_by_level(table: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-level sums of the port's level table, recomputed from its own rows."""

    keys = (
        "source_gate_count",
        "source_depth",
        "port_gate_count",
        "port_depth",
    )
    return [
        {
            "optimization_level": level,
            **{
                key: sum(
                    row[key]
                    for row in table["rows"]
                    if row["optimization_level"] == level
                )
                for key in keys
            },
        }
        for level in _LEVELS
    ]


# ---------------------------------------------------------------------------
# One mechanism at a time, and which pass produced it
# ---------------------------------------------------------------------------

#: The attribution table. Each row is one small program and one mechanism, small enough
#: that the difference it produces can only come from that mechanism. Without these rows a
#: level-table gap is a number with no cause, and "the port is behind" would be an
#: interpretation rather than a reading.
#:
#: The element grammar is ``opcode:wires`` with an optional ``@angle``, so ``"rz:0@0.3"``
#: is ``rz(0.3)`` on wire 0 and ``"cz:0,1"`` is ``cz`` on wires 0 and 1.
_MECHANISM_SPECS: dict[str, tuple[str, ...]] = {
    "adjacent_self_inverse_h_h": ("h:0", "h:0"),
    "adjacent_self_inverse_cx_cx": ("cx:0,1", "cx:0,1"),
    "same_opcode_rotation_run": ("rz:0@0.3", "rz:0@0.4"),
    "square_is_another_gate_sx_sx": ("sx:0", "sx:0"),
    "square_is_another_gate_s_s": ("s:0", "s:0"),
    "product_is_another_gate_s_sx": ("s:0", "sx:0"),
    "rotation_merged_across_opcodes": ("rz:0@0.3", "z:0"),
    "commuting_gap_merge": ("rz:0@0.3", "cz:0,1", "rz:0@0.4"),
    "two_qubit_block_conjugation_h_cz_h": ("h:1", "cz:0,1", "h:1"),
    "two_qubit_block_conjugation_h_cx_h": ("h:1", "cx:0,1", "h:1"),
    "swap_cz_swap_is_not_folded": ("swap:1,2", "cz:2,1", "swap:1,2"),
    "diagonal_then_measure_z": ("z:0", "measure:0"),
    "diagonal_then_measure_rz": ("rz:0@0.7", "measure:0"),
    "non_diagonal_then_measure_h": ("h:0", "measure:0"),
}

_MECHANISM_CASES: tuple[str, ...] = tuple(_MECHANISM_SPECS)


def _mechanism_program(spec: tuple[str, ...]) -> CircuitIR:
    """Build one mechanism case. ``opcode:wires@angle`` is the element grammar."""

    instructions: list[Instruction] = []
    width = 1
    for element in spec:
        opcode_text, _, angle_text = element.partition("@")
        opcode, _, wire_text = opcode_text.partition(":")
        wires = tuple(int(part) for part in wire_text.split(",")) if wire_text else ()
        width = max(width, (max(wires) + 1) if wires else 1)
        if opcode == "measure":
            instructions.append(_measure(wires[0]))
        elif angle_text:
            instructions.append(_instruction(opcode, wires, theta=float(angle_text)))
        else:
            instructions.append(_instruction(opcode, wires))
    return CircuitIR(width, tuple(instructions))


def _opcodes(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir.instructions]


def pass_trace(program: CircuitIR, level: int) -> list[dict[str, Any]]:
    """Every pass that changed this program at this level, in the order it ran.

    The level's declared stage tuple is run one pass at a time, to the same fixed point
    ``optimize`` reaches, and only the passes whose output differs from their input are
    recorded. Without this a level-table difference has no cause attached, and this
    module's own prose would be an interpretation of two numbers rather than a reading of
    which rule produced them.

    The trace compares opcode sequences and not whole instructions, so a pass that rewrote
    an angle without changing the opcode list is not recorded here. That is the right
    granularity for the table it feeds -- every row's claim is about an opcode count -- and
    ``execution_control`` is what covers the angles.
    """

    passes = _round_passes()
    stages = OPTIMIZATION_LEVEL_STAGES[level]
    current = program
    trace: list[dict[str, Any]] = []
    previous = _opcodes(current)
    for round_index in range(1, len(program) + 2):
        for name in stages:
            produced = passes[name](current)
            after = _opcodes(produced)
            if after != _opcodes(current):
                trace.append(
                    {
                        "round": round_index,
                        "pass": name,
                        "opcodes_before": _opcodes(current),
                        "opcodes_after": after,
                    }
                )
            current = produced
        if _opcodes(current) == previous:
            break
        previous = _opcodes(current)
    return trace


def mechanism_table(
    population_totals: dict[tuple[str, int], tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Each mechanism in isolation, on both ports at both implemented levels.

    The port's rows are always measured. Qiskit's rows are measured when the anchor is
    importable and reported as absent otherwise, and the count of rows carrying a
    second-port reading is stated separately so a missing port cannot read as agreement.
    """

    rows: list[dict[str, Any]] = []
    for label in _MECHANISM_CASES:
        program = _mechanism_program(_MECHANISM_SPECS[label])
        row: dict[str, Any] = {
            "label": label,
            "program": list(_MECHANISM_SPECS[label]),
            "source_gate_count": len(program),
            "source_depth": program_depth(program),
        }
        for level in _LEVELS:
            optimized = optimize(program, optimization_level=level)
            row[f"port_gate_count_level_{level}"] = len(optimized)
            row[f"port_depth_level_{level}"] = program_depth(optimized)
            row[f"port_opcodes_level_{level}"] = _opcodes(optimized)
        rows.append(row)

    attribution: list[dict[str, Any]] = []
    for row in rows:
        program = _mechanism_program(tuple(row["program"]))
        for level in _LEVELS:
            trace = pass_trace(program, level)
            attribution.append(
                {
                    "label": row["label"],
                    "optimization_level": level,
                    "changed": bool(trace),
                    "trace": trace,
                }
            )

    # Every mechanism case must be convertible, because a case the second port cannot be
    # asked about would silently drop out of `comparisons` and the "port is level on N
    # readings" count would be a count over a smaller table.
    unconvertible: list[dict[str, Any]] = []
    for row in rows:
        try:
            _to_qiskit(_mechanism_program(tuple(row["program"])))
        except Exception as error:
            unconvertible.append(
                {"label": row["label"], "error_type": type(error).__name__}
            )

    try:
        import qiskit  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {
            "rows": rows,
            "row_count": len(rows),
            "rows_with_a_second_port_reading": 0,
            "qiskit_available": False,
            "reason": f"{type(error).__name__}: {error}",
            "comparisons": [],
            "attribution": attribution,
            "unconvertible_cases": unconvertible,
        }

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        circuit = _to_qiskit(_mechanism_program(tuple(row["program"])))
        for level in _LEVELS:
            transpiled = _transpile_to_compare(circuit, level)
            comparisons.append(
                {
                    "label": row["label"],
                    "optimization_level": level,
                    "port_gate_count": row[f"port_gate_count_level_{level}"],
                    "qiskit_gate_count": transpiled.size(),
                    "port_depth": row[f"port_depth_level_{level}"],
                    "qiskit_depth": transpiled.depth(),
                    "qiskit_opcodes": [item.operation.name for item in transpiled.data],
                    "gate_count_delta_port_minus_qiskit": (
                        row[f"port_gate_count_level_{level}"] - transpiled.size()
                    ),
                    "depth_delta_port_minus_qiskit": (
                        row[f"port_depth_level_{level}"] - transpiled.depth()
                    ),
                }
            )
    return {
        "rows": rows,
        "row_count": len(rows),
        "rows_with_a_second_port_reading": len(rows),
        "qiskit_available": True,
        "qiskit_version": qiskit.__version__,
        "comparisons": comparisons,
        "attribution": attribution,
        "unconvertible_cases": unconvertible,
        "level_count": len(_LEVELS),
        "port_ahead_count": sum(
            1 for row in comparisons if row["gate_count_delta_port_minus_qiskit"] < 0
        ),
        "port_behind_count": sum(
            1 for row in comparisons if row["gate_count_delta_port_minus_qiskit"] > 0
        ),
        "port_agrees_count": sum(
            1 for row in comparisons if row["gate_count_delta_port_minus_qiskit"] == 0
        ),
        "never_changed_labels": sorted(
            {row["label"] for row in attribution if not row["changed"]}
        ),
        "changed_at_level_2_only": sorted(
            {
                row["label"]
                for row in attribution
                if row["optimization_level"] == 2 and row["changed"]
            }
            - {
                row["label"]
                for row in attribution
                if row["optimization_level"] == 1 and row["changed"]
            }
        ),
        "population_totals_unused": population_totals is None,
    }


# ---------------------------------------------------------------------------
# The refused level
# ---------------------------------------------------------------------------


def level_refusal() -> dict[str, Any]:
    """What each port does when asked for level 3, which is where they differ structurally.

    The port refuses and the refusal names the missing stage, so the reading is the
    exception's own text rather than a re-description of it. Qiskit accepts level 3 and,
    with no basis and no coupling map, produces the same reading it produces at level 2 --
    which is what makes the refusal a *capability* statement and not a claim that level 3
    has no optimization to offer.
    """

    program = _mechanism_program(_MECHANISM_SPECS["commuting_gap_merge"])
    try:
        optimize(program, optimization_level=_RESERVED_LEVEL)
    except Exception as error:
        port: dict[str, Any] = {
            "raises": True,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    else:  # pragma: no cover - the ladder's contract is that level 3 raises
        port = {"raises": False, "error_type": None, "message": None}

    try:
        transpiled = _transpile_to_compare(_to_qiskit(program), _RESERVED_LEVEL)
        twice = _transpile_to_compare(_to_qiskit(program), 2)
        anchor: dict[str, Any] = {
            "available": True,
            "gate_count": transpiled.size(),
            "depth": transpiled.depth(),
            "same_as_level_2": (
                transpiled.size() == twice.size() and transpiled.depth() == twice.depth()
            ),
        }
    except Exception as error:  # pragma: no cover - the anchor is optional
        anchor = {"available": False, "reason": f"{type(error).__name__}: {error}"}

    return {
        "reserved_level": _RESERVED_LEVEL,
        "declared_levels": list(_LEVELS) + list(RESERVED_OPTIMIZATION_LEVELS),
        "port": port,
        "port_stage_tuple_absent_for_the_reserved_level": (
            _RESERVED_LEVEL not in OPTIMIZATION_LEVEL_STAGES
        ),
        "anchor": anchor,
        "note": (
            "With no basis and no coupling map, level 3 has no work left that level 2 does "
            "not already do, so Qiskit's level-3 reading equalling its level-2 reading says "
            "nothing about the passes level 3 moves. The port's refusal is about the "
            "target-aware stage its ladder cannot contain, not about whether level 3 would "
            "help on this corpus."
        ),
    }


# ---------------------------------------------------------------------------
# The control: a smaller program is only a win if it is the same program
# ---------------------------------------------------------------------------


def _gate_only(ir: CircuitIR) -> CircuitIR:
    return CircuitIR(
        ir.n_wires,
        tuple(
            item for item in ir.instructions if item.name not in {"measure", "barrier"}
        ),
    )


def _statevector(ir: CircuitIR) -> torch.Tensor:
    from flagquantum.simulation.statevector.local import run_local_statevector

    return run_local_statevector(
        _gate_only(ir),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )[0]


def _outcome_distribution(ir: CircuitIR) -> dict[tuple[int, ...], float]:
    """The exact joint distribution over the measured wires, in classical-bit order.

    Wire 0 is the most significant bit, matching the statevector's basis-index
    convention, and the classical bits are ordered by their declared index rather than by
    the order the measurements appear -- the two differ in the two-wire branch of
    ``diagonal_before_measure``, which is exactly where a wrong ordering would be
    invisible.
    """

    probabilities = (_statevector(ir).abs() ** 2).detach().tolist()
    ordered: list[tuple[int, int]] = []
    for instruction in ir.instructions:
        if instruction.name == "measure":
            ordered.append(
                (instruction.wires[0], int(instruction.metadata.get("classical_bit", 0)))
            )
    ordered.sort(key=lambda pair: pair[1])
    width = ir.n_wires
    distribution: dict[tuple[int, ...], float] = {}
    for index, probability in enumerate(probabilities):
        key = tuple((index >> (width - 1 - wire)) & 1 for wire, _ in ordered)
        distribution[key] = distribution.get(key, 0.0) + probability
    return distribution


def execution_control() -> dict[str, Any]:
    """The port's own rewrite checked against an execution, per level and per population.

    Two differences are reported and they are different claims. The **statevector**
    difference is what a reader would check if the IR carried a global phase; the IR does
    not, so a rewrite that moves only a phase shows up here and is *not* a defect. The
    **outcome distribution** difference is what has to be zero for a program with
    measurements, because that is what the measurements report.

    Without both columns, a pass that changed the program could be counted as a gate
    saving -- which is the one way a count-only comparison goes wrong in the direction
    that flatters the port.
    """

    programs = corpus()
    rows: list[dict[str, Any]] = []
    for name in _POPULATIONS:
        for level in _LEVELS:
            worst_statevector = 0.0
            worst_distribution = 0.0
            measured_programs = 0
            for ir in programs[name]:
                source = _statevector(ir)
                optimized_ir = optimize(ir, optimization_level=level)
                worst_statevector = max(
                    worst_statevector,
                    float((source - _statevector(optimized_ir)).abs().max().item()),
                )
                if any(item.name == "measure" for item in ir.instructions):
                    measured_programs += 1
                    before = _outcome_distribution(ir)
                    after = _outcome_distribution(optimized_ir)
                    worst_distribution = max(
                        worst_distribution,
                        max(
                            (
                                abs(before.get(key, 0.0) - after.get(key, 0.0))
                                for key in set(before) | set(after)
                            ),
                            default=0.0,
                        ),
                    )
            rows.append(
                {
                    "population": name,
                    "optimization_level": level,
                    "measured_program_count": measured_programs,
                    "max_statevector_difference": worst_statevector,
                    "max_outcome_distribution_difference": worst_distribution,
                    "statevector_within_floor": worst_statevector <= _CONTROL_FLOOR,
                    "distribution_within_floor": worst_distribution <= _CONTROL_FLOOR,
                }
            )
    return {
        "statevector_floor": _STATEVECTOR_FLOOR,
        "distribution_floor": _DISTRIBUTION_FLOOR,
        "control_floor": _CONTROL_FLOOR,
        "rows": rows,
        "row_count": len(rows),
        "statevector_outside_floor": [
            [row["population"], row["optimization_level"]]
            for row in rows
            if not row["statevector_within_floor"]
        ],
        "distribution_outside_floor": [
            [row["population"], row["optimization_level"]]
            for row in rows
            if row["measured_program_count"] and not row["distribution_within_floor"]
        ],
        "max_statevector_difference": max(
            row["max_statevector_difference"] for row in rows
        ),
        "max_outcome_distribution_difference": max(
            row["max_outcome_distribution_difference"] for row in rows
        ),
    }


def phase_control() -> dict[str, Any]:
    """The one rewrite here that moves the statevector and must, with its magnitude.

    ``remove_diagonal_gates_before_measure`` is exact on the distribution and is not exact
    on the statevector, because a diagonal gate multiplies amplitudes by phases. Naming
    that as a measured row is what keeps the statevector column above from reading as a
    failure: the population where it is non-zero is exactly the population the pass exists
    for.
    """

    row = next(
        candidate
        for candidate in execution_control()["rows"]
        if candidate["population"] == "diagonal_before_measure"
        and candidate["optimization_level"] == 2
    )
    return {
        "population": "diagonal_before_measure",
        "optimization_level": 2,
        "max_statevector_difference": row["max_statevector_difference"],
        "max_outcome_distribution_difference": row[
            "max_outcome_distribution_difference"
        ],
        "statevector_moves_deliberately": (
            row["max_statevector_difference"] > _STATEVECTOR_FLOOR
        ),
        "distribution_still_exact": row["distribution_within_floor"],
    }


# ---------------------------------------------------------------------------
# Two ways the unit could have been chosen wrongly, measured
# ---------------------------------------------------------------------------


def conversion_coverage() -> dict[str, Any]:
    """Which declared single-qubit rotation opcodes the Qiskit exporter can lower.

    Measured because the corpus *depends* on the answer. ``u1`` and ``phase`` are both
    declared FlagQuantum opcodes with the same parameter, both are merged by the same
    ladder pass, and the exporter renames ``phase`` to Qiskit's ``p`` but has no entry for
    ``u1`` -- so a corpus that spelled its rotations ``u1`` would fail to convert and one
    that spelled them ``phase`` would not, and the gate-count comparison would silently
    depend on which spelling was picked.

    The check is per opcode rather than a claim about the exporter: one single-instruction
    program per declared rotation opcode, converted on its own, with the outcome recorded.
    Whether ``u1`` should gain a lowering is a separate proposal; this benchmark records
    that it has none and does not rest on it.
    """

    try:
        from flagquantum.ecosystem.qiskit.conversion import to_qiskit
    except Exception as error:  # pragma: no cover - qiskit is optional
        return {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
            "rows": [],
            "unlowered_opcodes": [],
        }

    rows: list[dict[str, Any]] = []
    for opcode in _DECLARED_SINGLE_QUBIT_ROTATIONS:
        program = CircuitIR(_REGISTER_WIDTH, (_instruction(opcode, (0,), theta=0.5),))
        try:
            to_qiskit(program)
        except Exception as error:
            rows.append(
                {
                    "opcode": opcode,
                    "has_a_qiskit_lowering": False,
                    "error_type": type(error).__name__,
                }
            )
        else:
            rows.append(
                {"opcode": opcode, "has_a_qiskit_lowering": True, "error_type": None}
            )
    unlowered = [row["opcode"] for row in rows if not row["has_a_qiskit_lowering"]]
    return {
        "available": True,
        "rows": rows,
        "row_count": len(rows),
        "unlowered_opcodes": unlowered,
        "corpus_rotation_family": list(_ROTATIONS),
        "corpus_avoids_unlowered": not (set(_ROTATIONS) & set(unlowered)),
        "note": (
            "The corpus uses " + ", ".join(_ROTATIONS) + " because every member has a "
            "lowering; a corpus on " + ", ".join(unlowered) + " would not be askable of "
            "the second port at all."
        ),
    }


def measurement_node_cost() -> dict[str, Any]:
    """How much the two measurement representations differ, in the unit compared.

    Reported because the choice was made rather than forced: a corpus that put its
    measurements in ``ir.measurements`` would report the same optimized program with a
    smaller gate count and a smaller depth, and the difference would be a property of the
    IR's shape and not of the optimization. The size of that difference is the number
    below, so a reader can see what the convention is worth.
    """

    population = corpus()["diagonal_before_measure"]
    program = next(
        (
            ir
            for ir in population
            if any(item.name == "measure" for item in ir.instructions)
        ),
        None,
    )
    if program is None:  # pragma: no cover - the population always measures
        return {"available": False}
    gate_only = CircuitIR(
        program.n_wires,
        tuple(item for item in program.instructions if item.name != "measure"),
    )
    return {
        "available": True,
        "program": [
            f"{item.name}:{','.join(str(wire) for wire in item.wires)}"
            for item in program.instructions
        ],
        "instruction_form_gate_count": len(program),
        "node_form_gate_count": len(gate_only),
        "measurement_count": len(program) - len(gate_only),
        "instruction_form_depth": program_depth(program),
        "node_form_depth": program_depth(gate_only),
        "note": (
            "One program, measured both ways. The node form is smaller by exactly its "
            "measurement count and shallower by its measurement depth, which is why mixing "
            "the two representations across the two ports would produce a difference with "
            "no cause."
        ),
    }


# ---------------------------------------------------------------------------
# What the numbers are not
# ---------------------------------------------------------------------------


def boundary() -> dict[str, Any]:
    """The scope statement, as data, so the contract test can hold it in place."""

    return {
        "basis_gates": None,
        "coupling_map": None,
        "layout_method": "trivial",
        "routing_method": "none",
        "unrolls_into_a_basis": False,
        "routes": False,
        "comparison_unit": "gate count and critical-path depth of the IR",
        "comparable_population": (
            "programs the target-independent ladder can be asked about at all; both ports "
            "are driven with no basis, so neither is asked what a device can run"
        ),
        "not_claimed": [
            "that either port is closer to a device's native gates",
            "that the port's gate counts are achievable on any target",
            "qiskit parity",
            "a performance gate or a scalability claim",
            "that level 3 would not help: the port refuses it and Qiskit's level 3 moves a "
            "target-aware stage this corpus cannot exercise",
            "that a smaller count in the port's column is an improvement; the execution "
            "control and the phase control are what make that a separate claim",
        ],
        "measurement_representation": (
            "every measurement is an instruction in both ports, because ir.measurements is "
            "a node form len(ir) cannot count while Qiskit's size() counts measure ops; the "
            "node form's cost is reported separately so the mismatch is measured rather "
            "than hidden"
        ),
    }


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    table = level_table()
    totals = level_totals(table)
    anchor = reference_anchor()
    # Measured once, here, and passed to both consumers: two callers measuring the same
    # corpus separately is how two reported numbers stop agreeing.
    population_totals = (
        _population_totals(corpus()) if anchor.get("available") else None
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": _NOW.isoformat(),
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "seed": {"sweep": _SWEEP_SEED},
        "reference_algorithm": "qiskit_transpile_preset_pass_manager",
        "reference_revision": (
            "qiskit 2.0.3 transpile(optimization_level=..., basis_gates=None, "
            "coupling_map=None, layout_method='trivial', routing_method='none')"
        ),
        "implemented_optimization_levels": list(_LEVELS),
        "reserved_optimization_levels": list(RESERVED_OPTIMIZATION_LEVELS),
        "depth_metric": {
            "function": "benchmarks.compiler_optimization_levels.program_depth",
            "conventions": depth_conventions(),
        },
        "corpus": {
            "populations": list(_POPULATIONS),
            "programs_per_population": _PROGRAMS_PER_POPULATION,
            "program_count": len(_POPULATIONS) * _PROGRAMS_PER_POPULATION,
            "register_width": _REGISTER_WIDTH,
            "dtype": _DTYPE,
        },
        "level_table": table,
        "level_totals": totals,
        "comparison": comparison(table, anchor, population_totals),
        "mechanism_table": mechanism_table(population_totals),
        "level_refusal": level_refusal(),
        "execution_control": execution_control(),
        "phase_control": phase_control(),
        "conversion_coverage": conversion_coverage(),
        "measurement_node_cost": measurement_node_cost(),
        "boundary": boundary(),
        "reference_anchor": {
            key: value for key, value in anchor.items() if not key.startswith("_")
        },
        # The anchor's calibration scope is stated only when the anchor exists: a missing
        # second port reported as a scope with zero mismatches would read as agreement.
        "anchor_row_count": 0,
        "anchor_agreement_scope": (
            "the second port was not importable, so no cross-port calibration was taken"
        ),
    }
    return _with_anchor_scope(payload)


def _with_anchor_scope(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        return payload
    rows = anchor["calibration_row_count"]
    payload["anchor_row_count"] = rows
    payload["anchor_agreement_scope"] = (
        f"{rows} corpus programs converted once and measured by both instruments; "
        f"program_depth equals QuantumCircuit.depth on "
        f"{rows - anchor['depth_mismatch_count']} of them and len(ir) equals "
        f"QuantumCircuit.size on {rows - anchor['gate_count_mismatch_count']} of them, so "
        "the two columns that are compared are the same instrument and not two"
    )
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"schema: {payload['schema']}")
    conventions = payload["depth_metric"]["conventions"]
    print(
        f"depth metric: {conventions['row_count']} conventions exercised, "
        f"{conventions['disagreement_count']} disagreeing"
    )
    anchor = payload["reference_anchor"]
    if anchor["available"]:
        print(
            f"calibration against qiskit {anchor['qiskit_version']}: "
            f"{anchor['calibration_row_count']} programs, "
            f"{anchor['unconvertible_program_count']} unconvertible, depth mismatches "
            f"{anchor['depth_mismatch_count']}, gate-count mismatches "
            f"{anchor['gate_count_mismatch_count']}, unaskable conventions "
            f"{len(anchor['unaskable_convention_labels'])}"
        )
    else:
        print(f"no second port: {anchor['reason']}")
    print()

    source = payload["level_totals"]["source_totals"]
    print(
        f"source: {source['gate_count']} gates over depth {source['depth']} "
        f"({payload['corpus']['program_count']} programs)"
    )
    print(
        f"{'level':>5}  {'port gates':>10}  {'qiskit':>7}  {'delta':>6}  "
        f"{'port depth':>10}  {'qiskit':>7}  {'delta':>6}"
    )
    for row in payload["comparison"]["per_level"]:
        print(
            f"{row['optimization_level']:>5}  {row['port_gate_count']:>10}  "
            f"{row['qiskit_gate_count']:>7}  {row['gate_count_delta']:>+6}  "
            f"{row['port_depth']:>10}  {row['qiskit_depth']:>7}  "
            f"{row['depth_delta']:>+6}"
        )
    refusal = payload["level_refusal"]
    if refusal["anchor"].get("available"):
        print(
            f"{refusal['reserved_level']:>5}  {'refused by the port':>10}  "
            f"{refusal['anchor']['gate_count']:>7}  {'--':>6}  "
            f"{'--':>10}  {refusal['anchor']['depth']:>7}  {'--':>6}"
        )
    print()

    comparison = payload["comparison"]
    print(
        f"{comparison['population_level_readings_identical']} of "
        f"{comparison['population_level_readings']} population-level readings are "
        "identical in both ports; identical at every level: "
        + ", ".join(comparison["populations_identical_at_every_level"])
    )
    for row in comparison["per_population"]:
        if row["gate_count_delta"] == 0 and row["depth_delta"] == 0:
            continue
        print(
            f"  {row['population']:<32} level {row['optimization_level']}  "
            f"port {row['port_gate_count']}/{row['port_depth']}  "
            f"qiskit {row['qiskit_gate_count']}/{row['qiskit_depth']}  "
            f"delta {row['gate_count_delta']:+d}/{row['depth_delta']:+d}"
        )
    print()

    mechanisms = payload["mechanism_table"]
    print(f"{'mechanism':<40} {'port':>5} {'qiskit':>7} {'delta':>6}  changed at L2 by")
    for row in mechanisms["rows"]:
        matching = [
            item
            for item in mechanisms["comparisons"]
            if item["label"] == row["label"] and item["optimization_level"] == 2
        ]
        delta = (
            matching[0]["gate_count_delta_port_minus_qiskit"] if matching else None
        )
        qiskit_count = (
            matching[0]["qiskit_gate_count"]
            if matching
            else row["port_gate_count_level_2"]
        )
        trace = next(
            item["trace"]
            for item in mechanisms["attribution"]
            if item["label"] == row["label"] and item["optimization_level"] == 2
        )
        cause = ", ".join(item["pass"] for item in trace) or "no pass changed it"
        print(
            f"  {row['label']:<38} {row['port_gate_count_level_2']:>5} "
            f"{qiskit_count:>7} {delta:>+6}  {cause}"
            if delta is not None
            else f"  {row['label']:<38} {row['port_gate_count_level_2']:>5} "
            f"{'n/a':>7} {'n/a':>6}  {cause}"
        )
    if mechanisms["qiskit_available"]:
        print(
            f"  port ahead on {mechanisms['port_ahead_count']}, behind on "
            f"{mechanisms['port_behind_count']}, level on "
            f"{mechanisms['port_agrees_count']} of "
            f"{len(mechanisms['comparisons'])} level readings"
        )
    print()

    control = payload["execution_control"]
    print(
        f"execution control: worst statevector move "
        f"{control['max_statevector_difference']:.3e}, worst distribution move "
        f"{control['max_outcome_distribution_difference']:.3e} "
        f"(floors {control['statevector_floor']:.0e} / "
        f"{control['distribution_floor']:.0e})"
    )
    phase = payload["phase_control"]
    print(
        "phase control: diagonal-before-measure at level 2 moves the statevector by "
        f"{phase['max_statevector_difference']:.3e} and the distribution by "
        f"{phase['max_outcome_distribution_difference']:.3e}"
    )
    totals = payload["level_totals"]
    print(
        "irreducible control: unchanged at every level "
        f"{totals['irreducible_unchanged_at_every_level']}, readings "
        + ", ".join(
            f"{row['optimization_level']}:{row['gate_count']}/{row['depth']}"
            for row in totals["irreducible_readings"]
        )
    )
    coverage = payload["conversion_coverage"]
    if coverage.get("available"):
        print(
            "exporter coverage of the declared rotation opcodes: unlowered "
            f"{coverage['unlowered_opcodes']}, corpus avoids them "
            f"{coverage['corpus_avoids_unlowered']}"
        )
    node = payload["measurement_node_cost"]
    if node.get("available"):
        print(
            f"measurement representation: instruction form "
            f"{node['instruction_form_gate_count']}/{node['instruction_form_depth']}, "
            f"node form {node['node_form_gate_count']}/{node['node_form_depth']}"
        )
    print(
        f"level {refusal['reserved_level']}: port refuses "
        f"({refusal['port']['error_type']}), second port accepts and matches its own "
        f"level 2: {refusal['anchor'].get('same_as_level_2', 'n/a')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    _print_summary(payload)

    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
