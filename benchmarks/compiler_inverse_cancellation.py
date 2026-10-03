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
  instruction is checked against an execution of the circuit. The pipeline also
  carries a same-wire single-qubit fold, `collapse_one_qubit_runs`, which reaches
  most of the shapes that motivated this pass. That fold is part of the baseline
  the delta is measured against rather than a pass this module may quietly omit,
  so what the delta reports is this pass's *marginal* reach and not a comparison
  against a pipeline that no longer exists: **zero** on the four populations the
  fold already reaches. Two of the populations frame a pair inside a half-pi pulse,
  because that is where the two passes do not subsume each other: folding a framed
  pair and cancelling the pair first are different answers, and the table reports
  which is shorter on each shape instead of claiming that one order dominates.
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
  cannot be asked is not an answer of "no". These columns are read off the pass
  itself rather than off the pipeline, because a pass can only be answerable for
  what it decides; the pipeline's own removals on these rows are a different
  question, answered by the delta above.

* **What the delta is measured in.** An instruction count is not the number that
  decides whether a compiler change is worth having, so each population is also
  lowered into one declared native basis (`rz` plus the half-pi pulse) and counted
  there. The two counts disagree in sign, and on the same shape: `sx s sdg` reaches
  `u3 rz` -- 2 instructions, 6 native gates -- without this pass and `sx` -- 1
  instruction, 1 native gate -- with it, while `sx s sdg sx` reaches one `u3` (4
  native gates) without the pass and `sx sx` (2 native gates) with it, which is a
  *longer* compiler program and a shorter native one. Both numbers are reported for
  that reason. A population the declared basis cannot express reports an absent count
  rather than a zero.

* **That the baselines are still the shipped pipeline.** The comparison pipelines
  below mirror `pipeline._optimize_to_fixed_point` by hand, because it cannot be
  asked for a variant of itself. A hand mirror drifts, and it drifts in the
  direction that flatters this pass: a pass added to the real pipeline and not to
  the mirror makes the baselines weaker and credits this pass with work another pass
  now does. So the mirror is driven against the real pipeline on every seeded
  population plus a reset program, and required to agree program for program. This
  is not hypothetical -- the merge that produced this version of the module added
  `remove_zero_state_resets` to the real pipeline while the mirror kept the old pass
  list, and this check is what caught it.

* **Which of the two passes should run first.** A third column re-runs each
  population with the one-qubit fold ahead of this pass, so the order chosen in
  `pipeline._optimize_to_fixed_point` is measured rather than assumed. It comes out
  as arithmetic: the fold is the baseline these deltas are measured against, so
  whichever of the two runs first takes the program and the fold-first order makes
  this pass's marginal delta zero on every population. The compiler-instruction
  totals are then equal -- the orders differ by 60 on two populations and in opposite
  directions -- and the native totals are not: **1320** native gates in the shipped
  order against **2026** fold first, over the five populations the declared basis can
  express. That is the whole of the argument for the shipped order, and it is the
  kind of argument that has to be re-measured rather than assumed.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch

from flagquantum.compiler.inverse_cancellation import inverse_pairs, merge_inverse_pairs
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    legalize_native_gates,
)
from flagquantum.compiler.one_qubit_optimization import collapse_one_qubit_runs
from flagquantum.compiler.one_qubit_synthesis import HALF_PI_PULSE_OPCODES
from flagquantum.compiler.pipeline import (
    _SELF_INVERSE,
    _optimize_to_fixed_point,
    merge_adjacent_rotations,
    merge_self_inverse,
    remove_identity_gates,
)
from flagquantum.compiler.zero_state_reset import remove_zero_state_resets
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import (
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
    inverse_operator,
)
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

SCHEMA = "flagquantum_compiler_inverse_cancellation_benchmark_v1"

_NOW = datetime(2026, 10, 8, 8, 0, tzinfo=timezone.utc)
COMPLEX = torch.complex128

#: Seeded so the counts below are reproducible, which is what lets them be pinned in
#: a benchmark contract.
_SWEEP_SEED = 20261008
_CIRCUIT_SEED = 20261013
_REGISTER_WIDTH = 3
_ANGLE = 0.7137

#: A reset and a measure are IR instructions rather than declared gates: the schema
#: does not know them, so the IR requires the metadata that says they read a draw.
_DYNAMIC = {"is_dynamic": True}

#: The operator comparison is exact, so it uses one absolute tolerance and no
#: relative slack. Two operators that differ are a shape that was not redundant.
_UNITARY_ATOL = 1.0e-12

#: Angle draws per opcode in the identity proof. The residual is exact, so this
#: number controls coverage rather than precision.
_DRAW_COUNT = 20

#: The opcodes the pass can act on, read from the pass rather than restated.
_PAIRS = dict(inverse_pairs())

#: The half-pi pulse a transcompiled run is written in, and the frame the two
#: populations below use to separate this pass's reach from the one-qubit fold's.
#: Read from the fold's own declaration rather than restated: the fold declines a
#: run already spelled in this vocabulary, so which opcode it is decides the shapes
#: below, and a second copy here could disagree with the pass being measured.
_PULSE = HALF_PI_PULSE_OPCODES[0]

#: The one target basis the native-gate column is measured in: a z-rotation and the
#: same half-pi pulse. It is the basis the two pulse populations are written in, so
#: a program this pass leaves alone is already native there, and it is declared here
#: rather than discovered because a gate count without its basis is not a number.
_NATIVE_Z_ROTATION = "rz"
_NATIVE_BASIS = (
    {"name": _NATIVE_Z_ROTATION, "parameters": ("theta",)},
    {"name": _PULSE, "parameters": ()},
)
_NATIVE_TARGET_ID = "inverse-cancellation-benchmark-target"


def _native_basis_snapshot() -> TargetCapabilitySnapshot:
    """A synthetic declared target whose only native gates are the measured basis.

    Synthetic because this module measures a compiler rewriting, not a device: the
    facts are declared and marked as such, and no provider is contacted.
    """

    scope = CapabilityScope(device_ids=(f"{_NATIVE_TARGET_ID}:0",))
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=_NATIVE_TARGET_ID,
            target_class="benchmark",
            provider="flagquantum.benchmark",
            provider_version="1",
            target_revision="1",
            environment_id="inverse-cancellation-benchmark",
        ),
        scope=scope,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=_NATIVE_BASIS,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=FactSource(kind="benchmark", ref="inverse-cancellation-basis"),
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="inverse-cancellation-basis",
                sha256="c" * 64,
                level=EvidenceLevel.BASIC,
                scope=scope,
            ),
        ),
    )


def _native_gate_count(ir: CircuitIR) -> int | None:
    """``ir``'s length after lowering into the declared basis, or None.

    None means the declared basis cannot express the program -- a `measure` is the
    case in these populations -- so the count is absent rather than zero. A zero
    would read as "this basis needs no gates for it", which is a different claim.
    """

    try:
        return len(
            legalize_native_gates(
                ir, snapshot=_native_basis_snapshot(), evaluated_at=_NOW
            ).program
        )
    except NativeGateLegalizationError:
        return None


def _native_column(count: int | None) -> str:
    """A native gate count for the printed table; an absent count stays visible."""

    return "n/a" if count is None else str(count)


def _add_native_count(total: int | None, ir: CircuitIR) -> int | None:
    """``total`` plus ``ir``'s native gate count, staying None once it is None.

    A population sums to None as soon as one of its circuits cannot be expressed in
    the declared basis, so the table never reports a partial total as if it were the
    population's own number.
    """

    if total is None:
        return None
    counted = _native_gate_count(ir)
    return None if counted is None else total + counted


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


#: The shipped optimization round, as an ordered list of passes.
#:
#: This mirrors `pipeline._optimize_to_fixed_point` rather than calling it, because
#: the comparison pipelines below have to be that pipeline with one pass moved or
#: removed, and there is no way to ask it for a variant of itself. Mirroring is
#: therefore unavoidable and also the exact hazard this module keeps running into: a
#: pass added to the real pipeline would silently be missing from the baselines, and
#: the delta would credit this pass with work another pass now does. So the mirror is
#: checked rather than trusted -- `_pipeline_as_shipped` applies this list and the
#: benchmark asserts it agrees with `pipeline._optimize_to_fixed_point` on every
#: seeded circuit, and that assertion is what fails the day the two diverge.
_PIPELINE_PASSES = (
    remove_zero_state_resets,
    remove_identity_gates,
    merge_self_inverse,
    merge_inverse_pairs,
    merge_adjacent_rotations,
    remove_identity_gates,
    collapse_one_qubit_runs,
)


def _run_rounds(ir: CircuitIR, passes: tuple[Any, ...]) -> CircuitIR:
    """Apply ``passes`` in order until the instruction count stops falling."""

    for _ in range(len(ir) + 1):
        previous = len(ir)
        for apply_pass in passes:
            ir = apply_pass(ir)
        if len(ir) == previous:
            return ir
    raise AssertionError("a comparison pipeline did not reach a fixed point")


def _pipeline_as_shipped(ir: CircuitIR) -> CircuitIR:
    """The pass list above applied to a program, for the fidelity check to compare."""

    return _run_rounds(ir, _PIPELINE_PASSES)


def _pipeline_without_this_pass(ir: CircuitIR) -> CircuitIR:
    """The pipeline with this pass removed, and nothing else changed.

    Every other pass in `pipeline._optimize_to_fixed_point` is applied, in its
    pipeline order, including `collapse_one_qubit_runs`. Leaving the fold out would
    have measured this pass against a pipeline the project no longer ships, and the
    delta would have credited it with removals the fold already performs on its own.
    """

    return _run_rounds(
        ir,
        tuple(
            apply_pass
            for apply_pass in _PIPELINE_PASSES
            if apply_pass is not merge_inverse_pairs
        ),
    )


def _fold_first_fixed_point(ir: CircuitIR) -> CircuitIR:
    """The same pipeline with the one-qubit fold at the head of the round.

    The alternative order, measured rather than argued about.
    ``_optimize_to_fixed_point`` runs this pass before ``merge_adjacent_rotations``
    and the fold last, so a framed run is decided by this pass and the fold then
    declines the half-pi vocabulary it leaves standing. Here the fold runs first in
    the round -- after the reset pass, which neither order is about and which the
    shipped pipeline also runs first -- so the same run is decided by the fold and
    this pass never sees it. Both orders still terminate in a fixed point, so the
    comparison is between two complete pipelines rather than between one that
    terminates and one that does not.
    """

    return _run_rounds(
        ir,
        (remove_zero_state_resets, collapse_one_qubit_runs)
        + tuple(
            apply_pass
            for apply_pass in _PIPELINE_PASSES
            if apply_pass is not collapse_one_qubit_runs
            and apply_pass is not remove_zero_state_resets
        ),
    )


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


def _pulse_framed_pair(seed: random.Random) -> CircuitIR:
    """A pair behind one half-pi pulse on its own wire.

    `collapse_one_qubit_runs` emits its own vocabulary and declines a run already
    written in the pulse basis, so a pair framed by a pulse is the shape where the
    fold and this pass disagree: folding the run and cancelling the pair first are
    different programs, and which is shorter is measured rather than assumed.
    """

    opcode = seed.choice(sorted(_PAIRS))
    qubit = seed.randrange(_REGISTER_WIDTH)
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(_PULSE, (qubit,)),
            Instruction(opcode, (qubit,)),
            Instruction(_PAIRS[opcode], (qubit,)),
        ),
    )


def _pulse_framed_pair_both_sides(seed: random.Random) -> CircuitIR:
    """The same pair with a pulse on *each* side, so the run's frame is closed."""

    opcode = seed.choice(sorted(_PAIRS))
    qubit = seed.randrange(_REGISTER_WIDTH)
    return CircuitIR(
        _REGISTER_WIDTH,
        (
            Instruction(_PULSE, (qubit,)),
            Instruction(opcode, (qubit,)),
            Instruction(_PAIRS[opcode], (qubit,)),
            Instruction(_PULSE, (qubit,)),
        ),
    )


def pipeline_delta() -> list[dict[str, Any]]:
    """The pipeline's own reach on the seeded populations, with and without the pass.

    ``removed_by_declared_inverse_pass`` is the shipped pipeline minus the pipeline
    without this pass: the marginal instruction count this pass contributes to the
    whole pipeline, not the count it would contribute to a pipeline that lacked the
    one-qubit fold. A negative value is a population where the shipped order reaches
    a *longer* compiler program than the fold alone, and it is reported as it stands.

    ``fold_first_instruction_count`` and its two siblings record the same
    populations through the opposite order -- the one-qubit fold ahead of this pass
    -- because the order of two passes that rewrite the same wire is a decision and
    a decision should be measured. That order makes this pass's marginal delta zero
    on every population, which is the expected result rather than a finding: the fold
    is the pipeline this pass is being measured against, so whoever runs first takes
    the program. The compiler-instruction totals are then equal (the two orders
    differ by 60 on two populations in opposite directions), and it is the native
    column that separates them: 1320 native gates shipped against 2026 fold first,
    over the five populations a declared basis can express. That is why the shipped
    order is the one in `pipeline._optimize_to_fixed_point`, and it is recorded here
    so the choice is a measurement and not a preference.
    """

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
        (
            "pair_behind_a_half_pi_pulse",
            [_pulse_framed_pair(seed) for _ in range(60)],
        ),
        (
            "pair_between_two_half_pi_pulses",
            [_pulse_framed_pair_both_sides(seed) for _ in range(60)],
        ),
    ]

    rows = []
    for label, circuits in populations:
        source = without_pass = with_pass = fold_first = 0
        native_without_pass: int | None = 0
        native_with_pass: int | None = 0
        native_fold_first: int | None = 0
        changed = 0
        executed = 0
        worst = 0.0
        for ir in circuits:
            source += len(ir)
            without_pass_ir = _pipeline_without_this_pass(ir)
            with_pass_ir = _optimize_to_fixed_point(ir)
            fold_first_ir = _fold_first_fixed_point(ir)
            without_pass += len(without_pass_ir)
            with_pass += len(with_pass_ir)
            fold_first += len(fold_first_ir)
            # The native counts are summed only while every circuit in the
            # population has one: a partial sum would be a number for a population
            # the basis cannot express, and None keeps that visible.
            native_without_pass = _add_native_count(
                native_without_pass, without_pass_ir
            )
            native_with_pass = _add_native_count(native_with_pass, with_pass_ir)
            native_fold_first = _add_native_count(native_fold_first, fold_first_ir)
            # The soundness check runs on every circuit and not only the changed
            # ones: a pass that removed nothing cannot be caught by comparing states.
            # It runs only where the program is gate-only, and the count of those is
            # reported, so a population that silently stopped being executable would
            # show up as a shrinking denominator rather than as a quiet pass.
            if _executable(ir) and _executable(with_pass_ir):
                executed += 1
                worst = max(
                    worst, float((_state(with_pass_ir) - _state(ir)).abs().max())
                )
            if len(with_pass_ir) != len(without_pass_ir):
                changed += 1
                if label == "pair_across_a_declared_barrier":
                    raise AssertionError("a declared barrier must stop the pass")
        rows.append(
            {
                "label": label,
                "circuit_count": len(circuits),
                "source_instruction_count": source,
                "without_pass_instruction_count": without_pass,
                "with_pass_instruction_count": with_pass,
                "removed_by_rest_of_pipeline": source - without_pass,
                "removed_by_declared_inverse_pass": without_pass - with_pass,
                "fold_first_instruction_count": fold_first,
                "removed_by_declared_inverse_pass_fold_first": without_pass
                - fold_first,
                "native_gate_count_without_pass": native_without_pass,
                "native_gate_count_with_pass": native_with_pass,
                "native_gate_count_fold_first": native_fold_first,
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

    ``removed`` is read off ``merge_inverse_pairs`` directly and not off
    `pipeline.optimize`. The question this table asks is where *this pass* stops, and
    only the pass can answer it: the pipeline's one-qubit fold also removes
    instructions on these shapes, and counting those here would credit this pass
    with reach the fold owns -- and, on the rows the fold does touch, would turn a
    deliberate decline into a removal.

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
            optimized = merge_inverse_pairs(ir)
            stripped = CircuitIR(_REGISTER_WIDTH, ir.instructions[1:-1])
            removable = None
            if optimized is ir:
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
                    "pass_instruction_count": len(optimized),
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


def pipeline_fidelity() -> dict[str, Any]:
    """Whether the mirrored pass list still is the pipeline the project ships.

    The comparison pipelines in this module mirror `pipeline._optimize_to_fixed_point`
    instead of calling it, and a mirror drifts: a pass added to the real pipeline and
    not to the mirror would quietly make the baselines weaker, and this pass would be
    credited with removals another pass now performs. So the mirror is driven here as
    well and compared against the real pipeline, program by program, on the same
    seeded populations the delta uses plus a reset program, so that a newly added pass
    has to be mirrored or this check fails and says so.
    """

    seed = random.Random(_CIRCUIT_SEED)
    programs: list[CircuitIR] = [
        CircuitIR(_REGISTER_WIDTH, tuple(_pair_sequence(seed, 4))) for _ in range(8)
    ]
    programs += [
        _prepared(_straddled_pair(seed).instructions),
        _prepared(_interleaved_pairs(seed).instructions),
        _prepared(_barrier_pair(seed).instructions),
        _pulse_framed_pair(seed),
        _pulse_framed_pair_both_sides(seed),
        # Reset programs, which is the pass the mirror gained last: a mirror that
        # omitted it would agree with the real pipeline on every population above and
        # disagree here. The second one needs a second round, because the wire is only
        # emptied of instructions by the self-inverse pass that runs after it.
        CircuitIR(_REGISTER_WIDTH, (Instruction("reset", (0,), metadata=_DYNAMIC),)),
        CircuitIR(
            _REGISTER_WIDTH,
            (
                Instruction("x", (0,)),
                Instruction("x", (0,)),
                Instruction("reset", (0,), metadata=_DYNAMIC),
            ),
        ),
    ]
    mismatches = [
        index
        for index, program in enumerate(programs)
        if _pipeline_as_shipped(program).instructions
        != _optimize_to_fixed_point(program).instructions
    ]
    return {
        "checked_program_count": len(programs),
        "mirrored_pass_count": len(_PIPELINE_PASSES),
        "mismatch_count": len(mismatches),
        "mismatch_indices": mismatches,
    }


def _order_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum the two pass orders over the populations both of them can be counted on.

    Only populations with a native count on both sides are summed: a basis that
    cannot express a circuit has no native number to add, and a total that quietly
    skipped such a population would read as a smaller program rather than as a
    missing measurement. ``covered_population_count`` says how many were summed, so
    the total is never read as covering more than it does.
    """

    covered = [
        row
        for row in rows
        if row["native_gate_count_with_pass"] is not None
        and row["native_gate_count_fold_first"] is not None
    ]
    return {
        "covered_population_count": len(covered),
        "population_count": len(rows),
        "native_gate_count_with_pass": sum(
            row["native_gate_count_with_pass"] for row in covered
        ),
        "native_gate_count_fold_first": sum(
            row["native_gate_count_fold_first"] for row in covered
        ),
        "instruction_count_with_pass": sum(
            row["with_pass_instruction_count"] for row in covered
        ),
        "instruction_count_fold_first": sum(
            row["fold_first_instruction_count"] for row in covered
        ),
    }


def run_benchmark() -> dict[str, Any]:
    shapes = shape_table()
    anchor = shapes["reference_anchor"]
    delta = pipeline_delta()
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
        "pipeline_fidelity": pipeline_fidelity(),
        "pipeline_delta": delta,
        "pass_order_totals": _order_totals(delta),
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
        f"{'population':32s} {'circuits':>8s} {'source':>7s} {'w/o pass':>8s} "
        f"{'with pass':>9s} {'delta':>6s} {'native w/o':>10s} {'native with':>11s} "
        f"{'changed':>8s} {'exec':>5s} {'max diff':>10s}"
    )
    for row in payload["pipeline_delta"]:
        print(
            f"{row['label']:32s} {row['circuit_count']:>8d} "
            f"{row['source_instruction_count']:>7d} "
            f"{row['without_pass_instruction_count']:>8d} "
            f"{row['with_pass_instruction_count']:>9d} "
            f"{row['removed_by_declared_inverse_pass']:>6d} "
            f"{_native_column(row['native_gate_count_without_pass']):>10s} "
            f"{_native_column(row['native_gate_count_with_pass']):>11s} "
            f"{row['changed_circuit_count']:>8d} "
            f"{row['executed_circuit_count']:>5d} "
            f"{row['max_state_difference']:>10.3e}"
        )
    print()
    print(
        "the same populations with the one-qubit fold moved ahead of this pass. Both "
        "orders are the whole pipeline; the shipped one is measured against the "
        "fold-first one:"
    )
    print(
        f"{'population':32s} {'delta shipped':>13s} {'delta fold first':>17s} "
        f"{'native shipped':>15s} {'native fold first':>18s}"
    )
    for row in payload["pipeline_delta"]:
        print(
            f"{row['label']:32s} "
            f"{row['removed_by_declared_inverse_pass']:>13d} "
            f"{row['removed_by_declared_inverse_pass_fold_first']:>17d} "
            f"{_native_column(row['native_gate_count_with_pass']):>15s} "
            f"{_native_column(row['native_gate_count_fold_first']):>18s}"
        )
    totals = payload["pass_order_totals"]
    print(
        f"  over the {totals['covered_population_count']} of "
        f"{totals['population_count']} populations the declared basis can express: "
        f"native gates {totals['native_gate_count_with_pass']} shipped vs "
        f"{totals['native_gate_count_fold_first']} fold first; compiler instructions "
        f"{totals['instruction_count_with_pass']} shipped vs "
        f"{totals['instruction_count_fold_first']} fold first"
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
