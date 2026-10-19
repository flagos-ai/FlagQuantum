"""Gate and circuit folding: realize a scale factor by lengthening the program.

Zero-noise extrapolation needs the same observable at several noise strengths,
and there are two ways to get one. The first, owned by
:mod:`flagquantum.algorithms.error_mitigation`, multiplies the one
error-probability parameter each channel declares and refuses by name the
channels whose parameters are probabilities of nothing. This unit is the
second: it leaves the noise model alone and makes the *program* longer, so every
instruction the model covers carries its noise more often while the ideal signal
does not move. Folding is what CUDA-Q's ``cudaq.zne`` does with its ``folding``
strategy, and it is the leg the error-mitigation unit named as absent.

**Why the ideal signal does not move.** Write the body as ``U`` and its inverse
as ``V``. The folded program is ``U`` followed by copies of ``V U``, and ``V U``
is the identity, so the copies cancel among themselves and ``U``'s own action is
untouched. The argument needs one property of ``U`` -- that it has an inverse --
and nothing else. The inverse is not restated here:
:meth:`flagquantum.circuit.Circuit.adjoint` reads each opcode's declared rule
from :data:`flagquantum.core.OPERATOR_SCHEMAS` and inverts a custom operation
through its own matrix, so this module owns no table of adjoints. A second one
would be a second source of truth for one fact, free to drift from the first.

**Why the realized factor is reported beside the requested one.** Both folds add
instructions two at a time and can only add them, so neither reaches an arbitrary
factor. ``gate`` folding replaces each instruction ``g`` with ``g`` followed by
its pairs, ``g V g V g``, distributing the pairs over the body leftmost first, so
its realized factor is ``1 + 2p/n`` for a body of ``n`` instructions and it
reaches the factors in between. ``circuit`` folding repeats the whole body,
``U (V U)^m``, so its realized factor is exactly ``1 + 2m`` and it never lands
between odd integers. A request of 2.0 on a five-instruction body therefore
realizes 2.2 under ``gate`` folding, because ten instructions cannot be added in
pairs to reach exactly ten, and 3.0 under ``circuit`` folding. The realized
factor, not the requested one, is the length ratio the noise saw, so it is the
abscissa the extrapolation is given and the value
:attr:`FoldingPlan.scale_factor` reports.

**What has been measured, rather than asserted.** Both folds are exact
identities. On a five-instruction two-qubit body at a factor of three the folded
program agrees with the body to 6.664e-08 in the largest amplitude difference at
``complex64``, which is that dtype's rounding floor, and it agrees exactly at
``complex128``. The same fold under a declared model moves the observable the way
a longer program should: with ``bit_flip`` at 0.1 on one ``x``, the
excited-state population reads 0.899999917, 0.755999804, 0.663839638 and
0.604857206 at realized factors one, three, five and seven, against the closed
form ``(1 - (1 - 2p)**m) / 2`` of 0.1, 0.244, 0.33616 and 0.3951424. The ideal
signal is unchanged and the noise is not.

**What is not here.** A scale factor below one is not offered: a fold can only
add instructions, and removing them would change the ideal signal rather than the
noise. Combining folding with channel-parameter scaling is not offered either,
because the two realize one strength in two different units -- a length ratio and
a channel parameter -- and their product names neither; a caller who wants both
applies one and passes the result to the other. Shot-based execution stays with
the executors and readout-error mitigation stays with
:func:`flagquantum.algorithms.plan_readout_mitigation`; this unit produces a
program and one record about it, and the extrapolation it feeds lives in
:mod:`flagquantum.algorithms.error_mitigation`.

**What the fold amplifies, measured rather than assumed.** A plan reports a length
ratio, and the strength a curve sees is not that ratio: the fold repeats
instructions, so it repeats the error of exactly the instructions the noise model
names. :func:`flagquantum.algorithms.fold_noise_amplification` is the reading that
distinguishes the two, and it counts channel *locations* -- one channel the model
applies at one instruction -- on the body and on the folded program. Counting
needs no arithmetic on a channel parameter, so it holds for a depolarizing
channel, a damping channel and a mixture alike, and it is exact rather than first
order. The cases it separates are the ones a caller cannot read off the plan. A
model that names only part of the body makes the real gain *below* the ratio. A
model that names the repeated instruction but not the inverse the fold inserts
makes it below the ratio too, because the inverse is a different gate name -- a
fold of ``t`` inserts ``t``, ``tdg``, ``t`` and a model naming only ``t`` reaches
two of the three instructions. And a gate fold, which spends its pairs on the
body's leading instructions, can make the gain *above* the ratio on the qubits the
pairs landed on while leaving the rest at one. Measured on the five-instruction
body ``h``, ``cx``, ``rz``, ``ry``, ``t``: with a model naming exactly ``h`` and
``cx``, a gate fold to a realized ratio of 1.4 amplifies qubit 0 by 2.0 and qubit 1
by 1.0, so the program is 1.6667 times as noisy where the ratio says 1.4 and no
single number is the strength, while the same request under a circuit fold
realizes 3.0 and amplifies both qubits by 3.0; with a model naming only ``h``, a
gate fold to 3.0 amplifies qubit 0 by 3.0, leaves qubit 1 silent, and reports four
of the five instructions uncovered. The reading names which of those a plan is in,
per qubit, and reports the instructions the model does not cover rather than
averaging them away.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..circuit import Circuit

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from ..core.ir import CircuitIR

__all__ = (
    "FOLDING_ASSUMPTIONS",
    "FOLDING_SCHEMA",
    "FOLDING_STRATEGIES",
    "FOLD_NOISE_SCHEMA",
    "FoldNoiseAmplification",
    "FoldingPlan",
    "fold_noise_amplification",
    "fold_program",
)

FoldingStrategy = Literal["gate", "circuit"]

#: The foldings this unit implements, in the order
#: :func:`flagquantum.algorithms.run_zne` accepts them. One vocabulary: the value
#: a caller passes is the value :attr:`FoldingPlan.strategy` reports.
FOLDING_STRATEGIES: tuple[FoldingStrategy, ...] = ("gate", "circuit")

#: The schema a folded plan's identity is taken under.
FOLDING_SCHEMA = "flagquantum.folding_plan.v1"

#: What folding adds to the extrapolation's own assumptions. These are premises
#: about the *scaling* rather than about the curve; the curve's premise is
#: :data:`flagquantum.algorithms.ZNE_ASSUMPTIONS` and applies unchanged, because a
#: fold changes which program is measured and not what kind of number the
#: measurement is.
FOLDING_ASSUMPTIONS: tuple[str, ...] = (
    "The scale factor is a length ratio rather than a channel parameter. The "
    "folded program is the body followed by an exact identity, so the ideal "
    "signal is the body's while every instruction the model covers carries its "
    "noise as many extra times as the fold repeats it. The abscissa is the ratio "
    "the fold realized, which is at least the one asked for and is not always "
    "equal to it.",
    "The fold scales the noise of every instruction the model covers and only of "
    "those. An instruction the model does not name is repeated without gaining "
    "any noise, so the strength the curve really sees is below the length ratio "
    "the plan reports; nothing here checks that the model covers the whole body, "
    "and a model that covers part of it is a curve whose slope is not the "
    "factor's.",
)


def _require_scale_factor(scale_factor: object) -> float:
    """Return ``scale_factor`` as a finite float at or above one.

    Raises:
        ValueError: if the value is not finite, or is below one, which no fold can
            realize because a fold only ever adds instructions.
    """

    try:
        number = float(scale_factor)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"a scale factor must be a number, and {scale_factor!r} was given"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"a scale factor must be finite, and {number!r} was given")
    if number < 1.0:
        raise ValueError(
            "a fold can only add instructions, so a scale factor below one would "
            "have to remove them and would change the ideal signal, and "
            f"{number!r} was given"
        )
    return number


def _pair_count(requested_length: float, body: int, *, strategy: str) -> int:
    """Return how many ``U^dagger U`` products reach ``requested_length``.

    A pair adds two instructions, so the lengths a body of ``body`` instructions
    can reach are ``body + 2p`` under ``gate`` folding and ``body * (1 + 2m)``
    under ``circuit`` folding. This returns the smallest count that reaches the
    requested length, or zero when the body already reaches it.

    The requested length is nudged one floating-point step down and the shortfall
    is nudged down again before it is rounded up, because a request that lands
    exactly on a reachable length can arrive a few steps above it in binary
    floating point -- ``1.4 * 5`` is ``7.000000000000001`` -- and rounding that up
    would add a pair nobody asked for.
    """

    needed = math.nextafter(requested_length, -math.inf)
    if strategy == "circuit":
        if needed <= body:
            return 0
        copies = (needed / body - 1.0) / 2.0
    else:
        copies = (needed - body) / 2.0
    return max(0, math.ceil(math.nextafter(copies, -math.inf)))


def _require_unitary(name: str, qubits: tuple[int, ...], matrix: Any) -> None:
    """Refuse a custom operation whose matrix is not unitary.

    The fold repeats an instruction with its own conjugate transpose, which
    cancels it only where ``U^dagger U`` is the identity. A matrix that is not
    unitary is accepted by the IR and inverted by
    :meth:`flagquantum.circuit.Circuit.adjoint`, so the fold would build a
    program that is a different ideal signal under the name of the original; the
    deviation is measured and named instead. The tolerance is scaled to the
    matrix's own largest entry and to the rounding of its real dtype, because a
    matrix that is unitary to that many steps is unitary for this purpose and one
    that is not is not.

    A channel instruction carries its Kraus operators rather than one matrix and
    is refused by the adjoint path instead, by name, because a channel has no
    unitary inverse; it is skipped by the caller rather than reinterpreted here,
    since its operator list has no conjugate transpose to check.
    """

    value = torch.as_tensor(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(
            f"the custom operation {name!r} on qubits {qubits} carries a matrix of "
            f"shape {tuple(value.shape)}, which is not square and so has no "
            "conjugate transpose to fold with"
        )
    residual = value.mH @ value
    identity = torch.eye(value.shape[-1], dtype=value.dtype, device=value.device)
    scale = max(1.0, float(value.abs().max()))
    tolerance = 64.0 * float(torch.finfo(value.real.dtype).eps) * scale
    deviation = float((residual - identity).abs().max())
    if deviation > tolerance:
        raise ValueError(
            f"the custom operation {name!r} on qubits {qubits} is not unitary: the "
            "fold repeats it with its own conjugate transpose, which cancels only "
            f"a unitary, and U^dagger U is {deviation:.3e} away from the identity "
            f"against a tolerance of {tolerance:.3e}. Folding it would return a "
            "different ideal signal under the name of the original program."
        )


def _plan_identity(
    strategy: str,
    requested: float,
    realized: float,
    body: int,
    folded: int,
    program_identity: str,
) -> str:
    """Return the content hash that names a plan's inputs and its output."""

    payload = json.dumps(
        {
            "schema": FOLDING_SCHEMA,
            "strategy": strategy,
            "requested_scale_factor": requested,
            "scale_factor": realized,
            "body_instructions": body,
            "folded_instructions": folded,
            "program_identity": program_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FoldingPlan:
    """One program folded to one realized scale factor.

    The plan is a value: it carries the program the fold built, the factor that
    was asked for, the factor the built program realizes, and the identity of
    exactly those inputs. Two requests differing only in a strategy or a factor
    have different identities.

    Attributes:
        strategy: Which fold produced the program, one of
            :data:`FOLDING_STRATEGIES`.
        requested_scale_factor: The factor the caller asked for.
        scale_factor: The factor the folded program realizes, which is the length
            ratio ``folded_instructions / body_instructions``. It is at least the
            requested one, and it is the abscissa an extrapolation is given
            because it is the ratio the noise saw rather than the one asked for.
        body_instructions: How many instructions the program had before folding.
        folded_instructions: How many it has after.
        added_pairs: How many ``U^dagger U`` products the fold inserted, which is
            the fold count a cost estimate is a function of.
        program: The folded program, as IR. Its declared observables and
            measurements are the body's, unchanged: a fold lengthens a program
            and does not change what is asked of its state.
        identity: ``sha256`` over the schema, the strategy, both factors, both
            instruction counts and the folded program's own content hash.
    """

    strategy: str
    requested_scale_factor: float
    scale_factor: float
    body_instructions: int
    folded_instructions: int
    added_pairs: int
    program: "CircuitIR"
    identity: str

    def __post_init__(self) -> None:
        if self.strategy not in FOLDING_STRATEGIES:
            raise ValueError(
                f"unsupported folding strategy {self.strategy!r}; expected one of "
                f"{', '.join(FOLDING_STRATEGIES)}"
            )
        requested = _require_scale_factor(self.requested_scale_factor)
        object.__setattr__(self, "requested_scale_factor", requested)
        realized = _require_scale_factor(self.scale_factor)
        object.__setattr__(self, "scale_factor", realized)
        if self.body_instructions < 1:
            raise ValueError(
                "a fold needs a body of at least one instruction, and "
                f"{self.body_instructions} was given; an empty program's scale "
                "factor is a ratio to zero"
            )
        if self.folded_instructions < self.body_instructions:
            raise ValueError(
                "a fold can only lengthen a program, and "
                f"{self.folded_instructions} instructions is fewer than the body's "
                f"{self.body_instructions}"
            )
        added = self.folded_instructions - self.body_instructions
        if added % 2 or self.added_pairs != added // 2:
            raise ValueError(
                "a fold adds instructions in pairs, so the difference between the "
                f"folded and body lengths must be twice added_pairs, and "
                f"{self.folded_instructions} - {self.body_instructions} is not "
                f"twice {self.added_pairs}"
            )
        ratio = self.folded_instructions / self.body_instructions
        if abs(realized - ratio) > 1e-12:
            raise ValueError(
                "the realized scale factor must be the folded program's length "
                f"ratio, {ratio!r}, and {realized!r} was given"
            )
        if realized < requested:
            raise ValueError(
                "a fold can only reach a scale factor at or above the one asked "
                f"for, and {realized!r} is below {requested!r}"
            )
        if len(self.program.instructions) != self.folded_instructions:
            raise ValueError(
                "the plan must carry the program it describes, and its program has "
                f"{len(self.program.instructions)} instructions where the plan "
                f"records {self.folded_instructions}"
            )
        if not self.identity:
            raise ValueError("a plan must carry the identity of its inputs")

    @property
    def program_identity(self) -> str:
        """The folded program's own content hash."""

        return self.program.content_hash

    @property
    def overshoot(self) -> float:
        """How far the realized factor is above the requested one."""

        return self.scale_factor - self.requested_scale_factor

    @property
    def exact(self) -> bool:
        """Whether the fold realized the factor that was asked for."""

        return self.scale_factor == self.requested_scale_factor

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as plain JSON-serializable data.

        The folded program is named by its content hash rather than embedded: it
        is an IR of its own, and a record whose point is *which* program was built
        does not need a second copy of it.
        """

        return {
            "schema": FOLDING_SCHEMA,
            "strategy": self.strategy,
            "requested_scale_factor": self.requested_scale_factor,
            "scale_factor": self.scale_factor,
            "body_instructions": self.body_instructions,
            "folded_instructions": self.folded_instructions,
            "added_pairs": self.added_pairs,
            "program_identity": self.program_identity,
            "identity": self.identity,
        }


def fold_program(
    program: Circuit | "CircuitIR",
    *,
    scale_factor: float,
    strategy: FoldingStrategy = "gate",
) -> FoldingPlan:
    """Return ``program`` folded to at least ``scale_factor`` times its length.

    The fold appends instructions and removes none, so the ideal signal is the
    body's and the noise is not: every instruction the model covers is executed
    as many extra times as the fold repeats it. ``gate`` folding replaces each
    instruction ``g`` with ``g`` followed by its pairs, ``g V g V g``, so one
    instruction's noise grows where that instruction is; ``circuit`` folding
    repeats the whole body, ``U (V U)^m``, so the noise grows uniformly and the
    realized factor is always an odd integer.

    Args:
        program: The program to fold, as a
            :class:`~flagquantum.circuit.Circuit` or a
            :class:`~flagquantum.core.ir.CircuitIR`. Its declared observables and
            measurements are carried into the folded program unchanged.
        scale_factor: How many times the body's instruction count to reach. It is
            a request rather than a promise: the realized factor is
            :attr:`FoldingPlan.scale_factor` and is at least this one.
        strategy: Which fold to use, one of :data:`FOLDING_STRATEGIES`.

    Returns:
        The plan, carrying the folded program, both factors, the fold count and
        the identity of all of them.

    Raises:
        TypeError: if ``program`` is neither a circuit nor an IR.
        ValueError: if the scale factor is not a finite number at or above one;
            if the strategy is not one of :data:`FOLDING_STRATEGIES`; if the
            program declares no instructions; or if a custom operation's matrix
            is not unitary, since such a program's fold would be a different
            ideal signal under the same name.
        CapabilityError: if an instruction has no inverse at all -- a noise
            channel, a mid-circuit measurement or reset, or a classically
            conditioned operation. The refusal names the instruction and the
            reason, and it is raised before any program is built.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms import fold_program
        >>> plan = fold_program(fq.Circuit(1).x(0), scale_factor=3.0)
        >>> plan.body_instructions, plan.folded_instructions, plan.scale_factor
        (1, 3, 3.0)
    """

    from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir

    if not isinstance(program, (Circuit, CircuitIR)):
        raise TypeError(
            "folding needs a Circuit or a CircuitIR, and "
            f"{type(program).__name__} was given"
        )
    if strategy not in FOLDING_STRATEGIES:
        raise ValueError(
            f"unsupported folding strategy {strategy!r}; expected one of "
            f"{', '.join(FOLDING_STRATEGIES)}"
        )
    factor = _require_scale_factor(scale_factor)
    body_ir = ensure_circuit_ir(program)
    body = tuple(body_ir.instructions)
    if not body:
        raise ValueError(
            "an empty program has no instruction to fold, so its scale factor "
            "would be a ratio to zero; fold a program with at least one "
            "instruction"
        )
    for instruction in body:
        # A channel carries its Kraus operators rather than one matrix and is
        # refused by the adjoint read below, by name.
        if instruction.matrix is not None and not instruction.metadata.get(
            "is_channel"
        ):
            _require_unitary(instruction.name, instruction.wires, instruction.matrix)

    # The adjoint is taken once for the whole body and read back in body order.
    # That is what keeps this module free of an adjoint table: every inverse is
    # the one ``Circuit.adjoint`` declares, read from OPERATOR_SCHEMAS or taken
    # from the instruction's own matrix, and a non-invertible instruction is
    # refused there by name before any program is built. It emits the inverses in
    # reverse, which the reversal here undoes.
    inverses = tuple(reversed(Circuit.from_ir(body_ir).adjoint().to_ir().instructions))
    if len(inverses) != len(body):
        raise ValueError(
            f"the adjoint of the body has {len(inverses)} instructions where the "
            f"body has {len(body)}, so each instruction's inverse cannot be read "
            "off it"
        )

    pairs = _pair_count(factor * len(body), len(body), strategy=strategy)
    sequence: list[Instruction] = []
    if strategy == "circuit":
        sequence.extend(body)
        for _ in range(pairs):
            sequence.extend(reversed(inverses))
            sequence.extend(body)
    else:
        # The pairs are distributed leftmost first, one instruction at a time, so
        # the folds sit where the body's own early instructions are and the
        # pattern does not depend on anything but the count.
        visits = [0] * len(body)
        for position in range(pairs):
            visits[position % len(body)] += 1
        for index, instruction in enumerate(body):
            sequence.append(instruction)
            for _ in range(visits[index]):
                sequence.append(inverses[index])
                sequence.append(instruction)

    folded = CircuitIR(
        n_wires=body_ir.n_wires,
        instructions=tuple(sequence),
        version=body_ir.version,
        dtype=body_ir.dtype,
        shape=body_ir.shape,
        observables=body_ir.observables,
        measurements=body_ir.measurements,
        metadata=body_ir.metadata,
    )
    realized = len(sequence) / len(body)
    return FoldingPlan(
        strategy=strategy,
        requested_scale_factor=factor,
        scale_factor=realized,
        body_instructions=len(body),
        folded_instructions=len(sequence),
        added_pairs=(len(sequence) - len(body)) // 2,
        program=folded,
        identity=_plan_identity(
            strategy,
            factor,
            realized,
            len(body),
            len(sequence),
            folded.content_hash,
        ),
    )


#: The schema a fold's noise reading is taken under.
FOLD_NOISE_SCHEMA = "flagquantum.fold_noise_amplification.v1"


def _channel_locations(
    program: "CircuitIR", noise_model: Any
) -> tuple[dict[int, int], int]:
    """Return the per-qubit and total count of channel applications on ``program``.

    A location is one channel the model applies at one instruction, so a rule
    matched on two qubits contributes two locations and a rule the model states
    on a qubit the instruction does not touch contributes none. The count is the
    unit a fold acts in: a fold repeats instructions, so it repeats exactly the
    channel applications those instructions carry, and no arithmetic on a channel
    parameter is involved.
    """

    per_qubit: dict[int, int] = {}
    for instruction in program.instructions:
        for _channel, wires in noise_model.channels_for(instruction):
            for qubit in wires:
                per_qubit[qubit] = per_qubit.get(qubit, 0) + 1
    return per_qubit, sum(per_qubit.values())


def _noise_identity(
    plan_identity: str,
    noise_identity: str,
    per_qubit: tuple[tuple[int, int, int], ...],
    covered: int,
    uncovered: int,
) -> str:
    """Return the content hash that names what a noise reading was taken over.

    The model's own identity is hashed and not only the counts: two models of the
    same shape and different probabilities apply the same number of channels, so a
    hash over counts alone would give two readings of two different noise strengths
    one name. The counts say *where* the fold's repeats landed, the model says what
    they were worth, and a reading names both or it names neither.
    """

    payload = json.dumps(
        {
            "schema": FOLD_NOISE_SCHEMA,
            "plan_identity": plan_identity,
            "noise_identity": noise_identity,
            "per_qubit": [list(entry) for entry in per_qubit],
            "covered_instructions": covered,
            "uncovered_instructions": uncovered,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FoldNoiseAmplification:
    """How much noise one fold added, and where the added noise landed.

    :func:`fold_program` returns a length ratio and takes no noise model, because
    a fold is a program transform and the ratio is a property of the program. The
    ratio is therefore *not* the strength the curve sees, and the two differ in a
    direction a caller cannot read off the plan: a fold repeats instructions, so
    it repeats the error of exactly the instructions the noise model names, and an
    instruction the model does not name is repeated with no added error at all.
    This record is that difference, measured against a model rather than assumed.

    The unit is a channel *location*: one channel the model applies at one
    instruction. It is the unit the fold acts in, so it needs no arithmetic on a
    channel parameter and no ordering between channels, and it is exact.

    Attributes:
        strategy: The strategy of the plan this reading was taken over.
        scale_factor: That plan's realized length ratio, so the reading can be
            compared with the ratio it is not.
        body_locations: Channel applications on the body.
        folded_locations: Channel applications on the folded program.
        covered_instructions: Body instructions the model applies at least one
            channel to.
        uncovered_instructions: Body instructions it applies none to. A fold
            repeats these without adding error, so this is the count
            :data:`FOLDING_ASSUMPTIONS` names as unchecked.
        per_qubit: One ``(qubit, body, folded)`` entry per qubit the model
            touches, in ascending qubit order. A qubit the fold did not reach has
            equal counts, which is a curve that cannot move on that qubit however
            many points it has.
        silent_qubits: The qubits the body's instructions act on that the model
            never applies a channel to, in ascending order. A readout supported
            only on these reads the same expectation at every scale factor
            because the fold adds no noise there, not because the fold failed.
        noise_identity: The noise model's own identity, so the reading names the
            model it was taken over even though it does not carry the model. Two
            models of the same shape and different probabilities apply the same
            number of channels, so the counts alone would not tell them apart.
        identity: ``sha256`` over the schema, the plan's own identity, the model's
            identity, the per-qubit counts and the coverage counts.
    """

    strategy: str
    scale_factor: float
    body_locations: int
    folded_locations: int
    covered_instructions: int
    uncovered_instructions: int
    per_qubit: tuple[tuple[int, int, int], ...]
    silent_qubits: tuple[int, ...]
    noise_identity: str
    identity: str

    def __post_init__(self) -> None:
        if self.strategy not in FOLDING_STRATEGIES:
            raise ValueError(
                f"unsupported folding strategy {self.strategy!r}; expected one of "
                f"{', '.join(FOLDING_STRATEGIES)}"
            )
        if self.body_locations < 1:
            raise ValueError(
                "a noise reading needs at least one channel application on the "
                f"body, and {self.body_locations} was given; a ratio with no noise "
                "in its denominator is not an amplification"
            )
        if self.folded_locations < self.body_locations:
            raise ValueError(
                "a fold can only repeat instructions, so it cannot apply a channel "
                f"fewer times than the body did, and {self.folded_locations} is "
                f"fewer than {self.body_locations}"
            )
        counted = sum(folded for _qubit, _body, folded in self.per_qubit)
        if counted != self.folded_locations:
            raise ValueError(
                "the per-qubit counts must add up to the folded total, and "
                f"{counted} does not equal {self.folded_locations}"
            )
        if not self.noise_identity:
            raise ValueError(
                "a noise reading must name the noise model it was taken over"
            )
        if not self.identity:
            raise ValueError("a noise reading must carry the identity of its inputs")

    @property
    def amplification(self) -> float:
        """The ratio of folded to body channel applications.

        It is at least one and it is not the plan's ``scale_factor``: the ratio is
        a length statement, this is a noise statement, and they agree only when
        the model names every instruction of the body and every inverse the fold
        inserts.
        """

        return self.folded_locations / self.body_locations

    def amplification_on(self, qubits: Any) -> float | None:
        """The same ratio restricted to ``qubits``, or ``None`` where there is none.

        This is the reading a caller with one observable wants, because a fold
        spends its pairs where the strategy puts them and not where the
        observable is sensitive. ``None`` means the model applies no channel to
        any of these qubits, so the fold amplifies nothing there and the curve is
        flat by construction rather than by measurement.

        Raises:
            ValueError: if ``qubits`` is not an iterable of qubit indices.
        """

        try:
            selected = {int(qubit) for qubit in qubits}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"qubits must be an iterable of qubit indices, and {qubits!r} was given"
            ) from exc
        body = sum(entry[1] for entry in self.per_qubit if entry[0] in selected)
        folded = sum(entry[2] for entry in self.per_qubit if entry[0] in selected)
        if body == 0:
            return None
        return folded / body

    @property
    def reached_uniformly(self) -> bool:
        """Whether every qubit the model touches gained noise by the same factor.

        A gate fold reaches the body's leading instructions first, so its pairs
        land on a subset of the qubits; a circuit fold repeats the whole body and
        so cannot be uneven. This is the property that decides whether one
        observable's curve is representative of another's.
        """

        ratios = {folded / body for _qubit, body, folded in self.per_qubit if body}
        return len(ratios) <= 1

    def to_dict(self) -> dict[str, Any]:
        """Return the reading as plain JSON-serializable data."""

        return {
            "schema": FOLD_NOISE_SCHEMA,
            "strategy": self.strategy,
            "scale_factor": self.scale_factor,
            "body_locations": self.body_locations,
            "folded_locations": self.folded_locations,
            "amplification": self.amplification,
            "covered_instructions": self.covered_instructions,
            "uncovered_instructions": self.uncovered_instructions,
            "per_qubit": [list(entry) for entry in self.per_qubit],
            "silent_qubits": list(self.silent_qubits),
            "reached_uniformly": self.reached_uniformly,
            "noise_identity": self.noise_identity,
            "identity": self.identity,
        }


def fold_noise_amplification(
    body: Circuit | "CircuitIR",
    plan: FoldingPlan,
    noise_model: Any,
) -> FoldNoiseAmplification:
    """Measure how much noise ``plan``'s fold added to ``body``, and where.

    A fold's own record states a length ratio, and the strength a zero-noise
    extrapolation sees is not that ratio: the fold repeats instructions, so it
    repeats the error of the instructions the model names and of no others.
    A model that names part of the body makes the real gain below the ratio, a
    model that names the repeated instruction but not the inverse the fold
    inserts makes it below the ratio too, and a model that names the instructions
    a gate fold happens to reach makes it above the ratio. This function states
    which case a caller is in by counting channel applications on both programs
    and reporting the ratio it measured rather than the ratio the plan claims.

    Counting is the unit because it needs no arithmetic on a channel parameter:
    it holds for a depolarizing channel, a damping channel and a mixture alike,
    and it is exact rather than first order.

    Args:
        body: The program that was folded, as a
            :class:`~flagquantum.circuit.Circuit` or a
            :class:`~flagquantum.core.ir.CircuitIR`.
        plan: The plan :func:`fold_program` returned for it.
        noise_model: The model whose channels the fold repeats, a
            :class:`~flagquantum.noise.NoiseModel`.

    Returns:
        The reading, carrying both location totals, the per-qubit counts, the
        coverage counts and the identity of all of them.

    Raises:
        TypeError: if ``body`` is neither a circuit nor an IR.
        ValueError: if ``body`` is empty; if the plan is not this body's fold,
            which is decided by re-deriving the fold and comparing identities
            rather than by trusting the recorded counts, at the cost of one more
            fold; or if the model applies a channel to no instruction of the
            body, because a ratio whose denominator is zero is not an
            amplification.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms import fold_noise_amplification, fold_program
        >>> from flagquantum.noise import NoiseModel, depolarizing_channel
        >>> body = fq.Circuit(2).h(0).h(1)
        >>> plan = fold_program(body, scale_factor=3.0)
        >>> model = NoiseModel().add("h", depolarizing_channel(0.03))
        >>> reading = fold_noise_amplification(body, plan, model)
        >>> reading.body_locations, reading.folded_locations, reading.amplification
        (2, 6, 3.0)
    """

    from ..core.ir import CircuitIR, ensure_circuit_ir

    if not isinstance(body, (Circuit, CircuitIR)):
        raise TypeError(
            "a noise reading needs the built body as a Circuit or a CircuitIR, and "
            f"{type(body).__name__} was given"
        )
    body_ir = ensure_circuit_ir(body)
    if not body_ir.instructions:
        raise ValueError(
            "an empty program applies no channel, so a fold of it amplifies no "
            "noise and the reading would be a ratio to zero"
        )
    # The plan is checked by re-deriving it rather than by comparing the counts it
    # records: a plan pasted from another body of the same length agrees on every
    # count and disagrees on the program, and the identity covers the program.
    derived = fold_program(
        body_ir,
        scale_factor=plan.requested_scale_factor,
        strategy=plan.strategy,  # type: ignore[arg-type]
    )
    if derived.identity != plan.identity:
        raise ValueError(
            "the plan is not this body's fold: re-deriving it at the requested "
            f"scale factor {plan.requested_scale_factor!r} under the "
            f"{plan.strategy!r} strategy returned identity {derived.identity[:12]} "
            f"where the plan carries {plan.identity[:12]}. A reading taken over a "
            "plan built from another program would report that program's noise "
            "under this one's name."
        )

    body_per_qubit, body_total = _channel_locations(body_ir, noise_model)
    folded_per_qubit, folded_total = _channel_locations(plan.program, noise_model)
    if body_total == 0:
        raise ValueError(
            "the noise model applies a channel to no instruction of the body, so "
            "the fold amplifies no noise and the ratio would be a division by "
            f"zero; the body has {len(body_ir.instructions)} instructions and the "
            "model names none of them"
        )

    per_qubit = tuple(
        (qubit, body_per_qubit.get(qubit, 0), folded_per_qubit.get(qubit, 0))
        for qubit in sorted(set(body_per_qubit) | set(folded_per_qubit))
    )
    covered = sum(
        1
        for instruction in body_ir.instructions
        if any(True for _ in noise_model.channels_for(instruction))
    )
    silent = tuple(
        sorted(
            {
                qubit
                for instruction in body_ir.instructions
                for qubit in instruction.wires
            }
            - set(body_per_qubit)
        )
    )
    return FoldNoiseAmplification(
        strategy=plan.strategy,
        scale_factor=plan.scale_factor,
        body_locations=body_total,
        folded_locations=folded_total,
        covered_instructions=covered,
        uncovered_instructions=len(body_ir.instructions) - covered,
        per_qubit=per_qubit,
        silent_qubits=silent,
        noise_identity=noise_model.identity,
        identity=_noise_identity(
            plan.identity,
            noise_model.identity,
            per_qubit,
            covered,
            len(body_ir.instructions) - covered,
        ),
    )
