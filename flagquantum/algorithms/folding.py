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
    "FoldingPlan",
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
