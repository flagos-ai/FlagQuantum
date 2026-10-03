"""Logical-layer resource estimation: a Clifford+T tally and its surface-code cost.

:func:`estimate_logical_resources` answers the question a fault-tolerant plan is
costed with -- how many Clifford operations and how many T operations a program
applies, how many logical layers that is, and what a rotated surface code at a
chosen distance spends to run those layers: a physical qubit count, a
surface-code cycle count, and the product of the two.  It runs nothing.  Every
number it returns is a property of the program text and of the code model the
caller named, and the record says so.

**The gate-level tally is not re-derived here.**  Counting operations, their
dependency depth, the T family, and the register width is what
:func:`flagquantum.compiler.resource_estimation.estimate_resources` already
does, against the compiler's own scheduler, and it is the estimator the Trotter
example and its tests already read a product formula with.  This module reads
that record, classifies the opcodes it reports into the two families a
fault-tolerant cost is defined over, and adds the code layer on top.  It defines
no second depth rule and no second T rule: ``t_count`` and ``t_depth`` are the
compiler's numbers, passed through rather than recomputed, and the tests assert
that agreement against the compiler's own private T-family table so the two
cannot drift apart silently.

**The classification is total, and that is the point of it.**  A logical
resource estimate is defined over a Clifford+T program -- that is the premise
of every fault-tolerant cost model -- so an operation outside those two families
is refused by name rather than counted as "other".  Two kinds of operation are
exactly that, and neither is a rounding question.

A *parametric rotation* -- ``rx``, ``rz``, ``u3``, and their controlled forms --
is a gate whose Clifford+T form is angle synthesis's answer.  This unit reads
opcodes and runs no legalization, so it never asks that question: a quarter
turn does have an exact Clifford+T form now, in the compiler's own angle
synthesis, and this unit refuses the rotation carrying it anyway, because a
T-count taken over a rotation the program still contains would be the count of
a circuit nobody will run.  A *compound operation* -- a Toffoli, a controlled
swap -- is a gate whose Clifford+T form is its decomposition's answer, and the
decompositions a synthesis pipeline may pick do not agree on it.  A Toffoli is
**not** a Clifford gate, and counting one as a single Clifford operation would
understate the T-count of every logical program that contains one, which is the
single number a logical resource estimate exists to report.

So the unit reads *opcodes* and refuses both families outright, naming every
offending opcode and the reason that applies to it: a caller who means a T gate
writes the T gate, and the alternatives -- classifying a float angle against a
tolerance, or choosing a decomposition on the caller's behalf -- are decisions
about the caller's own synthesis pipeline rather than ones this unit can make.

**The code model is one named model, and its assumptions travel with its
numbers.**  A distance-``d`` rotated surface code spends ``2 d^2 - 1`` physical
qubits per logical qubit -- ``d^2`` data and ``d^2 - 1`` measure -- and one
logical layer costs ``d`` surface-code cycles, which is one round of syndrome
extraction.  Both are the model's own conventions rather than measured
quantities, so every report carries the list of statements its arithmetic rests
on, and the surface-code cycle count is named for what it is: the compiler's
schedule depth times the distance, which overestimates a compiled patch whose
operations fit inside a round.  No physical error rate is read and no logical
error rate is reported: a threshold fit's prefactor and threshold are a
device's numbers, and a report that invented them would be quoting a hardware
claim this module has no evidence for.

**What is deliberately absent, and is owned rather than silent.**  There is no
angle synthesis for an arbitrary angle, and this unit consumes no synthesis at
all, so the input has to already be Clifford+T; there is no distillation factory, magic-state budget, or routing
overhead, so the physical figure is a floor for a circuit of these layers rather
than a compiled estimate; there is no placement, scheduling, or layout pass, and
no device model, so the physical qubit count is a patch count rather than a
layout and no pitch, connectivity, or yield is modelled; and there is no logical
error rate, because that needs a device's error rate and this unit has none.
Each is recorded in the report's own ``limitations`` field.

The report's ``to_dict`` carries its statements in a ``capability_evidence``
block whose keys are the field names ``capability-maturity.toml`` requires of a
capability's own evidence, so a reader moving between a report and the registry
entry that owns it does not have to translate between two vocabularies, and a
key this unit invents fails the test that checks the block against the schema's
own level tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compiler.resource_estimation import ResourceEstimate, estimate_resources
from ..core.ir import ensure_circuit_ir
from ..errors import CapabilityError

LOGICAL_RESOURCE_BASIS = "clifford_t_tally_times_surface_code_distance"
# Names what a :class:`LogicalResourceReport` was computed from.
#
# A record carrying this value states that its operation counts are a property of
# the program text and that its footprint is arithmetic over a named code model.
# No execution, allocation, device query, or timing produced either, so neither is
# measured evidence about any hardware.

SURFACE_CODE_MODEL = "rotated_surface_code_2d"
# The one code model the footprint arithmetic is stated for.
#
# Named on every report because ``2 d^2 - 1`` and ``d`` cycles per logical layer
# are this model's own conventions.  A different code family -- a colour code, a
# repetition code, a qLDPC code -- spends differently, and a caller reading a
# physical qubit count has to be able to see which convention produced it.

CLIFFORD_OPCODES = frozenset(
    {
        "cx",
        "cy",
        "cz",
        "h",
        "i",
        "s",
        "sdg",
        "swap",
        "sx",
        "sxdg",
        "x",
        "y",
        "z",
    }
)
# The Clifford operations this repository's IR can apply, by opcode.
#
# ``ccx`` and ``cswap`` are deliberately **not** here.  A Toffoli is not a
# Clifford gate: it is a compound operation whose Clifford+T form is a
# decomposition's answer, and the decompositions disagree.  Counting one as a
# single Clifford operation would understate the T-count of every program that
# contains one, so it is refused in :data:`DECOMPOSITION_OPCODES` instead.
#
# Held here rather than on :class:`~flagquantum.core.operator_schema.OperatorSchema`
# because a schema marker would change the operator manifest and every generated
# capability document for the benefit of one reader, which is the same ground the
# compiler's T-family table is held on.  The tests assert that every name here is a
# declared opcode and that this set, the T family, the refused families, and the
# declared channels partition the whole schema, so the classification cannot
# silently fall behind a new opcode: an opcode no family here names is refused.

DECOMPOSITION_OPCODES = frozenset({"ccx", "cswap"})
# The compound operations whose Clifford+T form is a decomposition's answer.
#
# Refused rather than counted, and refused by name, because the T-count of a
# Toffoli is a property of the decomposition a synthesis pipeline picked rather
# than of the program this unit was handed.  A caller costing a logical adder
# decomposes its carries first and gets a number; a caller who leaves the
# compound gate in place gets this refusal instead of a plausible wrong figure.

PARAMETRIC_OPCODES = frozenset(
    {
        "cphase",
        "crx",
        "cry",
        "crz",
        "phase",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rz",
        "rzz",
        "u1",
        "u2",
        "u3",
    }
)
# The parameterised rotations, whose Clifford+T form is angle synthesis's answer.
#
# Named here so the partition is explicit rather than implied by omission: an
# opcode added to the schema lands in one of these families or in the channels,
# and the tests fail until it does.

T_FAMILY_OPCODES = frozenset({"t", "tdg"})
# The opcodes a T-count and a T-depth are defined over.
#
# The *classification* is restated here because this module has to decide whether an
# opcode is inside the Clifford+T program class at all; the *counts* are not.  They
# come from the compiler's estimator, which is the only implementation of what a
# T-depth means, and a test asserts this set is exactly the compiler's own.

CLIFFORD_T_OPCODES = CLIFFORD_OPCODES | T_FAMILY_OPCODES
# The opcodes a logical resource estimate is defined over, which is a closed set.
#
# Its complement over the declared unitaries is exactly
# ``DECOMPOSITION_OPCODES | PARAMETRIC_OPCODES``, and the tests assert that.

_LIMITATIONS = (
    "A count over a program's operation sequence and a footprint over a stated "
    "code model, and nothing measured: no wall-clock time, no resident memory, "
    "no allocation, and no fidelity. No physical error rate is read and no "
    "logical error rate is reported, because a threshold fit's prefactor and "
    "threshold are a device's numbers rather than this unit's, so this report "
    "says how much hardware a logical program would occupy and for how long, and "
    "never how often it would fail. The input has to already be Clifford+T, and "
    "two families are refused by name rather than counted: a parametric rotation "
    "is refused rather than synthesised, because this unit reads opcodes and runs "
    "no legalization, so a T-count taken over a rotation the program still "
    "contains would be the count of a circuit nobody will run; angle synthesis "
    "for an arbitrary angle is still absent here, and a compound operation whose "
    "T-count is its "
    "decomposition's -- a Toffoli, a controlled swap -- is refused because a "
    "Toffoli is not a Clifford gate and the decompositions do not agree on what "
    "it costs. No distillation factory, "
    "magic-state budget, or routing overhead is included, so the physical figure "
    "is a floor for a circuit of these layers rather than a compiled estimate. No "
    "placement, scheduling, or layout pass runs and no device is modelled, so the "
    "physical qubit count is a patch count: a device's pitch, connectivity, yield, "
    "and component choices are absent, and so is every code family other than the "
    "rotated surface code named on the report."
)


def surface_code_qubits_per_logical(distance: int) -> int:
    """Return the physical qubits one rotated surface-code patch spends.

    A distance-``d`` rotated surface code holds ``d**2`` data qubits and
    ``d**2 - 1`` measure qubits, so one logical qubit costs ``2 * d**2 - 1``
    physical qubits.  The figure is the model's own convention, not a
    measurement, and it is exposed on its own because tabulating several
    distances needs it without a program to estimate.

    Args:
        distance: The code distance, an odd integer of at least 3.

    Returns:
        The physical qubit count for one logical qubit.

    Raises:
        TypeError: ``distance`` is not an integer.
        ValueError: ``distance`` is below 3, or is even.

    Examples:
        Cost one logical qubit at three distances:

        >>> from flagquantum.algorithms.logical_resources import (
        ...     surface_code_qubits_per_logical,
        ... )
        >>> [surface_code_qubits_per_logical(distance) for distance in (3, 5, 7)]
        [17, 49, 97]
    """

    if isinstance(distance, bool) or not isinstance(distance, int):
        raise TypeError(f"the code distance must be an integer, got {distance!r}")
    if distance < 3:
        raise ValueError(
            f"the code distance must be at least 3, got {distance}: a "
            f"distance-1 patch carries no redundancy and corrects nothing, and a "
            f"distance-2 patch detects an error without correcting one"
        )
    if distance % 2 == 0:
        raise ValueError(
            f"the code distance must be odd, got {distance}: {SURFACE_CODE_MODEL} "
            f"defines its patch and its logical operators on odd distances, and an "
            f"even distance names a different patch than the one this arithmetic is "
            f"stated for"
        )
    return 2 * distance * distance - 1


@dataclass(frozen=True)
class LogicalResourceReport:
    """A Clifford+T tally plus the surface-code cost of running its layers.

    Every field is derived from either the program text or the code model, and
    the report carries the statements its arithmetic rests on rather than leaving
    them to be inferred.  ``estimate`` is the compiler's own
    :class:`~flagquantum.compiler.resource_estimation.ResourceEstimate` for the
    same program, nested rather than copied, so a reader who wants per-wire
    depths or the raw operation counts reads them from the record that owns them.

    ``n_qubits`` is how many logical qubits the footprint was charged for, which
    is the register width :attr:`estimate` reports unless the caller named a
    count: the two differ whenever a register carries ancillas that are not
    logical qubits.  ``logical_depth`` is the compiler's schedule depth plus one
    logical layer per measurement record, because a measurement is an IR-level
    record outside the instruction sequence the schedule covers.  ``surface_code_cycles`` is
    ``logical_depth`` times the code distance, ``physical_qubits`` is the patch
    count times the logical qubit count, and ``spacetime_volume`` is their
    product, in physical-qubit cycles.  The last of the three is the figure a
    fault-tolerant plan compares between two compilations of the same algorithm.

    ``assumptions`` and ``limitations`` are the report's own statement of what
    its numbers rest on and what they are not.  ``limitations`` is written under
    the field name ``capability-maturity.toml`` requires at every level, so the
    text a maturity entry carries and the text a caller reads here can be the
    same text.
    """

    basis: str
    estimate: ResourceEstimate
    clifford_count: int
    t_count: int
    t_depth: int
    n_measurements: int
    code_distance: int
    n_qubits: int
    logical_depth: int
    surface_code_cycles: int
    physical_qubits: int
    spacetime_volume: int
    assumptions: tuple[str, ...]
    limitations: str

    @property
    def n_clifford_t(self) -> int:
        """Return the number of operations the Clifford+T tally covers."""

        return self.clifford_count + self.t_count

    @property
    def physical_qubits_per_logical(self) -> int:
        """Return the patch size the physical qubit count was multiplied by."""

        return surface_code_qubits_per_logical(self.code_distance)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record that states the basis of the count.

        The ``capability_evidence`` block holds the report's statements under the
        field names ``capability-maturity.toml`` uses, and it is nested rather
        than flattened so a reader can tell this unit's evidence fields from the
        report's own arithmetic.
        """

        return {
            "kind": "flagquantum.logical_resource_report",
            "basis": self.basis,
            "surface_code_model": SURFACE_CODE_MODEL,
            "capability_evidence": {"limitations": self.limitations},
            "assumptions": list(self.assumptions),
            "estimate": self.estimate.to_dict(),
            "clifford_count": self.clifford_count,
            "t_count": self.t_count,
            "t_depth": self.t_depth,
            "n_clifford_t": self.n_clifford_t,
            "n_measurements": self.n_measurements,
            "code_distance": self.code_distance,
            "n_qubits": self.n_qubits,
            "logical_depth": self.logical_depth,
            "physical_qubits_per_logical": self.physical_qubits_per_logical,
            "surface_code_cycles": self.surface_code_cycles,
            "physical_qubits": self.physical_qubits,
            "spacetime_volume": self.spacetime_volume,
        }


def _refuse(estimate: ResourceEstimate, family: frozenset[str], reason: str) -> None:
    """Refuse a program that applies an opcode of one refused family.

    Every offending opcode of the family is named rather than the first, because a
    caller who wrote one rotation, or one Toffoli, usually wrote several and needs
    the whole list to act on.
    """

    offending = sorted(name for name in estimate.operation_counts if name in family)
    if not offending:
        return
    listed = ", ".join(repr(name) for name in offending)
    raise CapabilityError(
        f"a logical resource estimate is defined over Clifford+T programs, and this "
        f"one applies {listed}: {reason}"
    )


_DECOMPOSITION_REASON = (
    "a Toffoli or a controlled swap is a compound operation rather than a "
    "Clifford+T gate, and its T-count is whatever decomposition the caller's "
    "synthesis pipeline picked -- the decompositions do not agree, so the T-count "
    "of this program is not a number this unit can report. Decompose the gate into "
    "the Clifford and T gates you mean and the count follows: this unit reads "
    "opcodes rather than decompositions, because choosing a decomposition is a "
    "decision about your synthesis pipeline rather than one it can make for you"
)
# The reason a compound operation is refused, stated once and quoted in the refusal.

_PARAMETRIC_REASON = (
    "turning a parametric rotation into Clifford+T within an error is angle "
    "synthesis, which is not implemented for an arbitrary angle, so the T-count "
    "of this program is not a "
    "number this unit can report. Decompose the rotation into the Clifford+T gates "
    "you mean -- this unit classifies opcodes and does not classify angles, "
    "because deciding whether a float angle is a Clifford+T angle is a decision "
    "about your synthesis pipeline rather than one it can make for you"
)
# The reason a parameterised rotation is refused, stated once and quoted in it.


def _require_clifford_t(estimate: ResourceEstimate) -> None:
    """Refuse a program whose operation sequence is not Clifford+T.

    The decomposition family is checked first.  A program carrying both kinds is
    reported by its compound operations rather than by its rotations, which is a
    deliberate ordering rather than the program's own first offending opcode: a
    caller who wrote a Toffoli most likely believed its T-count was well defined,
    and that is the more useful of the two refusals to see first.  Either way the
    refusal is raised before any figure is computed, so nothing partial escapes.
    """

    _refuse(estimate, DECOMPOSITION_OPCODES, _DECOMPOSITION_REASON)
    _refuse(estimate, PARAMETRIC_OPCODES, _PARAMETRIC_REASON)


def estimate_logical_resources(
    program: Any, *, distance: int, n_qubits: int | None = None
) -> LogicalResourceReport:
    """Cost a Clifford+T program as a rotated surface-code footprint.

    Args:
        program: A :class:`~flagquantum.core.ir.CircuitIR` or any object exposing
            ``to_ir()``.  Its operation sequence has to be Clifford+T; unbound
            parameter angles need no substitution, because the operations a
            program applies do not depend on the values of its angles.
        distance: The code distance, an odd integer of at least 3.
        n_qubits: How many logical qubits to charge for.  Defaults to the
            program's declared register width, which is the honest default for a
            program that holds no ancillas; a caller whose register carries
            ancillas that are not logical qubits passes the logical count
            explicitly, and the assumptions on the report say so.

    Returns:
        A :class:`LogicalResourceReport` over the program's operation sequence
        and the named code model.

    Raises:
        CapabilityError: The program's operation sequence is data-dependent, or
            it carries a noise channel, or it applies an operation outside the
            Clifford and T families.
        TypeError: ``program`` is neither a ``CircuitIR`` nor an object exposing
            ``to_ir()``, or ``distance`` or ``n_qubits`` is not an integer.
        ValueError: ``distance`` is below 3 or even, or ``n_qubits`` is
            below 1.

    Examples:
        Cost a two-layer Clifford+T program at distance 5:

        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.logical_resources import (
        ...     estimate_logical_resources,
        ... )
        >>> circuit = fq.Circuit(2).h(0).t(0).cx(0, 1).t(1)
        >>> report = estimate_logical_resources(circuit, distance=5)
        >>> (
        ...     report.clifford_count,
        ...     report.t_count,
        ...     report.logical_depth,
        ...     report.surface_code_cycles,
        ... )
        (2, 2, 4, 20)
        >>> report.physical_qubits, report.spacetime_volume
        (98, 1960)
    """

    # The caller's own arguments are checked before the program is read, so a bad
    # distance is reported as a bad distance rather than as a property of a
    # program that was never the problem.
    per_logical = surface_code_qubits_per_logical(distance)

    ir = ensure_circuit_ir(program)
    estimate = estimate_resources(ir)
    if estimate.channel_count:
        raise CapabilityError(
            f"the program carries {estimate.channel_count} channel instruction(s), "
            f"and a channel is a physical-layer model rather than a logical "
            f"operation: a logical resource estimate is taken over the circuit "
            f"before a noise model is lowered onto it, because the surface-code "
            f"figure is derived from the circuit's layers and a lowered noise model "
            f"adds layers that no logical operation corresponds to"
        )
    _require_clifford_t(estimate)

    if n_qubits is None:
        n_qubits = estimate.n_qubits
    elif isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise TypeError(f"n_qubits must be an integer, got {n_qubits!r}")
    if n_qubits < 1:
        raise ValueError(
            f"the logical qubit count must be at least 1, got {n_qubits}: a "
            f"program that occupies no logical qubit has no footprint to report"
        )

    clifford_count = sum(
        count
        for name, count in estimate.operation_counts.items()
        if name in CLIFFORD_OPCODES
    )
    n_measurements = len(ir.measurements)
    # A measurement is an IR-level record rather than an instruction, so the
    # compiler's schedule does not cover it and its layer has to be added here.
    logical_depth = estimate.depth + (1 if n_measurements else 0)
    surface_code_cycles = logical_depth * distance
    physical_qubits = per_logical * n_qubits

    assumptions = (
        f"one logical layer is charged {distance} surface-code cycles, which is one "
        f"round of syndrome extraction on a distance-{distance} patch",
        f"one logical qubit is one {SURFACE_CODE_MODEL} patch of {per_logical} "
        f"physical qubits, and that model is the rotated surface code rather than "
        f"any hardware implementation of it",
        f"the {n_measurements} measurement record(s) the program carries are outside "
        f"the instruction sequence the schedule covers and each is charged one "
        f"logical layer, so a program with no measurement is charged none",
        f"the {n_qubits} logical qubit(s) are patches of the same distance, and "
        f"a register that holds ancillas which are not logical qubits is charged for "
        f"them unless the caller passes n_qubits",
        f"the {surface_code_cycles} surface-code cycles are the compiler's own "
        f"schedule depth times the distance, which overestimates a compiled patch "
        f"whose operations fit inside one round and includes no routing, distillation, "
        f"or magic-state overhead",
    )

    return LogicalResourceReport(
        basis=LOGICAL_RESOURCE_BASIS,
        estimate=estimate,
        clifford_count=clifford_count,
        t_count=estimate.t_count,
        t_depth=estimate.t_depth,
        n_measurements=n_measurements,
        code_distance=distance,
        n_qubits=n_qubits,
        logical_depth=logical_depth,
        surface_code_cycles=surface_code_cycles,
        physical_qubits=physical_qubits,
        spacetime_volume=physical_qubits * surface_code_cycles,
        assumptions=assumptions,
        limitations=_LIMITATIONS,
    )


__all__ = (
    "CLIFFORD_OPCODES",
    "CLIFFORD_T_OPCODES",
    "DECOMPOSITION_OPCODES",
    "LOGICAL_RESOURCE_BASIS",
    "PARAMETRIC_OPCODES",
    "SURFACE_CODE_MODEL",
    "T_FAMILY_OPCODES",
    "LogicalResourceReport",
    "estimate_logical_resources",
    "surface_code_qubits_per_logical",
)
