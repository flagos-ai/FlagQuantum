"""Clifford data regression: an affine map from noisy to ideal expectations.

CUDA-Q names three error-mitigation methods. Zero-noise extrapolation scales the
noise and extrapolates back to zero, probabilistic error cancellation inverts a
declared channel exactly, and Clifford data regression -- this unit -- fits the
relation between what a circuit *should* give and what the noise makes it give,
and applies that relation to the circuit under study.

The method has two halves. The first builds a *training set*: circuits that are
near the target and cheap where the target is not. The construction here snaps
each rotation of the target to the nearest multiple of a quarter turn and
rewrites it into named Clifford gates, so every training circuit is a Clifford
circuit the repository's own stabilizer engine accepts -- and that membership is
asked of that engine (:func:`~flagquantum.simulation.stabilizer.require_clifford_program`)
rather than restated as a second gate list. The second half measures each
training circuit twice, once on the exact ideal density matrix and once under the
declared noise model, fits ``ideal = slope * noisy + intercept`` over those
points by ordinary least squares, and applies the fitted line to the target's own
noisy expectation.

**Why the training circuits are Clifford, and what that does and does not buy.**
A Clifford circuit's ideal expectation is exactly computable without full state
vector simulation, and that is the reason the method can be run at all on a
device. In this repository the ideal values are read off the exact density
simulation of each Clifford training circuit, so the Clifford structure is
*enforced* -- a training circuit outside the Clifford vocabulary is refused --
but it is not *exploited*: no simulation cost is saved by it here, and no
capacity or scaling claim follows. The membership requirement is still the right
one, because it is what makes the training set a Clifford data set rather than an
arbitrary set of nearby circuits.

**What each training circuit changes.** Each rotation of the target contributes
one *site*. The first training circuit puts every site at its nearest quarter
turn, and each further training circuit puts exactly one site at that site's
other neighbouring quarter turn and leaves the rest where they were. So a target
with ``k`` rotations yields ``1 + k`` training circuits, each differing from the
first in exactly one angle, and each record carries the largest angle it moved --
the "nearby" in "near-Clifford" is a number on the record, not a word in the
prose.

**The method's assumptions, stated as obligations.** The relation the fit
describes is the one the target experiences, and the training circuits experience
the same noise the target does. The second obligation is where this construction
is most exposed, because a noise model attaches its channels to *gate names*: a
rotation rewritten into named Clifford gates stops matching a rule that named the
rotation. The default training set is therefore refused outright when the model
names an operation the rewrite removes, and a caller who wants a training set
built another way supplies ``variants`` and owns the claim that those circuits
receive the declared noise. Refusing is the fail-closed choice this repository
makes elsewhere for the same reason: a training set measured under less noise
than the target fits a correction for a difference that is not there.

**What is not here.** The estimate is a point value with no error bound and no
confidence interval, because this slice evaluates exact state expectations rather
than samples; nothing guarantees it is closer to the ideal value than the
unmitigated one. Gate and circuit folding is a separate unit beside this one,
:func:`flagquantum.algorithms.fold_program`, and readout-error mitigation is a
third, :func:`flagquantum.algorithms.plan_readout_mitigation`.

Only the single-parameter rotations ``rx``, ``ry``, ``rz``, ``phase`` and ``u1``
are snapped, and a program containing none of them is refused with that list,
because the only training circuit this construction can build from it is the
program itself. A program that does contain one is still refused if any other
operation leaves it outside the Clifford vocabulary the stabilizer engine
executes -- ``ccx``, a custom matrix, a noise channel -- and that refusal comes
from the engine that owns the vocabulary rather than from a second gate list
here. A program that declares a requested output -- a measurement node or an
observable node -- is refused too: this unit reads ``Tr(O rho)`` from the state
and consumes no request, so the estimate would be returned while the declared
request stayed silently unused.

Dense matrix mathematics lives in :mod:`flagquantum.simulation.density_matrix`
and the Clifford vocabulary in :mod:`flagquantum.simulation.stabilizer`; both are
reused here rather than restated. What this module owns is the training-set
construction, the affine fit, and the refusals that keep either from being read
as more than it is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..compiler.noise import lower_noise_model
from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..noise import NoiseModel
from ..simulation.density_matrix import density_matrix_from_ir
from ..simulation.stabilizer import require_clifford_program
from .core import Hamiltonian

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from ..circuit import Circuit

__all__ = (
    "CDR_ASSUMPTIONS",
    "CDR_LIMITATIONS",
    "CDR_SNAP_OPCODES",
    "CdrResult",
    "CliffordFit",
    "CliffordTrainingPoint",
    "CliffordVariant",
    "clifford_variants",
    "run_cdr",
)

#: A quarter turn, the angle grid a snapped rotation lands on. Every rotation
#: this unit rewrites is a multiple of this, and multiples of it are exactly the
#: single-qubit rotations the Clifford group contains.
_QUARTER_TURN = math.pi / 2.0

#: Opcode -> the named-Clifford word that opcode's rotation becomes at each
#: residue of ``k`` quarter turns, indexed by ``k % 4``.
#:
#: The table is measured, not quoted: for every opcode and every residue, the
#: word below reproduces that rotation's unitary to ``1.6e-16``, which is the
#: complex128 floor, up to a single global phase (``tests/unit/test_algorithms_cdr.py``
#: derives the whole table from the gate matrices and would fail if a matrix
#: convention moved). A global phase on the whole program leaves its density
#: matrix unchanged, so a training circuit is the snapped circuit exactly and not
#: merely up to an unobservable phase. Residue zero is the empty word because the
#: rotation is the identity there; the operation is dropped rather than emitted
#: as ``i``, so a site whose nearest quarter turn is zero disappears from the
#: training circuit.
#:
#: ``phase`` and ``u1`` are separate opcodes with the same quarter-turn ladder,
#: which is why the two rows are identical; keeping them apart states that the
#: ladder is a property of each opcode rather than an aliasing this module
#: invents.
_QUARTER_TURN_WORDS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "rx": ((), ("sx",), ("x",), ("sxdg",)),
    "ry": ((), ("s", "sxdg", "sdg"), ("y",), ("s", "sx", "sdg")),
    "rz": ((), ("s",), ("z",), ("sdg",)),
    "phase": ((), ("s",), ("z",), ("sdg",)),
    "u1": ((), ("s",), ("z",), ("sdg",)),
}

#: The opcodes :func:`clifford_variants` snaps, sorted.
CDR_SNAP_OPCODES: tuple[str, ...] = tuple(sorted(_QUARTER_TURN_WORDS))

#: The fewest training circuits the affine fit is defined on. Two points fix two
#: coefficients and leave no way for the line to miss either one, so a two-point
#: fit is reported with its residual absent rather than as an arithmetic zero;
#: fewer than two cannot be fitted at all.
_MINIMUM_TRAINING_POINTS = 2

#: The most training circuits one call may evaluate. Each point costs two exact
#: density simulations -- one ideal and one noisy -- so this is 128 simulations,
#: and a request beyond it is refused rather than run, because the caller cannot
#: have intended a training set that large. A target with more rotations than
#: this yields a larger default set and is refused with the count it wanted.
_MAXIMUM_TRAINING_POINTS = 64

#: How far apart the training circuits' noisy values must lie for a slope to be
#: determined. The slope is a ratio whose denominator is the spread of those
#: values, so a spread at the arithmetic floor yields a slope that is arithmetic
#: noise amplified by whatever the floor's reciprocal is. The floor a single
#: precision density simulation itself carries is of order 1e-7, so a spread
#: below 1e-9 is smaller than the noise the simulation introduces -- a fit on it
#: would describe the rounding rather than the model. The training sets in
#: ``tests/unit/test_algorithms_cdr.py`` separate by 1.1e-02 or more where the
#: model has effect and by exactly 0.0 where it does not.
_SEPARATION_TOLERANCE = 1e-9

#: What must hold for the fitted line to describe the target. These are the
#: method's obligations, not the unit's properties, and every result carries a
#: copy so a reader sees them where the number is.
CDR_ASSUMPTIONS: tuple[str, ...] = (
    "The training circuits and the target experience the same noise, so the "
    "relation fitted on the training circuits is the relation the target is "
    "under. The noise model is what states that process here, and nothing checks "
    "that the device agreed with the declaration.",
    "Over the region the training circuits occupy, the ideal expectation of the "
    "observable is an affine function of the noisy one. Clifford data regression "
    "fits a line and nothing else, so a relation that curves inside that region "
    "is fitted as a line, and the residual is the only diagnostic that exposes it.",
    "The observable is the same operator on every training circuit as on the "
    "target, and the training circuits exercise the wires its terms act on. A "
    "training set that leaves a measured wire untouched fits a map for a "
    "different measurement, and nothing in the fit detects that.",
)

#: The limitations every result carries.
CDR_LIMITATIONS: tuple[str, ...] = (
    "This unit evaluates exact state expectations, Tr(O rho). It consumes no "
    "shots, reports no confidence interval, and its estimate carries no measured "
    "uncertainty. Nothing here bounds the estimate's distance to the ideal "
    "value, and a fitted line can move the estimate further from it than the "
    "unmitigated value was.",
    "The ideal values come from the exact density simulation of each Clifford "
    "training circuit, not from the stabilizer representation, so the method's "
    "premise -- that a Clifford circuit is cheaply simulable -- is enforced here "
    "and not exploited. No simulation cost, capacity, or scaling claim follows "
    "from the training circuits being Clifford.",
    "A noise model attaches its channels to gate names, and the default training "
    "set rewrites rotations into named Clifford gates, so a rule that named a "
    "rewritten rotation stops matching it. The default set is refused when the "
    "model names an operation the rewrite removes, rather than trained under "
    "less noise than the target; a caller who needs it supplies variants and owns "
    "the claim that those circuits receive the declared noise.",
    "Readout-error mitigation is a separate unit beside this one -- "
    ":func:`flagquantum.algorithms.plan_readout_mitigation` inverts a declared "
    "classical confusion on a measured vector, after measurement -- and a model "
    "that declares a readout rule is refused rather than measured without it, "
    "because readout confusion is a classical "
    "misassignment applied after measurement, so it is not part of rho and this "
    "path cannot see it, while correcting a value that omits it would return a "
    "state-preparation estimate under the name of a measured one.",
    "Only the single-parameter rotations rx, ry, rz, phase and u1 are snapped to "
    "the quarter-turn grid, so a program containing none of them is refused "
    "rather than trained on itself. A program that declares a requested output "
    "-- a measurement node or an observable node -- is refused as well, because "
    "this unit reads Tr(O rho) from the state and consumes no request, and "
    "returning an estimate while the declared request went unused would report a "
    "mitigation of something other than what was asked for.",
    "The estimate is a point value from an affine fit. Zero-noise extrapolation "
    "is provided beside it by flagquantum.algorithms.run_zne and probabilistic "
    "error cancellation by flagquantum.algorithms.run_pec; neither is applied "
    "here, and neither a channel-parameter scale factor nor the gate and circuit "
    "folding of flagquantum.algorithms.fold_program is.",
)


def _finite_floats(values: Sequence[Any], *, owner: str) -> tuple[float, ...]:
    """Return ``values`` as floats, refusing anything that is not a finite real."""

    converted: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{owner} must be a real number, and a bool is not one")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{owner} must be a real number, got {value!r}") from error
        if not math.isfinite(number):
            raise ValueError(f"{owner} must be finite, and {number!r} is not")
        converted.append(number)
    return tuple(converted)


def _angle_of(instruction: Instruction) -> float:
    """Return the one angle a snappable rotation carries, or refuse the site."""

    if instruction.matrix is not None:
        raise ValueError(
            f"instruction {instruction.name!r} on wire(s) {instruction.wires} "
            "carries its own matrix, so it is whatever that matrix is and not the "
            "declared rotation the quarter-turn grid is defined on; a training "
            "circuit cannot be built by snapping an angle this operation does not "
            "have"
        )
    # The IR already refuses an angle that is not a finite real, so this read is a
    # cast rather than a second validation -- with one exception it deliberately
    # leaves open. A traced Parameter or ParameterExpression has no number yet and
    # is stored as it is, and the quarter-turn grid is not defined on a symbol.
    try:
        return float(instruction.params["theta"])
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"instruction {instruction.name!r} on wire(s) {instruction.wires} "
            f"declares theta = {instruction.params['theta']!r}, which is not a "
            "number; a rotation this unit cannot read has no nearest quarter "
            "turn, and the quarter-turn grid is not defined on an unbound "
            "parameter"
        ) from error


@dataclass(frozen=True, slots=True)
class _Site:
    """One rotation of the target, and the two quarter turns it may take."""

    index: int
    opcode: str
    wires: tuple[int, ...]
    angle: float
    nearest: int
    other: int

    @property
    def nearest_angle(self) -> float:
        return self.nearest * _QUARTER_TURN

    @property
    def other_angle(self) -> float:
        return self.other * _QUARTER_TURN

    @property
    def nearest_shift(self) -> float:
        return abs(self.angle - self.nearest_angle)

    @property
    def other_shift(self) -> float:
        return abs(self.angle - self.other_angle)


def _plan_sites(ir: CircuitIR) -> tuple[_Site, ...]:
    """Return one site per snappable rotation, in instruction order."""

    sites: list[_Site] = []
    for index, instruction in enumerate(ir.instructions):
        opcode = instruction.name
        if opcode not in _QUARTER_TURN_WORDS:
            continue
        if len(instruction.wires) != 1:
            raise ValueError(
                f"instruction {index} {opcode!r} names {len(instruction.wires)} "
                "wire(s), and every opcode with a quarter-turn ladder is a "
                "single-wire rotation"
            )
        angle = _angle_of(instruction)
        lo = math.floor(angle / _QUARTER_TURN)
        hi = lo + 1
        # Ties go to the lower turn so the rule is a function of the angle alone:
        # a half-turn boundary is equidistant, and choosing the upper one there
        # would make the nearest turn depend on the comparison's direction.
        nearest = lo if angle - lo * _QUARTER_TURN <= hi * _QUARTER_TURN - angle else hi
        other = hi if nearest == lo else lo
        sites.append(
            _Site(
                index=index,
                opcode=opcode,
                wires=instruction.wires,
                angle=angle,
                nearest=nearest,
                other=other,
            )
        )
    return tuple(sites)


def _require_no_request(ir: CircuitIR, *, owner: str) -> None:
    """Refuse a program that declares an output this unit will not consume."""

    if ir.measurements:
        kinds = ", ".join(sorted({node.kind for node in ir.measurements}))
        raise ValueError(
            f"{owner} declares measurement request(s) of kind {kinds}, and this "
            "unit reads the observable as Tr(O rho) from the state: the estimate "
            "would be returned while the declared request stayed silently "
            "unused, which is a mitigation of something other than what was "
            "asked for. Pass the program's unitary part alone to mitigate it."
        )
    if ir.observables:
        names = ", ".join(sorted({node.name for node in ir.observables}))
        raise ValueError(
            f"{owner} declares observable node(s) named {names}, and this unit "
            "reads the observable it was given as an argument instead: the "
            "declared nodes would be silently unused, so the estimate would "
            "describe a different observable than the program asked to be "
            "measured. Pass the program's unitary part alone and name the "
            "observable on the call."
        )


def _snapped(
    ir: CircuitIR, sites: tuple[_Site, ...], *, flipped: int | None
) -> CircuitIR:
    """Rebuild ``ir`` with every site snapped, and site ``flipped`` on its other turn."""

    chosen = {
        site.index: (site.other if flipped == position else site.nearest)
        for position, site in enumerate(sites)
    }
    instructions: list[Instruction] = []
    for index, instruction in enumerate(ir.instructions):
        if index not in chosen:
            instructions.append(instruction)
            continue
        word = _QUARTER_TURN_WORDS[instruction.name][chosen[index] % 4]
        for name in word:
            instructions.append(Instruction(name, instruction.wires))
    return CircuitIR(
        n_wires=ir.n_wires,
        instructions=tuple(instructions),
        dtype=ir.dtype,
        shape=ir.shape,
        metadata=ir.metadata,
    )


@dataclass(frozen=True, eq=False)
class CliffordVariant:
    """One Clifford training circuit and how far it moved from the target.

    Attributes:
        name: Which site this variant moved, or ``"nearest"`` for the variant
            that moved none of them beyond its nearest quarter turn.
        circuit: The training circuit, already refused if the stabilizer engine
            cannot hold it, so a variant that exists is a Clifford circuit.
        angle_shift: The largest angle change any site of this variant made,
            in radians. It is the distance from the target this variant sits at,
            and it is not zero for the first variant unless every rotation of the
            target was already on the quarter-turn grid.
    """

    name: str
    circuit: CircuitIR
    angle_shift: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a variant must be named, and its name is empty")
        if not isinstance(self.circuit, CircuitIR):
            raise TypeError("a variant needs the IR of its training circuit")
        _finite_floats((self.angle_shift,), owner="the angle shift")


def clifford_variants(circuit: Circuit | CircuitIR) -> tuple[CliffordVariant, ...]:
    """Return the default Clifford training circuits for ``circuit``.

    The first variant snaps every rotation to its nearest quarter turn. Each
    further variant moves exactly one rotation to that rotation's other
    neighbouring quarter turn, so a program with ``k`` snappable rotations yields
    ``1 + k`` variants and each carries the largest angle it moved.

    Args:
        circuit: The target program. Its rotations are read, not modified.

    Returns:
        The training circuits, in the order they were built, each validated by
        :func:`~flagquantum.simulation.stabilizer.require_clifford_program`.

    Raises:
        ValueError: if the program has no snappable rotation, if a rotation
            carries its own matrix, if the program declares a measurement or
            observable request, or if the training set would exceed the point
            ceiling.
        TypeError: if a rotation's angle is an unbound parameter rather than a
            number.
        CapabilityError: if the stabilizer engine cannot hold the snapped
            program, which is how a program containing an operation outside its
            Clifford vocabulary is refused.
    """

    ir = ensure_circuit_ir(circuit)
    _require_no_request(ir, owner="the target program")
    sites = _plan_sites(ir)
    if not sites:
        raise ValueError(
            "clifford data regression snaps a program's rotations to the "
            f"quarter-turn grid, and this program has none: it contains no "
            f"operation from {', '.join(CDR_SNAP_OPCODES)}, so the only training "
            "circuit this construction builds is the program itself and there is "
            "nothing to regress"
        )
    count = 1 + len(sites)
    if count > _MAXIMUM_TRAINING_POINTS:
        raise ValueError(
            f"the default training set for this program has {count} circuits, one "
            f"per rotation plus the all-nearest one, and this unit evaluates at "
            f"most {_MAXIMUM_TRAINING_POINTS}; pass an explicit training set to "
            "choose which circuits to spend the budget on"
        )
    variants: list[CliffordVariant] = []
    for position in range(-1, len(sites)):
        flipped = None if position < 0 else position
        variant_ir = _snapped(ir, sites, flipped=flipped)
        require_clifford_program(variant_ir)
        shifts = [
            (site.other_shift if flipped == index else site.nearest_shift)
            for index, site in enumerate(sites)
        ]
        name = (
            "nearest"
            if flipped is None
            else f"{sites[flipped].opcode}-at-instruction-{sites[flipped].index}"
        )
        variants.append(
            CliffordVariant(
                name=name,
                circuit=variant_ir,
                angle_shift=max(shifts, default=0.0),
            )
        )
    return tuple(variants)


@dataclass(frozen=True, slots=True)
class CliffordTrainingPoint:
    """One Clifford training circuit's ideal and noisy expectations.

    Attributes:
        name: The training circuit's name, as the variant record gives it.
        ideal: The observable's exact expectation on the training circuit with no
            noise, as ``Tr(O rho)``.
        noisy: The same expectation under the declared noise model.
    """

    name: str
    ideal: float
    noisy: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a training point must be named, and its name is empty")
        _finite_floats((self.ideal,), owner="the ideal expectation")
        _finite_floats((self.noisy,), owner="the noisy expectation")


@dataclass(frozen=True, slots=True)
class CliffordFit:
    """The affine map from a noisy expectation to an ideal one.

    Attributes:
        slope: The fitted coefficient of the noisy value.
        intercept: The fitted constant term.
        points: The training points the fit was taken over, in the order they
            were measured.
        noisy_spread: The largest minus the smallest noisy value. It is the
            denominator's square root: a fit on a spread near the arithmetic
            floor describes the rounding rather than the model, and a spread far
            below the model's own effect is refused before the fit is taken.
        max_residual: The largest absolute difference between a point's ideal
            value and the fitted line at its noisy value, or ``None`` when no
            degree of freedom is left. A two-point fit passes through both points
            by construction, and reporting its residual as zero would present an
            arithmetic identity as a check, so it is reported as absent instead.
        degrees_of_freedom: The number of points beyond the two the fit consumes,
            which is the number of ways it can fail to pass through the data.
    """

    slope: float
    intercept: float
    points: tuple[CliffordTrainingPoint, ...]
    noisy_spread: float
    max_residual: float | None
    degrees_of_freedom: int

    def __post_init__(self) -> None:
        _finite_floats((self.slope, self.intercept), owner="a fitted coefficient")
        points = tuple(self.points)
        if len(points) < _MINIMUM_TRAINING_POINTS:
            raise ValueError(
                f"an affine fit needs at least {_MINIMUM_TRAINING_POINTS} training "
                f"points, and {len(points)} were given"
            )
        if not all(isinstance(point, CliffordTrainingPoint) for point in points):
            raise TypeError("a fit's points must be CliffordTrainingPoint records")
        object.__setattr__(self, "points", points)
        spread = _finite_floats((self.noisy_spread,), owner="the noisy spread")[0]
        if spread < 0.0:
            raise ValueError(f"a spread cannot be negative, and {spread!r} is")
        object.__setattr__(self, "noisy_spread", spread)
        if self.max_residual is not None:
            residual = _finite_floats(
                (self.max_residual,), owner="the maximum residual"
            )[0]
            if residual < 0.0:
                raise ValueError(f"a residual cannot be negative, and {residual!r} is")
            object.__setattr__(self, "max_residual", residual)
        if self.degrees_of_freedom != len(points) - 2:
            raise ValueError(
                "a fit has one degree of freedom per point beyond the two it "
                f"consumes, so {len(points)} points leave "
                f"{len(points) - 2} and the record says {self.degrees_of_freedom}"
            )
        if (self.max_residual is None) != (self.degrees_of_freedom == 0):
            raise ValueError(
                "a residual is reported exactly when the fit has a degree of "
                "freedom to leave one, so a fit with "
                f"{self.degrees_of_freedom} degree(s) of freedom must "
                f"{'report' if self.degrees_of_freedom else 'not report'} one"
            )

    def predict(self, noisy: float) -> float:
        """Return the ideal value the fit assigns to a noisy expectation."""

        value = _finite_floats((noisy,), owner="the noisy expectation")[0]
        return self.slope * value + self.intercept


@dataclass(frozen=True, slots=True)
class CdrResult:
    """A mitigated estimate together with the fit and the caveats it rests on.

    Attributes:
        estimate: The fitted ideal value at the target's own noisy expectation.
        unmitigated: The target's noisy expectation. It is what the fit corrected
            and it is not a noiseless value, so the two are reported together.
        fit: The affine map, carrying its training points and its residual.
        variants: The training circuits' names, in the order they were measured.
        assumptions: What had to hold for the estimate to mean anything.
        limitations: What the estimate does not carry.
    """

    estimate: float
    unmitigated: float
    fit: CliffordFit
    variants: tuple[str, ...]
    assumptions: tuple[str, ...] = CDR_ASSUMPTIONS
    limitations: tuple[str, ...] = CDR_LIMITATIONS

    def __post_init__(self) -> None:
        _finite_floats(
            (self.estimate, self.unmitigated),
            owner="an expectation",
        )
        if not isinstance(self.fit, CliffordFit):
            raise TypeError("a result needs the fit its estimate came from")
        variants = tuple(self.variants)
        if len(variants) != len(self.fit.points):
            raise ValueError(
                f"a result names one training circuit per point, and "
                f"{len(self.fit.points)} points carry {len(variants)} name(s)"
            )
        if len(set(variants)) != len(variants):
            raise ValueError("training circuits must be named distinctly")
        object.__setattr__(self, "variants", variants)
        if not self.assumptions:
            raise ValueError("a result must carry the assumptions it rests on")
        if not self.limitations:
            raise ValueError("a result must carry its limitations")

    @property
    def correction(self) -> float:
        """Return how far the estimate moved from the unmitigated value."""

        return self.estimate - self.unmitigated


def _expectation(hamiltonian: Hamiltonian, density: torch.Tensor) -> float:
    """Read one expectation value off a density matrix."""

    values = torch.as_tensor(hamiltonian.expectation(density)).reshape(-1)
    if values.numel() != 1:
        raise ValueError(
            "Clifford data regression needs one expectation value per training "
            f"circuit, and the Hamiltonian returned {values.numel()}; run it at "
            "batch size one"
        )
    return float(values[0])


def _require_no_readout(noise_model: NoiseModel) -> None:
    if noise_model.readout_rules:
        raise ValueError(
            "the noise model declares a readout rule, and this unit evaluates the "
            "observable as Tr(O rho) before measurement, where classical readout "
            "confusion is not represented; measuring anyway would report a "
            "state-preparation estimate while the rule stayed silently unused, so "
            "the model is refused instead. Drop the readout rules to mitigate the "
            "state-preparation estimate, or mitigate readout separately."
        )


def _require_rewritten_names_survive(
    ir: CircuitIR, noise_model: NoiseModel, sites: tuple[_Site, ...]
) -> None:
    """Refuse a default training set that would receive less noise than the target."""

    if not sites:
        return
    rewritten = {site.opcode for site in sites}
    present = sorted(name for name in rewritten if any(i.name == name for i in ir))
    if not present:
        return
    declared = sorted(
        {
            name
            for rule in noise_model.rules
            for name in rule.gate_names
            if name in rewritten
        }
    )
    if not declared:
        return
    raise ValueError(
        "the noise model attaches noise after "
        f"{', '.join(declared)}, and the default training circuits replace exactly "
        "those operations with named Clifford gates, so every training point would "
        "be measured under less noise than the target and the line would correct "
        "for a difference that is not there. Name the model after operations the "
        "training circuits keep, or pass variants to supply training circuits that "
        "receive the declared noise."
    )


def _as_variant_irs(variants: Sequence[Circuit | CircuitIR]) -> tuple[CircuitIR, ...]:
    """Return the caller's training circuits as validated IRs."""

    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        raise TypeError(
            "the training set must be a sequence of circuits, and "
            f"{type(variants).__name__} is not one"
        )
    if not variants:
        raise ValueError(
            "the training set is empty, and an affine fit needs at least "
            f"{_MINIMUM_TRAINING_POINTS} circuits"
        )
    if len(variants) > _MAXIMUM_TRAINING_POINTS:
        raise ValueError(
            f"the training set has {len(variants)} circuits and this unit "
            f"evaluates at most {_MAXIMUM_TRAINING_POINTS}"
        )
    return tuple(ensure_circuit_ir(variant) for variant in variants)


def run_cdr(
    circuit: Circuit | CircuitIR,
    hamiltonian: Hamiltonian,
    *,
    noise_model: NoiseModel,
    variants: Sequence[Circuit | CircuitIR] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> CdrResult:
    """Fit an affine noisy-to-ideal map on Clifford circuits and apply it here.

    Args:
        circuit: The program whose observable is mitigated. It is executed
            unchanged once, so the value the fit corrects is the one the program
            produces and not a value from a rewritten program.
        hamiltonian: The observable, evaluated exactly on each circuit's density
            matrix. Its expectation is a single number, so the batch size is one.
        noise_model: The model every circuit is measured under. It must not
            declare a readout rule: the observable is read as ``Tr(O rho)``,
            which is before measurement, so a classical readout confusion would
            be silently unused.
        variants: The Clifford training circuits. Omitting it builds the default
            set with :func:`clifford_variants` from the target's own rotations.
            Supplying it takes over the claim that those circuits receive the
            noise the model declares, which is why the default set refuses a
            model that names an operation the rewrite removes. Every supplied
            circuit is still required to be a Clifford circuit.
        device: The device the density simulations run on.
        dtype: The density simulations' complex dtype. The default is the
            runtime configuration's, which is single precision; the fit's own
            arithmetic is double precision regardless.

    Returns:
        The estimate, the target's unmitigated value, the fit, and the
        assumptions and limitations that go with them.

    Raises:
        ValueError: if the model declares a readout rule, if the target or a
            training circuit declares a measurement or observable request, if
            the default training set would receive less noise than the target,
            if the training points do not separate enough to determine a slope,
            or if the default set cannot be built. All of it is checked before
            the first simulation runs, except the separation check, which needs
            the measured values.
        TypeError: if the observable, the model, or the training set is of the
            wrong type.
        CapabilityError: if a training circuit is not a Clifford circuit.
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("Clifford data regression needs a Hamiltonian observable")
    if not isinstance(noise_model, NoiseModel):
        raise TypeError("Clifford data regression needs a noise model to measure")
    _require_no_readout(noise_model)
    ir = ensure_circuit_ir(circuit)
    _require_no_request(ir, owner="the target program")

    if variants is None:
        sites = _plan_sites(ir)
        _require_rewritten_names_survive(ir, noise_model, sites)
        records = clifford_variants(ir)
        training_irs = tuple(record.circuit for record in records)
        names = tuple(record.name for record in records)
    else:
        training_irs = _as_variant_irs(variants)
        if len(training_irs) < _MINIMUM_TRAINING_POINTS:
            raise ValueError(
                "an affine fit needs at least "
                f"{_MINIMUM_TRAINING_POINTS} training circuits, and the training "
                f"set has {len(training_irs)}"
            )
        names = tuple(f"variant-{index}" for index in range(len(training_irs)))
    for training_ir in training_irs:
        _require_no_request(training_ir, owner="a training circuit")
        require_clifford_program(training_ir)

    points: list[CliffordTrainingPoint] = []
    for name, training_ir in zip(names, training_irs, strict=True):
        ideal_density = density_matrix_from_ir(
            training_ir, bsz=1, device=device, dtype=dtype
        )
        noisy_density = density_matrix_from_ir(
            _lowered(training_ir, noise_model), bsz=1, device=device, dtype=dtype
        )
        points.append(
            CliffordTrainingPoint(
                name=name,
                ideal=_expectation(hamiltonian, ideal_density),
                noisy=_expectation(hamiltonian, noisy_density),
            )
        )

    noisy_values = [point.noisy for point in points]
    spread = max(noisy_values) - min(noisy_values)
    scale = max(1.0, max(abs(value) for value in noisy_values))
    if spread <= _SEPARATION_TOLERANCE * scale:
        raise ValueError(
            "the training circuits do not separate under the noise model: their "
            f"noisy expectations span {spread:.3e}, which is at the arithmetic "
            f"floor rather than an effect of the noise, so no slope is "
            f"determined. The points span {scale:.3e} in magnitude, and a spread "
            f"below {_SEPARATION_TOLERANCE:.0e} of that is not evidence of "
            "anything."
        )

    fit = _fit(tuple(points), spread)
    unmitigated = _expectation(
        hamiltonian,
        density_matrix_from_ir(
            _lowered(ir, noise_model), bsz=1, device=device, dtype=dtype
        ),
    )
    return CdrResult(
        estimate=fit.predict(unmitigated),
        unmitigated=unmitigated,
        fit=fit,
        variants=names,
    )


def _lowered(ir: CircuitIR, noise_model: NoiseModel) -> CircuitIR:
    """Return ``ir`` with the model's channels inserted."""

    return lower_noise_model(ir, noise_model)


def _fit(points: tuple[CliffordTrainingPoint, ...], spread: float) -> CliffordFit:
    """Least-squares the ideal values onto the noisy ones."""

    noisy = torch.tensor([point.noisy for point in points], dtype=torch.float64)
    ideal = torch.tensor([point.ideal for point in points], dtype=torch.float64)
    centred = noisy - noisy.mean()
    variance = float((centred * centred).sum())
    slope = float((centred * (ideal - ideal.mean())).sum() / variance)
    intercept = float(ideal.mean()) - slope * float(noisy.mean())
    degrees = len(points) - 2
    residual: float | None = None
    if degrees:
        predicted = slope * noisy + intercept
        residual = float((ideal - predicted).abs().max())
    return CliffordFit(
        slope=slope,
        intercept=intercept,
        points=points,
        noisy_spread=spread,
        max_residual=residual,
        degrees_of_freedom=degrees,
    )
