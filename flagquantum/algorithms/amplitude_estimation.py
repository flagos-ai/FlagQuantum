"""Amplitude estimation by phase estimation over the Grover operator.

Quantum amplitude estimation for a state-preparation unitary ``A``, following Gilles
Brassard, Peter Høyer, Michele Mosca and Alain Tapp, "Quantum Amplitude Amplification and
Estimation", in *Quantum Computation and Information*, S. J. Lomonaco Jr. (ed.), AMS
Contemporary Mathematics vol. 305, pp. 53-74, 2002, DOI 10.1090/conm/305/05215,
arXiv:quant-ph/0005055. The paper generalizes Grover's iteration to an arbitrary
state-preparation unitary ``A`` and adds Shor-style phase estimation to it, so the
amplitude of the subspace a marking operator selects is read out of a counting register
rather than amplified until it is sampled.

**The premise is the state-preparation unitary, and it is not free.** The quadratic
speedup over classical sampling is a statement about the number of applications of ``A``
*and its adjoint*, against an ``A`` whose cost the query model does not count. A real
distribution needs QRAM to be loaded in, and this unit supplies none, so nothing here
shows that a Monte Carlo integral is estimated faster than classically: what the unit
demonstrates is the circuit and the readout, not a speedup.

**The readout is a phase, never the amplitude directly.** The two eigenphases of the
Grover operator ``Q`` are conjugate, so a counting outcome ``y`` of ``m`` counting wires
is a phase, and the amplitude is ``a = sin²(theta)`` with ``theta`` recovered from that
phase. Reading the register's value as an amplitude is a silently wrong answer rather
than an error, which is why :func:`maximum_likelihood_estimate` inverts the model below
instead.

**The likelihood model.** For ``m = n_counting_wires``, ``M = 2**m``, ``t = asin(sqrt(a))``
and a counting outcome ``y`` out of ``0 .. M - 1``::

    P(y | a) = (1 / (2 * M**2)) * [ sin²(M * t - pi * y) / sin²(t - pi * y / M)
                                  + sin²(M * t + pi * y) / sin²(t + pi * y / M) ]

with each term taken as ``M**2`` where its denominator vanishes, which is the limit of
``sin²(M * x) / sin²(x)`` at ``x = 0``. Both eigenphases contribute and both terms are
summed: a single-term model is wrong by a phase-dependent factor. The amplitude grid the
estimate is drawn from is ``a_j = sin²(pi * j / 2**(m+1))`` for ``j`` in ``0 .. 2**m``.

**The accuracy is a resolution, not an interval.** :func:`amplitude_resolution` is the
widest gap between adjacent grid amplitudes, and the estimate is accurate to about one
such step. That is a property of the grid, not a coverage-calibrated error bar: no
confidence interval is computed or reported anywhere in this module.

Sample keys are big-endian bit strings, one character per wire with wire 0 the most
significant, as :meth:`flagquantum.circuit.Circuit.counts` returns them; the counting
register's bits lead the string and the evaluation register's follow.

This unit is demonstration scale. It makes no performance, convergence, or hardware
claim, and it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..circuit import Circuit
from .primitives.phase_estimation import append_phase_estimation
from .primitives.types import AmplitudeOperator

__all__ = [
    "AmplitudeEstimationResult",
    "amplitude_estimation_circuit",
    "amplitude_resolution",
    "maximum_likelihood_estimate",
    "run_amplitude_estimation",
]

# A grid amplitude can give an observed outcome exactly zero probability -- at the ends of
# the range it does so for half the outcomes -- and ``log(0.0)`` raises rather than
# returning minus infinity. The floor sits far below any probability these widths can
# produce, so clamping to it never reorders two grid points that the model separates.
_LIKELIHOOD_FLOOR = 1e-300


class _GroverOperator:
    """``Q = -A (I - 2|0><0|) A^dagger S_chi``, in the primitive layer's controlled form.

    The primitive layer consumes a :class:`~flagquantum.algorithms.primitives.types.
    ControlledUnitary` and applies its controlled powers, so ``Q`` is presented as one
    instead of being re-derived there. Each factor is emitted by the operator's own
    protocol methods, and the leading ``-1`` becomes a ``z`` on the control wire: a
    controlled ``Q`` needs that phase only on the branch where it is applied, which is
    exactly what a ``z`` on the control supplies.

    Only the controlled power form is meaningful here. Every gate the amplitude operator
    exposes is controlled, and ``Q``'s ``-1`` is emitted on a control wire, so there is no
    wire-free form of ``Q`` to write: ``apply_controlled`` is the controlled form at its
    first power, and ``apply`` has no implementation and refuses. Phase estimation, the
    only consumer, takes the power form.
    """

    def __init__(self, operator: AmplitudeOperator) -> None:
        self._operator = operator

    @property
    def n_wires(self) -> int:
        """The number of wires the operator acts on."""
        return self._operator.n_wires

    def apply(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Refuse to append ``Q`` without a control wire.

        Raises:
            NotImplementedError: Always. See the class docstring: ``Q`` is written here in
                its controlled form only, and its factors have no uncontrolled emission.
        """
        raise NotImplementedError(
            "the Grover operator is only expressed here in its controlled power form; "
            "the amplitude operator's gates and the leading -1 both need a control wire"
        )

    def apply_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        """Append ``Q`` controlled on ``control``, its power form at exponent one.

        Args:
            circuit: The circuit to extend.
            control: The wire ``Q`` is controlled on.
            wires: The evaluation register's wires.
        """
        self.apply_power_controlled(circuit, control, wires, 1)

    def apply_power_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int], power: int
    ) -> None:
        """Append ``Q`` raised to ``power``, controlled on ``control``.

        One factor is emitted per power, in the order the product is written: the ``-1``,
        the marking operator, the adjoint of ``A``, the reflection about the zero state,
        and ``A``.

        Args:
            circuit: The circuit to extend.
            control: The wire ``Q`` is controlled on.
            wires: The evaluation register's wires.
            power: The exponent, at least zero.
        """
        for _ in range(power):
            circuit.gate("z", control)
            self._operator.apply_mark(circuit, control, wires)
            self._operator.apply_a_dagger(circuit, control, wires)
            self._operator.apply_zero_reflection(circuit, control, wires)
            self._operator.apply_a(circuit, control, wires)


@dataclass(frozen=True)
class AmplitudeEstimationResult:
    """The outcome of one sampled :func:`run_amplitude_estimation` run.

    Attributes:
        estimate: The amplitude the sample's most likely grid point carries, in ``[0, 1]``.
        resolution: The largest gap between adjacent grid amplitudes at this counting
            width. The estimate is accurate to about one such step; this is a resolution,
            not a coverage-calibrated confidence interval.
        n_counting_wires: The width of the counting register the estimate came from.
        n_evaluation_wires: The width of the operator's own register.
    """

    estimate: float
    resolution: float
    n_counting_wires: int
    n_evaluation_wires: int

    def __post_init__(self) -> None:
        """Reject a result that is not an amplitude on a resolvable grid.

        Raises:
            ValueError: If ``estimate`` is outside ``[0, 1]``, or if ``resolution`` is not
                positive.
        """
        if not 0.0 <= self.estimate <= 1.0:
            raise ValueError(
                f"an amplitude estimate must lie in [0, 1], got estimate={self.estimate}"
            )
        if self.resolution <= 0.0:
            raise ValueError(
                "the grid resolution must be positive, got "
                f"resolution={self.resolution}"
            )

    def within(self, amplitude: float) -> bool:
        """Return whether ``amplitude`` is within one grid step of the estimate.

        This is the accuracy contract the module makes, expressed once here rather than
        re-derived by every caller. It is a resolution statement and not a statistical
        interval: a value it rejects is not excluded at any stated confidence level, and
        one it accepts is not covered with any stated probability.

        Args:
            amplitude: The amplitude to compare against, normally a known one.

        Returns:
            ``True`` if ``abs(self.estimate - amplitude) <= self.resolution``.
        """
        return abs(self.estimate - amplitude) <= self.resolution


def amplitude_resolution(n_counting_wires: int) -> float:
    """Return the largest gap between adjacent grid amplitudes.

    The estimate is drawn from the grid ``sin²(pi * j / 2**(m+1))`` for ``j`` in
    ``0 .. 2**m``, whose widest step sits near an amplitude of one half and tracks
    ``pi / 2**(m+1)``. The resolution narrows as the counting register grows; it is a
    property of the grid, not a measurement of any particular run.

    Args:
        n_counting_wires: The width of the counting register, at least one.

    Returns:
        The widest gap between two adjacent grid amplitudes.

    Raises:
        ValueError: If ``n_counting_wires`` is less than one.
    """
    _validate_counting_width(n_counting_wires)
    grid = _amplitude_grid(n_counting_wires)
    return max(grid[index + 1] - grid[index] for index in range(len(grid) - 1))


def maximum_likelihood_estimate(
    counts: Mapping[str, int], n_counting_wires: int, n_evaluation_wires: int
) -> float:
    """Return the grid amplitude that best explains ``counts``.

    Each count key carries every wire, so the evaluation register's bits are marginalised
    away before the model is fitted: a key is truncated to its leading
    ``n_counting_wires`` characters and its count accumulated onto that counting outcome.
    Reading the keys without folding loses the evaluation register's whole share of the
    probability mass and returns a silently wrong estimate rather than an error.

    The likelihood is evaluated on the grid, and the maximiser is returned. Ties resolve
    to the smallest amplitude, so the result is deterministic.

    Args:
        counts: The sample counts, keyed by the big-endian bit string of all
            ``n_counting_wires + n_evaluation_wires`` wires.
        n_counting_wires: The width of the counting register, at least one.
        n_evaluation_wires: The width of the operator's own register, at least one.

    Returns:
        The grid amplitude with the largest likelihood, exactly one of
        ``sin²(pi * j / 2**(n_counting_wires+1))``.

    Raises:
        ValueError: If either register width is less than one, if ``counts`` is empty, or
            if a count key does not carry one bit per wire.
    """
    _validate_counting_width(n_counting_wires)
    if n_evaluation_wires < 1:
        raise ValueError(
            "the evaluation register needs at least one wire, got "
            f"n_evaluation_wires={n_evaluation_wires}; the operator acts on a register "
            "and its amplitudes are what is being estimated"
        )
    if not counts:
        raise ValueError(
            "counts must not be empty; with no sample there is no likelihood to maximise"
        )
    expected = n_counting_wires + n_evaluation_wires
    folded: dict[str, int] = {}
    for key, count in counts.items():
        if len(key) != expected:
            raise ValueError(
                f"count key {key!r} has {len(key)} bits, expected {expected} for a "
                f"{n_counting_wires}-wire counting register and a "
                f"{n_evaluation_wires}-wire evaluation register"
            )
        if count == 0:
            continue
        counting_key = key[:n_counting_wires]
        folded[counting_key] = folded.get(counting_key, 0) + count
    # A strict improvement keeps the first, smallest amplitude on a tie.
    grid = _amplitude_grid(n_counting_wires)
    best = grid[0]
    best_log_likelihood = -math.inf
    for amplitude in grid:
        log_likelihood = 0.0
        for counting_key, count in folded.items():
            probability = _counting_probability(
                int(counting_key, 2), amplitude, n_counting_wires
            )
            log_likelihood += count * math.log(max(probability, _LIKELIHOOD_FLOOR))
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best = amplitude
    return best


def amplitude_estimation_circuit(
    operator: AmplitudeOperator, *, n_counting_wires: int
) -> Circuit:
    """Build the amplitude estimation circuit for ``operator``.

    The circuit prepares the operator's own register, then runs phase estimation over
    ``operator``'s Grover operator on the counting register. The counting register
    occupies the first ``n_counting_wires`` wires and the evaluation register follows it,
    which is the leading-block layout
    :func:`~flagquantum.algorithms.primitives.phase_estimation.append_phase_estimation`
    requires; that primitive emits the counting register's Hadamards, the controlled
    powers, and the inverse Fourier transform itself.

    Args:
        operator: The state-preparation unitary, its marking operator and its reflections,
            in the controlled forms amplitude estimation needs.
        n_counting_wires: The number of wires in the counting register, at least one.

    Returns:
        A circuit over ``n_counting_wires + operator.n_wires`` wires.

    Raises:
        ValueError: If ``n_counting_wires`` is less than one.
    """
    _validate_counting_width(n_counting_wires)
    evaluation = list(range(n_counting_wires, n_counting_wires + operator.n_wires))
    circuit = Circuit(n_counting_wires + operator.n_wires)
    operator.apply_plain(circuit, evaluation)
    append_phase_estimation(
        circuit,
        unitary=_GroverOperator(operator),
        counting_wires=list(range(n_counting_wires)),
        evaluation_wires=evaluation,
    )
    return circuit


def run_amplitude_estimation(
    operator: AmplitudeOperator,
    *,
    n_counting_wires: int,
    shots: int = 4096,
    seed: int | None = None,
) -> AmplitudeEstimationResult:
    """Sample an amplitude estimation circuit and estimate the amplitude.

    The circuit is built by :func:`amplitude_estimation_circuit`, sampled ``shots`` times,
    and the maximum likelihood estimate over the grid is returned along with the grid's
    own resolution. ``seed`` seeds the sampler's generator, so the same seed and the same
    arguments replay the same estimate exactly.

    Args:
        operator: The state-preparation unitary, its marking operator and its reflections,
            in the controlled forms amplitude estimation needs.
        n_counting_wires: The number of wires in the counting register, at least one.
        shots: The number of samples to draw, at least one.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator.

    Returns:
        The estimate, the grid resolution, and the register widths it came from.

    Raises:
        ValueError: If ``n_counting_wires`` is less than one, or if ``shots`` is less than
            one.
    """
    _validate_counting_width(n_counting_wires)
    if shots < 1:
        raise ValueError(
            f"an amplitude estimation run needs at least one shot, got shots={shots}; "
            "with no samples there is nothing to estimate"
        )
    circuit = amplitude_estimation_circuit(operator, n_counting_wires=n_counting_wires)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    observed = circuit.counts(shots, generator=generator)[0]
    # ``counts`` is typed ``dict[str | int, int]`` because its ``format`` can be "int";
    # this call takes the default "bin", so every key is already the bit string that this
    # mapping is typed as, and the coercion below is a no-op at run time.
    counts = {str(key): count for key, count in observed.items()}
    return AmplitudeEstimationResult(
        estimate=maximum_likelihood_estimate(
            counts, n_counting_wires, operator.n_wires
        ),
        resolution=amplitude_resolution(n_counting_wires),
        n_counting_wires=n_counting_wires,
        n_evaluation_wires=operator.n_wires,
    )


def _amplitude_grid(n_counting_wires: int) -> list[float]:
    """Return the amplitudes ``sin²(pi * j / 2**(m+1))`` for ``j`` in ``0 .. 2**m``.

    The grid is the image of the counting register's phase resolution under the readout
    ``a = sin²(theta)``, so an estimate is always one of these values.

    Args:
        n_counting_wires: The width of the counting register, at least one.

    Returns:
        The ``2**n_counting_wires + 1`` grid amplitudes, ascending.
    """
    # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
    # negative exponent yields a float, and an ``Any`` return fails the type gate.
    half_turns = 2.0 ** (n_counting_wires + 1)
    return [
        math.sin(math.pi * index / half_turns) ** 2
        for index in range(2**n_counting_wires + 1)
    ]


def _counting_probability(
    outcome: int, amplitude: float, n_counting_wires: int
) -> float:
    """Return ``P(y | a)`` for one counting outcome and one grid amplitude.

    The two conjugate eigenphases of the Grover operator put the same amplitude at
    ``y`` and at ``M - y``, so both terms of the model are summed; each is the resonant
    term that goes to ``M**2`` as its denominator vanishes.

    Args:
        outcome: The counting value ``y``, read from the leading register bits.
        amplitude: The grid amplitude ``a`` the model is evaluated at.
        n_counting_wires: The width of the counting register, at least one.

    Returns:
        The probability of reading ``outcome`` at that amplitude.
    """
    # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
    # negative exponent yields a float, and an ``Any`` return fails the type gate.
    size = 2.0**n_counting_wires
    theta = math.asin(math.sqrt(amplitude))
    shift = math.pi * outcome / size
    return (
        _resonant_term(theta - shift, size) + _resonant_term(theta + shift, size)
    ) / (2 * size * size)


def _resonant_term(angle: float, size: float) -> float:
    """Return ``sin²(size * angle) / sin²(angle)``.

    Args:
        angle: The denominator's angle, ``t - pi * y / M``.
        size: The counting register's width, ``2**n_counting_wires``.

    Returns:
        The term, or ``size**2`` at the resonance where the denominator vanishes, which
        is the limit of the quotient there.
    """
    denominator = math.sin(angle)
    if denominator == 0.0:
        return size * size
    return math.sin(size * angle) ** 2 / denominator**2


def _validate_counting_width(n_counting_wires: int) -> None:
    """Check the width of an amplitude estimation counting register.

    Args:
        n_counting_wires: The register width as given.

    Raises:
        ValueError: If ``n_counting_wires`` is less than one.
    """
    if n_counting_wires < 1:
        raise ValueError(
            f"amplitude estimation needs at least one counting wire, got "
            f"n_counting_wires={n_counting_wires}; a register with no wires resolves no "
            "phase and carries no amplitude"
        )
