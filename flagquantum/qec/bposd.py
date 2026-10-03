"""Belief propagation with ordered-statistics post-processing.

The minimum-weight matcher in :mod:`flagquantum.qec.matching` needs a graphlike
detector error model, so a mechanism that flips three detectors is refused rather
than decoded. This module decodes the same model without that restriction: a
belief-propagation pass estimates one log-likelihood ratio per mechanism from the
syndrome, and an order-zero ordered-statistics pass turns that estimate into a
mechanism set that reproduces the syndrome exactly.

The decoder mirrors ``MinimumWeightMatchingDecoder.decode``: both accept the
detector indices a syndrome names and both return a record carrying the logical
observables the correction flips, so either can be substituted at a call site. It
deliberately does not satisfy the repetition-code ``Decoder`` protocol, whose
``Correction`` names one data wire and cannot express the correction a detector
error model implies.

Two limits are stated rather than implied. Order-zero post-processing leaves the
free mechanism positions at zero, so the result explains the syndrome without
being its most likely explanation, and this module offers the syndrome
consistency invariant instead of an optimality claim. A mechanism that flips no
detector is invisible to any decoder that reads a syndrome, so it is never
selected, and a syndrome no combination of the model's mechanisms produces is
refused rather than projected.

Nothing outside the standard library and PyTorch is imported, so the ``stim`` and
``pymatching`` distributions are not required and no part of the decode depends
on an external decoder.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral, Real

import torch

from ..errors import CapabilityError
from .dem import DetectorErrorModel

__all__ = (
    "BeliefPropagationOsdDecodeResult",
    "BeliefPropagationOsdDecoder",
)

_DEFAULT_ITERATIONS = 30
_DEFAULT_SCALING = 0.75
_INFINITY = float("inf")
_MESSAGE_LIMIT = 1.0e6
_MINIMUM_PROBABILITY = 1.0e-12


def _indices(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be integer indices")
        index = int(value)
        if index < 0:
            raise ValueError(f"{name} must be non-negative")
        indices.append(index)
    ordered = tuple(sorted(set(indices)))
    if ordered != tuple(indices):
        raise ValueError(f"{name} must be unique and ascending")
    return ordered


def _syndrome(detection_events: Iterable[int], *, num_detectors: int) -> torch.Tensor:
    syndrome = torch.zeros(num_detectors, dtype=torch.float64)
    seen: set[int] = set()
    for value in detection_events:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("detection events must be integer detector indices")
        index = int(value)
        if index < 0 or index >= num_detectors:
            raise ValueError(
                f"detection event D{index} is outside the model's {num_detectors} "
                "detectors"
            )
        if index in seen:
            raise ValueError("a syndrome cannot name the same detector twice")
        seen.add(index)
        syndrome[index] = 1.0
    return syndrome


@dataclass(frozen=True)
class BeliefPropagationOsdDecodeResult:
    """The mechanisms that explain one syndrome and the flips they imply.

    Attributes:
        observables: The observables the selected mechanisms flip, ascending.
        mechanisms: The indices of the selected mechanisms, ascending.
        weight: The log-odds weight the selected mechanisms sum to.
        converged: Whether belief propagation's own estimate already reproduced
            the syndrome before the ordered-statistics pass ran.
        iterations: The belief-propagation passes that ran.
    """

    observables: tuple[int, ...]
    mechanisms: tuple[int, ...]
    weight: float
    converged: bool
    iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observables", _indices(self.observables, name="decoded observables")
        )
        object.__setattr__(
            self, "mechanisms", _indices(self.mechanisms, name="selected mechanisms")
        )
        if isinstance(self.weight, bool) or not isinstance(self.weight, Real):
            raise TypeError("decode weight must be a real number")
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError("decode weight must be finite")
        object.__setattr__(self, "weight", weight)
        if not isinstance(self.converged, bool):
            raise TypeError("decode convergence must be a boolean")
        if isinstance(self.iterations, bool) or not isinstance(
            self.iterations, Integral
        ):
            raise TypeError("belief-propagation passes must be an integer")
        if self.iterations < 0:
            raise ValueError("belief-propagation passes must be non-negative")
        object.__setattr__(self, "iterations", int(self.iterations))


@dataclass(frozen=True)
class BeliefPropagationOsdDecoder:
    """Decode a detector error model by belief propagation and ordered statistics.

    Args:
        model: The detector error model whose mechanisms explain a syndrome.
        max_iterations: The belief-propagation passes to run before the
            ordered-statistics pass decides the result.
        scaling: The min-sum factor damping each check message. One leaves
            belief propagation unnormalised.

    Raises:
        TypeError: The model or an option has the wrong type.
        ValueError: ``max_iterations`` is below one, or ``scaling`` is outside the
            half-open interval that keeps the min-sum iteration damped.
    """

    model: DetectorErrorModel
    max_iterations: int = _DEFAULT_ITERATIONS
    scaling: float = _DEFAULT_SCALING
    _flips: torch.Tensor = field(init=False, repr=False, compare=False)
    _observable_flips: torch.Tensor = field(init=False, repr=False, compare=False)
    _prior: torch.Tensor = field(init=False, repr=False, compare=False)
    _decodable: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, DetectorErrorModel):
            raise TypeError("model must be a DetectorErrorModel")
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, Integral
        ):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        if isinstance(self.scaling, bool) or not isinstance(self.scaling, Real):
            raise TypeError("scaling must be a real number")
        scaling = float(self.scaling)
        if not math.isfinite(scaling) or not 0.0 < scaling <= 1.0:
            raise ValueError("scaling must be greater than zero and at most one")
        object.__setattr__(self, "scaling", scaling)

        flips = self.model.detector_error_matrix().to(torch.bool)
        object.__setattr__(self, "_flips", flips)
        object.__setattr__(
            self,
            "_observable_flips",
            self.model.observables_flips_matrix().to(torch.bool),
        )
        # A mechanism that fires with probability one has no finite log-odds
        # ratio, and the floor also keeps a probability of one or zero from
        # producing an infinite message that the iteration cannot subtract.
        probabilities = torch.tensor(
            [error.probability for error in self.model.errors], dtype=torch.float64
        ).clamp(_MINIMUM_PROBABILITY, 1.0 - _MINIMUM_PROBABILITY)
        object.__setattr__(
            self, "_prior", torch.log((1.0 - probabilities) / probabilities)
        )
        object.__setattr__(self, "_decodable", flips.any(dim=0))

    def decode(
        self, detection_events: Iterable[int]
    ) -> BeliefPropagationOsdDecodeResult:
        """Explain one syndrome with the model's mechanisms.

        Args:
            detection_events: The detector indices this syndrome lights up, in
                any order, each at most once.

        Returns:
            The mechanisms that reproduce the syndrome together with the
            observables they flip.

        Raises:
            TypeError: An entry is not an integer detector index.
            ValueError: A detector index is outside the model, or the syndrome
                names the same detector twice.
            CapabilityError: No combination of the model's mechanisms produces
                the syndrome, so no correction explains it.
        """

        syndrome = _syndrome(detection_events, num_detectors=self.model.num_detectors)
        if not bool(syndrome.any()):
            # The zero syndrome is reproduced by selecting nothing, which is the
            # least we can ask of any correction.
            return BeliefPropagationOsdDecodeResult(
                observables=(),
                mechanisms=(),
                weight=0.0,
                converged=True,
                iterations=0,
            )
        posterior, converged, iterations = self._propagate(syndrome)
        mechanisms = self._ordered_statistics(syndrome, posterior)
        if mechanisms is None:
            raise CapabilityError(
                "the syndrome is not in the span of the detector error model's "
                "mechanisms, so no mechanism set reproduces it and no correction "
                "follows"
            )
        observables = self._observables(mechanisms)
        weight = float(sum(self._prior[index].item() for index in mechanisms))
        return BeliefPropagationOsdDecodeResult(
            observables=observables,
            mechanisms=mechanisms,
            weight=weight,
            converged=converged,
            iterations=iterations,
        )

    def _propagate(self, syndrome: torch.Tensor) -> tuple[torch.Tensor, bool, int]:
        """Estimate one log-likelihood ratio per mechanism from the syndrome.

        The iteration is min-sum with a damping factor: a detector sends each
        mechanism the magnitude of the smallest opposing message on that check,
        signed by the parity of the check's other messages. A check with one
        mechanism has no opposing message, so its magnitude is the message limit,
        which is how a detector that only one mechanism can flip forces it.
        """

        flips = self._flips
        prior = self._prior
        if flips.shape[1] == 0:
            # No mechanism exists, so no pass can reproduce a non-zero syndrome.
            return prior.clone(), False, 0
        positions = torch.arange(flips.shape[1]).unsqueeze(0)
        check_sign = 1.0 - 2.0 * syndrome
        posterior = prior.clone()
        check_to_variable = torch.zeros(flips.shape, dtype=torch.float64)
        variable_to_check = prior.unsqueeze(0).expand(flips.shape) * flips
        for iteration in range(1, self.max_iterations + 1):
            magnitudes = torch.where(flips, variable_to_check.abs(), _INFINITY)
            smallest, chosen = magnitudes.min(dim=1)
            exclusive = torch.where(
                chosen.unsqueeze(1) == positions, _INFINITY, magnitudes
            )
            excluded = torch.where(
                chosen.unsqueeze(1) == positions,
                exclusive.min(dim=1).values.unsqueeze(1),
                smallest.unsqueeze(1),
            ).clamp(max=_MESSAGE_LIMIT)
            negative = (variable_to_check < 0) & flips
            parity = negative.sum(dim=1).remainder(2).to(torch.float64)
            sign = (1.0 - 2.0 * parity).unsqueeze(1) * torch.where(
                variable_to_check < 0, -1.0, 1.0
            )
            check_to_variable = (
                self.scaling * check_sign.unsqueeze(1) * sign * excluded * flips
            )
            totals = check_to_variable.sum(dim=0)
            variable_to_check = (
                prior.unsqueeze(0) + totals.unsqueeze(0) - check_to_variable
            ) * flips
            posterior = prior + totals
            estimate = posterior < 0
            reproduced = (
                flips.to(torch.float64) @ estimate.to(torch.float64)
            ).remainder(2.0)
            if torch.equal(reproduced, syndrome):
                return posterior, True, iteration
        return posterior, False, self.max_iterations

    def _ordered_statistics(
        self, syndrome: torch.Tensor, posterior: torch.Tensor
    ) -> tuple[int, ...] | None:
        """Solve the syndrome over the most error-prone independent mechanisms.

        The mechanisms are ranked by their estimated log-likelihood ratio, most
        error-prone first, and Gaussian elimination in that rank order elects an
        information set: the earliest mechanisms that are linearly independent.
        Every mechanism outside that set is held at zero, which reproduces the
        syndrome exactly but need not be its most likely explanation.

        Returns:
            The selected mechanism indices, or ``None`` when the syndrome is not
            in the span of the model's mechanisms.
        """

        flips = self._flips
        ranked = torch.argsort(posterior, stable=True)
        ranked = ranked[self._decodable[ranked]]
        width = int(ranked.numel())
        if width == 0:
            return None
        augmented = torch.cat(
            [flips[:, ranked], (syndrome > 0.5).unsqueeze(1)], dim=1
        ).to(torch.bool)
        pivot = [-1] * width
        rank = 0
        rows = int(augmented.shape[0])
        for column in range(width):
            candidates = torch.nonzero(augmented[rank:, column], as_tuple=False)
            if candidates.numel() == 0:
                continue
            chosen = rank + int(candidates[0, 0])
            if chosen != rank:
                holding = augmented[rank].clone()
                augmented[rank] = augmented[chosen]
                augmented[chosen] = holding
            hits = torch.nonzero(augmented[:, column], as_tuple=False).flatten()
            hits = hits[hits != rank]
            if hits.numel():
                augmented[hits] ^= augmented[rank]
            pivot[column] = rank
            rank += 1
            if rank == rows:
                break
        if bool(augmented[rank:, :].any()):
            return None
        return tuple(
            sorted(
                int(ranked[column])
                for column in range(width)
                if pivot[column] >= 0 and bool(augmented[pivot[column], width])
            )
        )

    def _observables(self, mechanisms: tuple[int, ...]) -> tuple[int, ...]:
        """The observables the given mechanisms flip between them."""

        matrix = self._observable_flips
        if not mechanisms:
            return ()
        selected = torch.tensor(mechanisms, dtype=torch.long)
        flipped = matrix[:, selected].sum(dim=1).remainder(2) > 0
        indices = torch.nonzero(flipped, as_tuple=False).flatten()
        return tuple(int(index) for index in indices)
