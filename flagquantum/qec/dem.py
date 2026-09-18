"""Exact detector error models for code-independent memory experiments.

A detector error model names every independent physical error mechanism by the
detectors and logical observables it flips. Construction is exact and does not
sample: the reference gate set is Clifford and every configured channel is a
Pauli channel, so forcing one mechanism through the circuit yields a
deterministic signature.

The model is defined for Pauli noise only. It is built on the memory circuit
*without* in-circuit feedback, because the model describes the physical
noise-to-detection mapping that a decoder inverts; the compiled feedback layer
is what a decoder replaces.

Detector rates the model predicts are compared against rates sampled from the
circuit simulator by the tests, not by this module. The comparison shares the
injection helper with construction, so it does not independently re-derive the
signatures; what it tests is that mechanisms compose by XOR in the simulator and
that merging identical signatures yields the right marginals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral, Real

import torch

_DETECTOR_PREFIX = "D"
_OBSERVABLE_PREFIX = "L"


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return normalized


def _count(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _normalized_indices(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integer indices")
        index = int(value)
        if index < 0:
            raise ValueError(f"{name} must contain non-negative indices")
        indices.append(index)
    return tuple(sorted(set(indices)))


def _target_index(token: str, prefix: str) -> int | None:
    """Return the index a stim target names, or ``None`` if it names nothing.

    A target is a one-letter prefix followed by ASCII digits, so a token such
    as ``X0`` or ``D`` is a malformed target rather than a crash inside
    ``int``, which accepts the digits of other scripts.
    """

    digits = token[len(prefix) :]
    if token.startswith(prefix) and digits.isascii() and digits.isdigit():
        return int(digits)
    return None


def _parse_error_line(line: str) -> DemError:
    """Parse one ``error(<p>) <targets...>`` line."""

    body = line[len("error") :]
    if not body.startswith("("):
        raise ValueError("an error line must state its probability in parentheses")
    body = body[1:]
    closing = body.find(")")
    if closing < 0:
        raise ValueError("an error line must close its probability in parentheses")
    try:
        probability = float(body[:closing])
    except ValueError as error:
        raise ValueError("an error line must state a numeric probability") from error
    detectors: list[int] = []
    observables: list[int] = []
    for token in body[closing + 1 :].split():
        detector = _target_index(token, _DETECTOR_PREFIX)
        observable = _target_index(token, _OBSERVABLE_PREFIX)
        if detector is not None:
            detectors.append(detector)
        elif observable is not None:
            observables.append(observable)
        else:
            raise ValueError(f"error targets must be D or L indices, not {token!r}")
    if not detectors and not observables:
        raise ValueError(
            "an error mechanism must flip at least one detector or observable"
        )
    return DemError(
        probability=probability,
        detectors=tuple(detectors),
        observables=tuple(observables),
    )


def _parse_declaration_line(
    line: str, *, instruction: str, prefix: str, coordinates: bool
) -> int:
    """Parse one ``detector [<coordinates>] D<i>`` declaration into ``i``.

    Coordinates carry geometry the model does not represent, so a detector
    declaration accepts them and discards them: the declared index is the
    target, never the coordinate. A logical-observable declaration takes no
    coordinates, which is the form the format itself accepts.
    """

    remainder = line[len(instruction) :]
    if coordinates and remainder.startswith("("):
        closing = remainder.find(")")
        if closing < 0:
            raise ValueError(f"{instruction} coordinates must close in parentheses")
        remainder = remainder[closing + 1 :]
    tokens = remainder.split()
    if len(tokens) != 1:
        raise ValueError(
            f"a {instruction} declaration must name exactly one {prefix} index"
        )
    index = _target_index(tokens[0], prefix)
    if index is None:
        raise ValueError(
            f"a {instruction} declaration must name a {prefix} index, "
            f"not {tokens[0]!r}"
        )
    return index


def _declared_count(declared: set[int], *, name: str, prefix: str) -> int:
    """Return how many indices the text declared, refusing a hole.

    The count may only come from the declarations, so a text that declares
    ``D1`` without ``D0`` is malformed rather than a one-detector model that
    happens to call its detector one.
    """

    count = len(declared)
    if declared != set(range(count)):
        raise ValueError(f"{name} declarations must be consecutive from {prefix}0")
    return count


def _parse_stim_text(text: str) -> tuple[int, int, tuple[DemError, ...]]:
    """Parse stim text into a model shape and its error mechanisms.

    Only the instructions this model represents are accepted; every other
    construct is refused with a stated reason, so a text that carries
    information the model cannot hold fails closed instead of losing it.
    """

    declared_detectors: set[int] = set()
    declared_observables: set[int] = set()
    errors: list[DemError] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # The keyword is what precedes the first parenthesis, so the
        # coordinates of ``detector(1, 2) D0`` belong to the body.
        head = stripped.split("(", 1)[0].split()
        keyword = head[0] if head else ""
        if keyword == "error":
            errors.append(_parse_error_line(stripped))
        elif keyword == "detector":
            declared_detectors.add(
                _parse_declaration_line(
                    stripped,
                    instruction="detector",
                    prefix=_DETECTOR_PREFIX,
                    coordinates=True,
                )
            )
        elif keyword == "logical_observable":
            declared_observables.add(
                _parse_declaration_line(
                    stripped,
                    instruction="logical_observable",
                    prefix=_OBSERVABLE_PREFIX,
                    coordinates=False,
                )
            )
        elif keyword == "repeat":
            raise ValueError(
                "repeat blocks are not supported: expand the block into the "
                "instructions it repeats"
            )
        elif keyword == "shift_detectors":
            raise ValueError(
                "shift_detectors is not supported: every detector index in the "
                "text must be absolute"
            )
        else:
            raise ValueError(f"unsupported stim instruction {stripped!r}")
    num_detectors = _declared_count(
        declared_detectors, name="detector", prefix=_DETECTOR_PREFIX
    )
    num_observables = _declared_count(
        declared_observables, name="logical_observable", prefix=_OBSERVABLE_PREFIX
    )
    return num_detectors, num_observables, tuple(errors)


@dataclass(frozen=True)
class DemError:
    """One independent error mechanism and the signature it flips."""

    probability: float
    detectors: tuple[int, ...] = ()
    observables: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probability",
            _probability(self.probability, name="error probability"),
        )
        object.__setattr__(
            self,
            "detectors",
            _normalized_indices(self.detectors, name="error detectors"),
        )
        object.__setattr__(
            self,
            "observables",
            _normalized_indices(self.observables, name="error observables"),
        )
        if not self.detectors and not self.observables:
            raise ValueError(
                "an error mechanism must flip at least one detector or observable"
            )


@dataclass(frozen=True, eq=False)
class DemSample:
    """Sampled detector and observable flips from a detector error model.

    Equality compares the tensors by content, and instances are deliberately
    unhashable: the generated comparison would ask ``bool()`` of a tensor and
    the generated hash would not agree with a content comparison.
    """

    detectors: torch.Tensor
    observables: torch.Tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DemSample):
            return NotImplemented
        return torch.equal(self.detectors, other.detectors) and torch.equal(
            self.observables, other.observables
        )

    @property
    def shots(self) -> int:
        """Number of sampled shots."""

        return int(self.detectors.shape[0])


@dataclass(frozen=True)
class DetectorErrorModel:
    """A detector error model over a fixed number of detectors and observables."""

    num_detectors: int
    num_observables: int
    errors: tuple[DemError, ...] = ()

    def __post_init__(self) -> None:
        detectors = _count(self.num_detectors, name="num_detectors")
        observables = _count(self.num_observables, name="num_observables")
        if detectors < 1:
            raise ValueError("a detector error model requires at least one detector")
        object.__setattr__(self, "num_detectors", detectors)
        object.__setattr__(self, "num_observables", observables)

        for error in self.errors:
            if not isinstance(error, DemError):
                raise TypeError("errors must contain DemError records")
        object.__setattr__(
            self,
            "errors",
            tuple(
                sorted(
                    self.errors,
                    key=lambda error: (
                        error.detectors,
                        error.observables,
                        error.probability,
                    ),
                )
            ),
        )
        for error in self.errors:
            for index in error.detectors:
                if index >= detectors:
                    raise ValueError(
                        f"detector index {index} is outside the model shape"
                    )
            for index in error.observables:
                if index >= observables:
                    raise ValueError(
                        f"observable index {index} is outside the model shape"
                    )

    def detector_error_matrix(self) -> torch.Tensor:
        """Return the ``(num_detectors, num_errors)`` parity matrix.

        Entry ``[d, e]`` is one when error ``e`` flips detector ``d``. The
        orientation matches the stim ecosystem's detector error matrix.
        """

        matrix = torch.zeros((self.num_detectors, self.num_errors), dtype=torch.int8)
        for column, error in enumerate(self.errors):
            for index in error.detectors:
                matrix[index, column] = 1
        return matrix

    def observables_flips_matrix(self) -> torch.Tensor:
        """Return the ``(num_observables, num_errors)`` parity matrix.

        Entry ``[o, e]`` is one when error ``e`` flips observable ``o``.
        """

        matrix = torch.zeros((self.num_observables, self.num_errors), dtype=torch.int8)
        for column, error in enumerate(self.errors):
            for index in error.observables:
                matrix[index, column] = 1
        return matrix

    def _marginal_rates(
        self, count: int, select: Callable[[DemError], tuple[int, ...]]
    ) -> torch.Tensor:
        rates = torch.zeros((count,), dtype=torch.float64)
        if count == 0:
            # Reachable only for ``num_observables == 0``; detectors are never zero.
            return rates
        complements = torch.ones((count,), dtype=torch.float64)
        for error in self.errors:
            factor = 1.0 - 2.0 * error.probability
            for index in select(error):
                complements[index] *= factor
        return (1.0 - complements) / 2.0

    def detector_rates(self) -> torch.Tensor:
        """Return the exact marginal flip probability of every detector."""

        return self._marginal_rates(self.num_detectors, lambda error: error.detectors)

    def observable_rates(self) -> torch.Tensor:
        """Return the exact marginal flip probability of every observable."""

        return self._marginal_rates(
            self.num_observables, lambda error: error.observables
        )

    def dem_sampling(self, *, shots: int, seed: int | None = None) -> DemSample:
        """Sample ``shots`` shots by XORing the signatures that fired.

        Every mechanism is drawn independently per shot with its own
        probability. This is the model's own arithmetic, not the circuit
        simulator; the circuit-simulator comparison lives in the tests.
        """

        if isinstance(shots, bool) or not isinstance(shots, Integral):
            raise TypeError("shots must be a positive integer")
        if shots <= 0:
            raise ValueError("shots must be a positive integer")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise TypeError("seed must be an integer or None")

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))

        detector_bits = torch.zeros((shots, self.num_detectors), dtype=torch.int8)
        observable_bits = torch.zeros((shots, self.num_observables), dtype=torch.int8)
        if not self.errors:
            return DemSample(detectors=detector_bits, observables=observable_bits)

        probabilities = torch.tensor(
            [error.probability for error in self.errors], dtype=torch.float64
        )
        fired = torch.rand(
            (shots, self.num_errors), generator=generator, dtype=torch.float64
        )
        fired = fired < probabilities
        for column, error in enumerate(self.errors):
            active = fired[:, column]
            for index in error.detectors:
                detector_bits[:, index] ^= active.to(torch.int8)
            for index in error.observables:
                observable_bits[:, index] ^= active.to(torch.int8)
        return DemSample(detectors=detector_bits, observables=observable_bits)

    @classmethod
    def from_stim_text(cls, text: str) -> DetectorErrorModel:
        """Build a model from stim's text representation of a detector error model.

        The shape comes from the ``detector`` and ``logical_observable``
        declarations and from nowhere else. Reading it off the largest index an
        error mentions would accept a text that had lost its trailing detectors,
        so a text without a ``detector`` declaration is refused by the
        constructor instead.
        """

        num_detectors, num_observables, errors = _parse_stim_text(text)
        return cls(
            num_detectors=num_detectors,
            num_observables=num_observables,
            errors=errors,
        )

    def to_stim_text(self) -> str:
        """Render the model as stim text.

        One ``error(...)`` line per mechanism, with its detectors ascending and
        then its observables ascending, and the probability written with
        ``repr`` so the value survives the round trip. The declaration lines
        follow, one per detector and one per observable, so a model whose
        trailing detectors no error touches keeps its shape through a round
        trip. Every line, including the last, ends with a newline.
        """

        lines: list[str] = []
        for error in self.errors:
            targets = [f"{_DETECTOR_PREFIX}{index}" for index in error.detectors] + [
                f"{_OBSERVABLE_PREFIX}{index}" for index in error.observables
            ]
            lines.append(f"error({error.probability!r}) {' '.join(targets)}")
        lines.extend(
            f"detector {_DETECTOR_PREFIX}{index}" for index in range(self.num_detectors)
        )
        lines.extend(
            f"logical_observable {_OBSERVABLE_PREFIX}{index}"
            for index in range(self.num_observables)
        )
        return "".join(f"{line}\n" for line in lines)

    @property
    def num_errors(self) -> int:
        """Number of independent error mechanisms in the model."""

        return len(self.errors)


__all__ = ("DemError", "DemSample", "DetectorErrorModel")
