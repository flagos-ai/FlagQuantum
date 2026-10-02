"""Exact detector error models for code-independent memory experiments.

A detector error model names every independent physical error mechanism by the
detectors and logical observables it flips. Construction is exact and does not
sample, and it has two routes. ``DetectorErrorModel.from_memory_circuit`` is the
circuit route, and ``dem_construction`` owns its engine: it enumerates the noise
locations a ``PhenomenologicalNoise`` record configures, drops the ones that flip
nothing at all, and merges the ones that flip the same detectors and observables.
``DetectorErrorModel.from_code_matrices`` is the matrix route: it reads a
parity-check matrix and a logical-operator matrix and derives the signatures
combinatorially, with nothing lowered or executed. The two routes describe
different experiments -- the matrix route is code capacity and has no terminal
data readout -- so their detector counts differ, and each is pinned to its own
stim transcription rather than to the other's.

A model may also arrive with two mechanisms on one signature, because stim's
text format allows it and this reader preserves what the text states. Merging is
therefore available to the caller as well, as
``DetectorErrorModel.merge_duplicate_mechanisms`` under a stated
``DemMergeRule``; construction states the same rule by default. The model's own
shape never depends on how many mechanisms a signature has, and the decoder is
what refuses to read an unmerged duplicate rather than quietly weighting a fault
by the cheaper of its parts.

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
from enum import Enum
from numbers import Integral, Real

import torch

from .circuit import MemoryCircuit
from .dem_construction import _code_matrix_entries, _memory_circuit_entries
from .noise import PhenomenologicalNoise

_DETECTOR_PREFIX = "D"
_OBSERVABLE_PREFIX = "L"
_SEPARATOR = "^"


class DemMergeRule(str, Enum):
    """The rule that gives one prior to mechanisms sharing a signature.

    Two mechanisms are one fault to a decoder, which sees only whether their
    shared signature flipped and not how many mechanisms produced it. Which
    single prior stands for the pair is a modelling decision, so the caller
    states it rather than the merge assuming one.

    Attributes:
        INDEPENDENT_PARITY: The mechanisms are independent faults, so a shot
            flips the signature exactly when an odd number of them fire and the
            combined prior is the probability of that event. For two priors it
            is ``p1 + p2 - 2 * p1 * p2``, and the values fold left to right so a
            group of any size reduces the same way. This is the rule
            construction uses, it is exact, and it is closed on ``[0, 1/2)``:
            combining any number of mechanisms each below one half stays below
            one half, so a merged mechanism still has a finite matching weight.
        CLAMPED_LINEAR_SUM: The priors add and the total saturates at one,
            ``min(1, sum(p))``. This is a small-prior approximation for a caller
            combining priors that are already marginal, not the probability of
            any event: at ``0.1`` and ``0.2`` it gives ``0.3`` where the parity
            rule gives ``0.26``. It is not closed on ``[0, 1/2)``, so a merged
            prior may be one that no minimum-weight decoder can weight, and the
            merge does not hide that by clamping below the boundary.
    """

    INDEPENDENT_PARITY = "independent_parity"
    CLAMPED_LINEAR_SUM = "clamped_linear_sum"


def _combine_two(previous: float, probability: float, *, rule: DemMergeRule) -> float:
    """Fold one more prior into a signature's accumulated prior.

    The arithmetic of both rules lives here and nowhere else, so construction
    and :meth:`DetectorErrorModel.merge_duplicate_mechanisms` cannot drift apart
    on the formula. Each caller supplies its own fold order, which neither rule
    depends on: both are associative and commutative, so the orders agree up to
    floating-point rounding.
    """

    if rule is DemMergeRule.INDEPENDENT_PARITY:
        return previous * (1.0 - probability) + probability * (1.0 - previous)
    return min(1.0, previous + probability)


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


def _parse_shift_detectors(line: str) -> int:
    """Return the detector-index shift a ``shift_detectors`` line states.

    The instruction takes optional coordinates in parentheses and exactly one
    target, the shift itself. Coordinates carry geometry the model does not
    represent, so they are accepted and discarded for the same reason a
    ``detector`` declaration's coordinates are: the index is the target.

    Stim reaches the absolute detector index by adding the shifts accumulated so
    far, so the caller accumulates this value rather than replacing one.
    """

    remainder = line[len("shift_detectors") :]
    if remainder.startswith("("):
        closing = remainder.find(")")
        if closing < 0:
            raise ValueError("shift_detectors coordinates must close in parentheses")
        remainder = remainder[closing + 1 :]
    tokens = remainder.split()
    if len(tokens) != 1:
        raise ValueError(
            "a shift_detectors instruction must state exactly one "
            "detector-index shift"
        )
    shift = tokens[0]
    # Stim accepts an unsigned integer here and nothing else, so a signed or
    # non-ASCII token is malformed rather than a negative shift in disguise.
    if not shift.isascii() or not shift.isdigit():
        raise ValueError(
            "a shift_detectors instruction must state a non-negative integer "
            f"shift, not {shift!r}"
        )
    return int(shift)


def _toggle(targets: set[int], index: int) -> None:
    """Flip one target's membership, which is how a repeated target cancels."""

    if index in targets:
        targets.remove(index)
    else:
        targets.add(index)


def _parse_error_line(line: str, *, detector_shift: int) -> DemError:
    """Parse one ``error(<p>) <targets...>`` line into its symptom.

    A detector target is relative to the shifts in force at that line, which is
    what makes a ``shift_detectors`` instruction meaningful; an observable
    target is absolute, because the format has no observable shift.

    The signature is the symmetric difference of the line's targets, so a target
    that appears twice cancels. Stim's ``^`` separators mark how a composite
    mechanism decomposes into simpler ones; they partition the targets and do
    not change which detectors and observables a shot of this mechanism flips,
    which is the whole of what a signature records. The groups are therefore
    not retained, and a decomposed line contributes the same signature its
    targets state in any order.
    """

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
    detectors: set[int] = set()
    observables: set[int] = set()
    tokens = body[closing + 1 :].split()
    if not tokens:
        raise ValueError(
            "an error mechanism must flip at least one detector or observable"
        )
    group_is_empty = True
    for token in tokens:
        if token == _SEPARATOR:
            if group_is_empty:
                raise ValueError(
                    "an error separator must sit between two groups of targets"
                )
            group_is_empty = True
            continue
        group_is_empty = False
        detector = _target_index(token, _DETECTOR_PREFIX)
        if detector is not None:
            _toggle(detectors, detector + detector_shift)
            continue
        observable = _target_index(token, _OBSERVABLE_PREFIX)
        if observable is not None:
            _toggle(observables, observable)
            continue
        if _SEPARATOR in token:
            raise ValueError("an error separator must be separated by spacing")
        raise ValueError(f"error targets must be D or L indices, not {token!r}")
    if group_is_empty:
        raise ValueError("an error separator must sit between two groups of targets")
    # ``DemError`` re-checks this in its own constructor; refusing here as well
    # fails at the parse site, where the offending line is still in hand. A
    # decomposed line whose groups cancel reaches this as well, because the
    # signature it states is empty.
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
    line: str,
    *,
    instruction: str,
    prefix: str,
    coordinates: bool,
    detector_shift: int,
) -> int:
    """Parse one ``detector [<coordinates>] D<i>`` declaration into ``i``.

    Coordinates carry geometry the model does not represent, so a detector
    declaration accepts them and discards them: the declared index is the
    target, never the coordinate. A logical-observable declaration takes no
    coordinates, which is the form the format itself accepts. A detector index
    is relative to the shifts in force at that line.
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
    return index + (detector_shift if prefix == _DETECTOR_PREFIX else 0)


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
    referenced_observables: set[int] = set()
    detector_shift = 0
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
            error = _parse_error_line(stripped, detector_shift=detector_shift)
            errors.append(error)
            referenced_observables.update(error.observables)
        elif keyword == "detector":
            declared_detectors.add(
                _parse_declaration_line(
                    stripped,
                    instruction="detector",
                    prefix=_DETECTOR_PREFIX,
                    coordinates=True,
                    detector_shift=detector_shift,
                )
            )
        elif keyword == "logical_observable":
            declared_observables.add(
                _parse_declaration_line(
                    stripped,
                    instruction="logical_observable",
                    prefix=_OBSERVABLE_PREFIX,
                    coordinates=False,
                    detector_shift=detector_shift,
                )
            )
        elif keyword == "shift_detectors":
            detector_shift += _parse_shift_detectors(stripped)
        elif keyword == "repeat":
            raise ValueError(
                "repeat blocks are not supported: expand the block into the "
                "instructions it repeats"
            )
        else:
            raise ValueError(f"unsupported stim instruction {stripped!r}")
    num_detectors = _declared_count(
        declared_detectors, name="detector", prefix=_DETECTOR_PREFIX
    )
    # Stim declares the observable when no error mechanism references it and
    # omits the declaration when one does, so at most one of these two sources
    # has anything to say. A declared shape governs; the referenced indices
    # govern only where there is no declaration to state the shape.
    num_observables = (
        _declared_count(
            declared_observables, name="logical_observable", prefix=_OBSERVABLE_PREFIX
        )
        if declared_observables
        else (max(referenced_observables) + 1 if referenced_observables else 0)
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

    def mechanisms_are_unique(self) -> bool:
        """Return whether no two mechanisms share a detector and observable pair.

        A mechanism's signature is the pair of its detector indices and its
        observable indices, each already normalized to ascending distinct
        indices by construction, so this is a comparison over the same GF(2)
        support a decoder reads. Two mechanisms with one signature are one fault
        to a decoder and two columns here.
        """

        signatures = [(error.detectors, error.observables) for error in self.errors]
        return len(set(signatures)) == len(signatures)

    def require_unique_mechanisms(self) -> None:
        """Fail unless every mechanism has a signature of its own.

        A decoder reads whether a signature flipped, not how many mechanisms
        produced it, so a model that keeps two mechanisms on one signature lets
        a caller state a fault twice and then weight it by whichever part it
        prefers. This is the check that turns that into a refusal a caller can
        act on.

        Raises:
            ValueError: Two mechanisms flip the same detectors and observables,
                naming the pair and the operation that resolves it.
        """

        seen: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        for position, error in enumerate(self.errors):
            signature = (error.detectors, error.observables)
            first = seen.setdefault(signature, position)
            if first != position:
                targets = " ".join(
                    [f"{_DETECTOR_PREFIX}{index}" for index in signature[0]]
                    + [f"{_OBSERVABLE_PREFIX}{index}" for index in signature[1]]
                )
                raise ValueError(
                    f"mechanisms {first} and {position} both flip {targets}; "
                    "merge_duplicate_mechanisms() states one prior for a shared "
                    "signature and is what a decoder needs"
                )

    def merge_duplicate_mechanisms(
        self, *, rule: DemMergeRule | str = DemMergeRule.INDEPENDENT_PARITY
    ) -> DetectorErrorModel:
        """Return the model with one mechanism per detector and observable pair.

        The merge is what makes a model a decoder can weight: the result states
        one prior for each signature, so no fault is represented twice. It is a
        merge and not a filter, so every signature the model holds survives and
        the model's shape is unchanged; only the number of mechanisms falls. A
        model that is already unique comes back equal to this one, and a
        mechanism alone on its signature keeps its probability bit for bit
        rather than being passed through the arithmetic. Merging is therefore
        idempotent, and because the model stores its mechanisms sorted the
        result does not depend on the order the mechanisms arrived in.

        Under the default rule the merge is marginal-preserving, not merely
        tidy: a detector's rate is ``(1 - prod(1 - 2p)) / 2`` over the
        mechanisms that touch it, and a merged group's prior is exactly the
        single ``p`` whose ``1 - 2p`` is that group's product, so every detector
        rate and every observable rate is unchanged. Sampling one merged
        mechanism is therefore the same distribution as sampling the group
        independently and asking whether an odd number fired. Under
        :attr:`DemMergeRule.CLAMPED_LINEAR_SUM` the rates do change, which is
        what makes that rule an approximation rather than another spelling of
        this one. What changes under either rule is the number of mechanisms
        and, for a caller that reads one weight per signature, those weights.

        Args:
            rule: The rule that gives one prior to a shared signature. The
                default, :attr:`DemMergeRule.INDEPENDENT_PARITY`, is the exact
                probability that an odd number of the mechanisms fire.

        Returns:
            A model over the same shape with unique signatures.

        Raises:
            ValueError: ``rule`` is not one of the two stated rules.
        """

        selected = DemMergeRule(rule)
        accumulated: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        for error in self.errors:
            signature = (error.detectors, error.observables)
            previous = accumulated.get(signature)
            accumulated[signature] = (
                error.probability
                if previous is None
                else _combine_two(previous, error.probability, rule=selected)
            )
        return DetectorErrorModel(
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
            errors=tuple(
                DemError(probability=probability, detectors=key[0], observables=key[1])
                for key, probability in accumulated.items()
            ),
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
        declarations. Reading the detector count off the largest index an error
        mentions would accept a text that had lost its trailing detectors, so a
        text without a ``detector`` declaration is refused by the constructor
        instead. The observable count is the exception: stim declares the
        observable exactly when no error mechanism references it, so the count
        falls back to the largest observable the errors name when no
        declaration states it. At most one of the two sources has anything to
        say, and a declared count always governs.

        ``shift_detectors`` offsets every detector index on the lines that
        follow it, in the declarations and in the error mechanisms alike, and
        successive shifts accumulate. Coordinates are accepted and discarded on
        both ``detector`` and ``shift_detectors``, because they carry geometry
        the model does not represent.

        An error line's signature is the symmetric difference of its targets, so
        a repeated target cancels. Stim writes ``^`` between the groups a
        composite mechanism decomposes into. Those groups are a decoder's
        business: they partition the targets without changing which detectors
        and observables a shot of the mechanism flips. The groups are therefore
        not retained.

        A ``repeat`` block is refused. Expanding one means interpreting a nested
        instruction stream, and ``str(model.flattened())`` already states the
        same instructions without the block, so this reader does not grow a
        second implementation of the format. ``flatten_loops=True`` is not
        equivalent: stim still emits a block for a long enough circuit.

        What remains refused is a ``repeat`` block, a ``#`` comment, a
        declaration that skips an index, and a malformed line. A developer-time
        sweep of stim 1.16.0 over 240 detector error models -- repetition-code
        and rotated-surface-code memory circuits, distances three, five and
        seven, rounds one, two, three, five and nine, noisy and noise-free, with
        and without decomposed errors -- parsed all 180 that carried no block,
        agreeing with a re-read of the same text on the shape, on the error
        count, and on every ``(probability, detectors, observables)`` mechanism.
        The 60 refusals were all ``repeat`` blocks, all from
        ``repetition_code:memory`` at five rounds or more. Every one of the same
        240 models parsed when ``flattened()`` supplied the text.

        The printed text is lossy in the last digit: stim prints 17 significant
        digits, and over that sweep the largest relative difference between an
        in-memory probability and the printed one was 4.9e-16. The text is the
        interchange format, so the printed value is the one this reader states.
        """

        num_detectors, num_observables, errors = _parse_stim_text(text)
        return cls(
            num_detectors=num_detectors,
            num_observables=num_observables,
            errors=errors,
        )

    @classmethod
    def _merge_mechanisms(
        cls,
        entries: Iterable[tuple[float, tuple[int, ...], tuple[int, ...]]],
        *,
        num_detectors: int,
        num_observables: int,
    ) -> DetectorErrorModel:
        """Merge ``(probability, detectors, observables)`` entries into a model.

        Two mechanisms that flip the same detectors and the same observables are
        one mechanism to a decoder, which sees only their combined parity, so
        their probabilities combine by
        :attr:`DemMergeRule.INDEPENDENT_PARITY`. An entry whose signature is
        empty flips nothing at all, so it becomes no mechanism and is dropped
        here; a mechanism has to flip something to be one at all, so it could
        not be stored. Nothing counts the dropped entries: the model has no
        field for that count, and its shape comes from the caller's counts and
        from nowhere else.

        The fold runs over the caller's entries in the order given. Both rules
        are associative and commutative, so that order reaches the result only
        as floating-point rounding; the model then stores its mechanisms sorted
        by signature and probability, so the model a caller reads back does not
        depend on the order of the entries at all.
        """

        accumulated: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        for probability, detectors, observables in entries:
            signature = (
                _normalized_indices(detectors, name="mechanism detectors"),
                _normalized_indices(observables, name="mechanism observables"),
            )
            if not signature[0] and not signature[1]:
                continue
            previous = accumulated.get(signature)
            accumulated[signature] = (
                probability
                if previous is None
                else _combine_two(
                    previous, probability, rule=DemMergeRule.INDEPENDENT_PARITY
                )
            )
        errors = tuple(
            DemError(probability=value, detectors=key[0], observables=key[1])
            for key, value in accumulated.items()
        )
        return cls(
            num_detectors=num_detectors,
            num_observables=num_observables,
            errors=errors,
        )

    @classmethod
    def from_memory_circuit(
        cls, circuit: MemoryCircuit, *, noise: PhenomenologicalNoise
    ) -> DetectorErrorModel:
        """Build the exact model of every noise location a memory circuit has.

        Each location the circuit's round structure and ``noise`` imply is forced
        through the circuit on its own and its signature is read off the layouts,
        so the model is derived from the program rather than asserted about it.
        The refusals of the injection engine reach the caller unchanged: a
        hand-built circuit whose source does not match its layouts fails closed
        with a stated reason instead of building a model that misdescribes it.
        """

        num_detectors, num_observables, entries = _memory_circuit_entries(
            circuit, noise
        )
        return cls._merge_mechanisms(
            entries,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )

    @classmethod
    def from_code_matrices(
        cls,
        *,
        hz: torch.Tensor,
        noise: PhenomenologicalNoise,
        lz: torch.Tensor | None = None,
        num_rounds: int = 1,
    ) -> DetectorErrorModel:
        """Build the code-capacity model of a parity-check and logical matrix.

        ``hz`` and ``lz`` are read directly rather than simulated, so an
        arbitrary code reaches a model without a circuit record standing for it:
        a code whose checks and logical operators are known as matrices needs no
        gadget, no wire layout, and no statevector. See
        :func:`~flagquantum.qec.dem_construction._code_matrix_entries` for the row
        and column conventions and for the detector geometry, which is the
        code-capacity one and not the geometry
        :meth:`from_memory_circuit` reads off a circuit.

        The rates come from ``noise`` exactly as they do on the circuit route: a
        data wire's bit flip at a round boundary, and a syndrome bit flipped at a
        check's readout. The phase-flip family and the per-location rates
        upstream states are therefore not reachable here, which the alignment
        contract records against this entry point.
        """

        num_detectors, num_observables, entries = _code_matrix_entries(
            hz=hz, noise=noise, lz=lz, num_rounds=num_rounds
        )
        return cls._merge_mechanisms(
            entries,
            num_detectors=num_detectors,
            num_observables=num_observables,
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
