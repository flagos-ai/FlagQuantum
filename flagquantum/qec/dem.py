"""Exact detector error models for code-independent memory experiments.

A detector error model names every independent physical error mechanism by the
detectors and logical observables it flips. Construction is exact and does not
sample: the reference gate set is Clifford and every configured channel is a
Pauli channel, so forcing one mechanism through the circuit yields a
deterministic signature. ``DetectorErrorModel.from_memory_circuit`` builds a
model that way: it enumerates the noise locations a ``PhenomenologicalNoise``
record configures, drops the ones that flip nothing at all, and merges the ones
that flip the same detectors and observables.

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
from typing import Literal

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .circuit import MeasurementRef, MemoryCircuit
from .noise import PhenomenologicalNoise

_DETECTOR_PREFIX = "D"
_OBSERVABLE_PREFIX = "L"
_SEPARATOR = "^"


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
        their probabilities combine by ``p1 * (1 - p2) + p2 * (1 - p1)``. An
        entry whose signature is empty flips nothing at all, so it becomes no
        mechanism and is dropped here. Nothing counts the dropped entries: the
        model has no field for that count, and its shape comes from the caller's
        counts and from nowhere else.
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
                else previous * (1.0 - probability) + probability * (1.0 - previous)
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

        Each location :func:`_mechanisms` enumerates is forced through the
        circuit on its own and its signature read off the layouts, so the model
        is derived from the program rather than asserted about it. The refusals
        of the injection engine reach the caller unchanged: a hand-built circuit
        whose source does not match its layouts fails closed with a stated
        reason instead of building a model that misdescribes it.
        """

        if not isinstance(circuit, MemoryCircuit):
            raise TypeError("circuit must be a MemoryCircuit")
        if not isinstance(noise, PhenomenologicalNoise):
            raise TypeError("noise must be a PhenomenologicalNoise")
        entries: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for mechanism in _mechanisms(circuit, noise):
            if mechanism.kind == "data":
                source = _inject_data_flip(
                    circuit,
                    round_index=mechanism.round_index,
                    wire=mechanism.wire,
                )
            elif mechanism.kind == "measurement":
                source = _inject_measurement_flip(
                    circuit,
                    round_index=mechanism.round_index,
                    ancilla_wire=mechanism.wire,
                )
            else:
                # Unreachable through ``_mechanisms``, which sets the field from
                # the annotation; stated for a caller that builds records itself,
                # where a bare ``else`` would read an unknown kind as a
                # measurement flip whenever the wire is a declared ancilla.
                raise ValueError(f"unknown mechanism kind {mechanism.kind!r}")
            detectors, observables = _forced_signature(circuit, source)
            entries.append((mechanism.probability, detectors, observables))
        return cls._merge_mechanisms(
            entries,
            # The model describes the circuit's own layouts, not a count
            # re-derived from the code. Stage 1 pins both to each other, so
            # ``len(circuit.code.checks) * (circuit.rounds + 1)`` for the
            # detector count and ``len(circuit.code.logical_observables)`` for
            # the observable count are equivalent mutants.
            num_detectors=len(circuit.detectors),
            num_observables=len(circuit.observables),
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


# The forced-error engine drives the memory circuit one mechanism at a time. It
# injects a single physical error into the bounded hybrid-compiler program the
# circuit carries, executes the result, and reads the detector and observable
# flips off the Stage 1 layouts. Nothing here is public: a mechanism is a
# physical location, and the model above is what a caller is given.

_ROUND_LOOP_ANCHOR = "    for round_index in range(rounds):\n"
_FLIP_INDENT = "        "


@dataclass(frozen=True)
class _Mechanism:
    """One physical noise location, by kind, round, and wire.

    The record carries the coordinates the matching injector needs rather than a
    rendered source. A caller that fires a set of mechanisms at once has to
    re-inject them into the one program it executes, and it cannot rebuild a
    source from a string alone; carrying coordinates also keeps injection in
    exactly one place. ``kind`` is ``"data"`` for a flip on a data wire, where
    ``wire`` is that data wire, and ``"measurement"`` for a flip on a check's
    syndrome measurement, where ``wire`` is that check's ancilla. The kind is a
    literal rather than a free string, so a caller that builds a record by hand
    is told at the type checker which two flips exist.
    """

    kind: Literal["data", "measurement"]
    round_index: int
    wire: int
    probability: float


def _mechanisms(
    circuit: MemoryCircuit, noise: PhenomenologicalNoise
) -> tuple[_Mechanism, ...]:
    """Return every noise location a probability makes possible, in order.

    Data flips come first — one per round per data wire — and measurement flips
    second, one per round per check. Both loops walk the code's own tuples in
    order, which is the order the emitted program measures in. A location whose
    probability is zero cannot flip anything, so it is not enumerated at all: a
    caller that counts mechanisms then counts exactly what can happen.

    A code that declares a data wire twice is refused rather than enumerated
    twice. Its second copy would be the same physical location as the first
    with the same signature, so merging them states one location's rate as two
    independent flips, ``p * (1 - p) + p * (1 - p)``, instead of ``p`` — a wrong
    model with no signal. A code whose checks share an ancilla wire is refused
    for the same reason: both checks record their syndrome bit at one position
    in the classical register, so the later check's position overwrites the
    earlier one's and each round's detectors read that one bit for both. The
    injection engine refuses that shape too, but only where it injects — its
    anchor is the measurement line, so a model built from data flips alone never
    reaches it — which is why the refusal is stated here, before any mechanism
    is enumerated.
    """

    data_wires = circuit.code.data_wires
    if len(set(data_wires)) != len(data_wires):
        raise ValueError(
            "the code declares a repeated data wire, so a mechanism at that "
            "location would be enumerated twice and merged with itself"
        )
    ancilla_wires = [check.ancilla_wire for check in circuit.code.checks]
    if len(set(ancilla_wires)) != len(ancilla_wires):
        repeated = sorted(
            wire for wire in set(ancilla_wires) if ancilla_wires.count(wire) > 1
        )
        raise ValueError(
            f"the code declares a repeated check ancilla wire ({repeated[0]}): "
            "two checks record one syndrome bit, so the model would read the "
            "same bit for both"
        )
    mechanisms: list[_Mechanism] = []
    if noise.data_flip:
        for round_index in range(circuit.rounds):
            for wire in data_wires:
                mechanisms.append(
                    _Mechanism(
                        kind="data",
                        round_index=round_index,
                        wire=wire,
                        probability=noise.data_flip,
                    )
                )
    if noise.measurement_flip:
        for round_index in range(circuit.rounds):
            for check in circuit.code.checks:
                mechanisms.append(
                    _Mechanism(
                        kind="measurement",
                        round_index=round_index,
                        wire=check.ancilla_wire,
                        probability=noise.measurement_flip,
                    )
                )
    return tuple(mechanisms)


def _checked_round(circuit: MemoryCircuit, round_index: int, *, label: str) -> int:
    """Return ``round_index`` once it names a round this circuit configures."""

    if isinstance(round_index, bool) or not isinstance(round_index, Integral):
        raise TypeError(f"{label} must be an integer")
    if round_index not in range(circuit.rounds):
        raise ValueError(
            f"{label} {round_index} is outside the configured rounds "
            f"0..{circuit.rounds - 1}"
        )
    return int(round_index)


def _checked_wire(wires: tuple[int, ...], wire: int, *, label: str) -> int:
    """Return ``wire`` once the code declares it."""

    if isinstance(wire, bool) or not isinstance(wire, Integral):
        raise TypeError(f"{label} must be an integer")
    if wire not in wires:
        raise ValueError(f"{label} {wire} is not declared by the code")
    return int(wire)


def _single_anchor(source: str, anchor: str, *, label: str) -> int:
    """Return the offset of ``anchor``, refusing a source that repeats or lacks it.

    An anchor that occurs twice cannot say which occurrence the mechanism
    belongs to, so it is refused rather than resolved to the first one.
    """

    occurrences = source.count(anchor)
    if occurrences != 1:
        raise ValueError(
            f"{label} must occur exactly once in the source, found {occurrences}"
        )
    return source.index(anchor)


def _forced_flip(round_index: int, wire: int) -> str:
    """Return the guarded single-qubit ``X`` that forces one error."""

    return (
        f"{_FLIP_INDENT}if round_index == {round_index}:\n"
        f"{_FLIP_INDENT}    qp.X(wires={wire})\n"
    )


def _inject_data_flip(circuit: MemoryCircuit, *, round_index: int, wire: int) -> str:
    """Return ``circuit``'s source with one ``X`` forced on a data wire.

    The flip is guarded by ``if round_index == <round_index>:`` and inserted at
    the start of that round, before any of the round's CNOTs, so the error opens
    the frame the round's detectors compare against.
    """

    checked_round = _checked_round(circuit, round_index, label="data-flip round")
    checked_wire = _checked_wire(circuit.code.data_wires, wire, label="data-flip wire")
    offset = _single_anchor(
        circuit.source, _ROUND_LOOP_ANCHOR, label="the round loop anchor"
    )
    end = offset + len(_ROUND_LOOP_ANCHOR)
    return (
        f"{circuit.source[:end]}"
        f"{_forced_flip(checked_round, checked_wire)}"
        f"{circuit.source[end:]}"
    )


def _inject_measurement_flip(
    circuit: MemoryCircuit, *, round_index: int, ancilla_wire: int
) -> str:
    """Return ``circuit``'s source with one check measurement forced to flip.

    The flip is an ``X`` on the ancilla immediately before the check measures
    it, guarded by ``if round_index == <round_index>:``. A check's ancilla wire
    is unique to it, so the anchor names exactly one check; a source where that
    line is missing or repeated is refused rather than injected into the wrong
    check.
    """

    checked_round = _checked_round(circuit, round_index, label="measurement-flip round")
    checked_wire = _checked_wire(
        circuit.code.ancilla_wires, ancilla_wire, label="measurement-flip ancilla"
    )
    offset = _single_anchor(
        circuit.source,
        f"{_FLIP_INDENT}last = qp.measure(wires={checked_wire})\n",
        label="the check measurement anchor",
    )
    return (
        f"{circuit.source[:offset]}"
        f"{_forced_flip(checked_round, checked_wire)}"
        f"{circuit.source[offset:]}"
    )


def _forced_signature(
    circuit: MemoryCircuit, source: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the detectors and observables that ``source``'s forced error flips.

    The source is lowered and executed twice, and the two shots must agree on
    the *flip set*: a mechanism whose signature depends on the trajectory is not
    a Pauli mechanism in the reference gate set, so it is refused instead of
    contributing a signature. The shot is read through the layouts, never off the
    raw register: a detector XORs the measurements its parity names, and an
    observable XORs the terminal samples over the support of its Pauli.

    The register bits themselves need not agree, and for a code with X-type
    checks they do not: an X-type ancilla is prepared in ``|+>``, so its
    round-zero outcome is random. What the model records is the flip a mechanism
    causes relative to the noiseless outcome, and that is the parity the layouts
    define — deterministic for the steady-state X-type detectors precisely
    because the two rounds it compares were projected into the same eigenstate.
    Requiring the raw bits to agree would refuse every code with an X-type check,
    including the rotated surface code, for a randomness the model does not see.

    A syndrome measurement is read at ``round_index * len(checks) +
    position_in_checks``, where the position is the check's index in the code's
    ``checks`` tuple. That is the order the emitted loop measures in, which is
    what the classical register records; ``CodeCheck.index`` is a label the code
    chooses and need not be that order.
    """

    program = capture_source(source, (INDEX,))
    lowered = lower_dynamic_program(
        program,
        (circuit.rounds,),
        max_dynamic_measurements=circuit.rounds * len(circuit.code.checks),
    )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit, shots=2, seed=0, strategy="trajectory"
    )
    classical: list[list[int]] = execution.classical_bits.tolist()
    samples: list[list[int]] = execution.samples.tolist()
    positions = {
        check.ancilla_wire: position
        for position, check in enumerate(circuit.code.checks)
    }
    checks = len(circuit.code.checks)

    def measurement_bit(
        classical_row: list[int], sample_row: list[int], reference: MeasurementRef
    ) -> int:
        """Return the recorded bit one layout reference names.

        A syndrome reference is read at the position of the check that owns its
        ancilla, so an ancilla no check owns has no recorded bit to read. A
        layout may name one — ``MemoryCircuit`` ties the reference to the code's
        declared ancilla wires, not to its checks — so the lookup states the
        failure instead of raising a bare ``KeyError``.
        """

        if reference.round_index is None:
            return sample_row[reference.wire]
        position = positions.get(reference.wire)
        if position is None:
            raise ValueError(
                f"detector syndrome measurement names ancilla wire "
                f"{reference.wire}, which no check owns"
            )
        return classical_row[reference.round_index * checks + position]

    def flip_set(
        classical_row: list[int], sample_row: list[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        detectors = tuple(
            detector.index
            for detector in circuit.detectors.detectors
            if sum(
                measurement_bit(classical_row, sample_row, reference)
                for reference in detector.parity
            )
            % 2
            == 1
        )
        observables = tuple(
            observable.index
            for observable in circuit.observables.observables
            if sum(sample_row[wire] for wire in observable.pauli.support) % 2 == 1
        )
        return detectors, observables

    signature = flip_set(classical[0], samples[0])
    if signature != flip_set(classical[1], samples[1]):
        raise ValueError(
            "the forced error's detector and observable flips are not "
            "deterministic across trajectories, so it is not a Pauli mechanism "
            "in the reference gate set"
        )
    return signature


__all__ = ("DemError", "DemSample", "DetectorErrorModel")
