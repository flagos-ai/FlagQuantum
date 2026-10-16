"""Belief propagation over a detector error model, with an ordered-statistics fallback.

A detector error model states, for every independent physical error mechanism,
which detectors that mechanism flips. The matcher in
:mod:`flagquantum.qec.matching` reads a model only when every mechanism flips one
or two detectors, because it inverts the model by pairing defective detectors
along edges. This module reads the model as it is written -- one variable per
mechanism, one parity check per detector -- and estimates which mechanisms fired
by passing beliefs between the two, so a mechanism that flips three detectors is a
variable touching three checks rather than a shape it has to refuse.

**What is propagated.** Each mechanism carries the log-likelihood ratio of its own
rate, ``log((1 - p) / p)``, which is positive when the mechanism is more likely
quiet than fired, and which is the same quantity the matcher weighs an edge by. A
check sends a variable what the check's other variables imply about it; a variable
sends a check its prior combined with everything the other checks said. One round
of that exchange is one iteration, and the exchange is sum-product rather than
min-sum, so on a factor graph that is a tree the beliefs are that variable's exact
posterior rather than a bound on it and the hard decision is the
maximum-likelihood explanation of the syndrome.

**What is decided, and what is not.** The decoder reports the mechanisms it
selected, their total weight, and the logical observables those mechanisms flip.
The selected set is *a* correction that explains the syndrome -- the message
exchange settled on it, or ordered statistics solved for it -- and the observable
prediction is the parity of that correction's own labels, which is the convention
the matcher uses and not an estimate of the probability that an observable
flipped. An observable's marginal would need the joint distribution of the
mechanisms, which per-variable beliefs do not carry.

**Ordered statistics, and why it is not optional in practice.** Belief propagation
is exact on a tree and approximate on a graph with cycles, which is what a
two-dimensional code's syndrome history is, so the exchange can fail to settle and
the hard decision can fail to explain the syndrome. When that happens the decoder
does not return the last iterate: it ranks the mechanisms by how far the beliefs
have moved from the prior -- the larger the magnitude of a posterior, the more the
beliefs commit to that mechanism's value -- takes the most confident columns that
are linearly independent over GF(2) as an information set, fixes every other
mechanism at its own belief decision, and solves the information set exactly. That
is ordered-statistics decoding of order zero, the fallback upstream's
belief-propagation decoders carry, and it is visible in the result's ``converged``
flag rather than hidden: ``converged`` is ``False`` exactly when the fallback was
used, so a caller who needs the exact posterior can refuse that case instead of
trusting it.

**Refusals.** A model that states mechanisms are alternatives is refused, because
"at most one of these fires" is a correlation this factor graph cannot carry: the
beliefs exchanged below assume every mechanism is independent, and a group that is
not independent would be read as though it were. A mechanism of probability zero is
refused, as it is by the graph, because its prior ratio is infinite and no belief
can be propagated from it. A detector named outside the model is refused rather
than truncated, and a syndrome that no set of mechanisms explains is refused by
name rather than answered with the least-bad set.

**What this does not relax.** The matcher's own scope is unchanged: a graphlike
model is still the matcher's, and this is a second in-tree family for the models
the matcher refuses rather than a replacement for the first. A model that states
one signature twice is refused by the matcher and accepted here, which is not a
disagreement: the matcher minimizes over a graph whose edges *are* signatures, so
two mechanisms of one signature are one fault carrying two weights to it, while
here they are two variables, which is what "two independent faults that look the
same" means. Both families are registered in :mod:`flagquantum.qec.registry`, each
returns its own result record, and neither name is preferred over the other.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real

from ..errors import CapabilityError
from .decoding_graph import DecodingGraph
from .dem import DemError, DetectorErrorModel, _count, _normalized_indices

__all__ = ("BeliefPropagationDecodeResult", "BeliefPropagationDecoder")

#: Iterations of the message exchange before ordered statistics takes over. A
#: tree settles in a number of rounds bounded by its diameter, so the default is
#: generous for the models this decoder is for and finite for the ones it is not.
_DEFAULT_MAX_ITERATIONS = 64

#: Beliefs are exchanged as log-likelihood ratios, and a check's update takes a
#: product of the hyperbolic tangents of their halves. A message of large
#: magnitude would saturate that product to exactly one and make the inverted
#: tangent infinite, so messages are clipped to this magnitude and the product is
#: held off one. Sixty-four corresponds to a probability within 1e-28 of
#: certainty, far below any rate a detector error model states.
_CLIP = 64.0
_TANH_LIMIT = 1.0 - 1e-12


def _syndrome_mask(detection_events: Iterable[int], *, num_detectors: int) -> int:
    """Return a syndrome as a bit mask, refusing an unknown detector.

    A detector named twice states the same set as naming it once, because a
    syndrome is the set of detectors that flipped and not a multiset of reports; a
    detector outside the model is refused, because the model states no check for it
    and so no mechanism could be evidence about it.
    """

    mask = 0
    for index in detection_events:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("a syndrome must be a sequence of detector indices")
        if index < 0:
            raise ValueError("a syndrome must name non-negative detector indices")
        if index >= num_detectors:
            raise ValueError(
                f"detector {index} is outside the model's {num_detectors} detector(s)"
            )
        mask |= 1 << int(index)
    return mask


@dataclass(frozen=True)
class _FactorGraph:
    """The model as variables, checks, and the prior of each variable.

    ``checks`` holds, per detector, the mechanisms that flip it, and ``columns``
    holds, per mechanism, the detectors it flips. ``check_slots`` and
    ``variable_slots`` are the two alignments between those lists: a mechanism
    appears in several checks and a check holds several mechanisms, so passing a
    belief from one to the other needs to know where each sits in the other's
    list, and computing that once here keeps the exchange below a product rather
    than a search.
    """

    num_detectors: int
    num_mechanisms: int
    priors: tuple[float, ...]
    columns: tuple[tuple[int, ...], ...]
    checks: tuple[tuple[int, ...], ...]
    check_slots: tuple[tuple[int, ...], ...]
    variable_slots: tuple[tuple[int, ...], ...]

    @property
    def check_rows(self) -> tuple[int, ...]:
        """One bit mask over mechanisms per detector."""

        return tuple(
            sum(1 << mechanism for mechanism in touched) for touched in self.checks
        )


def _factor_graph(model: DetectorErrorModel) -> _FactorGraph:
    """Build the factor graph, refusing what it cannot carry.

    The two refusals are the ones the module docstring states -- grouped
    alternatives and a zero rate -- and neither is a size limit: both are
    properties of the model that would make the exchange answer a different
    question than the model asks.
    """

    if not isinstance(model, DetectorErrorModel):
        raise TypeError("a belief-propagation decoder reads a DetectorErrorModel")
    if model.error_ids is not None:
        raise CapabilityError(
            "belief propagation assumes every mechanism is an independent fault, and "
            "this model states mechanisms are alternatives to one another; reading "
            "such a group as independent would weigh every member as though the "
            "others could fire with it"
        )
    priors = []
    for index, error in enumerate(model.errors):
        probability = float(error.probability)
        if probability == 0.0:
            raise CapabilityError(
                f"mechanism {index} states a rate of zero, so it can never fire and "
                "carries no belief to propagate; a mechanism that no shot can select "
                "is not part of the model this decoder reads"
            )
        priors.append(math.log((1.0 - probability) / probability))
    checks: list[list[int]] = [[] for _ in range(model.num_detectors)]
    for mechanism, error in enumerate(model.errors):
        for detector in error.detectors:
            checks[detector].append(mechanism)
    columns = tuple(tuple(error.detectors) for error in model.errors)
    # ``check_slots[check][position]`` is where the mechanism at ``position`` in
    # that check's list sits in the mechanism's own list of checks, which is the
    # slot its belief for this check occupies. ``variable_slots`` is the same
    # alignment the other way round, filled in the same walk below.
    variable_slots: list[list[int]] = [[0] * len(touched) for touched in columns]
    for check, touched in enumerate(checks):
        for position, mechanism in enumerate(touched):
            variable_slots[mechanism][columns[mechanism].index(check)] = position
    check_slots = tuple(
        tuple(columns[mechanism].index(check) for mechanism in touched)
        for check, touched in enumerate(checks)
    )
    return _FactorGraph(
        num_detectors=model.num_detectors,
        num_mechanisms=model.num_errors,
        priors=tuple(priors),
        columns=columns,
        checks=tuple(tuple(touched) for touched in checks),
        check_slots=check_slots,
        variable_slots=tuple(tuple(slots) for slots in variable_slots),
    )


def _clip(value: float) -> float:
    """Hold a belief inside the range the tangents below stay finite in."""

    return max(-_CLIP, min(_CLIP, value))


def _check_messages(
    graph: _FactorGraph, syndrome: int, variable_messages: list[list[float]]
) -> list[list[float]]:
    """Return each check's belief about every variable it touches.

    The update is sum-product in the log domain, and the check's syndrome bit is
    part of it: the constraint a check states is that the variables it touches have
    the parity the syndrome bit records, so a set bit inverts the composed belief.
    Without that sign a check would answer as though its own detector had not
    fired, which is the difference between explaining the syndrome and explaining
    its complement.

    Excluding the variable's own report is what separates the check's evidence
    about it from what the variable already said.
    """

    messages: list[list[float]] = []
    for check, touched in enumerate(graph.checks):
        if not touched:
            messages.append([])
            continue
        tangents = [
            max(
                -_TANH_LIMIT,
                min(
                    _TANH_LIMIT,
                    math.tanh(
                        _clip(
                            variable_messages[mechanism][graph.check_slots[check][slot]]
                        )
                        / 2.0
                    ),
                ),
            )
            for slot, mechanism in enumerate(touched)
        ]
        sign = -1.0 if syndrome >> check & 1 else 1.0
        message = []
        for excluded in range(len(touched)):
            product = sign
            for slot, value in enumerate(tangents):
                if slot != excluded:
                    product *= value
            product = max(-_TANH_LIMIT, min(_TANH_LIMIT, product))
            message.append(2.0 * math.atanh(product))
        messages.append(message)
    return messages


def _variable_messages(
    graph: _FactorGraph, check_messages: list[list[float]]
) -> list[list[float]]:
    """Return each variable's belief for every check it touches.

    A variable's belief for one check is its prior plus what every *other* check
    said about it; the exclusion keeps a check from hearing its own belief back,
    which would count the same evidence twice.
    """

    totals = list(graph.priors)
    for check, touched in enumerate(graph.checks):
        for slot, mechanism in enumerate(touched):
            totals[mechanism] += check_messages[check][slot]
    messages = [[0.0] * len(touched) for touched in graph.columns]
    for mechanism, touched in enumerate(graph.columns):
        for slot, check in enumerate(touched):
            messages[mechanism][slot] = (
                totals[mechanism]
                - check_messages[check][graph.variable_slots[mechanism][slot]]
            )
    return messages


def _posteriors(graph: _FactorGraph, check_messages: list[list[float]]) -> list[float]:
    """Return one posterior log-likelihood ratio per variable."""

    posteriors = list(graph.priors)
    for check, touched in enumerate(graph.checks):
        for slot, mechanism in enumerate(touched):
            posteriors[mechanism] += check_messages[check][slot]
    return posteriors


def _decision(posteriors: list[float]) -> int:
    """The most likely value of every variable under the given beliefs."""

    selected = 0
    for mechanism, posterior in enumerate(posteriors):
        if posterior < 0.0:
            selected |= 1 << mechanism
    return selected


def _flips(graph: _FactorGraph, selected: int) -> int:
    """The detectors the selected mechanisms flip."""

    flipped = 0
    for mechanism, touched in enumerate(graph.columns):
        if selected >> mechanism & 1:
            for detector in touched:
                flipped ^= 1 << detector
    return flipped


def _explains(graph: _FactorGraph, syndrome: int, selected: int) -> bool:
    """Whether the selected mechanisms' flips are exactly the syndrome."""

    return _flips(graph, selected) == syndrome


def _weight(graph: _FactorGraph, selected: int) -> float:
    """The total prior weight of the selected mechanisms."""

    return sum(
        prior
        for mechanism, prior in enumerate(graph.priors)
        if selected >> mechanism & 1
    )


def _ordered_statistics(
    graph: _FactorGraph, syndrome: int, posteriors: list[float]
) -> int:
    """Solve for the most confident independent columns that explain the syndrome.

    The mechanisms are ranked by how far their posterior has moved from their
    prior, and the most confident columns that are linearly independent over GF(2)
    become the information set. Every other mechanism is fixed at its own belief
    decision, and the information set is then solved exactly, so the result
    explains the syndrome whenever any set of mechanisms does.

    Raises:
        CapabilityError: No set of mechanisms explains the syndrome, which the
            elimination detects as a check row that has become dependent on the
            others and whose syndrome bit disagrees with the values already fixed.
    """

    order = sorted(
        range(graph.num_mechanisms),
        key=lambda mechanism: (-abs(posteriors[mechanism]), mechanism),
    )
    rows = [
        [row, (syndrome >> check) & 1] for check, row in enumerate(graph.check_rows)
    ]
    pivot_row: dict[int, int] = {}
    used = [False] * len(rows)
    for mechanism in order:
        pivot = None
        for index, row in enumerate(rows):
            if not used[index] and row[0] >> mechanism & 1:
                pivot = index
                break
        if pivot is None:
            # Linearly dependent on the columns already chosen, so it stays
            # outside the information set and keeps its belief decision.
            continue
        used[pivot] = True
        pivot_row[mechanism] = pivot
        for index, row in enumerate(rows):
            if index != pivot and row[0] >> mechanism & 1:
                row[0] ^= rows[pivot][0]
                row[1] ^= rows[pivot][1]
    free = _decision(posteriors) & ~_pivot_mask(pivot_row)
    for index, (mask, parity) in enumerate(rows):
        if used[index]:
            continue
        if bin(mask & free).count("1") % 2 != parity:
            raise CapabilityError(
                "no set of mechanisms explains this syndrome: the detector error "
                "model's checks state an inconsistency no correction can satisfy, so "
                "this syndrome did not come from this model"
            )
    selected = free
    for mechanism, index in pivot_row.items():
        value = rows[index][1]
        if bin(rows[index][0] & free).count("1") % 2:
            value ^= 1
        if value:
            selected |= 1 << mechanism
    return selected


def _pivot_mask(pivot_row: dict[int, int]) -> int:
    """The information set as a bit mask over mechanisms."""

    mask = 0
    for mechanism in pivot_row:
        mask |= 1 << mechanism
    return mask


def _model_from_graph(graph: DecodingGraph) -> DetectorErrorModel:
    """Read a decoding graph as the detector error model it states.

    The lift is the graph's own definition read backwards, and it is exact for
    what the graph carries: every edge becomes one mechanism, the graph's boundary
    node becomes the fact that the mechanism flips one detector, and the edge's
    observable labels become the mechanism's. It is deliberately *not* the
    inverse of :meth:`DecodingGraph.from_detector_error_model`, which drops a
    mechanism of zero probability and a mechanism that flips no detector because a
    graph has no edge for either; a graph therefore lifts to the model of its own
    edges rather than to whatever model it was derived from.
    """

    if not isinstance(graph, DecodingGraph):
        raise TypeError("a decoding graph is built from a DecodingGraph")
    boundary = graph.boundary_node
    errors = []
    for edge in graph.edges:
        first, second = edge.detectors
        detectors = (first,) if second == boundary else (first, second)
        errors.append(
            DemError(
                probability=edge.probability,
                detectors=detectors,
                observables=edge.observables,
            )
        )
    return DetectorErrorModel(
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class BeliefPropagationDecodeResult:
    """The mechanisms one decode selected and the observables they flip.

    ``mechanisms`` holds the indices of the selected mechanisms, ascending, in the
    order the model's own parity matrices number their columns, so a caller reads
    one column's support, labels and rate from one index. ``weight`` is the total
    log-likelihood ratio of those mechanisms, the same quantity the matcher weighs
    an edge by and therefore comparable between the two. ``converged`` is ``True``
    exactly when the message exchange settled on a set that explains the syndrome,
    and ``iterations`` is how many exchanges were run: zero means the priors alone
    explained it, and a value equal to the decoder's iteration limit means the
    exchange was cut off and ordered statistics answered instead.
    """

    converged: bool
    observables: tuple[int, ...]
    mechanisms: tuple[int, ...]
    weight: float
    iterations: int

    def __post_init__(self) -> None:
        if not isinstance(self.converged, bool):
            raise TypeError("decode convergence must be a boolean")
        object.__setattr__(
            self,
            "mechanisms",
            _normalized_indices(self.mechanisms, name="selected mechanisms"),
        )
        object.__setattr__(
            self,
            "observables",
            _normalized_indices(self.observables, name="decoded observables"),
        )
        if isinstance(self.weight, bool) or not isinstance(self.weight, Real):
            raise TypeError("decode weight must be a real number")
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError("decode weight must be finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self,
            "iterations",
            _count(self.iterations, name="belief-propagation iterations"),
        )


@dataclass(frozen=True)
class BeliefPropagationDecoder:
    """Decode a syndrome by exchanging beliefs over the model's own checks.

    The decoder reads a detector error model and nothing else, and it applies to
    any model rather than to the graphlike ones: a mechanism that flips three
    detectors is a variable touching three checks, which is the shape the model
    states rather than a shape to refuse.

    Attributes:
        model: The detector error model the decoder reads.
        max_iterations: Exchanges of beliefs before ordered statistics takes over.
        osd: Whether to fall back to ordered statistics rather than refuse when the
            exchange does not settle.
    """

    model: DetectorErrorModel
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    osd: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, DetectorErrorModel):
            raise TypeError("a belief-propagation decoder reads a DetectorErrorModel")
        limit = _count(self.max_iterations, name="iteration limit")
        if limit < 1:
            raise ValueError("belief propagation needs at least one iteration")
        object.__setattr__(self, "max_iterations", limit)
        if not isinstance(self.osd, bool):
            raise TypeError("the ordered-statistics flag must be a boolean")
        # The refusals that are properties of the model are answered here, while
        # the caller still holds the model, rather than at the first syndrome.
        _factor_graph(self.model)

    @classmethod
    def from_detector_error_model(
        cls,
        model: DetectorErrorModel,
        *,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        osd: bool = True,
    ) -> BeliefPropagationDecoder:
        """Build the decoder for one detector error model.

        Raises:
            CapabilityError: The model states mechanisms are alternatives to one
                another, or a mechanism's rate is zero.
            TypeError: The argument is not a detector error model.
        """

        return cls(model=model, max_iterations=max_iterations, osd=osd)

    @classmethod
    def from_decoding_graph(
        cls,
        graph: DecodingGraph,
        *,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        osd: bool = True,
    ) -> BeliefPropagationDecoder:
        """Build the decoder for the detector error model a graph states.

        A graph is a narrower statement than a model -- it holds one edge per
        graphlike mechanism of positive rate -- so this route reads the graph's
        own edges as the model, which is exact for what the graph carries.

        Raises:
            TypeError: The argument is not a decoding graph.
            CapabilityError: An edge states a rate of zero.
        """

        return cls(
            model=_model_from_graph(graph), max_iterations=max_iterations, osd=osd
        )

    @property
    def factor_graph(self) -> _FactorGraph:
        """The variables, checks and priors the exchange runs over."""

        return _factor_graph(self.model)

    def decode(self, detection_events: Iterable[int]) -> BeliefPropagationDecodeResult:
        """Return the mechanisms that explain a syndrome and their logical flips.

        Args:
            detection_events: Indices of the detectors that flipped, in any order.

        Returns:
            The selected mechanisms, the logical observables they flip, their total
            weight, whether the exchange settled, and how many exchanges ran.

        Raises:
            CapabilityError: No set of mechanisms explains the syndrome, or the
                exchange did not settle and ``osd`` is off.
            ValueError: A detector index is outside the model.
        """

        graph = _factor_graph(self.model)
        syndrome = _syndrome_mask(detection_events, num_detectors=graph.num_detectors)
        posteriors = list(graph.priors)
        selected = _decision(posteriors)
        if _explains(graph, syndrome, selected):
            # The priors alone explain this syndrome, so no exchange is run: the
            # empty syndrome is the case a caller meets most often, and it settles
            # before the first message rather than after one.
            return self._result(graph, selected, converged=True, iterations=0)
        variable_messages = [[0.0] * len(touched) for touched in graph.columns]
        iterations = 0
        while iterations < self.max_iterations:
            check_messages = _check_messages(graph, syndrome, variable_messages)
            variable_messages = _variable_messages(graph, check_messages)
            posteriors = _posteriors(graph, check_messages)
            selected = _decision(posteriors)
            iterations += 1
            if _explains(graph, syndrome, selected):
                return self._result(
                    graph, selected, converged=True, iterations=iterations
                )
        if not self.osd:
            raise CapabilityError(
                "belief propagation did not settle in "
                f"{self.max_iterations} iteration(s) on this syndrome and ordered "
                "statistics is off; the last iterate does not explain the syndrome, "
                "so it is refused rather than returned"
            )
        selected = _ordered_statistics(graph, syndrome, posteriors)
        if not _explains(graph, syndrome, selected):
            # Ordered statistics solved the reduced system exactly, so this can
            # only mean the elimination and the parity checks disagree; returning
            # the set anyway would hand the caller a correction that does not
            # explain the syndrome it was asked about.
            raise CapabilityError(
                "ordered statistics produced a set of mechanisms that does not "
                "explain this syndrome, which means the model's own checks are "
                "inconsistent with it"
            )
        return self._result(graph, selected, converged=False, iterations=iterations)

    def _result(
        self,
        graph: _FactorGraph,
        selected: int,
        *,
        converged: bool,
        iterations: int,
    ) -> BeliefPropagationDecodeResult:
        """Read a selected mechanism set as the decoder's result record."""

        mechanisms = tuple(
            mechanism
            for mechanism in range(graph.num_mechanisms)
            if selected >> mechanism & 1
        )
        flipped: set[int] = set()
        for mechanism in mechanisms:
            for index in self.model.errors[mechanism].observables:
                if index in flipped:
                    flipped.remove(index)
                else:
                    flipped.add(index)
        return BeliefPropagationDecodeResult(
            converged=converged,
            observables=tuple(sorted(flipped)),
            mechanisms=mechanisms,
            weight=_weight(graph, selected),
            iterations=iterations,
        )
