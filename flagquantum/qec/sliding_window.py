"""Sliding-window minimum-weight matching over a decoding graph.

A syndrome is the set of detectors that flipped, and a graphlike detector error
model explains one with a minimum-weight T-join. :mod:`flagquantum.qec.matching`
computes that join exactly by enumerating the ways to pair the defective
detectors, and it therefore refuses a syndrome whose defect count exceeds its
budget. That budget is what a long-lived memory experiment runs into: the
syndrome grows with the number of rounds, and this package's own distance-three
patch carries well over a hundred defective detectors over 200 rounds at three
percent noise -- 110 to 191 across forty seeded shots, against a budget of 20 --
far past any enumeration the exact matcher will pay for. The history a memory
experiment produces is exactly the regime where the exact matcher stops
answering, and it is also the regime a decoder for a real experiment works in.

This module answers those syndromes by deciding the history a band at a time. The
detectors are cut into windows of ``window`` consecutive detectors advancing by
``commit``; each window is decoded by the exact matcher over the graph of the
mechanisms that begin inside it, and the mechanisms of that answer whose first
endpoint lies in the window's leading ``commit`` detectors are committed while
the rest are left for a later window. A committed mechanism toggles its endpoints
out of the residual syndrome, and each window decides its band against what the
windows before it left behind. Upstream states this family as a decoder named
``sliding_window``; this is the same technique over a decoding graph this package
already builds, registered as ``sliding_window_matching``.

**Why the band is decided exactly.** Three facts hold for the window that decides
the band ``[start, start + commit)``. Its graph holds every mechanism of the
model whose first endpoint lies in the window, and a mechanism incident to a
detector of the band has its first endpoint at or before that detector, so every
mechanism incident to the band is in the window. Every such mechanism is
committed, because committing is exactly the test ``first < start + commit``.
And the window's answer is a T-join of the window's syndrome, so a band detector
has odd degree in that answer precisely when the residual called it defective.
Together those say the committed mechanisms toggle each band detector exactly as
often as it needs, so each band is explained as it is decided and the residual is
empty once the last window has been decided. That is checked rather than assumed:
a non-empty residual is refused rather than returned as a correction.

**Why the mechanisms a window reaches past its last detector are kept.** A
mechanism that begins inside the window and ends past its last detector is still
a mechanism of the model, and the window offers it to the matcher as a step to
the model's boundary instead of dropping it. Dropping it is what would leave a
detector of the band with nothing to pair against, so keeping it is what makes
the band answerable at all. Such a mechanism is never committed by the window
that cannot follow it, and the floor is what proves that rather than a separate
test. If its first endpoint were inside the band then its last would be before
``start + commit + span``, which the floor puts at or before the window's last
detector, so the window would have been able to follow it. What a caller receives
is therefore never a mechanism the model does not have. The window's own answer does treat the region past its last detector as
boundary, which is the approximation a window makes: the mechanisms of that
region are seen as reaching the model's boundary rather than as continuing, which
can make a band's decision cheaper than the history supports. What survives that
approximation is the record itself. Every committed mechanism is a mechanism of
the model, and the committed mechanisms have the whole syndrome as their
odd-degree set, so they are a genuine T-join and their total weight is at least
the exact matcher's for the same syndrome.

**Why a window has a floor.** A mechanism begins at or before the band's last
detector and spans at most the model's widest mechanism, so with
``window >= commit + span`` every mechanism starting inside the band ends before
the window's last detector and the window never treats it as a boundary step. The
constructor measures ``span`` from the graph and refuses a narrower window by
name, because below that floor the decoder would commit a mechanism whose far
endpoint the window rewrote. One exemption is admitted and it is sound for the
same reason: a window that already reaches the end of the graph has nothing past
its last detector, so it rewrites nothing whatever its width. That exemption is
what keeps this decoder exact when it is asked to be, since one window covering
the whole graph is ``MinimumWeightMatchingDecoder`` mechanism for mechanism, and
that identity is what its replacement test asserts.

**What the narrow window buys, and what it costs.** The defect budget bounds a
window's syndrome rather than the whole history's, which is how a syndrome the
exact matcher refuses becomes a syndrome this decoder answers. What it costs is
that a narrow window is a decision about a window: the mechanisms it selects can
differ from the exact matcher's even where the observables agree, and the
agreement a caller gets is bought with the window's width rather than guaranteed.
No threshold and no logical error rate is estimated here, and no accuracy claim
is made for a window narrower than the whole graph.

**What is not here.** The window is measured in detectors, and the default band
is one syndrome round only because the widest mechanism of a memory model
connects a detector to its counterpart one round later, which makes the widest
span the detectors-per-round; a graph built some other way states no rounds, and a
caller who wants a band of rounds states that count from the code the model was
built from. There is also no streaming entry point over a live detection stream
and no chunk seam to feed one: this decoder takes a whole syndrome and answers it
band by band, and the seams a streaming decoder needs are the ``DemChunkSpec``
family upstream states and this package does not have.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from ..errors import CapabilityError
from .decoding_graph import DecodingGraph, DecodingGraphEdge
from .dem import DetectorErrorModel
from .matching import (
    _DEFAULT_DEFECT_BUDGET,
    MatchingDecodeResult,
    MinimumWeightMatchingDecoder,
    _defects,
)

__all__ = ("SlidingWindowMatchingDecoder",)


def _mechanism_span(graph: DecodingGraph) -> int:
    """How many detectors the model's widest mechanism spans.

    A boundary mechanism spans no pair of detectors, so it contributes nothing:
    it is the model's own way out of the patch rather than a distance between two
    detectors.
    """

    return max(
        (
            second - first
            for first, second in (edge.detectors for edge in graph.edges)
            if second != graph.boundary_node
        ),
        default=0,
    )


def _window_graph(graph: DecodingGraph, *, start: int, stop: int) -> DecodingGraph:
    """The model's mechanisms that begin in ``[start, stop)``.

    The window keeps the model's detector and observable numbering, so a
    mechanism it selects is a mechanism of the model at its own endpoints. Its
    boundary is the model's boundary, and a mechanism that ends past the window's
    last detector is offered at the model's boundary as well, because past a
    window's last detector is where that window stops seeing.
    """

    boundary = graph.boundary_node
    edges = []
    for edge in graph.edges:
        first, second = edge.detectors
        if first < start or first >= stop:
            continue
        # The window sees the detectors before `stop` and nothing past them, and
        # past `stop` is where the model's boundary already sits for a mechanism
        # the window cannot follow. `stop <= num_detectors == boundary`, so this
        # test also sends a mechanism to the model's boundary to itself.
        edges.append(
            DecodingGraphEdge(
                detectors=(first, boundary if second >= stop else second),
                probability=edge.probability,
                observables=edge.observables,
            )
        )
    return DecodingGraph(
        num_detectors=boundary,
        num_observables=graph.num_observables,
        edges=tuple(edges),
    )


def _ordering_key(
    edge: DecodingGraphEdge,
) -> tuple[tuple[int, int], tuple[int, ...], float]:
    """The key ``DecodingGraph`` orders its own edges by."""

    return (edge.detectors, edge.observables, edge.probability)


@dataclass(frozen=True)
class SlidingWindowMatchingDecoder:
    """Decode a syndrome by matching one band of detectors at a time.

    The decoder reads the graph a detector error model defines and nothing else,
    and it answers a syndrome of any size by deciding ``commit`` detectors at a
    time against a window of ``window``. The defect budget bounds a window's
    syndrome rather than the whole syndrome, which is the property that makes a
    history longer than the exact matcher's budget decodable at all.

    Attributes:
        graph: The weighted graph the decoder searches, in its own numbering.
        commit: How many detectors one window decides. The default is the number
            of detectors the model's widest mechanism spans, which is one
            syndrome round for a memory circuit, or one detector for a graph
            whose mechanisms all reach the boundary.
        window: How many detectors one window sees. The default is three times
            ``commit``, a band with a lookahead of two bands. Both numbers are
            the trade this decoder makes between the work one syndrome costs and
            how closely it agrees with the exact matcher, so a caller who needs a
            particular trade states them.
        max_defects: Largest syndrome one window's exact matching accepts.

    Raises:
        CapabilityError: ``window`` is narrower than the band plus the model's
            widest mechanism, and does not already reach the end of the graph,
            so the window would commit a mechanism whose far endpoint it
            rewrote.
        ValueError: ``commit`` is not positive, ``window`` is smaller than
            ``commit``, or the defect budget is not positive.
    """

    graph: DecodingGraph
    commit: int | None = None
    window: int | None = None
    max_defects: int = _DEFAULT_DEFECT_BUDGET

    def __post_init__(self) -> None:
        if not isinstance(self.graph, DecodingGraph):
            raise TypeError("graph must be a DecodingGraph")
        if isinstance(self.max_defects, bool) or not isinstance(
            self.max_defects, Integral
        ):
            raise TypeError("defect budget must be an integer")
        if self.max_defects < 1:
            raise ValueError("defect budget must be at least one")
        object.__setattr__(self, "max_defects", int(self.max_defects))
        span = _mechanism_span(self.graph)
        commit = max(span, 1) if self.commit is None else self.commit
        window = 3 * commit if self.window is None else self.window
        for name, value in (("commit", commit), ("window", window)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"the {name} must be an integer")
            if value < 1:
                raise ValueError(f"the {name} must be at least one")
        commit = int(commit)
        window = int(window)
        if window < commit:
            raise ValueError(
                f"a window of {window} detectors cannot decide a band of {commit} "
                "detectors: the window must cover at least the band it decides"
            )
        if window < commit + span and window < self.graph.num_detectors:
            raise CapabilityError(
                f"sliding-window matching needs a window of at least {commit + span} "
                f"detectors to decide a band of {commit} on this graph and was given "
                f"{window}: the model's widest mechanism spans {span} detectors, so a "
                "narrower window would commit a mechanism whose far endpoint it had "
                "rewritten as a step to the boundary"
            )
        object.__setattr__(self, "commit", commit)
        object.__setattr__(self, "window", window)

    @property
    def mechanism_span(self) -> int:
        """How many detectors the model's widest mechanism spans."""

        return _mechanism_span(self.graph)

    @classmethod
    def from_detector_error_model(
        cls,
        model: DetectorErrorModel,
        *,
        commit: int | None = None,
        window: int | None = None,
        max_defects: int = _DEFAULT_DEFECT_BUDGET,
    ) -> SlidingWindowMatchingDecoder:
        """Build the decoder for the graph a detector error model defines.

        The model is required to state each mechanism's signature once, for the
        reason the exact matcher states: two mechanisms with one signature are
        one fault carrying two weights, and a window that matched against either
        of them would be choosing a weight the model does not have.

        Raises:
            CapabilityError: The model is not graphlike, or the window is
                narrower than the band plus the model's widest mechanism.
            ValueError: A mechanism's probability has no finite non-negative edge
                weight, or two mechanisms flip the same detectors and
                observables.
        """

        model.require_unique_mechanisms()

        return cls(
            graph=DecodingGraph.from_detector_error_model(model),
            commit=commit,
            window=window,
            max_defects=max_defects,
        )

    def decode(self, detection_events: Iterable[int]) -> MatchingDecodeResult:
        """Return the mechanisms the windows commit and the observables they flip.

        The windows advance by ``commit`` detectors, and each commits exactly the
        mechanisms whose first endpoint lies in its leading ``commit`` detectors,
        whether or not the window reaches the end of the graph, so a mechanism is
        decided by exactly one window: the one whose band holds its first
        endpoint.

        Args:
            detection_events: Indices of the detectors that flipped, in any
                order.

        Returns:
            The predicted logical observables, the mechanisms the windows
            committed, and their total weight.

        Raises:
            CapabilityError: One window's syndrome holds more defective detectors
                than ``max_defects``, one window cannot explain its syndrome --
                which is what a detector whose only mechanism begins before the
                window looks like -- or the bands did not explain the whole
                syndrome, which this method checks rather than assumes.
            ValueError: A detector index is outside the graph, or the syndrome
                names one detector twice.
        """

        num_detectors = self.graph.num_detectors
        defects = _defects(detection_events, num_detectors=num_detectors)
        if not defects:
            return MatchingDecodeResult(observables=(), error_edges=(), weight=0.0)
        # `__post_init__` resolves both of these, so the record never holds a
        # window this method has to choose.
        assert self.commit is not None and self.window is not None
        boundary = self.graph.boundary_node
        residual = set(defects)
        committed: dict[
            tuple[tuple[int, int], tuple[int, ...], float], DecodingGraphEdge
        ] = {}
        for start in range(0, num_detectors, self.commit):
            if not residual:
                break
            stop = min(start + self.window, num_detectors)
            band = start + self.commit
            local = tuple(index for index in sorted(residual) if start <= index < stop)
            if not local:
                continue
            answer = MinimumWeightMatchingDecoder(
                graph=_window_graph(self.graph, start=start, stop=stop),
                max_defects=self.max_defects,
            ).decode(local)
            for edge in answer.error_edges:
                first, second = edge.detectors
                if first >= band:
                    continue
                committed[_ordering_key(edge)] = edge
                residual.symmetric_difference_update((first,))
                if second != boundary:
                    residual.symmetric_difference_update((second,))
        if residual:
            raise CapabilityError(
                "sliding-window matching left "
                + ", ".join(f"D{index}" for index in sorted(residual))
                + " defective after deciding every band, which the committed "
                "mechanisms' parity makes unreachable; the graph or the window "
                "arithmetic is inconsistent"
            )
        selected = tuple(committed[key] for key in sorted(committed))
        flipped: set[int] = set()
        for edge in selected:
            flipped.symmetric_difference_update(edge.observables)
        return MatchingDecodeResult(
            observables=tuple(sorted(flipped)),
            error_edges=selected,
            weight=math.fsum(edge.weight for edge in selected),
        )
