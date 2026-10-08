"""Window-scoped projections of a chunked model, and the decoder that reads them.

Upstream `cudaq-qec` ships a ``sliding_window`` decoder over the extended DEM
that :mod:`flagquantum.qec.chunks` reshapes: it cuts a model into round windows,
hands each window its own matrices and its own inner decoder, commits the part of
a window the window that follows cannot revise, and folds the committed
corrections into a logical-frame prediction. This module is that decoder, over
the window records this repository already has, plus the three chunk-scoped
projections the same baseline lists.

**A window's matrices are already cut.** Upstream slices detector rows and error
columns out of one flat matrix to build a window and re-indexes both, which is
why its per-window ``H_round`` and ``first_column`` are separate facts. Here the
decomposition already did that: :func:`~flagquantum.qec.dem_chunks_from_spec`
hands each window its own :class:`~flagquantum.qec.DetectorErrorModel` whose
mechanisms are that window's alone and whose detector rows are re-based to zero,
with the window's detector identity being ``first_row`` plus a local row. So the
inner decoder of a window is built from that window's model and from nothing
else -- nothing is inherited, because there is nothing to inherit.

**Columns are the model's own, rows are the model's own.** A window's mechanism
is a *fault column* of the model the sequence decomposes, and the two sparse
projections here index that one column space: the order
:func:`~flagquantum.qec.dem_close_all` states, which is the order the closed
model's :meth:`~flagquantum.qec.DetectorErrorModel.detector_error_matrix` and
:meth:`~flagquantum.qec.DetectorErrorModel.observables_flips_matrix` use. A
window enumerates its own mechanisms in its own order, so a projection maps that
local column onto the model's global one; the mapping is computed once, from the
mechanisms themselves -- the stable sort that takes the concatenated window order
to the order the model sorts by -- so it is well defined without a second
statement of which mechanisms the sequence holds, and the closed model's shape is
still checked against the model it came from. Rows are detector rows of the model
throughout.

**A measurement bit is a third number, and only one function uses it.** A
*measurement bit* is a position in the flat ``rounds * d`` buffer a memory
experiment's raw measurements occupy, which :func:`dem_chunks_to_d_sparse` maps
detectors onto. It is not a detector row and not a fault column, and that
function is the only place either of the other two numberings meets it. The map
holds when every round of the sequence is the same ``d`` detectors wide, which is
the precondition upstream's own arithmetic states; a sequence whose boundary
rounds are narrower than its interior ones -- a surface code's, for instance --
is refused by name, and :meth:`~flagquantum.qec.MeasurementMap.flattened` is the
map that covers that case, because there the correspondence comes from the
circuit's measurement handles rather than from a round width.

**The streaming shape is two entry points here.** Upstream's ``decode_batch``
accepts either a whole block of syndromes or one round of them and decides
between the two by comparing the first entry's length against the block size.
Here :meth:`SlidingWindowDecoder.decode` takes a whole block in the model's
detector numbering and :meth:`SlidingWindowDecoder.decode_round` takes one round
in *that round's own* numbering, which is the numbering upstream's per-round
vector has. Neither is the repetition-decoder protocol in
:mod:`flagquantum.qec.decoders`, whose ``decode_round`` is keyed on ordered
:class:`~flagquantum.qec.SyndromeRound` records and returns a correction on a
known data qubit; a sliding window over a detector error model has neither a data
qubit nor a round record, so it does not implement that protocol and does not
pretend to. What it shares with it is the word "round".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .chunks import (
    DemChunk,
    DemChunksSpec,
    SeamId,
    dem_chunks_from_spec,
    dem_close_all,
)
from .dem import DetectorErrorModel
from .registry import get_decoder

__all__ = (
    "SlidingWindowDecodeResult",
    "SlidingWindowDecoder",
    "dem_chunks_to_d_sparse",
    "dem_chunks_to_o_sparse",
    "dem_chunks_to_pcm",
)


# ---------------------------------------------------------------------------
# the round geometry of a sequence of windows
# ---------------------------------------------------------------------------


def _seam_width(chunk: DemChunk, from_seam: SeamId, to_seam: SeamId) -> int:
    """The width a chunk's boundary bands state, preferring the outgoing one."""

    for seam_id in (to_seam, from_seam):
        if chunk.has_seam(seam_id):
            width = chunk.get_seam(seam_id).width
            if width > 0:
                return width
    return 0


def _rounds_of(chunk: DemChunk, width: int, to_seam: SeamId) -> int:
    """How many rounds one window spans at a stated round width.

    A boundary band counts as one round and a window's interior as a whole number
    of rounds beyond it, which is upstream's ``dem_chunk_rounds``: the band counts
    only when the window carries one, because an absent boundary is a window with
    no neighbour on that side rather than a round holding no detector.
    """

    interior = len(chunk.interior_rows())
    return (1 if chunk.has_seam(to_seam) else 0) + interior // width


def _round_sequence(
    chunks: tuple[DemChunk, ...], from_seam: SeamId, to_seam: SeamId, fn: str
) -> tuple[int, int]:
    """Read the round width and round count a sequence of windows spans.

    Every non-empty boundary band is the sequence's width, every interior is a
    whole number of those rounds, and the first window states the width at all --
    upstream's ``validate_dem_chunk_sequence`` to the letter, so a sequence this
    accepts is one its arithmetic accepts too.
    """

    if not chunks:
        raise ValueError(f"{fn}: a sequence requires at least one window")
    width = _seam_width(chunks[0], from_seam, to_seam)
    if width == 0:
        raise ValueError(
            f"{fn}: the first window carries no boundary band on either side, so "
            "the rounds of the sequence cannot be counted"
        )
    rounds = 0
    for index, chunk in enumerate(chunks):
        interior = len(chunk.interior_rows())
        if interior % width != 0:
            raise ValueError(
                f"{fn}: window {index} holds {interior} interior rows, which is "
                f"not a whole number of {width}-row rounds; a memory map needs "
                "every round of the sequence to be the same width"
            )
        for seam in chunk.seams:
            if seam.width != width:
                raise ValueError(
                    f"{fn}: window {index} carries a {seam.width}-row "
                    f"{seam.seam_id} band where this sequence is {width} rows wide"
                )
        rounds += _rounds_of(chunk, width, to_seam)
    return width, rounds


# ---------------------------------------------------------------------------
# the shared column space
# ---------------------------------------------------------------------------


def _window_columns(chunks: tuple[DemChunk, ...]) -> tuple[tuple[int, ...], ...]:
    """Map each window's local mechanism onto the closed model's fault column.

    A window numbers its mechanisms from zero and the model it is a window of
    numbers them from zero too, and the two orders differ: a decomposition
    concatenates its windows in window order, while a model sorts its mechanisms
    by their detectors, their observables and their probability. The two lists
    hold the same mechanisms by construction, because
    :func:`~flagquantum.qec.dem_close_all` builds the closed model out of exactly
    these windows without dropping or duplicating any of them, so the mapping is
    the stable sort that takes the concatenated order to the sorted one. The
    detectors take part in the key because that is one of the terms the model
    sorts by; two mechanisms that agree on every term are duplicates, and a
    duplicate pair flips the same detectors and the same observables, so which of
    the two a column name lands on cannot change any projection made from it.
    """

    keys: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
    for chunk in chunks:
        for error in chunk.model.errors:
            keys.append(
                (
                    tuple(chunk.first_row + row for row in error.detectors),
                    tuple(error.observables),
                    float(error.probability),
                )
            )
    order = sorted(range(len(keys)), key=keys.__getitem__)
    column = [0] * len(keys)
    for position, index in enumerate(order):
        column[index] = position
    windows: list[tuple[int, ...]] = []
    offset = 0
    for chunk in chunks:
        count = len(chunk.model.errors)
        windows.append(tuple(column[offset : offset + count]))
        offset += count
    return tuple(windows)


def _projected_columns(
    chunks: tuple[DemChunk, ...],
    *,
    by_row: bool,
) -> tuple[tuple[int, ...], ...]:
    """Read a sequence's windows as one matrix of the model, in sparse form."""

    fn = "dem_chunks_to_pcm" if by_row else "dem_chunks_to_o_sparse"
    if not chunks:
        raise ValueError(f"{fn}: a sequence requires at least one window")
    observables = chunks[0].model.num_observables
    for index, chunk in enumerate(chunks):
        if chunk.model.num_observables != observables:
            raise ValueError(
                f"{fn}: window {index} states {chunk.model.num_observables} "
                f"observables where the sequence states {observables}, and an "
                "observable is a property of the experiment rather than of one "
                "window"
            )
    closed = dem_close_all(chunks)
    columns = _window_columns(chunks)
    entries: list[list[int]] = [
        [] for _ in range(closed.num_detectors if by_row else closed.num_observables)
    ]
    for index, chunk in enumerate(chunks):
        for local, error in enumerate(chunk.model.errors):
            column = columns[index][local]
            if by_row:
                for row in error.detectors:
                    entries[chunk.first_row + row].append(column)
            else:
                for observable in error.observables:
                    entries[observable].append(column)
    return tuple(tuple(sorted(entry)) for entry in entries)


# ---------------------------------------------------------------------------
# the three chunk-scoped projections
# ---------------------------------------------------------------------------


def dem_chunks_to_d_sparse(
    chunks: Sequence[DemChunk],
    *,
    from_seam: SeamId = SeamId.next_round,
    to_seam: SeamId = SeamId.prev_round,
) -> tuple[tuple[int, ...], ...]:
    """The memory XOR map: each detector's measurement bits, in round order.

    Detector ``r * d + k`` of the sequence brings together measurement bit ``k``
    of round zero and nothing else in the first round, and measurement bit ``k``
    of rounds ``r - 1`` and ``r`` after it, because a memory experiment's detector
    in a round compares that round's syndrome against the round before it and its
    terminal detector compares the readout against the last round.

    The numbering returned is the *measurement bit* of a flat ``rounds * d``
    buffer -- round zero's bits ``0 .. d-1``, round one's ``d .. 2d-1`` -- laid
    over the sequence's rounds. The outer index is the detector row of the model.
    The two numberings line up only when every round is ``d`` detectors wide and
    the sequence spans ``rounds * d`` detector rows, and both are checked here
    rather than assumed: a decomposition whose boundary rounds are narrower than
    its interior ones -- ``(4, 8, 8, 8, 4)`` for a distance-three surface code --
    describes a model whose detectors are not pairs of uniform rounds, and this
    map is the wrong one for it. That is upstream's own precondition, since its
    ``dem_chunk_rounds`` divides a window's interior rows by the same width.

    Args:
        chunks: Windows of one model, in order, from
            :func:`~flagquantum.qec.dem_chunks_from_spec`.
        from_seam: The band a window contracts forward with.
        to_seam: The band a window contracts backward with.

    Returns:
        One entry per detector row, holding that detector's measurement bits in
        ascending order.

    Raises:
        ValueError: If the sequence is empty, states no boundary width, has an
            interior that is not a whole number of rounds or a band of another
            width, does not span ``rounds * width`` detector rows, or would
            number a measurement bit past what an unsigned 32-bit position holds.
    """

    fn = "dem_chunks_to_d_sparse"
    items = tuple(chunks)
    width, rounds = _round_sequence(items, from_seam, to_seam, fn)
    detectors = dem_close_all(items).num_detectors
    if width * rounds != detectors:
        raise ValueError(
            f"{fn}: {rounds} rounds of {width} measurement bits describe "
            f"{width * rounds} detector rows and the sequence spans {detectors}, "
            f"so its rounds are not all {width} rows wide"
        )
    if width * rounds > 2**32:
        raise ValueError(
            f"{fn}: measurement bit {width * rounds - 1} exceeds the largest "
            "unsigned 32-bit position a realtime decoder can address"
        )
    rows: list[tuple[int, ...]] = []
    for round_index in range(rounds):
        for bit in range(width):
            row = round_index * width + bit
            rows.append((bit,) if round_index == 0 else (row - width, row))
    return tuple(rows)


def dem_chunks_to_pcm(
    chunks: Sequence[DemChunk],
) -> tuple[tuple[int, ...], ...]:
    """The sequence's parity-check matrix, read one detector row at a time.

    Entry ``[row]`` lists the fault columns that flip detector ``row``, ascending.
    A fault column is a column of the model the sequence decomposes, so the whole
    return value is
    :meth:`~flagquantum.qec.DetectorErrorModel.detector_error_matrix` of
    :func:`~flagquantum.qec.dem_close_all` of the same sequence in sparse form,
    and the sparse form is what a sliding window reads: a committed column's
    support is what has to come out of the next window's syndrome, and the dense
    detector-by-fault product is never built.

    Args:
        chunks: Windows of one model, in order.

    Raises:
        ValueError: If the sequence is empty, is not a decomposition of one model
            -- adjacent windows that do not share exactly one boundary layer,
            windows that disagree on their observable count, or a shared boundary
            whose rows are not the same detectors in the same order.
    """

    return _projected_columns(tuple(chunks), by_row=True)


def dem_chunks_to_o_sparse(chunks: Sequence[DemChunk]) -> tuple[tuple[int, ...], ...]:
    """The sequence's observable matrix, read one observable at a time.

    Entry ``[observable]`` lists the fault columns that flip that observable,
    ascending, over the same column space :func:`dem_chunks_to_pcm` indexes. The
    whole return value is
    :meth:`~flagquantum.qec.DetectorErrorModel.observables_flips_matrix` of
    :func:`~flagquantum.qec.dem_close_all` of the same sequence in sparse form.

    This is the projection a sliding window applies to the columns it has
    committed, which is what makes its answer a logical-frame prediction rather
    than a list of mechanisms.

    Raises:
        ValueError: If the sequence is empty, is not a decomposition of one
            model, or its windows disagree on how many observables the experiment
            has.
    """

    return _projected_columns(tuple(chunks), by_row=False)


# ---------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlidingWindowDecodeResult:
    """What the windows decoded so far jointly claim about one experiment.

    ``faults`` is the committed part of the correction: the model's fault columns
    every window decoded so far selected, ascending. ``observables`` is that
    vector projected through the model's observable matrix, so the two are one
    statement read twice rather than two answers free to disagree.

    ``converged`` is ``None`` exactly when no inner decoder's record stated
    convergence at all. It is not ``True`` by default: upstream ands its windows'
    flags, and a decoder that never reports one would otherwise be credited with
    settling when it has said nothing.

    ``complete`` is ``False`` while windows remain. A record is issued as soon as
    one window is committed rather than only at the end, because a committed
    window's own mechanisms are that window's alone: upstream returns an empty
    vector while it slides, since its result vector is indexed by a column space
    in which the shared columns are still open, and no column here is shared.
    """

    observables: tuple[int, ...]
    faults: tuple[int, ...]
    windows_decoded: int
    complete: bool
    converged: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise TypeError("a sliding window result states completion as a boolean")
        object.__setattr__(self, "observables", tuple(sorted(set(self.observables))))
        object.__setattr__(self, "faults", tuple(sorted(set(self.faults))))
        windows = int(self.windows_decoded)
        if windows < 0:
            raise ValueError("a window count cannot be negative")
        object.__setattr__(self, "windows_decoded", windows)
        if self.converged is not None and not isinstance(self.converged, bool):
            raise TypeError(
                "a sliding window result states convergence as a boolean, or as "
                "None when its inner decoder states none"
            )


class SlidingWindowDecoder:
    """Decode a windowed model window by window, committing as it goes.

    The decoder is built from a :class:`~flagquantum.qec.DemChunksSpec` rather
    than from a whole model, because the windows, their widths and the round each
    detector sits in are what a sliding window needs and a model does not state.
    Every window gets its own inner decoder, built from that window's own model,
    so no window is ever handed a row or a mechanism that is not its own. The
    name is a registered one and is resolved through
    :func:`~flagquantum.qec.get_decoder`, so a name the registry does not hold is
    refused where the decoder is built.

    **What an inner decoder must report.** A window's syndrome is the events of
    its detector rows less the support of the mechanisms the window before it
    committed, because two adjacent windows share the boundary round's detectors
    and the same event may not be explained twice. Forming that residue needs to
    know which mechanisms a window's decoder selected, so an inner record must
    state them as ``mechanisms`` in that window's own column order, which is what
    :class:`~flagquantum.qec.BeliefPropagationDecodeResult` holds. A record that
    states only observables cannot be used: those observables are an answer and
    not a correction, and this decoder projects the correction itself through the
    model's observable matrix so that one run has one answer. The authority
    matcher's record states its correction as ``error_edges`` records rather than
    as column indices, so it is not yet usable as an inner decoder; that is a
    property of that record and is stated in the implementation notes rather than
    worked around here.

    **Why it is not in the registry.** A registered decoder is built from a model
    or from the graph one defines, and a sliding window cannot be: it needs the
    windows of a decomposition, and a model does not state them. Upstream's
    ``sliding_window`` takes its window geometry as constructor arguments for the
    same reason, so this is where it is reached rather than what it is.

    Attributes:
        spec: The decomposition the windows come from.
        chunks: One window per entry, in order.
        inner_decoder: The registered name of the decoder each window builds.
        model: The model the sequence decomposes, read through
            :func:`~flagquantum.qec.dem_close_all`.
    """

    __slots__ = (
        "_buffer",
        "_chunks",
        "_columns",
        "_converged",
        "_decoders",
        "_faults",
        "_layout",
        "_model",
        "_mods",
        "_next_layer",
        "_next_window",
        "_observables",
        "_spec",
        "_support",
        "inner_decoder",
    )

    def __init__(self, spec: DemChunksSpec, inner_decoder: str, **options: Any) -> None:
        if not isinstance(spec, DemChunksSpec):
            raise TypeError("a sliding window slides over a DemChunksSpec")
        if not isinstance(inner_decoder, str) or not inner_decoder.strip():
            raise ValueError(
                "a sliding window needs the name of an inner decoder; an empty "
                "name selects nothing, and defaulting to some implementation "
                "would make the name a decoration"
            )
        chunks = dem_chunks_from_spec(spec)
        layout = spec.layout
        for chunk in chunks:
            rows = layout.rows_for(chunk.first_layer, chunk.last_layer)
            if rows != chunk.rows:
                raise ValueError(
                    f"the window spanning layers {chunk.first_layer}.."
                    f"{chunk.last_layer} covers detector rows {rows.start}.."
                    f"{rows.stop - 1} of the layout and holds rows "
                    f"{chunk.first_row}.."
                    f"{chunk.first_row + chunk.model.num_detectors - 1}, so the "
                    "rounds of one do not describe the other"
                )
        self._spec = spec
        self._chunks = chunks
        self.inner_decoder = inner_decoder
        self._layout = layout
        self._model = dem_close_all(chunks)
        self._columns = _window_columns(chunks)
        support: list[list[int]] = [[] for _ in range(len(self._model.errors))]
        observables: list[list[int]] = [[] for _ in range(len(self._model.errors))]
        for index, chunk in enumerate(chunks):
            for local, error in enumerate(chunk.model.errors):
                column = self._columns[index][local]
                for row in error.detectors:
                    support[column].append(chunk.first_row + row)
                observables[column] = list(error.observables)
        self._support = tuple(tuple(sorted(rows)) for rows in support)
        self._observables = tuple(tuple(sorted(set(k))) for k in observables)
        self._decoders = tuple(
            get_decoder(inner_decoder, chunk.model, **options) for chunk in chunks
        )
        self._buffer: dict[int, tuple[int, ...]] = {}
        self._mods = [0] * self._model.num_detectors
        self._faults: set[int] = set()
        self._converged: bool | None = None
        self._next_layer = 0
        self._next_window = 0

    # -- what the decoder is ------------------------------------------------

    @property
    def spec(self) -> DemChunksSpec:
        """The decomposition the windows come from."""

        return self._spec

    @property
    def chunks(self) -> tuple[DemChunk, ...]:
        """One window per entry, in order."""

        return self._chunks

    @property
    def model(self) -> DetectorErrorModel:
        """The model the sequence decomposes."""

        return self._model

    @property
    def num_windows(self) -> int:
        """Number of windows one block is decoded in."""

        return self._spec.num_windows

    @property
    def windows_decoded(self) -> int:
        """How many windows of the current block have been committed."""

        return self._next_window

    # -- the two entry points ----------------------------------------------

    def decode(self, detection_events: Iterable[int]) -> SlidingWindowDecodeResult:
        """Decode a whole block of detector events, committing every window.

        The block is split into the layout's rounds and fed to
        :meth:`decode_round` one round at a time, which is what upstream's
        whole-block branch does rather than a second decoding path: one block and
        one round per round produce the same record by construction.

        Args:
            detection_events: Indices of the detectors that fired, in any order,
                in the numbering of :attr:`model`.

        Returns:
            The complete record, with ``complete`` set.

        Raises:
            ValueError: If an index is outside the model, or an entry repeats.
            RuntimeError: If a block is decoded while a block is already in
                progress, since the buffered rounds of the two cannot mix.
            CapabilityError: If an inner decoder refuses a window's residue, in
                which case the stream has been reset.
            TypeError: If an inner decoder's result record states no selected
                mechanisms.
        """

        if self._next_layer != 0:
            raise RuntimeError(
                f"a whole block cannot be decoded {self._next_layer} round(s) into "
                "a block that is already in progress; reset the stream first"
            )
        events = self._events(detection_events, self._model.num_detectors, "the model")
        result: SlidingWindowDecodeResult | None = None
        for layer in range(self._layout.num_layers):
            start = self._layout.first_row_of(layer)
            width = self._layout.width_of(layer)
            local = tuple(
                sorted(row - start for row in events if start <= row < start + width)
            )
            result = self._feed(layer, local)
        if result is None or not result.complete:
            raise RuntimeError(
                "the sliding window fed every round of a block without closing its "
                "last window, which means its stream state was changed under it"
            )
        return result

    def decode_round(
        self, detection_events: Iterable[int]
    ) -> SlidingWindowDecodeResult | None:
        """Feed the next round of detector events and commit any window it closes.

        The events are numbered within the round being fed, which is the
        numbering the round's own syndrome has: a round of ``w`` rows takes
        indices in ``0 .. w-1``, not detector rows of the model. That is
        upstream's per-round vector, and it is the numbering a per-round syndrome
        is already in; :meth:`decode` is the entry point that takes the model's.

        Args:
            detection_events: Indices of the detectors that fired in the next
                round, in any order, counted within that round.

        Returns:
            The record held after this round, or ``None`` while no window has
            closed. Every window that closes returns a record, because a
            committed window's correction is final and its contribution is
            reported as soon as it exists.

        Raises:
            ValueError: If an index is outside the round being fed, or repeats.
            RuntimeError: If every round of a block has been fed without the block
                completing, which means the stream state was changed under it.
            CapabilityError: If an inner decoder refuses a window's residue, in
                which case the stream has been reset.
            TypeError: If an inner decoder's result record states no selected
                mechanisms.
        """

        layer = self._next_layer
        if layer >= self._layout.num_layers:
            raise RuntimeError(
                "the sliding window has fed every round of a block without "
                "completing it, which means its stream state was changed under it"
            )
        width = self._layout.width_of(layer)
        events = self._events(detection_events, width, f"round {layer}")
        return self._feed(layer, events)

    def reset_stream(self) -> None:
        """Discard the buffered rounds and start the next block fresh.

        A block that finished is reset by itself, and so is a block whose window
        an inner decoder refused, so this is for a caller that abandons one: the
        rounds it had fed are dropped, the residue carried between windows is
        cleared, and the next :meth:`decode_round` is round zero of a new block.
        The inner decoders are not rebuilt, because each reads one window's model
        and that model does not change between blocks.
        """

        self._buffer = {}
        self._mods = [0] * self._model.num_detectors
        self._faults = set()
        self._converged = None
        self._next_layer = 0
        self._next_window = 0

    # -- one round and one window -------------------------------------------

    def _events(
        self, detection_events: Iterable[int], width: int, where: str
    ) -> tuple[int, ...]:
        """Read a syndrome as ascending, unique, in-range detector indices."""

        events: list[int] = []
        for event in detection_events:
            index = int(event)
            if not 0 <= index < width:
                raise ValueError(
                    f"detector {index} is outside the {width} detector(s) of {where}"
                )
            events.append(index)
        if len(set(events)) != len(events):
            raise ValueError(
                "a syndrome cannot name the same detector twice, and this one "
                "repeats a detector"
            )
        return tuple(sorted(events))

    def _feed(
        self, layer: int, events: tuple[int, ...]
    ) -> SlidingWindowDecodeResult | None:
        """Buffer one round and commit the window it completes, if it completes one."""

        self._buffer[layer] = events
        self._next_layer = layer + 1
        window = self._next_window
        last_layer = window * self._spec.stride + self._spec.window - 1
        if self._next_layer <= last_layer:
            return None
        result = self._commit(window)
        self._next_window = window + 1
        self._release_layers()
        if self._next_window == self._spec.num_windows:
            self.reset_stream()
        return result

    def _commit(self, window: int) -> SlidingWindowDecodeResult:
        """Decode one window against the residue and fold in its correction."""

        first_layer = window * self._spec.stride
        last_layer = first_layer + self._spec.window - 1
        first_row = self._layout.first_row_of(first_layer)
        rows = self._layout.rows_for(first_layer, last_layer)
        fired = {
            self._layout.first_row_of(layer) + event
            for layer in range(first_layer, last_layer + 1)
            for event in self._buffer[layer]
        }
        if window > 0:
            # The shared round is this window's leading band and the previous
            # window's trailing one, so what the previous window already explained
            # comes out of it. Nothing else of this window was ever committed, and
            # the residue is a symmetric difference rather than a subtraction
            # because an event the previous window explained and one it committed
            # a mechanism flipping cancel out.
            fired.symmetric_difference_update(row for row in rows if self._mods[row])
        try:
            inner = self._decoders[window].decode(
                sorted(row - first_row for row in fired)
            )
            mechanisms = self._selected(window, inner)
        except Exception:
            # Upstream abandons the block here, returning no result and dropping
            # the buffered rounds; an exception carries the inner decoder's own
            # reason, so the stream is reset and that reason is re-raised.
            self.reset_stream()
            raise
        for local in mechanisms:
            column = self._columns[window][local]
            for row in self._support[column]:
                self._mods[row] ^= 1
            self._faults.symmetric_difference_update((column,))
        converged = getattr(inner, "converged", None)
        if converged is not None:
            settled = True if self._converged is None else self._converged
            self._converged = settled and bool(converged)
        return self._record(window + 1)

    def _selected(self, window: int, inner: Any) -> tuple[int, ...]:
        """Read one inner record's selected mechanism columns."""

        mechanisms = getattr(inner, "mechanisms", None)
        if mechanisms is None:
            raise TypeError(
                f"the inner decoder {self.inner_decoder!r} returned "
                f"{type(inner).__name__} for window {window}, which states no "
                "selected mechanisms; a sliding window needs them because the next "
                "window's syndrome has their support removed, and a record that "
                "states only observables is an answer rather than a correction"
            )
        count = len(self._chunks[window].model.errors)
        selected: list[int] = []
        for mechanism in mechanisms:
            local = int(mechanism)
            if not 0 <= local < count:
                raise ValueError(
                    f"the inner decoder {self.inner_decoder!r} selected mechanism "
                    f"{local} of window {window}, which holds {count}"
                )
            selected.append(local)
        return tuple(sorted(set(selected)))

    def _record(self, committed: int) -> SlidingWindowDecodeResult:
        """Read the committed columns as one result record."""

        flipped: set[int] = set()
        for column in self._faults:
            flipped.symmetric_difference_update(self._observables[column])
        return SlidingWindowDecodeResult(
            observables=tuple(sorted(flipped)),
            faults=tuple(sorted(self._faults)),
            windows_decoded=committed,
            complete=committed == self._spec.num_windows,
            converged=self._converged,
        )

    def _release_layers(self) -> None:
        """Drop the rounds no window that follows this one still needs."""

        keep_from = self._next_window * self._spec.stride
        for layer in [layer for layer in self._buffer if layer < keep_from]:
            del self._buffer[layer]
