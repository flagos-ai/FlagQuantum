"""Round windows of a detector error model, joined by named seam boundaries.

A detector error model states one flat parity per mechanism, which is what a
decoder wants for a short experiment and not what a streaming decoder wants: it
wants the mechanisms of one round window at a time, and it wants the detector
rows that window shares with its neighbours to be *named*, so that a window can
be read, committed and slid forward without re-reading the model.

This module is that decomposition. The unit is a :class:`DemChunk`: a detector
error model whose rows are the detector layers of one window, whose mechanisms
are the ones that window owns, and whose first and last layers are its two
seams. Adjacent windows share those layers, so the seam between them is one
detector boundary seen from both sides rather than two boundaries that happen to
line up: :func:`dem_stitch` compares the two bands' detector identities and
refuses the contraction when they differ.

Bands, and why the order is fixed
---------------------------------
A chunk's local rows are grouped into three bands, in this order: its leading
seam band (``prev_round``), its interior rows, and its trailing seam band
(``next_round``). The order is fixed rather than recorded, because it is what
makes the stitch a concatenation: contracting two adjacent chunks lays out
``left.leading | left.interior | shared boundary | right.interior |
right.trailing``, so the result is again a chunk in this same shape and a chain
of windows folds left without a special case for the middle.

A chunk carries no ``prev_round`` seam when its first layer is the layout's
first, and no ``next_round`` seam when its last layer is the layout's last,
because there is then no neighbour to contract that boundary against. That is a
consequence of the layer layout and not of a phase template: an init window has
no incoming boundary because no layer precedes it, not because a different matrix
was built for it.

The fixed order also settles what a contraction may leave behind. A stitch lays
the shared boundary out once and carries the two *outer* boundaries through, so
its result holds at most those two seams, and a seam that is neither the left
side's leading boundary nor the right side's trailing one is not a shape a chunk
can be in: :class:`DemChunk` admits ``prev_round`` and ``next_round`` at most
once each and refuses anything else. There is therefore no "leftover seam" check
in :func:`dem_stitch`. One was written and removed, because no reachable input
can satisfy its condition and a check that cannot fail asserts nothing; the shape
rule it was guarding is enforced where the shape is built rather than after the
fact.

Identity
--------
Rows are identified by the *detector index they have in the whole model*, and a
chunk records the first of them in :attr:`DemChunk.first_row`. Because a window
is a range of layers and a layer is a range of rows, a chunk's rows are always
the contiguous block that starts there, so the block is a starting point rather
than a list that could disagree with the model. Closing a chunk is then a
relabel rather than a renumbering: :func:`dem_close` produces the flat model
numbered by those identities, which is the numbering the model had before it was
cut. A stitched chain of windows closes back to the model it was cut from, and
that round trip is what the tests hold :func:`dem_chunks_from_spec` and
:func:`dem_close_all` to.

Ownership, and why it is not "every window that contains it"
------------------------------------------------------------
One mechanism belongs to one window. A fault that flips detectors in two
adjacent layers lies inside every window whose layer range covers both, and with
windows that share a boundary layer that is usually more than one; the owner here
is the *first* window in the sequence that contains the whole mechanism.
Stitching then produces each mechanism exactly once, where a rule that let two
windows both claim it would produce it twice and quietly double its prior. A
mechanism no window contains whole is refused rather than split, because
splitting it would be two weaker mechanisms where the model stated one.

What is deliberately not mirrored
---------------------------------
Upstream is ``libs/qec/`` of ``NVIDIA/cudaq-qec``, whose ``extended_dem``,
``dem_chunk_spec`` and ``dem_chunks_spec`` are the surface this module lines up
with. Four differences are decisions rather than omissions:

* A seam here is identified by its name, kept as the name. Upstream hashes the
  name into an ``uint32`` at compile time so that a call site never assigns an
  id, and keeps a name registry only so that a diagnostic can turn the hash back
  into text. A record that keeps the text needs neither the hash nor the
  registry, and cannot collide.
* Upstream's per-seam ``tags`` are labels a caller may choose, and it numbers
  them positionally within a seam, which makes its stitching tag check the same
  statement as its width check. The rows here carry the global detector index
  instead, so the check compares identities and a contraction of two boundaries
  that merely have the same width is refused.
* Upstream's ``dem_chunk_spec`` accepts a sparse shorthand and expands it for a
  given list of seam ids, with a second, non-shorthand form carrying a separate
  matrix per seam. A chunk here is cut from a model whose rows are already laid
  out in layers, so there is no shorthand to expand and no second form.
* Upstream's ``dem_chunks_spec`` is a phase *graph*: phases named by the caller,
  edges between them, a self-loop for the repeating phase, and a round count
  supplied at expansion time. This module keeps the linear case a memory
  experiment actually has -- init, a repetition of bulk, final -- and refuses
  anything else rather than approximating it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

from .circuit import MemoryCircuit
from .dem import DemError, DemMergeRule, DetectorErrorModel, _count

__all__ = (
    "ChunkLayout",
    "DemChunk",
    "DemChunkSpec",
    "DemChunksSpec",
    "DemSeam",
    "PhaseId",
    "SeamId",
    "dem_chunk_from_spec",
    "dem_chunks_from_spec",
    "dem_close",
    "dem_close_all",
    "dem_stitch",
    "dem_stitch_all",
    "dem_stitch_merged",
)


@dataclass(frozen=True)
class SeamId:
    """The identity of one named detector-row boundary.

    Two seams are the same seam exactly when their names are equal, so the name
    is the identity and is kept: nothing here hashes it, interns it, or needs a
    second lookup to spell it again for a message.
    """

    name: str

    prev_round: ClassVar["SeamId"]
    """Incoming syndrome boundary: the leading band of a window."""

    next_round: ClassVar["SeamId"]
    """Outgoing syndrome boundary: the trailing band of a window."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("a seam name must be a string")
        if not self.name.strip():
            raise ValueError("a seam name must not be empty")
        if self.name != self.name.strip():
            raise ValueError(
                "a seam name must not carry surrounding whitespace, because two "
                "names differing only in whitespace would be two seams"
            )

    def __str__(self) -> str:
        return self.name


SeamId.prev_round = SeamId("prev_round")
SeamId.next_round = SeamId("next_round")


@dataclass(frozen=True)
class PhaseId:
    """The label of a window's position in a chunk sequence.

    A label and not a template: a window's two boundary bands fall out of the
    layer layout it was cut from, so an init window carries no ``prev_round``
    seam because no detector layer precedes it. Upstream's phases are separate
    chunk templates joined by a graph, which is a more general record than a
    memory experiment needs.
    """

    name: str

    init: ClassVar["PhaseId"]
    """First window of a sequence."""

    bulk: ClassVar["PhaseId"]
    """A repeated interior window, and the whole sequence when there is one."""

    final: ClassVar["PhaseId"]
    """Last window of a sequence."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("a phase name must be a string")
        if not self.name.strip():
            raise ValueError("a phase name must not be empty")

    def __str__(self) -> str:
        return self.name


PhaseId.init = PhaseId("init")
PhaseId.bulk = PhaseId("bulk")
PhaseId.final = PhaseId("final")


@dataclass(frozen=True)
class DemSeam:
    """One named boundary band: a half-open local row range within a chunk."""

    seam_id: SeamId
    row_begin: int
    row_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.seam_id, SeamId):
            raise TypeError("a seam must be identified by a SeamId record")
        begin = _count(self.row_begin, name="seam row_begin")
        end = _count(self.row_end, name="seam row_end")
        if end <= begin:
            raise ValueError(
                "a seam band must hold at least one row; a boundary with no row "
                "is not a boundary and is left out rather than recorded empty"
            )
        object.__setattr__(self, "row_begin", begin)
        object.__setattr__(self, "row_end", end)

    @property
    def width(self) -> int:
        """Number of detector rows in this band."""

        return self.row_end - self.row_begin

    @property
    def rows(self) -> range:
        """The local row indices this band covers."""

        return range(self.row_begin, self.row_end)

    def __len__(self) -> int:
        return self.width


@dataclass(frozen=True)
class ChunkLayout:
    """A model's detector rows, divided into ordered boundary layers.

    A layer is the unit a window is measured in, and its number is the round the
    detectors sit in -- with the terminal readout layer numbered one past the last
    syndrome round, because the readout detectors are compared against that round
    rather than belonging to it. The layer numbers are therefore the numbers a
    streaming decoder already holds, and a gap in them is refused rather than
    closed up: a round whose detectors are missing would otherwise be renumbered,
    which would silently move every window that follows it.

    A layer is stored as its width, because that is all a layer is once the
    offset is read from the widths before it, and a list of row ranges would be
    the same numbers written twice.
    """

    model: DetectorErrorModel
    widths: tuple[int, ...]

    _first_row: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, DetectorErrorModel):
            raise TypeError("a layout must describe a DetectorErrorModel")
        widths = tuple(self.widths)
        if not widths:
            raise ValueError("a chunk layout requires at least one detector layer")
        for width in widths:
            if _count(width, name="layer width") < 1:
                raise ValueError("a detector layer must hold at least one row")
        if sum(widths) != self.model.num_detectors:
            raise ValueError(
                f"layer widths sum to {sum(widths)} rows and the model states "
                f"{self.model.num_detectors} detectors, so the layers of one do "
                "not describe the other"
            )
        object.__setattr__(self, "widths", widths)
        first_row: list[int] = []
        cursor = 0
        for width in widths:
            first_row.append(cursor)
            cursor += width
        object.__setattr__(self, "_first_row", tuple(first_row))

    @classmethod
    def from_memory_circuit(
        cls, model: DetectorErrorModel, circuit: MemoryCircuit
    ) -> "ChunkLayout":
        """Read the layers of a memory circuit out of its detector layout.

        A detector's layer is the syndrome round it belongs to, and the terminal
        layer is the one the readout detectors sit in. The round is read from the
        measurement references the detector is built from, so the layout follows
        the circuit's own rounds instead of a count stated a second time here.
        """

        if not isinstance(circuit, MemoryCircuit):
            raise TypeError("a memory layer layout must describe a MemoryCircuit")
        layout = circuit.detectors
        if len(layout) != model.num_detectors:
            raise ValueError(
                f"the circuit declares {len(layout)} detectors and the model "
                f"states {model.num_detectors}, so the layers of one do not "
                "describe the other"
            )
        rounds = circuit.rounds
        layers: list[int] = []
        for detector in layout.detectors:
            references = tuple(item.round_index for item in detector.parity)
            if any(item is None for item in references):
                layers.append(rounds)
            else:
                layers.append(max(item for item in references if item is not None))
        if layers != sorted(layers):
            raise ValueError(
                "detector layers must not decrease along the detector index "
                "order, because a window is a contiguous range of them"
            )
        widths: list[int] = []
        previous: int | None = None
        for layer in layers:
            if layer == previous:
                widths[-1] += 1
                continue
            if widths and layer != len(widths):
                raise ValueError(
                    "detector layers must be dense and ordered from zero: a "
                    f"layout that reaches layer {layer} after {len(widths)} layers "
                    "names a round with no detector, and this decomposition "
                    "refuses rather than renumbering the rounds after it"
                )
            widths.append(1)
            previous = layer
        return cls(model=model, widths=tuple(widths))

    @property
    def num_layers(self) -> int:
        """Number of detector layers in the layout."""

        return len(self.widths)

    @property
    def num_detectors(self) -> int:
        """Number of detector rows the layers cover."""

        return self.model.num_detectors

    @property
    def rows(self) -> range:
        """Every detector row, in layer order."""

        return range(self.model.num_detectors)

    def width_of(self, layer: int) -> int:
        """Number of detector rows in one layer."""

        index = _count(layer, name="layer")
        if index >= self.num_layers:
            raise ValueError(
                f"layer {index} is outside a layout of {self.num_layers} layers"
            )
        return self.widths[index]

    def first_row_of(self, layer: int) -> int:
        """Lowest detector row of one layer."""

        index = _count(layer, name="layer")
        if index >= self.num_layers:
            raise ValueError(
                f"layer {index} is outside a layout of {self.num_layers} layers"
            )
        return self._first_row[index]

    def layer_of(self, row: int) -> int:
        """The layer a detector row belongs to."""

        index = _count(row, name="detector row")
        if index >= self.model.num_detectors:
            raise ValueError(
                f"detector row {index} is outside the model's "
                f"{self.model.num_detectors} detectors"
            )
        layer = 0
        while row >= self._first_row[layer] + self.widths[layer]:
            layer += 1
        return layer

    def rows_for(self, first_layer: int, last_layer: int) -> range:
        """The detector rows of a whole range of layers, in layer order."""

        first = _count(first_layer, name="first layer")
        last = _count(last_layer, name="last layer")
        if last < first:
            raise ValueError("a window's last layer must not precede its first")
        if last >= self.num_layers:
            raise ValueError(
                f"layer {last} is outside a layout of {self.num_layers} layers"
            )
        return range(self._first_row[first], self._first_row[last] + self.widths[last])


@dataclass(frozen=True)
class DemChunk:
    """One window of a model: local rows, their identity, and its seams.

    ``model`` is the window on its own, so its detector indices are local and
    number from zero; ``first_row`` says which detector of the whole model local
    row zero is, and ``seams`` names the leading and trailing bands of the
    window.
    """

    model: DetectorErrorModel
    first_row: int
    seams: tuple[DemSeam, ...]
    first_layer: int
    last_layer: int

    def __post_init__(self) -> None:
        if not isinstance(self.model, DetectorErrorModel):
            raise TypeError("a chunk must hold a DetectorErrorModel")
        first_row = _count(self.first_row, name="chunk first_row")
        first = _count(self.first_layer, name="chunk first_layer")
        last = _count(self.last_layer, name="chunk last_layer")
        if last < first:
            raise ValueError("a chunk's last layer must not precede its first")
        object.__setattr__(self, "first_row", first_row)
        object.__setattr__(self, "first_layer", first)
        object.__setattr__(self, "last_layer", last)
        seams = tuple(self.seams)
        for seam in seams:
            if not isinstance(seam, DemSeam):
                raise TypeError("chunk seams must be DemSeam records")
            if seam.row_end > self.model.num_detectors:
                raise ValueError(
                    f"seam {seam.seam_id} ends at row {seam.row_end}, outside the "
                    f"chunk's {self.model.num_detectors} rows"
                )
        identifiers = [seam.seam_id for seam in seams]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a chunk may name a seam id at most once")
        for seam_id in identifiers:
            if seam_id not in (SeamId.prev_round, SeamId.next_round):
                raise ValueError(
                    f"chunk seam {seam_id} is not a standard boundary; a window "
                    "here has exactly the two layers flanking it, so its seams "
                    "are prev_round and next_round and nothing else"
                )
        if list(seams) != sorted(seams, key=lambda item: item.row_begin):
            raise ValueError("chunk seams must be ordered by their first row")
        leading = [item for item in seams if item.seam_id == SeamId.prev_round]
        trailing = [item for item in seams if item.seam_id == SeamId.next_round]
        if leading and leading[0].row_begin != 0:
            raise ValueError(
                "a prev_round seam is the chunk's leading band, so it must begin "
                "at local row zero"
            )
        if trailing and trailing[0].row_end != self.model.num_detectors:
            raise ValueError(
                "a next_round seam is the chunk's trailing band, so it must end "
                "at the chunk's last local row"
            )
        if leading and trailing and leading[0].row_end > trailing[0].row_begin:
            raise ValueError("a chunk's two seam bands must not overlap")
        object.__setattr__(self, "seams", seams)

    @property
    def rows(self) -> range:
        """The whole-model detector index of every local row, in row order."""

        return range(self.first_row, self.first_row + self.model.num_detectors)

    @property
    def num_layers(self) -> int:
        """Number of detector layers this window spans."""

        return self.last_layer - self.first_layer + 1

    def has_seam(self, seam_id: SeamId) -> bool:
        """Whether this chunk carries the named boundary."""

        return any(seam.seam_id == seam_id for seam in self.seams)

    def get_seam(self, seam_id: SeamId) -> DemSeam:
        """The named boundary band, or a stated refusal when it is not here."""

        for seam in self.seams:
            if seam.seam_id == seam_id:
                return seam
        raise KeyError(
            f"this chunk carries no {seam_id} seam; it has "
            f"{', '.join(str(item.seam_id) for item in self.seams) or 'none'}"
        )

    def interior_rows(self) -> tuple[int, ...]:
        """The local rows no seam covers, in row order.

        These are the rows a window owns alone: a boundary band is shared with a
        neighbour and an interior row is not, so this is the part of a chunk that
        survives a stitch unchanged and the part a decoder may commit without
        waiting for the window that follows.
        """

        covered = {row for seam in self.seams for row in seam.rows}
        return tuple(
            row for row in range(self.model.num_detectors) if row not in covered
        )

    def seam_rows(self, seam_id: SeamId) -> tuple[int, ...]:
        """The detector identities of a boundary band, in band order."""

        seam = self.get_seam(seam_id)
        return tuple(self.rows[row] for row in seam.rows)


@dataclass(frozen=True)
class DemChunkSpec:
    """One window of a layout, stated as a layer range, before it is cut.

    Nothing else about the window is stated here: its rows come from the
    layout's layers and its mechanisms come from the model those layers
    describe, so a spec cannot introduce a row or a mechanism the model does not
    have.
    """

    layout: ChunkLayout
    first_layer: int
    last_layer: int
    phase: PhaseId

    def __post_init__(self) -> None:
        if not isinstance(self.layout, ChunkLayout):
            raise TypeError("a chunk spec must cut a ChunkLayout")
        if not isinstance(self.phase, PhaseId):
            raise TypeError("a chunk spec must be labelled by a PhaseId")
        first = _count(self.first_layer, name="spec first_layer")
        last = _count(self.last_layer, name="spec last_layer")
        if last < first:
            raise ValueError("a window's last layer must not precede its first")
        if last >= self.layout.num_layers:
            raise ValueError(
                f"layer {last} is outside a layout of "
                f"{self.layout.num_layers} layers"
            )
        object.__setattr__(self, "first_layer", first)
        object.__setattr__(self, "last_layer", last)

    @property
    def num_layers(self) -> int:
        """Number of detector layers this window spans."""

        return self.last_layer - self.first_layer + 1


def _cut(
    spec: DemChunkSpec,
    *,
    owned: dict[int, int] | None = None,
    owner: int | None = None,
) -> DemChunk:
    """Cut one window out of a layout.

    Without an ownership map every mechanism whose detectors lie wholly inside
    the window is kept, which is what a single stated window means. With one, a
    mechanism is kept only where the map gives this window as its owner, which is
    what makes a sequence of windows a partition rather than a set of overlapping
    cuts.
    """

    layout = spec.layout
    rows = layout.rows_for(spec.first_layer, spec.last_layer)
    index_of = {row: index for index, row in enumerate(rows)}
    errors: list[DemError] = []
    for index, error in enumerate(layout.model.errors):
        if owned is not None and owned.get(index) != owner:
            continue
        if not error.detectors:
            raise ValueError(
                f"mechanism {index} flips no detector, so no window can hold it: "
                "an observable-only fault has no round to be placed in, and this "
                "decomposition refuses rather than choosing one"
            )
        layers = [layout.layer_of(row) for row in error.detectors]
        if min(layers) < spec.first_layer or max(layers) > spec.last_layer:
            continue
        errors.append(
            DemError(
                probability=error.probability,
                detectors=tuple(index_of[row] for row in error.detectors),
                observables=error.observables,
                error_id=error.error_id,
            )
        )
    seams: list[DemSeam] = []
    if spec.first_layer > 0:
        seams.append(
            DemSeam(
                seam_id=SeamId.prev_round,
                row_begin=0,
                row_end=layout.width_of(spec.first_layer),
            )
        )
    if spec.last_layer < layout.num_layers - 1:
        width = layout.width_of(spec.last_layer)
        seams.append(
            DemSeam(
                seam_id=SeamId.next_round,
                row_begin=len(rows) - width,
                row_end=len(rows),
            )
        )
    return DemChunk(
        model=DetectorErrorModel(
            num_detectors=len(rows),
            num_observables=layout.model.num_observables,
            errors=tuple(errors),
        ),
        first_row=rows.start,
        seams=tuple(seams),
        first_layer=spec.first_layer,
        last_layer=spec.last_layer,
    )


def dem_chunk_from_spec(spec: DemChunkSpec) -> DemChunk:
    """Cut one chunk out of a spec.

    This is a cut and not a partition: a mechanism spanning a shared boundary is
    a whole mechanism inside both windows that flank it, so cutting two
    overlapping windows separately lists it twice. :func:`dem_chunks_from_spec`
    is the operation that assigns every mechanism to one window.
    """

    if not isinstance(spec, DemChunkSpec):
        raise TypeError("dem_chunk_from_spec requires a DemChunkSpec")
    return _cut(spec)


@dataclass(frozen=True)
class DemChunksSpec:
    """A whole model's windows: one width, tiled across the layers.

    The stride is the window width minus one, so consecutive windows share
    exactly one detector layer -- the trailing boundary of one being the leading
    boundary of the next. The stride is not a free parameter here: a stride wider
    than that leaves layers in no window, and a narrower one puts a layer inside
    two windows that are not adjacent, which is a window a decoder may read but
    not one this algebra can stitch. A decoder that wants to slide by less than a
    window reads windows without stitching them, and the stride belongs to that
    decoder rather than to this record.
    """

    layout: ChunkLayout
    window: int
    phases: tuple[PhaseId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.layout, ChunkLayout):
            raise TypeError("a chunk sequence must cut a ChunkLayout")
        window = _count(self.window, name="window")
        if window < 2:
            raise ValueError(
                "a window must span at least two detector layers, because a window "
                "of one layer has one boundary shared with itself and no interior "
                "row"
            )
        if window > self.layout.num_layers:
            raise ValueError(
                f"a window of {window} layers does not fit a layout of "
                f"{self.layout.num_layers} layers"
            )
        stride = window - 1
        if (self.layout.num_layers - window) % stride != 0:
            raise ValueError(
                f"a layout of {self.layout.num_layers} layers does not tile into "
                f"windows of {window} at a stride of {stride}; leave the layer "
                "count a whole number of strides past the first window"
            )
        object.__setattr__(self, "window", window)
        phases = tuple(self.phases)
        for phase in phases:
            if not isinstance(phase, PhaseId):
                raise TypeError("window labels must be PhaseId records")
        if phases and len(phases) != (self.layout.num_layers - window) // stride + 1:
            raise ValueError(
                f"{len(phases)} window labels were stated for "
                f"{(self.layout.num_layers - window) // stride + 1} windows"
            )
        object.__setattr__(self, "phases", phases)

    @property
    def stride(self) -> int:
        """Layers a window advances by, which is its width minus one."""

        return self.window - 1

    @property
    def num_windows(self) -> int:
        """Number of windows this spec tiles the layout into."""

        return (self.layout.num_layers - self.window) // self.stride + 1

    @property
    def layer_ranges(self) -> tuple[tuple[int, int], ...]:
        """Each window's layer range, in sequence order."""

        return tuple(
            (index * self.stride, index * self.stride + self.window - 1)
            for index in range(self.num_windows)
        )

    def phase_sequence(self) -> tuple[PhaseId, ...]:
        """The window labels, stated or derived.

        The derived sequence is the linear case a memory experiment has: the
        first window is ``init``, the last is ``final``, and everything between
        is ``bulk``. A single window is the whole model and has no neighbour to
        contract with, so it is labelled ``bulk`` rather than being called both
        an init and a final.
        """

        if self.phases:
            return self.phases
        if self.num_windows == 1:
            return (PhaseId.bulk,)
        return (
            (PhaseId.init,)
            + (PhaseId.bulk,) * (self.num_windows - 2)
            + (PhaseId.final,)
        )

    def chunk_specs(self) -> tuple[DemChunkSpec, ...]:
        """One spec per window, in sequence order."""

        return tuple(
            DemChunkSpec(
                layout=self.layout,
                first_layer=first,
                last_layer=last,
                phase=phase,
            )
            for (first, last), phase in zip(
                self.layer_ranges, self.phase_sequence(), strict=True
            )
        )


def dem_chunks_from_spec(spec: DemChunksSpec) -> tuple[DemChunk, ...]:
    """Expand a sequence spec into chunks that partition the model.

    Every mechanism of the model lands in exactly one window: the first window
    whose layer range contains all of its detectors. A mechanism no window
    contains whole is refused rather than split.
    """

    if not isinstance(spec, DemChunksSpec):
        raise TypeError("dem_chunks_from_spec requires a DemChunksSpec")
    layout = spec.layout
    ranges = spec.layer_ranges
    owned: dict[int, int] = {}
    for index, error in enumerate(layout.model.errors):
        if not error.detectors:
            raise ValueError(
                f"mechanism {index} flips no detector, so no window can hold it: "
                "an observable-only fault has no round to be placed in, and this "
                "decomposition refuses rather than choosing one"
            )
        layers = [layout.layer_of(row) for row in error.detectors]
        first, last = min(layers), max(layers)
        owners = [
            position
            for position, (begin, end) in enumerate(ranges)
            if begin <= first and last <= end
        ]
        if not owners:
            raise ValueError(
                f"mechanism {index} flips detector layers {first}..{last}, which no "
                f"window of {spec.window} layers covers; widen the window or state "
                "the fault per round"
            )
        owned[index] = owners[0]
    return tuple(
        _cut(chunk_spec, owned=owned, owner=position)
        for position, chunk_spec in enumerate(spec.chunk_specs())
    )


def dem_stitch(
    left: DemChunk,
    right: DemChunk,
    from_seam: SeamId = SeamId.next_round,
    to_seam: SeamId = SeamId.prev_round,
) -> DemChunk:
    """Contract two adjacent chunks into one chunk spanning both.

    The left chunk's ``from_seam`` band and the right chunk's ``to_seam`` band
    must be the same detector boundary: the same rows, in the same order, not
    merely the same number of them. The result lays the shared boundary out once,
    between the two interiors, and keeps the outer boundaries as its own seams,
    so a chain folds left.
    """

    for chunk, side in ((left, "left"), (right, "right")):
        if not isinstance(chunk, DemChunk):
            raise TypeError(f"the {side} side of a stitch must be a DemChunk")
    if left.model.num_observables != right.model.num_observables:
        raise ValueError(
            f"cannot contract chunks over {left.model.num_observables} and "
            f"{right.model.num_observables} observables; an observable is a "
            "property of the whole experiment and not of a window"
        )
    if not left.has_seam(from_seam):
        raise ValueError(
            f"the left chunk carries no {from_seam} seam, so it has no boundary "
            "to contract forward"
        )
    if not right.has_seam(to_seam):
        raise ValueError(
            f"the right chunk carries no {to_seam} seam, so it has no boundary "
            "to contract backward"
        )
    if left.last_layer != right.first_layer:
        raise ValueError(
            f"cannot contract chunks spanning layers "
            f"{left.first_layer}..{left.last_layer} and "
            f"{right.first_layer}..{right.last_layer}: adjacent windows share "
            "exactly one detector layer, the trailing boundary of one being the "
            "leading boundary of the next, and that shared layer is what is "
            "contracted"
        )
    left_band = left.get_seam(from_seam)
    right_band = right.get_seam(to_seam)
    left_identities = left.seam_rows(from_seam)
    right_identities = right.seam_rows(to_seam)
    if left_identities != right_identities:
        raise ValueError(
            "the contracted boundary is not the same detector row band: the left "
            f"chunk's {from_seam} seam holds detectors {list(left_identities)} and "
            f"the right chunk's {to_seam} seam holds {list(right_identities)}"
        )
    if right_band.row_begin != 0:
        raise ValueError(
            "a right-hand chunk's contracted band must be its leading band, "
            "because the result lays the shared boundary out between the two "
            "interiors"
        )
    removed = right_band.row_end
    width = left.model.num_detectors

    def shift(row: int) -> int:
        if row < removed:
            return left_band.row_begin + row
        return width + row - removed

    errors = list(left.model.errors)
    errors.extend(
        DemError(
            probability=error.probability,
            detectors=tuple(shift(row) for row in error.detectors),
            observables=error.observables,
            error_id=error.error_id,
        )
        for error in right.model.errors
    )
    seams: list[DemSeam] = []
    if left.has_seam(to_seam):
        leading = left.get_seam(to_seam)
        seams.append(DemSeam(to_seam, leading.row_begin, leading.row_end))
    if right.has_seam(from_seam):
        trailing = right.get_seam(from_seam)
        begin = shift(trailing.row_begin)
        seams.append(DemSeam(from_seam, begin, begin + trailing.width))
    detectors = width + right.model.num_detectors - removed
    return DemChunk(
        model=DetectorErrorModel(
            num_detectors=detectors,
            num_observables=left.model.num_observables,
            errors=tuple(errors),
        ),
        first_row=left.first_row,
        seams=tuple(seams),
        first_layer=left.first_layer,
        last_layer=right.last_layer,
    )


def dem_stitch_all(
    chunks: Sequence[DemChunk],
    from_seam: SeamId = SeamId.next_round,
    to_seam: SeamId = SeamId.prev_round,
) -> DemChunk:
    """Fold a sequence of adjacent chunks into one, left to right.

    The sequence must be in order and each chunk must be adjacent to the one
    before it, which is what a spec's windows are and what a hand-built list has
    to state.
    """

    items = tuple(chunks)
    if not items:
        raise ValueError("a stitch needs at least one chunk")
    stitched = items[0]
    for chunk in items[1:]:
        stitched = dem_stitch(stitched, chunk, from_seam, to_seam)
    return stitched


def dem_stitch_merged(
    chunks: Sequence[DemChunk],
    *,
    rule: DemMergeRule | str = DemMergeRule.INDEPENDENT_PARITY,
    from_seam: SeamId = SeamId.next_round,
    to_seam: SeamId = SeamId.prev_round,
) -> DemChunk:
    """Stitch a sequence, then give each signature one mechanism.

    A partition of a model holds one mechanism per fault, so a merged stitch of
    one is that model with its duplicate signatures collected -- which is what a
    decoder wants to weight, and what a stitch of chunks that were cut
    independently needs before it is one.
    """

    stitched = dem_stitch_all(chunks, from_seam, to_seam)
    return replace(stitched, model=stitched.model.merge_duplicate_mechanisms(rule=rule))


def dem_close(chunk: DemChunk) -> DetectorErrorModel:
    """Collapse one chunk to a flat model numbered by detector identity.

    The rows keep the identity the chunk recorded for them rather than being
    renumbered from zero, so closing a window cut out of the middle of a model
    gives a model whose detector indices are still that model's. No row order is
    chosen here: a chunk's rows are increasing already, so the flat order is the
    identity order.
    """

    if not isinstance(chunk, DemChunk):
        raise TypeError("dem_close requires a DemChunk")
    errors = tuple(
        DemError(
            probability=error.probability,
            detectors=tuple(chunk.rows[row] for row in error.detectors),
            observables=error.observables,
            error_id=error.error_id,
        )
        for error in chunk.model.errors
    )
    return DetectorErrorModel(
        num_detectors=chunk.first_row + chunk.model.num_detectors,
        num_observables=chunk.model.num_observables,
        errors=errors,
    )


def dem_close_all(chunks: Sequence[DemChunk]) -> DetectorErrorModel:
    """Close a whole sequence of chunks at once.

    This is what :func:`dem_close` of :func:`dem_stitch_all` returns, built in
    one pass rather than by stitching and then relabelling, so a long sequence
    does not pay for a chain of intermediate models. The sequence must be
    adjacent window by window and the contracted bands must be the same
    detectors, because a sequence that is not is not a decomposition of one
    model however its windows were built.
    """

    items = tuple(chunks)
    if not items:
        raise ValueError("a close needs at least one chunk")
    for chunk in items:
        if not isinstance(chunk, DemChunk):
            raise TypeError("every element of a close must be a DemChunk")
    observables = items[0].model.num_observables
    for chunk in items:
        if chunk.model.num_observables != observables:
            raise ValueError(
                "a chunk sequence must agree on its observable count, because an "
                "observable is a property of the whole experiment"
            )
    for left, right in zip(items, items[1:], strict=False):
        if left.last_layer != right.first_layer:
            raise ValueError(
                f"chunks spanning layers {left.first_layer}..{left.last_layer} and "
                f"{right.first_layer}..{right.last_layer} do not share exactly one "
                "boundary layer, so a layer of the model is in no window or the "
                "two windows are not adjacent"
            )
        if not left.has_seam(SeamId.next_round) or not right.has_seam(
            SeamId.prev_round
        ):
            raise ValueError(
                "adjacent windows must share a boundary, so every chunk but the "
                "last needs a next_round seam and every chunk but the first needs "
                "a prev_round seam"
            )
        if left.seam_rows(SeamId.next_round) != right.seam_rows(SeamId.prev_round):
            raise ValueError(
                "adjacent windows do not agree on the detectors of their shared "
                f"boundary: {list(left.seam_rows(SeamId.next_round))} against "
                f"{list(right.seam_rows(SeamId.prev_round))}"
            )
    errors: list[DemError] = []
    for chunk in items:
        errors.extend(
            DemError(
                probability=error.probability,
                detectors=tuple(chunk.rows[row] for row in error.detectors),
                observables=error.observables,
                error_id=error.error_id,
            )
            for error in chunk.model.errors
        )
    last = max(chunk.first_row + chunk.model.num_detectors for chunk in items)
    return DetectorErrorModel(
        num_detectors=last,
        num_observables=observables,
        errors=tuple(errors),
    )
