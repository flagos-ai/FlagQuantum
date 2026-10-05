"""The decoder inputs a memory experiment implies, read one basis at a time.

A detector error model states which mechanisms a decoder can see. It does not
state where in a shot each detector's parity is read, and that is the other half
of what a decoder needs: a matcher is handed a syndrome over measurements and
has to know which measurements compose each detector, and which compose each
observable.

Upstream cudaq-qec hands both halves over together.
``decoder_context_from_memory_circuit`` runs the circuit analysis once and
returns a ``decoder_context`` whose ``x_component()``, ``z_component()`` and
``full_component()`` each return a ``decoder_inputs`` -- a canonicalized
detector error model plus the measurement-to-detector and
measurement-to-observable maps for that component -- and whose
``num_measurements()`` states the width of the buffer the maps index.

This module supplies the same four members from a memory circuit:

* :class:`MeasurementMap` is one of the two maps: for each detector (or
  observable) row, the measurements whose parity it is. It is upstream's sparse
  D and O, and it projects to the dense matrix and to the ``-1``-terminated
  sparse vector a realtime decoder configuration takes.
* :class:`DecoderInputs` carries a model together with both maps.
* :class:`DecoderContext` holds a circuit and the model built from it, and
  projects both onto one basis.

The numbering is the sampler's. That is not a coincidence to be hoped for:
:func:`flagquantum.qec.sample_memory_circuit` records its bits in the same flat
buffer -- round by round, each round in the code's declared check order, with
the terminal data readouts appended -- and the two are pinned to each other by a
test rather than by a shared constant. They are independent readings of one
convention: the sampler derives its numbering from the *lowered program*, and
this module derives it from the circuit's own *declaration*. Deriving it here
from the declaration is what lets a caller hold the maps without lowering
anything, and the test is what keeps the two readings one convention.

Two things this module does not do, both stated rather than implied. It does not
execute the circuit: the maps are read off the layouts, so a context costs one
model construction and no simulation. And it does not merge two mechanisms that
a projection collapses into one signature silently -- see
:func:`_projected`, which refuses an id-carrying model rather than either
dropping the correlation or merging two members of a group of alternatives.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import torch

from .circuit import MeasurementRef, MemoryCircuit
from .codes import CodeCheck
from .dem import DetectorErrorModel
from .noise import PhenomenologicalNoise

__all__ = (
    "DecoderContext",
    "DecoderInputs",
    "MeasurementMap",
    "decoder_context_from_memory_circuit",
)


def _measurement_row(
    values: Iterable[int], *, num_measurements: int
) -> tuple[int, ...]:
    """Normalize one row of a measurement map.

    The row is sorted because a parity is a set of measurements and the order
    they were listed in is not part of it. It is *not* de-duplicated: two reads
    of one measurement cancel rather than add, so a row that names a measurement
    twice is refused instead of being read as naming it once. Zero is not a
    special case of that -- a measurement read no times is simply absent from
    the row, which is why the empty row is a legal one.
    """

    entries: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("a measurement map row must contain integer indices")
        index = int(value)
        if index < 0 or index >= num_measurements:
            raise ValueError(
                f"a measurement map row names measurement {index}, which is "
                f"outside the {num_measurements} measurement(s) the map states"
            )
        if index in seen:
            raise ValueError(
                f"a measurement map row names measurement {index} twice: two "
                "reads of one measurement cancel, so a row cannot state one"
            )
        seen.add(index)
        entries.append(index)
    return tuple(sorted(entries))


@dataclass(frozen=True)
class MeasurementMap:
    """The measurements whose parity is each detector or observable row.

    Row order is the row order of the thing the map describes: row ``i`` of a
    measurement-to-detector map is detector ``i``, which is the layout's own
    index, so a caller never translates between a row number and a detector
    number.

    ``num_measurements`` is the width of the buffer the indices address. It is
    the number of measurements one shot of the experiment records, so it is the
    same value for both maps of one experiment and is stated once per map rather
    than inferred from the largest index a row happens to name -- a detector
    that reads nothing from the last round still has a buffer of that width.
    """

    num_measurements: int
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if isinstance(self.num_measurements, bool) or not isinstance(
            self.num_measurements, Integral
        ):
            raise TypeError("the measurement count must be an integer")
        count = int(self.num_measurements)
        if count < 0:
            raise ValueError("the measurement count must be non-negative")
        object.__setattr__(self, "num_measurements", count)
        object.__setattr__(
            self,
            "rows",
            tuple(_measurement_row(row, num_measurements=count) for row in self.rows),
        )

    @property
    def num_rows(self) -> int:
        """Return how many detectors or observables the map has a row for."""

        return len(self.rows)

    @property
    def num_entries(self) -> int:
        """Return how many measurement references the map holds in total."""

        return sum(len(row) for row in self.rows)

    def dense(self) -> torch.Tensor:
        """Return the map as a dense ``(rows, measurements)`` binary tensor.

        The orientation is upstream's D and O: one row per detector or
        observable, one column per measurement, and an entry is one exactly when
        that row's parity reads that measurement.
        """

        matrix = torch.zeros((self.num_rows, self.num_measurements), dtype=torch.int8)
        for index, row in enumerate(self.rows):
            if row:
                matrix[index, list(row)] = 1
        return matrix

    def flattened(self) -> tuple[int, ...]:
        """Return the ``-1``-terminated sparse form, one row after another.

        This is the shape a realtime decoder configuration takes for its
        detector list, and upstream's ``d_sparse`` helper produces it from the
        same map: each row's measurement indices in ascending order, then
        ``-1`` to end the row. The terminator is what makes an empty row
        readable -- without it a row that reads nothing would be invisible and
        the next row's first index would be read as its own.
        """

        entries: list[int] = []
        for row in self.rows:
            entries.extend(row)
            entries.append(-1)
        return tuple(entries)


@dataclass(frozen=True)
class DecoderInputs:
    """A model together with the maps that say where its parities are read.

    The three agree on shape by construction rather than by convention: the
    detector map has one row per detector the model declares and the observable
    map one row per observable, so a caller who reads row ``i`` of the detector
    map is reading detector ``i`` of the model and never has to be told which
    numbering it is holding.
    """

    dem: DetectorErrorModel
    measurement_to_detectors: MeasurementMap
    measurement_to_observables: MeasurementMap

    def __post_init__(self) -> None:
        if not isinstance(self.dem, DetectorErrorModel):
            raise TypeError("decoder inputs require a detector error model")
        if not isinstance(self.measurement_to_detectors, MeasurementMap):
            raise TypeError("the detector map must be a MeasurementMap")
        if not isinstance(self.measurement_to_observables, MeasurementMap):
            raise TypeError("the observable map must be a MeasurementMap")
        if self.measurement_to_detectors.num_rows != self.dem.num_detectors:
            raise ValueError(
                "the detector map must have one row per detector the model "
                f"declares: {self.measurement_to_detectors.num_rows} row(s) "
                f"against {self.dem.num_detectors} detector(s)"
            )
        if self.measurement_to_observables.num_rows != self.dem.num_observables:
            raise ValueError(
                "the observable map must have one row per observable the model "
                f"declares: {self.measurement_to_observables.num_rows} row(s) "
                f"against {self.dem.num_observables} observable(s)"
            )
        if (
            self.measurement_to_detectors.num_measurements
            != self.measurement_to_observables.num_measurements
        ):
            raise ValueError(
                "the two maps must be read from the same measurement buffer, "
                "because one shot records one set of measurements"
            )

    @property
    def num_measurements(self) -> int:
        """Return the width of the measurement buffer both maps index."""

        return self.measurement_to_detectors.num_measurements


def _measurement_columns(
    circuit: MemoryCircuit,
) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    """Return the record column of every measurement the experiment takes.

    One syndrome round records one measurement per check, in the code's declared
    check order, so round ``r``'s check at position ``p`` is column
    ``r * n_checks + p``. The terminal data readout follows the last round and
    is column ``rounds * n_checks + q`` for the data qubit at position ``q``.

    Both halves are read from the circuit's declaration rather than from a
    lowered program, which is what makes a context cheap; that the sampler's
    independently derived numbering agrees with this one is asserted by a test
    rather than assumed here.
    """

    checks = tuple(circuit.code.checks)
    per_round = len(checks)
    syndrome = {
        (round_index, check.ancilla_qubit): round_index * per_round + position
        for round_index in range(circuit.rounds)
        for position, check in enumerate(checks)
    }
    terminal = {
        wire: per_round * circuit.rounds + position
        for position, wire in enumerate(circuit.code.data_qubits)
    }
    return syndrome, terminal


def _measurement_index(
    reference: MeasurementRef,
    syndrome: dict[tuple[int, int], int],
    terminal: dict[int, int],
) -> int:
    """Return the record column one measurement reference addresses."""

    if reference.round_index is None:
        column = terminal.get(reference.qubit)
        described = f"the terminal readout of data qubit {reference.qubit}"
    else:
        column = syndrome.get((reference.round_index, reference.qubit))
        described = (
            f"the syndrome measurement of qubit {reference.qubit} in round "
            f"{reference.round_index}"
        )
    if column is None:
        raise ValueError(
            f"a detector or observable reads {described}, which this memory "
            "experiment does not record"
        )
    return column


def _parity_row(
    parity: Iterable[MeasurementRef],
    syndrome: dict[tuple[int, int], int],
    terminal: dict[int, int],
) -> tuple[int, ...]:
    """Return the record columns one detector's or observable's parity reads."""

    return tuple(
        _measurement_index(reference, syndrome, terminal) for reference in parity
    )


def _detector_checks(circuit: MemoryCircuit) -> tuple[CodeCheck, ...]:
    """Return the check each detector belongs to, in the layout's own order.

    A detector compares one check's readout with what that check's readout was
    before it -- in round zero with the known prior state, in the last round with
    the terminal data readout -- so every detector is carried by exactly one
    check, and that check is the one whose ancilla the detector's syndrome
    measurements read. The check is looked up from the layout's own declaration
    rather than inferred from the detector's position, so a code that declares
    its checks in an unusual order cannot have a detector attributed to a
    neighbour.
    """

    by_ancilla = {check.ancilla_qubit: check for check in circuit.code.checks}
    owners: list[CodeCheck] = []
    for detector in circuit.detectors.detectors:
        ancillas = {
            reference.qubit
            for reference in detector.parity
            if reference.round_index is not None
        }
        if len(ancillas) != 1:
            raise ValueError(
                "a detector's syndrome measurements must all read one check's "
                "ancilla, so that it belongs to exactly one check"
            )
        check = by_ancilla.get(next(iter(ancillas)))
        if check is None:
            raise ValueError(
                "a detector reads a syndrome measurement on a wire that no "
                "check declares"
            )
        owners.append(check)
    return tuple(owners)


def _projected(dem: DetectorErrorModel, kept: tuple[int, ...]) -> DetectorErrorModel:
    """Return ``dem`` read over the detectors ``kept``, in the given order.

    A mechanism keeps the kept detectors it flips, renumbered to the new rows,
    and its observables unchanged. A mechanism that no longer flips a kept
    detector but still flips an observable survives: it is an undetectable
    logical fault, and whether to drop it is a modelling decision the caller
    makes rather than one a projection makes for them -- upstream's
    ``remove_zero_syndrome_errors`` flag is that decision, and its default keeps
    them.

    Two mechanisms whose projected signatures coincide are one fault to a
    decoder, which sees only whether the shared signature flipped, so they are
    merged by the parity rule construction uses. That is what makes the
    projection margin-preserving rather than merely tidy: every signature keeps
    the rate it had.

    An id-carrying model is refused. A projection can either drop the
    correlation or merge two members of a group of alternatives, and both are
    changes to what the model states rather than readings of it; naming the ids
    is what the three other operations that would lose them already do.
    """

    stated = dem.stated_error_ids()
    if stated:
        raise ValueError(
            "a basis component cannot be read from a model that states error "
            f"ids {stated}: projecting a group of alternatives onto one basis "
            "would either drop the correlation or merge two of its members, so "
            "it is refused rather than approximated"
        )
    index_of = {original: new for new, original in enumerate(kept)}
    entries = [
        (
            error.probability,
            tuple(index_of[index] for index in error.detectors if index in index_of),
            error.observables,
        )
        for error in dem.errors
    ]
    return DetectorErrorModel._merge_mechanisms(
        entries, num_detectors=len(kept), num_observables=dem.num_observables
    )


@dataclass(frozen=True)
class DecoderContext:
    """A memory circuit and the model built from it, projected per basis.

    The model is built once, by :func:`decoder_context_from_memory_circuit`, and
    every component is a projection of it, so a caller who reads all three pays
    for construction once. That is the whole content of the handle upstream
    calls lazy: the expensive half is the circuit analysis, and the basis is
    chosen afterwards.

    The basis decides *which detectors* are read, not which mechanisms exist:
    a detector is carried by the check whose ancilla it reads, so the X
    component is the detectors of the code's X-type checks and the Z component
    the detectors of its Z-type checks. The terminal detectors -- the ones
    comparing the last syndrome round with the data readout -- are carried by the
    checks of the experiment's own readout basis, so they belong to that basis'
    component rather than to a third category. Upstream needs a separate
    boundary-aware variant because its D matrix is laid out in uniform per-round
    blocks; here the layout states the boundary detectors themselves, so the
    union needs no separate bookkeeping, and ``full_component`` is the model as
    built.
    """

    circuit: MemoryCircuit
    dem: DetectorErrorModel

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, MemoryCircuit):
            raise TypeError("a decoder context requires a MemoryCircuit")
        if not isinstance(self.dem, DetectorErrorModel):
            raise TypeError("a decoder context requires a detector error model")
        if self.dem.num_detectors != len(self.circuit.detectors):
            raise ValueError(
                "the model's detector count must be the circuit's: "
                f"{self.dem.num_detectors} against {len(self.circuit.detectors)}"
            )
        if self.dem.num_observables != len(self.circuit.observables):
            raise ValueError(
                "the model's observable count must be the circuit's: "
                f"{self.dem.num_observables} against "
                f"{len(self.circuit.observables)}"
            )

    def num_measurements(self) -> int:
        """Return how many measurements one shot of this experiment records.

        This is upstream's ``num_measurements``: the width of the buffer the
        component maps index, one measurement per check per round plus one
        terminal readout per data qubit.
        """

        return len(self.circuit.code.checks) * self.circuit.rounds + len(
            self.circuit.code.data_qubits
        )

    def full_component(self) -> DecoderInputs:
        """Return the model as built, with a map row for every detector.

        Nothing is projected away, so the model is the one the circuit produced
        and the maps describe every detector and observable the experiment has,
        boundary detectors included.
        """

        return self._inputs(self.dem, tuple(range(self.dem.num_detectors)))

    def x_component(self) -> DecoderInputs:
        """Return the model read over the detectors of the code's X-type checks."""

        return self._basis(x_type=True)

    def z_component(self) -> DecoderInputs:
        """Return the model read over the detectors of the code's Z-type checks.

        A Z-basis experiment's terminal detectors are included here, because they
        are carried by its Z-type checks: a terminal detector compares such a
        check's readout with the data readout taken in the same basis. In an
        X-basis experiment the same role is played by the X-type checks, so the
        terminal detectors are in the X component instead.
        """

        return self._basis(x_type=False)

    def _inputs(self, dem: DetectorErrorModel, kept: tuple[int, ...]) -> DecoderInputs:
        """Return ``dem`` with both maps over the detectors ``kept``."""

        syndrome, terminal = _measurement_columns(self.circuit)
        width = self.num_measurements()
        detector_rows = tuple(
            _parity_row(
                self.circuit.detectors.detectors[index].parity, syndrome, terminal
            )
            for index in kept
        )
        observable_rows = tuple(
            _parity_row(observable.measurement_parity, syndrome, terminal)
            for observable in self.circuit.observables.observables
        )
        return DecoderInputs(
            dem=dem,
            measurement_to_detectors=MeasurementMap(
                num_measurements=width, rows=detector_rows
            ),
            measurement_to_observables=MeasurementMap(
                num_measurements=width, rows=observable_rows
            ),
        )

    def _basis(self, *, x_type: bool) -> DecoderInputs:
        """Return the component for one basis, or refuse an empty one."""

        owners = _detector_checks(self.circuit)
        kept = tuple(
            detector.index
            for detector, owner in zip(
                self.circuit.detectors.detectors, owners, strict=True
            )
            if bool(owner.stabilizer.x_qubits) is x_type
        )
        if kept:
            return self._inputs(_projected(self.dem, kept), kept)
        basis, other = ("X", "Z") if x_type else ("Z", "X")
        raise ValueError(
            f"this memory experiment has no detector carried by an {basis}-type "
            f"check, so its {basis} component would hold no detector at all, "
            "which a detector error model here does not represent: every "
            f"detector this code's {self.circuit.rounds} round(s) declare is "
            f"carried by a {other}-type check"
        )


def decoder_context_from_memory_circuit(
    circuit: MemoryCircuit, *, noise: PhenomenologicalNoise
) -> DecoderContext:
    """Build a memory experiment's decoder inputs, constructing the model once.

    Upstream's name for this is
    ``decoder_context_from_memory_circuit(code, statePrep, numRounds, noise)``.
    Here the memory circuit already carries the code, the round count and the
    state preparation the experiment is configured with, so the circuit is the
    whole configuration and the noise is the only second argument. The model is
    built eagerly and stored, because a context is asked for exactly to be read
    a basis at a time, and every one of those reads needs the same model.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    return DecoderContext(
        circuit=circuit,
        dem=DetectorErrorModel.from_memory_circuit(circuit, noise=noise),
    )
