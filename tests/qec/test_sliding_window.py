"""Coverage for the chunk-scoped projections and the sliding-window decoder."""

from __future__ import annotations

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec.belief_propagation import BeliefPropagationDecoder
from flagquantum.qec.chunks import (
    ChunkLayout,
    DemChunkSpec,
    DemChunksSpec,
    PhaseId,
    dem_chunk_from_spec,
    dem_chunks_from_spec,
    dem_close_all,
)
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode, RotatedSurfaceCode
from flagquantum.qec.dem import DemError, DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.registry import AUTHORITY_NAME, BELIEF_PROPAGATION_NAME
from flagquantum.qec.sliding_window import (
    SlidingWindowDecoder,
    SlidingWindowDecodeResult,
    dem_chunks_to_d_sparse,
    dem_chunks_to_o_sparse,
    dem_chunks_to_pcm,
)

pytestmark = pytest.mark.unit


NOISE = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.02)

# The exact model of a distance-three repetition memory whose syndrome rounds are
# four, which is the shape every claim here is stated over: five detector layers
# of two rows, ten detector rows, twenty mechanisms, and every mechanism flips a
# pair of adjacent detectors.
REPETITION_ROUNDS = 4

# A model of uniform two-row rounds whose mechanisms span two adjacent detectors,
# which is the same shape without a circuit behind it.
SPANS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9))


def _segmented_model(num_layers: int = 4) -> DetectorErrorModel:
    """A uniform-round model whose faults span two detectors each."""

    return DetectorErrorModel(
        num_detectors=2 * num_layers,
        num_observables=1,
        errors=tuple(
            DemError(probability=0.01 * (index + 1), detectors=detectors)
            for index, detectors in enumerate(SPANS[: 2 * num_layers - 1])
        ),
    )


def _layout(num_layers: int = 4) -> ChunkLayout:
    return ChunkLayout(model=_segmented_model(num_layers), widths=(2,) * num_layers)


def _windows(window: int = 2, num_layers: int = 4) -> tuple:
    """The windows of a synthetic layout that tiles into the given window size."""

    return dem_chunks_from_spec(
        DemChunksSpec(layout=_layout(num_layers), window=window)
    )


def _memory_model(code: object, rounds: int) -> tuple[DetectorErrorModel, ChunkLayout]:
    """The exact model of a memory circuit and the layout of its rounds."""

    circuit = build_memory_circuit(code, rounds=rounds)  # type: ignore[arg-type]
    model = DetectorErrorModel.from_memory_circuit(circuit, noise=NOISE)
    return model, ChunkLayout.from_memory_circuit(model, circuit)


def _repetition(window: int = 3) -> tuple[DetectorErrorModel, DemChunksSpec, tuple]:
    """A repetition memory, the decomposition of its model, and its windows."""

    model, layout = _memory_model(RepetitionCode(distance=3), rounds=REPETITION_ROUNDS)
    spec = DemChunksSpec(layout=layout, window=window)
    return model, spec, dem_chunks_from_spec(spec)


def _column_support(pcm: tuple[tuple[int, ...], ...], columns: int) -> list[set[int]]:
    """Transpose a row-indexed parity-check matrix into its columns."""

    support: list[set[int]] = [set() for _ in range(columns)]
    for row, entry in enumerate(pcm):
        for column in entry:
            support[column].add(row)
    return support


def _layers(layout: ChunkLayout, events: list[int]) -> list[list[int]]:
    """Number a model syndrome the way each round of the layout numbers it."""

    return [
        [
            row - layout.first_row_of(layer)
            for row in events
            if layout.first_row_of(layer)
            <= row
            < layout.first_row_of(layer) + layout.width_of(layer)
        ]
        for layer in range(layout.num_layers)
    ]


# ---------------------------------------------------------------------------
# dem_chunks_to_d_sparse
# ---------------------------------------------------------------------------


def test_the_memory_map_pairs_each_round_with_the_one_before_it() -> None:
    rows = dem_chunks_to_d_sparse(_windows())
    # Four two-row rounds, so detector r*2 + k pairs bit k of round r-1 with bit
    # k of round r, and the first round's detectors compare against nothing.
    assert len(rows) == 8
    assert rows[0] == (0,)
    assert rows[1] == (1,)
    assert rows[2] == (0, 2)
    assert rows[3] == (1, 3)
    assert rows[6] == (4, 6)
    assert rows[7] == (5, 7)


def test_the_memory_map_covers_every_detector_exactly_once() -> None:
    rows = dem_chunks_to_d_sparse(_windows(window=3, num_layers=5))
    assert len(rows) == _layout(5).num_detectors == 10
    assert [len(entry) for entry in rows[:2]] == [1, 1]
    assert all(len(entry) == 2 for entry in rows[2:])


def test_the_memory_map_refuses_a_sequence_its_rounds_do_not_describe() -> None:
    # A surface code's boundary rounds are half as wide as its interior ones, so
    # its detectors are not pairs of uniform rounds and this memory map is not
    # the map that covers them: the circuit-derived MeasurementMap is.
    _, layout = _memory_model(RotatedSurfaceCode(distance=3), rounds=4)
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    assert layout.widths == (4, 8, 8, 8, 4)
    with pytest.raises(ValueError, match="not a whole number of 8-row rounds"):
        dem_chunks_to_d_sparse(chunks)


def test_the_memory_map_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="requires at least one window"):
        dem_chunks_to_d_sparse(())
    with pytest.raises(ValueError, match="requires at least one window"):
        dem_chunks_to_pcm(())
    with pytest.raises(ValueError, match="requires at least one window"):
        dem_chunks_to_o_sparse(())


def test_the_memory_map_refuses_a_window_with_no_boundary_at_all() -> None:
    # One window covering the whole layout: it has no neighbour on either side,
    # so it carries no boundary band and its rounds cannot be counted.
    whole = dem_chunk_from_spec(
        DemChunkSpec(layout=_layout(), first_layer=0, last_layer=3, phase=PhaseId.bulk)
    )
    assert whole.seams == ()
    with pytest.raises(ValueError, match="no boundary band on either side"):
        dem_chunks_to_d_sparse((whole,))
    # The same layout through a decomposition states a boundary, so it counts.
    assert dem_chunks_to_d_sparse(_windows())[0] == (0,)


# ---------------------------------------------------------------------------
# dem_chunks_to_pcm and dem_chunks_to_o_sparse
# ---------------------------------------------------------------------------


def test_the_row_projection_is_the_closed_models_own_matrix() -> None:
    chunks = _windows()
    closed = dem_close_all(chunks)
    dense = closed.detector_error_matrix().tolist()
    rows = dem_chunks_to_pcm(chunks)
    assert len(rows) == closed.num_detectors
    for row, entry in enumerate(rows):
        assert entry == tuple(index for index, value in enumerate(dense[row]) if value)


def test_the_observable_projection_is_the_closed_models_own_matrix() -> None:
    chunks = _windows()
    closed = dem_close_all(chunks)
    dense = closed.observables_flips_matrix().tolist()
    rows = dem_chunks_to_o_sparse(chunks)
    assert len(rows) == closed.num_observables
    for observable, entry in enumerate(rows):
        assert entry == tuple(
            index for index, value in enumerate(dense[observable]) if value
        )


def test_the_two_projections_share_one_fault_column_space() -> None:
    chunks = _windows(window=3, num_layers=5)
    pcm = dem_chunks_to_pcm(chunks)
    observables = dem_chunks_to_o_sparse(chunks)
    columns = len(dem_close_all(chunks).errors)
    assert all(column < columns for entry in pcm for column in entry)
    assert all(column < columns for entry in observables for column in entry)
    # Every mechanism flips at least one detector, so the row projection covers
    # every mechanism of the model.
    assert {column for entry in pcm for column in entry} == set(range(columns))


def test_a_row_projection_refuses_a_sequence_that_is_not_a_decomposition() -> None:
    # Windows one and three of a four-window decomposition: the layer between
    # them is in no window, so they are not two windows of one model however
    # well formed each is on its own.
    first, _, third, _ = _windows(window=2, num_layers=5)
    with pytest.raises(ValueError, match="do not share exactly one boundary layer"):
        dem_chunks_to_pcm((first, third))
    with pytest.raises(ValueError, match="do not share exactly one boundary layer"):
        dem_chunks_to_o_sparse((first, third))


def test_a_projection_refuses_windows_that_disagree_on_the_observables() -> None:
    one = dem_chunk_from_spec(
        DemChunkSpec(layout=_layout(), first_layer=0, last_layer=1, phase=PhaseId.init)
    )
    two = DetectorErrorModel(
        num_detectors=8,
        num_observables=2,
        errors=(DemError(probability=0.01, detectors=(0, 1)),),
    )
    other = dem_chunk_from_spec(
        DemChunkSpec(
            layout=ChunkLayout(model=two, widths=(2, 2, 2, 2)),
            first_layer=0,
            last_layer=1,
            phase=PhaseId.init,
        )
    )
    with pytest.raises(ValueError, match="window 1 states 2 observables"):
        dem_chunks_to_pcm((one, other))
    with pytest.raises(ValueError, match="window 1 states 2 observables"):
        dem_chunks_to_o_sparse((one, other))


# ---------------------------------------------------------------------------
# SlidingWindowDecodeResult
# ---------------------------------------------------------------------------


def test_a_result_record_normalises_what_it_was_handed() -> None:
    record = SlidingWindowDecodeResult(
        observables=(2, 1, 1),
        faults=(3, 0, 3),
        windows_decoded=2,
        complete=False,
        converged=None,
    )
    assert record.observables == (1, 2)
    assert record.faults == (0, 3)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {"windows_decoded": 0, "complete": 1, "converged": None},
            TypeError,
            "completion as a boolean",
        ),
        (
            {"windows_decoded": -1, "complete": True, "converged": None},
            ValueError,
            "cannot be negative",
        ),
        (
            {"windows_decoded": 0, "complete": True, "converged": "yes"},
            TypeError,
            "convergence as a boolean",
        ),
    ],
)
def test_a_result_record_refuses_what_it_cannot_mean(
    kwargs: dict, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        SlidingWindowDecodeResult(observables=(), faults=(), **kwargs)


# ---------------------------------------------------------------------------
# building the decoder
# ---------------------------------------------------------------------------


def test_a_decoder_refuses_a_spec_it_does_not_slide_over() -> None:
    with pytest.raises(TypeError, match="slides over a DemChunksSpec"):
        SlidingWindowDecoder("spec", BELIEF_PROPAGATION_NAME)  # type: ignore[arg-type]


def test_a_decoder_refuses_a_name_that_selects_nothing() -> None:
    spec = DemChunksSpec(layout=_layout(), window=2)
    with pytest.raises(ValueError, match="needs the name of an inner decoder"):
        SlidingWindowDecoder(spec, "")
    with pytest.raises(ValueError, match="needs the name of an inner decoder"):
        SlidingWindowDecoder(spec, "   ")


def test_a_decoder_refuses_a_name_the_registry_does_not_hold() -> None:
    spec = DemChunksSpec(layout=_layout(), window=2)
    with pytest.raises(ValueError, match="no decoder is registered as 'nope'"):
        SlidingWindowDecoder(spec, "nope")


def test_a_decoder_holds_the_model_the_windows_decompose() -> None:
    spec = DemChunksSpec(layout=_layout(), window=2)
    decoder = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME)
    chunks = dem_chunks_from_spec(spec)
    assert decoder.spec is spec
    assert decoder.chunks == chunks
    assert decoder.num_windows == 3
    assert decoder.model == dem_close_all(chunks)
    assert decoder.windows_decoded == 0


# ---------------------------------------------------------------------------
# feeding syndromes
# ---------------------------------------------------------------------------


def test_a_decoder_refuses_a_round_of_the_wrong_width() -> None:
    decoder = SlidingWindowDecoder(
        DemChunksSpec(layout=_layout(), window=2), BELIEF_PROPAGATION_NAME
    )
    with pytest.raises(ValueError, match="detector 2 is outside the 2 detector"):
        decoder.decode_round([0, 1, 2])
    with pytest.raises(ValueError, match="detector 5 is outside the 2 detector"):
        decoder.decode_round([5])


def test_a_decoder_refuses_a_syndrome_that_names_a_detector_twice() -> None:
    decoder = SlidingWindowDecoder(
        DemChunksSpec(layout=_layout(), window=2), BELIEF_PROPAGATION_NAME
    )
    with pytest.raises(ValueError, match="cannot name the same detector twice"):
        decoder.decode_round([0, 0])
    # Nothing was buffered, so the refused round is not half-counted.
    assert decoder.windows_decoded == 0


def test_a_decoder_refuses_an_event_outside_the_model_it_decomposes() -> None:
    decoder = SlidingWindowDecoder(
        DemChunksSpec(layout=_layout(), window=2), BELIEF_PROPAGATION_NAME
    )
    with pytest.raises(ValueError, match="outside the 8 detector"):
        decoder.decode([99])


def test_no_record_comes_out_before_the_first_window_closes() -> None:
    _, spec, _ = _repetition(window=3)
    decoder = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME)
    # A three-round window is not ready after two rounds.
    assert decoder.decode_round([]) is None
    assert decoder.decode_round([]) is None
    assert decoder.windows_decoded == 0
    record = decoder.decode_round([])
    assert record is not None
    assert record.windows_decoded == 1
    # The second and last window of the block has not closed yet.
    assert record.complete is False


def test_a_whole_block_cannot_be_decoded_into_a_block_in_progress() -> None:
    decoder = SlidingWindowDecoder(
        DemChunksSpec(layout=_layout(), window=2), BELIEF_PROPAGATION_NAME
    )
    assert decoder.decode_round([]) is None
    with pytest.raises(RuntimeError, match="already in progress"):
        decoder.decode([])


def test_a_block_and_a_stream_of_rounds_agree() -> None:
    model, layout = _memory_model(RepetitionCode(distance=3), rounds=REPETITION_ROUNDS)
    spec = DemChunksSpec(layout=layout, window=2)
    events = [0, 3, 4]
    whole = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode(events)
    streamed = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME)
    last = None
    for layer in _layers(layout, events):
        record = streamed.decode_round(layer)
        if record is not None:
            last = record
    assert last is not None
    assert last == whole
    assert whole.complete is True
    assert whole.windows_decoded == spec.num_windows


def test_a_finished_block_leaves_the_decoder_ready_for_the_next_one() -> None:
    _, spec, _ = _repetition()
    decoder = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME)
    assert decoder.decode([]).complete is True
    # The block reset itself, so the next one is a whole block again rather than
    # a continuation of the one that finished.
    assert decoder.windows_decoded == 0
    assert decoder.decode([]).faults == ()


def test_a_window_an_inner_decoder_cannot_answer_resets_the_stream() -> None:
    _, spec, _ = _repetition(window=2)
    decoder = SlidingWindowDecoder(spec, AUTHORITY_NAME)
    # The authority matcher states its correction as graph edges rather than as
    # mechanism columns, so a sliding window cannot carry its answer forward.
    with pytest.raises(TypeError, match="states no selected mechanisms"):
        decoder.decode([])
    assert decoder.windows_decoded == 0
    # The reset is what makes the next block start clean: the same refusal comes
    # from round zero rather than from a window the failed block had half filled.
    assert decoder.decode_round([]) is None
    with pytest.raises(TypeError, match="states no selected mechanisms"):
        decoder.decode_round([])
    assert decoder.windows_decoded == 0


def test_a_window_that_cannot_see_a_mechanism_refuses_rather_than_guesses() -> None:
    # The last window of a two-round decomposition holds only the terminal
    # mechanisms, so a fault of the round before it is outside what that window
    # can explain. Upstream abandons the block here, and so does this decoder.
    model, spec, chunks = _repetition(window=2)
    columns = _column_support(dem_chunks_to_pcm(chunks), len(model.errors))
    syndrome = sorted(columns[17])
    assert syndrome == [6, 8]
    with pytest.raises(CapabilityError, match="no set of mechanisms explains"):
        SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode(syndrome)


# ---------------------------------------------------------------------------
# what a finished record claims
# ---------------------------------------------------------------------------


def test_a_record_explains_the_whole_syndrome_it_was_given() -> None:
    model, spec, chunks = _repetition()
    columns = _column_support(dem_chunks_to_pcm(chunks), len(model.errors))
    # Syndromes drawn from the model's own mechanisms: a window's residue is a
    # subsystem the window need not explain, so an arbitrary detector subset is
    # not a syndrome any decoder here promises to read.
    for mechanism in range(len(model.errors)):
        events = sorted(columns[mechanism])
        record = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode(events)
        support: set[int] = set()
        for column in record.faults:
            support ^= columns[column]
        assert sorted(support) == events


def test_a_record_projects_its_own_faults_onto_the_observables() -> None:
    model, spec, chunks = _repetition()
    rows = dem_chunks_to_pcm(chunks)
    observables = dem_chunks_to_o_sparse(chunks)
    columns = _column_support(rows, len(model.errors))
    for mechanism in (0, 3, 17, 19):
        record = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode(
            sorted(columns[mechanism])
        )
        flipped: set[int] = set()
        for column in record.faults:
            flipped.symmetric_difference_update(
                observable
                for observable, entry in enumerate(observables)
                if column in entry
            )
        assert record.observables == tuple(sorted(flipped))


def test_a_record_reports_every_window_it_committed() -> None:
    _, spec, _ = _repetition()
    decoder = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME)
    seen: list[tuple[int, bool]] = []
    for layer in range(spec.layout.num_layers):
        record = decoder.decode_round([])
        if record is not None:
            seen.append((record.windows_decoded, record.complete))
    # One record per window, in order, and only the last one is complete.
    assert seen == [
        (index + 1, index + 1 == spec.num_windows) for index in range(spec.num_windows)
    ]
    assert decoder.windows_decoded == 0


def test_a_record_states_convergence_the_inner_decoder_reported() -> None:
    _, spec, _ = _repetition()
    record = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode([])
    # The exchange settles on the empty syndrome, and the flag is a boolean
    # rather than the None an inner decoder states nothing with.
    assert record.converged is True


# ---------------------------------------------------------------------------
# integration: the projections and the decoder over a real memory model
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_the_projections_reproduce_a_memory_models_own_matrices() -> None:
    model, layout = _memory_model(RepetitionCode(distance=3), rounds=REPETITION_ROUNDS)
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    dense_h = model.detector_error_matrix().tolist()
    dense_o = model.observables_flips_matrix().tolist()
    assert dem_chunks_to_pcm(chunks) == tuple(
        tuple(index for index, value in enumerate(row) if value) for row in dense_h
    )
    assert dem_chunks_to_o_sparse(chunks) == tuple(
        tuple(index for index, value in enumerate(row) if value) for row in dense_o
    )
    # The memory map's round width is the code's distance, its round count is the
    # number of detector rounds, and the two together are the detector rows.
    rounds = dem_chunks_to_d_sparse(chunks)
    assert len(rounds) == model.num_detectors == 10
    assert rounds[0] == (0,)
    assert rounds[2] == (0, 2)


@pytest.mark.integration
def test_a_windowed_decode_explains_every_mechanism_of_the_model() -> None:
    model, spec, chunks = _repetition()
    columns = _column_support(dem_chunks_to_pcm(chunks), len(model.errors))
    # A three-round window covers every mechanism of this model's chain, so each
    # mechanism's own syndrome comes back explained rather than abandoned.
    for mechanism in range(len(model.errors)):
        events = sorted(columns[mechanism])
        record = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode(events)
        support: set[int] = set()
        for column in record.faults:
            support ^= columns[column]
        assert sorted(support) == events


@pytest.mark.integration
def test_a_windowed_decode_and_a_whole_model_decode_agree() -> None:
    model, spec, chunks = _repetition()
    columns = _column_support(dem_chunks_to_pcm(chunks), len(model.errors))
    whole = BeliefPropagationDecoder(model)
    for mechanism in range(len(model.errors)):
        events = sorted(columns[mechanism])
        windowed = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME).decode(events)
        assert windowed.observables == tuple(whole.decode(events).observables)


@pytest.mark.integration
def test_the_memory_map_refuses_a_surface_codes_boundary_rounds() -> None:
    _, layout = _memory_model(RotatedSurfaceCode(distance=3), rounds=4)
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    with pytest.raises(ValueError, match="dem_chunks_to_d_sparse"):
        dem_chunks_to_d_sparse(chunks)
    # The two projections the surface code does have are still exact.
    assert len(dem_chunks_to_pcm(chunks)) == layout.num_detectors
