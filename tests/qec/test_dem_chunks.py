"""Coverage for round windows of a detector error model and their seams."""

from __future__ import annotations

from dataclasses import replace

import pytest

from flagquantum.qec.chunks import (
    ChunkLayout,
    DemChunk,
    DemChunkSpec,
    DemChunksSpec,
    DemSeam,
    PhaseId,
    SeamId,
    dem_chunk_from_spec,
    dem_chunks_from_spec,
    dem_close,
    dem_close_all,
    dem_stitch,
    dem_stitch_all,
    dem_stitch_merged,
)
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode, RotatedSurfaceCode
from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


# A four-layer model whose mechanisms span one or two adjacent layers, which is
# the shape every memory circuit's model has: a detector layer of two rows, and a
# fault that flips a pair of detectors inside a layer or across a boundary.
WIDTHS = (2, 2, 2, 2)
SPANS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))


def _segmented_model() -> DetectorErrorModel:
    """A layered model whose mechanisms span one or two adjacent layers."""

    return DetectorErrorModel(
        num_detectors=8,
        num_observables=1,
        errors=tuple(
            DemError(probability=0.01 * (index + 1), detectors=detectors)
            for index, detectors in enumerate(SPANS)
        ),
    )


def _layout() -> ChunkLayout:
    return ChunkLayout(model=_segmented_model(), widths=WIDTHS)


def _chunk(
    *,
    first_row: int,
    num_detectors: int,
    seams: tuple[DemSeam, ...] = (),
    first_layer: int,
    last_layer: int,
    detectors: tuple[tuple[int, ...], ...] = ((0, 1),),
    num_observables: int = 1,
) -> DemChunk:
    return DemChunk(
        model=DetectorErrorModel(
            num_detectors=num_detectors,
            num_observables=num_observables,
            errors=tuple(
                DemError(probability=0.01, detectors=item) for item in detectors
            ),
        ),
        first_row=first_row,
        seams=seams,
        first_layer=first_layer,
        last_layer=last_layer,
    )


def test_a_seam_name_is_the_identity() -> None:
    assert SeamId("round_1") == SeamId("round_1")
    assert SeamId("round_1") != SeamId("round_2")
    assert str(SeamId.prev_round) == "prev_round"
    assert str(SeamId.next_round) == "next_round"


def test_a_seam_name_must_be_a_non_empty_stripped_string() -> None:
    with pytest.raises(TypeError, match="seam name must be a string"):
        SeamId(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seam name must not be empty"):
        SeamId("   ")
    with pytest.raises(ValueError, match="must not carry surrounding whitespace"):
        SeamId(" prev_round")


def test_the_phase_labels_are_init_bulk_and_final() -> None:
    assert (str(PhaseId.init), str(PhaseId.bulk), str(PhaseId.final)) == (
        "init",
        "bulk",
        "final",
    )
    with pytest.raises(TypeError, match="phase name must be a string"):
        PhaseId(0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="phase name must not be empty"):
        PhaseId("")


def test_a_seam_band_must_hold_at_least_one_row() -> None:
    seam = DemSeam(SeamId.next_round, 2, 5)
    assert seam.width == 3
    assert list(seam.rows) == [2, 3, 4]
    with pytest.raises(TypeError, match="identified by a SeamId"):
        DemSeam("next_round", 2, 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must hold at least one row"):
        DemSeam(SeamId.next_round, 3, 3)
    with pytest.raises(TypeError, match="row_begin must be an integer"):
        DemSeam(SeamId.next_round, 0.5, 3)  # type: ignore[arg-type]


def test_layer_widths_must_cover_every_detector_once() -> None:
    model = _segmented_model()
    with pytest.raises(ValueError, match="layer widths sum to 6 rows"):
        ChunkLayout(model=model, widths=(2, 2, 2))
    with pytest.raises(ValueError, match="must hold at least one row"):
        ChunkLayout(model=model, widths=(2, 0, 2, 4))
    with pytest.raises(ValueError, match="requires at least one detector layer"):
        ChunkLayout(model=model, widths=())
    with pytest.raises(TypeError, match="must describe a DetectorErrorModel"):
        ChunkLayout(model="model", widths=WIDTHS)  # type: ignore[arg-type]


def test_layer_numbers_are_dense_from_zero() -> None:
    # The widths carry the offsets, so a layout cannot state a gap; the gap is
    # refused where it is read, from the detector rounds of a circuit.
    layout = ChunkLayout(model=_segmented_model(), widths=WIDTHS)
    assert layout.num_layers == 4
    assert [layout.width_of(index) for index in range(4)] == [2, 2, 2, 2]
    assert [layout.first_row_of(index) for index in range(4)] == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="outside a layout of 4 layers"):
        layout.width_of(4)
    with pytest.raises(ValueError, match="outside a layout of 4 layers"):
        layout.first_row_of(4)


def test_a_row_belongs_to_the_layer_its_width_places_it_in() -> None:
    layout = _layout()
    assert [layout.layer_of(row) for row in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError, match="outside the model's 8 detectors"):
        layout.layer_of(8)
    with pytest.raises(TypeError, match="detector row must be an integer"):
        layout.layer_of(True)


def test_a_layer_range_covers_the_rows_of_its_layers() -> None:
    layout = _layout()
    assert list(layout.rows_for(1, 2)) == [2, 3, 4, 5]
    assert list(layout.rows_for(0, 3)) == list(range(8))
    with pytest.raises(ValueError, match="must not precede its first"):
        layout.rows_for(2, 1)
    with pytest.raises(ValueError, match="outside a layout of 4 layers"):
        layout.rows_for(0, 4)


@pytest.mark.integration
def test_the_layers_come_from_the_detector_rounds_of_the_circuit() -> None:
    circuit = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel(
        num_detectors=len(circuit.detectors), num_observables=1, errors=()
    )
    layout = ChunkLayout.from_memory_circuit(model, circuit)
    # Three syndrome rounds of two checks, then the terminal readout layer.
    assert layout.widths == (2, 2, 2, 2)


@pytest.mark.integration
def test_a_boundary_of_the_circuit_is_one_readout_layer_wide() -> None:
    circuit = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=4)
    model = DetectorErrorModel(
        num_detectors=len(circuit.detectors), num_observables=1, errors=()
    )
    layout = ChunkLayout.from_memory_circuit(model, circuit)
    # Four X-type checks in round zero and four Z-type in the readout layer,
    # with eight in every round between them.
    assert layout.widths == (4, 8, 8, 8, 4)


@pytest.mark.integration
def test_a_layout_must_agree_with_the_circuit_it_reads() -> None:
    circuit = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel(
        num_detectors=len(circuit.detectors) + 1, num_observables=1, errors=()
    )
    with pytest.raises(ValueError, match="declares 8 detectors and the model states 9"):
        ChunkLayout.from_memory_circuit(model, circuit)
    with pytest.raises(TypeError, match="must describe a MemoryCircuit"):
        ChunkLayout.from_memory_circuit(_segmented_model(), "circuit")  # type: ignore[arg-type]


def test_a_cut_window_holds_the_mechanisms_inside_it() -> None:
    layout = _layout()
    chunk = dem_chunk_from_spec(
        DemChunkSpec(layout=layout, first_layer=1, last_layer=2, phase=PhaseId.bulk)
    )
    assert chunk.model.num_detectors == 4
    assert list(chunk.rows) == [2, 3, 4, 5]
    # Every mechanism whose detectors lie inside layers one and two, re-based.
    assert chunk.model.errors == (
        DemError(probability=0.03, detectors=(0, 1)),
        DemError(probability=0.04, detectors=(1, 2)),
        DemError(probability=0.05, detectors=(2, 3)),
    )


def test_a_single_cut_is_not_a_partition() -> None:
    spec = DemChunksSpec(layout=_layout(), window=2)
    middle = spec.chunk_specs()[1]
    # The middle window's own cut holds every mechanism wholly inside it, and two
    # of those are inside its neighbour as well, so the three cuts hold nine
    # mechanisms where the model states seven: a cut is not a partition.
    assert len(dem_chunk_from_spec(middle).model.errors) == 3
    cut = sum(
        len(dem_chunk_from_spec(item).model.errors) for item in spec.chunk_specs()
    )
    assert cut == 9
    assert len(dem_chunks_from_spec(spec)[1].model.errors) == 2


def test_a_window_names_only_the_boundaries_it_has_a_neighbour_for() -> None:
    layout = _layout()
    first, middle, last = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    assert [seam.seam_id for seam in first.seams] == [SeamId.next_round]
    assert [seam.seam_id for seam in middle.seams] == [
        SeamId.prev_round,
        SeamId.next_round,
    ]
    assert [seam.seam_id for seam in last.seams] == [SeamId.prev_round]
    assert first.get_seam(SeamId.next_round) == DemSeam(SeamId.next_round, 2, 4)
    assert middle.get_seam(SeamId.prev_round) == DemSeam(SeamId.prev_round, 0, 2)


def test_a_boundary_band_is_the_detector_layer_it_shares() -> None:
    _, middle, _ = dem_chunks_from_spec(DemChunksSpec(layout=_layout(), window=2))
    assert middle.seam_rows(SeamId.prev_round) == (2, 3)
    assert middle.seam_rows(SeamId.next_round) == (4, 5)
    assert middle.interior_rows() == ()
    with pytest.raises(KeyError, match="carries no prev_round seam"):
        dem_chunk_from_spec(
            DemChunkSpec(
                layout=_layout(), first_layer=0, last_layer=1, phase=PhaseId.init
            )
        ).get_seam(SeamId.prev_round)


def test_the_interior_rows_are_the_rows_no_seam_covers() -> None:
    first, middle, last = dem_chunks_from_spec(
        DemChunksSpec(layout=_layout(), window=2)
    )
    assert first.interior_rows() == (0, 1)
    assert middle.interior_rows() == ()
    assert last.interior_rows() == (2, 3)


def test_a_chunk_must_name_one_boundary_band_per_standard_seam() -> None:
    with pytest.raises(TypeError, match="chunk seams must be DemSeam records"):
        _chunk(
            first_row=0,
            num_detectors=4,
            seams=("next_round",),  # type: ignore[arg-type]
            first_layer=0,
            last_layer=1,
        )
    with pytest.raises(ValueError, match="may name a seam id at most once"):
        _chunk(
            first_row=0,
            num_detectors=4,
            seams=(
                DemSeam(SeamId.next_round, 2, 4),
                DemSeam(SeamId.next_round, 2, 4),
            ),
            first_layer=0,
            last_layer=1,
        )
    with pytest.raises(ValueError, match="must be ordered by their first row"):
        _chunk(
            first_row=2,
            num_detectors=4,
            seams=(
                DemSeam(SeamId.next_round, 2, 4),
                DemSeam(SeamId.prev_round, 0, 2),
            ),
            first_layer=1,
            last_layer=2,
        )
    with pytest.raises(ValueError, match="must begin at local row zero"):
        _chunk(
            first_row=2,
            num_detectors=4,
            seams=(DemSeam(SeamId.prev_round, 1, 2),),
            first_layer=1,
            last_layer=2,
        )
    with pytest.raises(ValueError, match="must end at the chunk's last local row"):
        _chunk(
            first_row=0,
            num_detectors=4,
            seams=(DemSeam(SeamId.next_round, 1, 3),),
            first_layer=0,
            last_layer=1,
        )
    with pytest.raises(ValueError, match="two seam bands must not overlap"):
        _chunk(
            first_row=0,
            num_detectors=4,
            seams=(
                DemSeam(SeamId.prev_round, 0, 3),
                DemSeam(SeamId.next_round, 2, 4),
            ),
            first_layer=0,
            last_layer=1,
        )


def test_a_seam_that_is_not_a_window_boundary_is_refused() -> None:
    with pytest.raises(ValueError, match="is not a standard boundary"):
        _chunk(
            first_row=0,
            num_detectors=4,
            seams=(DemSeam(SeamId("seam_7"), 2, 4),),
            first_layer=0,
            last_layer=1,
        )


def test_a_chunk_must_hold_a_model_and_an_ordered_layer_range() -> None:
    with pytest.raises(TypeError, match="must hold a DetectorErrorModel"):
        DemChunk(
            model="model",  # type: ignore[arg-type]
            first_row=0,
            seams=(),
            first_layer=0,
            last_layer=1,
        )
    with pytest.raises(ValueError, match="must not precede its first"):
        _chunk(first_row=0, num_detectors=4, first_layer=2, last_layer=1)
    with pytest.raises(ValueError, match="outside the chunk's 4 rows"):
        _chunk(
            first_row=0,
            num_detectors=4,
            seams=(DemSeam(SeamId.next_round, 2, 5),),
            first_layer=0,
            last_layer=1,
        )


def test_a_chunk_spec_must_cut_a_layout_inside_its_layers() -> None:
    layout = _layout()
    with pytest.raises(TypeError, match="must cut a ChunkLayout"):
        DemChunkSpec(layout="layout", first_layer=0, last_layer=1, phase=PhaseId.bulk)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="labelled by a PhaseId"):
        DemChunkSpec(layout=layout, first_layer=0, last_layer=1, phase="bulk")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside a layout of 4 layers"):
        DemChunkSpec(layout=layout, first_layer=0, last_layer=4, phase=PhaseId.bulk)
    with pytest.raises(TypeError, match="dem_chunk_from_spec requires a DemChunkSpec"):
        dem_chunk_from_spec("spec")  # type: ignore[arg-type]


def test_the_windows_partition_the_model() -> None:
    layout = _layout()
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    assert len(chunks) == 3
    assert dem_close_all(chunks) == layout.model


def test_a_mechanism_on_a_shared_boundary_lands_in_one_window() -> None:
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=_layout(), window=2))
    # Each window claims a mechanism only where its layer range covers every
    # detector of it, and the first such window wins, so a fault stated once is
    # held once.
    assert [len(chunk.model.errors) for chunk in chunks] == [3, 2, 2]
    assert sum(len(chunk.model.errors) for chunk in chunks) == len(SPANS)


def test_a_stitched_chain_closes_back_to_the_model() -> None:
    layout = _layout()
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    stitched = dem_stitch_all(chunks)
    assert stitched.model.num_detectors == 8
    assert stitched.seams == ()
    assert dem_close(stitched) == layout.model


def test_closing_all_at_once_agrees_with_stitching_first() -> None:
    layout = _layout()
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    assert dem_close_all(chunks) == dem_close(dem_stitch_all(chunks))


def test_closing_a_window_keeps_the_detector_numbering_it_was_cut_at() -> None:
    _, middle, _ = dem_chunks_from_spec(DemChunksSpec(layout=_layout(), window=2))
    closed = dem_close(middle)
    # One past the window's last row rather than the model's own count: the
    # indices are the model's, the extent is the window's.
    assert closed.num_detectors == 6
    assert closed.errors == (
        DemError(probability=0.04, detectors=(3, 4)),
        DemError(probability=0.05, detectors=(4, 5)),
    )
    with pytest.raises(TypeError, match="dem_close requires a DemChunk"):
        dem_close("chunk")  # type: ignore[arg-type]


def test_a_stitch_refuses_windows_that_are_not_adjacent() -> None:
    first, middle, last = dem_chunks_from_spec(
        DemChunksSpec(layout=_layout(), window=2)
    )
    assert dem_stitch(first, middle).last_layer == 2
    with pytest.raises(ValueError, match="cannot contract chunks spanning layers"):
        dem_stitch(first, last)


def test_a_stitch_refuses_an_observable_count_that_disagrees() -> None:
    left = _chunk(
        first_row=0,
        num_detectors=4,
        seams=(DemSeam(SeamId.next_round, 2, 4),),
        first_layer=0,
        last_layer=1,
    )
    right = _chunk(
        first_row=4,
        num_detectors=4,
        seams=(DemSeam(SeamId.prev_round, 0, 2),),
        first_layer=1,
        last_layer=2,
        num_observables=2,
    )
    with pytest.raises(ValueError, match="cannot contract chunks over 1 and 2"):
        dem_stitch(left, right)


def test_a_stitch_refuses_a_boundary_a_chunk_does_not_carry() -> None:
    first, middle, last = dem_chunks_from_spec(
        DemChunksSpec(layout=_layout(), window=2)
    )
    with pytest.raises(ValueError, match="the left chunk carries no next_round seam"):
        dem_stitch(last, middle)
    with pytest.raises(ValueError, match="the right chunk carries no prev_round seam"):
        dem_stitch(middle, first)


def test_a_stitch_keeps_the_outer_boundaries_of_both_sides() -> None:
    first, middle, last = dem_chunks_from_spec(
        DemChunksSpec(layout=_layout(), window=2)
    )
    # The shared boundary is laid out once, between the two interiors, and the
    # seams the contraction did not consume are the result's own.
    stitched = dem_stitch(first, middle)
    assert stitched.model.num_detectors == 6
    assert stitched.first_layer == 0
    assert stitched.last_layer == 2
    assert stitched.seams == (DemSeam(SeamId.next_round, 4, 6),)
    assert dem_stitch(stitched, last).seams == ()


def test_a_stitch_refuses_two_boundaries_of_one_width_that_differ() -> None:
    left = _chunk(
        first_row=0,
        num_detectors=4,
        seams=(DemSeam(SeamId.next_round, 2, 4),),
        first_layer=0,
        last_layer=1,
    )
    shifted = _chunk(
        first_row=5,
        num_detectors=4,
        seams=(DemSeam(SeamId.prev_round, 0, 2),),
        first_layer=1,
        last_layer=2,
    )
    with pytest.raises(ValueError, match="is not the same detector row band"):
        dem_stitch(left, shifted)


def test_a_stitch_needs_two_chunks_and_a_close_needs_a_non_empty_sequence() -> None:
    windows = dem_chunks_from_spec(DemChunksSpec(layout=_layout(), window=2))
    with pytest.raises(TypeError, match="the left side of a stitch must be a DemChunk"):
        dem_stitch("left", windows[0])  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="the right side of a stitch must be a DemChunk"
    ):
        dem_stitch(windows[0], "right")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="a stitch needs at least one chunk"):
        dem_stitch_all(())
    with pytest.raises(ValueError, match="a close needs at least one chunk"):
        dem_close_all(())
    with pytest.raises(TypeError, match="every element of a close must be a DemChunk"):
        dem_close_all(("chunk",))  # type: ignore[arg-type]


def test_a_stitch_of_one_window_is_that_window() -> None:
    single = dem_chunk_from_spec(
        DemChunkSpec(layout=_layout(), first_layer=0, last_layer=3, phase=PhaseId.bulk)
    )
    assert dem_stitch_all((single,)) == single
    assert dem_close_all((single,)) == _segmented_model()


def test_a_close_refuses_a_sequence_that_is_not_one_model() -> None:
    first, middle, last = dem_chunks_from_spec(
        DemChunksSpec(layout=_layout(), window=2)
    )
    with pytest.raises(ValueError, match="do not share exactly one boundary layer"):
        dem_close_all((first, last))
    without_a_boundary = replace(middle, seams=(DemSeam(SeamId.next_round, 2, 4),))
    with pytest.raises(
        ValueError, match="every chunk but the first needs a prev_round"
    ):
        dem_close_all((first, without_a_boundary))
    mismatched = replace(middle, first_row=4)
    with pytest.raises(ValueError, match="do not agree on the detectors of their"):
        dem_close_all((first, mismatched))


def test_a_close_refuses_windows_that_disagree_on_their_observables() -> None:
    first, middle, _ = dem_chunks_from_spec(DemChunksSpec(layout=_layout(), window=2))
    wider = replace(
        middle,
        model=replace(middle.model, num_observables=2),
    )
    with pytest.raises(ValueError, match="must agree on its observable count"):
        dem_close_all((first, wider))


def test_a_merged_stitch_gives_each_signature_one_mechanism() -> None:
    duplicated = DetectorErrorModel(
        num_detectors=8,
        num_observables=1,
        errors=(
            DemError(probability=0.1, detectors=(0, 1)),
            DemError(probability=0.2, detectors=(0, 1)),
            DemError(probability=0.3, detectors=(6, 7)),
        ),
    )
    chunks = dem_chunks_from_spec(
        DemChunksSpec(layout=ChunkLayout(model=duplicated, widths=WIDTHS), window=2)
    )
    assert len(dem_close_all(chunks).errors) == 3
    merged = dem_stitch_merged(chunks)
    assert len(merged.model.errors) == 2
    assert merged.model.errors[0].probability == pytest.approx(
        0.1 + 0.2 - 2 * 0.1 * 0.2
    )


def test_a_window_must_span_two_layers_and_tile_the_whole_layout() -> None:
    layout = _layout()
    with pytest.raises(ValueError, match="must span at least two detector layers"):
        DemChunksSpec(layout=layout, window=1)
    with pytest.raises(ValueError, match="does not fit a layout of 4 layers"):
        DemChunksSpec(layout=layout, window=5)
    with pytest.raises(ValueError, match="does not tile into windows of 3"):
        DemChunksSpec(layout=layout, window=3)
    with pytest.raises(TypeError, match="must cut a ChunkLayout"):
        DemChunksSpec(layout="layout", window=2)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="dem_chunks_from_spec requires a DemChunksSpec"
    ):
        dem_chunks_from_spec("spec")  # type: ignore[arg-type]


def test_the_phase_sequence_is_init_then_bulk_then_final() -> None:
    layout = _layout()
    spec = DemChunksSpec(layout=layout, window=2)
    assert spec.stride == 1
    assert spec.num_windows == 3
    assert spec.layer_ranges == ((0, 1), (1, 2), (2, 3))
    assert spec.phase_sequence() == (PhaseId.init, PhaseId.bulk, PhaseId.final)
    assert [item.phase for item in spec.chunk_specs()] == list(spec.phase_sequence())
    assert [item.num_layers for item in spec.chunk_specs()] == [2, 2, 2]


def test_one_window_is_labelled_bulk_rather_than_both_ends() -> None:
    spec = DemChunksSpec(layout=_layout(), window=4)
    assert spec.num_windows == 1
    assert spec.phase_sequence() == (PhaseId.bulk,)


def test_a_stated_phase_sequence_must_match_the_windows() -> None:
    layout = _layout()
    with pytest.raises(ValueError, match="2 window labels were stated for 3 windows"):
        DemChunksSpec(layout=layout, window=2, phases=(PhaseId.init, PhaseId.final))
    with pytest.raises(TypeError, match="window labels must be PhaseId records"):
        DemChunksSpec(layout=layout, window=2, phases=("init",))  # type: ignore[arg-type]
    named = DemChunksSpec(
        layout=layout,
        window=2,
        phases=(PhaseId("prep"), PhaseId("body"), PhaseId("readout")),
    )
    assert named.phase_sequence()[0] == PhaseId("prep")


def test_a_mechanism_no_window_holds_whole_is_refused() -> None:
    # Layers zero and three at once: no two-layer window covers it.
    straddling = DetectorErrorModel(
        num_detectors=8,
        num_observables=1,
        errors=(DemError(probability=0.01, detectors=(0, 7)),),
    )
    with pytest.raises(ValueError, match="no window of 2 layers covers"):
        dem_chunks_from_spec(
            DemChunksSpec(layout=ChunkLayout(model=straddling, widths=WIDTHS), window=2)
        )


def test_a_mechanism_that_flips_no_detector_is_refused() -> None:
    observable_only = DetectorErrorModel(
        num_detectors=8,
        num_observables=1,
        errors=(DemError(probability=0.01, detectors=(), observables=(0,)),),
    )
    layout = ChunkLayout(model=observable_only, widths=WIDTHS)
    with pytest.raises(ValueError, match="flips no detector, so no window can hold it"):
        dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    with pytest.raises(ValueError, match="flips no detector, so no window can hold it"):
        dem_chunk_from_spec(
            DemChunkSpec(layout=layout, first_layer=0, last_layer=1, phase=PhaseId.init)
        )
