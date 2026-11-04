"""Read a model, inspect rates and graph edges, and close a complete chunk."""

import argparse
import pickle

from flagquantum.qec import (
    ChunkLayout,
    DecodingGraph,
    DemChunk,
    DemChunkSpec,
    DemChunksSpec,
    DetectorErrorModel,
    PhaseId,
    dem_chunk_from_spec,
    dem_chunks_from_spec,
    dem_chunks_to_o_sparse,
    dem_chunks_to_pcm,
    dem_close,
    dem_close_all,
)


def _inspect_single_chunk() -> None:
    model = DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\nerror(0.25) D0 D1\n"
        "detector D0\ndetector D1\nlogical_observable L0\n"
    )
    print(
        "Detectors, observables, errors:",
        model.num_detectors,
        model.num_observables,
        model.num_errors,
    )
    print("Detector rates:", model.detector_rates().tolist())
    print("Observable rates:", model.observable_rates().tolist())
    print("Graph edges:", DecodingGraph.from_detector_error_model(model).num_edges)
    layout = ChunkLayout(model=model, widths=(1, 1))
    chunk = dem_chunk_from_spec(
        DemChunkSpec(layout=layout, first_layer=0, last_layer=1, phase=PhaseId.bulk)
    )
    print("Closed chunk:", dem_close(chunk).to_stim_text() == model.to_stim_text())
    restored = pickle.loads(pickle.dumps(model))
    print("Historical pickle class:", type(restored).__module__)
    print("Same class:", type(restored) is DetectorErrorModel)


def _fixed_chunks() -> tuple[DetectorErrorModel, tuple[DemChunk, ...]]:
    model = DetectorErrorModel.from_stim_text(
        "error(0.125) D0 D1\n"
        "error(0.25) D1 L0\n"
        "error(0.375) D1 D2\n"
        "detector D0\ndetector D1\ndetector D2\nlogical_observable L0\n"
    )
    layout = ChunkLayout(model=model, widths=(1, 1, 1))
    chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))
    return model, chunks


def _round_trip() -> None:
    model, chunks = _fixed_chunks()
    print([chunk.model.num_errors for chunk in chunks])
    print(dem_close_all(chunks) == model)


def _sparse_projection() -> None:
    _, chunks = _fixed_chunks()
    print(dem_chunks_to_pcm(chunks))
    print(dem_chunks_to_o_sparse(chunks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--round-trip", action="store_true")
    modes.add_argument("--sparse-projection", action="store_true")
    args = parser.parse_args()
    if args.sparse_projection:
        _sparse_projection()
    elif args.round_trip:
        _round_trip()
    else:
        _inspect_single_chunk()
