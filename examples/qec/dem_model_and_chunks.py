"""Read a model, inspect rates and graph edges, and close a complete chunk."""

import pickle

from flagquantum.qec import (
    ChunkLayout,
    DecodingGraph,
    DemChunkSpec,
    DetectorErrorModel,
    PhaseId,
    dem_chunk_from_spec,
    dem_close,
)

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
