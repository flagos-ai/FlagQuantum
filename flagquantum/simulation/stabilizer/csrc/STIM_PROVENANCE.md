# Embedded Stim source provenance

The measurement sampler and explicit-record converter embed unchanged Stim 1.16.0 files from the PyPI
source distribution. The source lock records the archive digests, upstream
commit, translation units, headers and Apache-2.0 license. The private
CPython bridge uses upstream reference sampling, frame sampling, RNG
construction and the b8 writer. The Stim Python extension is not linked
or imported by the embedded measurement route.

The first local embedded build profile targets macOS arm64 and is selected
with `FQ_BUILD_EMBEDDED_STIM=1`. The generated wheel marker selects the
private extension. An embedded profile with a missing or broken extension
fails at preparation. The standard build retains its existing optional
Stim SDK behavior.

The lock now contains 96 unchanged upstream files: the original 78 plus 18
conversion and determinism-analysis files. The bridge directly calls upstream
`measurements_to_detection_events_helper`, reference sampling and the error
analyzer with gauge detectors disallowed. The internal DEM representation is
needed by that check; it adds no public DEM API. No upstream file is patched.
The Python bridge uses CPython's limited API and owned byte buffers; no NumPy,
pybind11 or installed Stim distribution is needed by the embedded profile.

The initial 94-file candidate missed the same-directory include of
`sparse_rev_frame_tracker.h` and its companion translation unit. The explicitly
approved two-file expansion supplies both, unchanged from the pinned archives.
The source lock lists 37 translation units. The failed initial build is retained
alongside the successful expanded build; no determinism check was removed.

## Detector execution extension

The private bridge also calls `sample_batch_detection_events` and
`ErrorAnalyzer::circuit_to_detector_error_model` from the same locked 96 files.
No upstream bytes, translation-unit list, compiler flags or dependencies changed.
The detector producer rejects gauge parities on the noiseless circuit. Exact DEM
analysis separately disables decomposition, gauge acceptance, disjoint
approximation and ignored decomposition failures. Frame scratch is recreated
on each sample call; no performance advantage is asserted.


## Detector-file bridge candidate

The private `write_detectors` entry calls the already vendored
`sample_batch_detection_events_writing_results_to_disk` with the existing handle's
circuit and RNG. Six file formats and optional same-shot observable output use
Stim's writers. All 96 upstream files, their lock and license bytes are unchanged.
Python owns validation, staging and publication; C++ owns numerical sampling,
file writes and checked close. No Python dense table or new translation unit is
introduced. Build and no-SDK installed acceptance remain required before delivery.

## Measurement-to-detector file conversion

Private `convert_file` reuses the existing prepared Converter circuit, stats and
reference with `stream_measurements_to_detection_events_helper<width>`. This is
the pinned 1024-record upstream reader/converter/writer, not a Python dense loop.
Both output orientations are included in the pre-allocation estimate. Selected
PTB64 output refuses before opening outputs, matching the upstream single-record
writer restriction. The native bridge checks read/write/close errors; Python
owns immutable snapshots, framing checks and no-replace publication.

All 96 upstream files and 37 translation units remain byte-identical to Stim
1.16.0 archive SHA-256
`9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52`,
commit `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`. No source closure,
compiler flag, dependency, license or setup.py change is made by this package.

## Shot-file reader candidate

The private reader cursor calls MeasureRecordReader<width>::make and
start_and_read_entire_record from the current 96-file source closure. Counts and
validation use one complete pass before a fresh cursor delivers owned batches.
PTB64 retains the upstream 64-shot buffer across consumer batches. No pybind11,
NumPy, upstream source edits, dependency changes or extra translation units are
introduced. read_write.pybind.cc is a semantic reference, not a compiled input.
Native and installed acceptance remain required before delivery.
