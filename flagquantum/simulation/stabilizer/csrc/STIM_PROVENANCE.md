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
