# Embedded Stim source provenance

The measurement sampler embeds unchanged Stim 1.16.0 files from the PyPI
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
