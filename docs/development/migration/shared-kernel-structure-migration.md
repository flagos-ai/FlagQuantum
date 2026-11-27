# Shared kernel source ownership

Three internal Simulation-root modules move under the existing Kernels owner:
shared catalog admission to `dispatch.py`, complex BMM admission to
`complex_bmm.py`, and compiled/eager complex contraction to `real_imag.py`.
The previous files are removed, all operational consumers use the actual
locations, and no forwarding modules or public API concepts are introduced.
The package initializer remains unchanged and metadata-only.

The implementations match the audited #1084 source algorithm ASTs after import
normalization. They are unchanged in that PR's original base-to-head diff.
Before activation, recheck its latest increment for sources, targets and
absolute/relative/parent-package consumers. No active #1084 implementation is
integrated by this batch.

Keep frozen contracts and raw evidence unchanged. Whole-file relocation entries
resolve their historical source identifiers. Preserve numerical assertions,
cache semantics, opt-in controls, catalog IDs, fallback and rejection behavior.
This kernel batch adds no public compatibility entries; these internal modules
have no frozen public API or public serialized types. The combined follow-up
inventory has 102 entries after the separate experimental ownership closure,
including its two existing unstable experimental facades.

This batch is implemented on `refactor/structure-shared-implementation-owners`,
based on the independently CI-qualified #1332 head `85e8e32d4`. Complete formal
source regressions, clean installed-artifact checks and local quality gates
pass; the pushed commit still requires its own complete CI qualification.

The verified incoming head is
`7ba1a5ae147119e4523a370cc8812108779ad39b`. All 4,389 changed text files
were blob-SHA verified; the actual retired-import guard parsed 3,194 changed
Python trees for these three candidate modules. No incoming source/target
intersection or additional consumer was found. The guard scope is this batch,
not the other deferred #1084 families. All three source ASTs match the incoming
implementations after import normalization, and all three relocated module
ASTs match the parent after the same normalization.

Focused validation: 288 passed / 7 conditional skips in eleven existing
consumer, catalog, layout, cache and metadata-import test files. The existing
eleven quality checks pass, including strict mypy for 1,260 source files,
public API, source/retired-import guards and documentation. No public
compatibility entry is added. Installed-artifact verification is recorded in the
[formal acceptance](shared-implementation-ownership-acceptance.md); complete
branch CI remains required.

The latest incremental consumer audit reaches
`177ed24aba6c27d72b6fcb4b79b60747d8feec9e`: 4,399 changed texts and
3,202 Python import trees. The six added/modified complex-phase files have no
source/target intersection or new consumer of these retired kernel paths.

Activation audit: the next nine-file mean-estimation/complex-phase increment
through `0742c440a100ce56eaec9934f9cf1487a84e0bc5` is SHA-verified.
It has no source/target intersection or new candidate reference; the actual
retired-import guard passes on 3,204 incoming Python trees.
