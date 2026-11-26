# Dense measurement numerical ownership

Implemented on parent PR1334 head `10ba61639d984eda5250db6c130ae13008fc3175`,
after its applicable workflows qualified. This batch has its own formal source,
artifact and pushed-head CI requirements.

The private dense readout adapter, shared probability draw and instrument
branch-weight function live in the existing Simulation density-matrix module.
Four shared raw-tensor probability helpers live in the existing Simulation
Numerics package's `probabilities.py`: extraction, ordered joint reduction,
normalization and parity reconstruction. The original `_TRACE_TOLERANCE = 1e-4`
moves with those kernels, unchanged. No new package, public interface,
compatibility facade or contract type is introduced. The existing Dynamics
physical-state validator and Core SuperOperator remain authoritative.

Runtime retains request and marginal-size admission, backend/catalog selection,
private generator creation and seeds, result records, branch metadata,
capability refusals and evidence. Draw order, device/dtype, physical validation,
tolerances, error classes and branch ordering are unchanged. All four production
consumers and seven direct/private test consumers use canonical Simulation
owners. Eight old private Runtime bindings are removed without forwarding;
patches follow the same sampling authority, preserving all 198 assertions.

The seven moved definition ASTs and 15 existing destination definitions are
unchanged. All 46 remaining Runtime definitions are unchanged after normalizing
the direct-owner bindings. The ledger has 538 rows; the public-entry inventory
remains 102 with no additions.

The incoming audit reaches PR1084
`e92bf44bb5470535996a824e42d25bac6f871327`. The latest increment does not
intersect these 19 candidate paths; the seven definitions retain their original
ASTs across the incoming base and head after prior import normalization. There
are no destination definition collisions or scoped retired imports over 3246
changed incoming Python trees. A separate cumulative audit retains 15 gated
incoming old-import obligations. This independent batch does not close those
obligations, Construction/IR or the native/development distribution gate.

See [the acceptance record](dense-measurement-numerics-acceptance.md) for formal
regressions, artifact equivalence and completion boundaries.
