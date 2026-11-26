# Independent private-binding acceptance

This complete batch follows CI-qualified PR1336 parent
`4f04ec407066f7b93582713d05a867b6cc4f6bee`. It retires nine unused QEC aliases,
fourteen unused Compiler Topology aliases and two internal Runtime Dynamic
aliases, without forwarding or numerical/transform-definition moves. Existing
consumers import authoritative defining modules directly. Three private
production callers use `dem/model.py` and `decoders/repetition.py`, even though
the published compatibility globals remain. The no-reference-allocation
scenario patches the actual signature helper owner rather than an unused alias. The cumulative ledger
has 560 rows, with empty definition lists on retirement-only rows. The existing
102-entry historical facade inventory remains unchanged and is not an exhaustive
inventory of all QEC frozen globals. Thirty-three existing QEC pickle,
namespace and replacement-owner bindings are separately itemized in the
[QEC record](qec-private-binding-retirement.md).

## Formal source and regression evidence

All seventeen changed Python ASTs preserve 113 product definitions and 185 test
assertions after direct imports/owner bindings, the two internal-only Dynamic
export names and ten obsolete alias-test parameters are normalized. All three
parameterized test bodies and their assertions remain unchanged; only cases
requiring the retired private aliases are removed. Remaining actual/public
alias cases stay. Numerical/statevector assertions, public routing evidence and
historical pickle checks are unchanged. The mutable retired-root metadata
inventory now names its existing emission owner directly.

All 11,890 other parent tracked blobs are byte-identical, including all
157 separately counted contract/QEC-fixture blobs. Protected Dynamic and
experimental entry modules are byte-identical. No public contract, frozen
fixture, precision, RNG order, error/refusal policy, source definition identity,
scientific threshold or raw result changes.

Expanded focused preview: **318 passed / 2 existing conditional skips**,
plus thirteen original restoration scenarios. The first formal Unit/API/Team
run found one remaining compatibility-only `_restore_path` parameter; its
failure log is retained. That parameter was retired, with its test body and
public restoration cases unchanged. A complete finite-parameter getattr audit
then found zero remaining consumers of the 25 retired names.
Independent formal regression ran in the actual Git checkout:

- unit-api-team-final: **19,410 passed / 957 skipped**, 395.23 seconds.
- qec-full: **2,801 passed / 329 skipped**, 85.75 seconds.
- root-local-integration: **1,592 passed / 223 skipped**, 89.58 seconds.

Existing optional/hardware policies and the credential-dependent Kaiwu
live-smoke exclusion remain unchanged. Eleven quality gates pass, including
strict mypy over 1262 sources. Both Black 24.1 and 26.10 pass all 3663 Python
files; the operator manifest, runtime schema, legacy-root usage and correctness
certification gates pass. Applicable pinned source hooks and final documentation
checks must pass before committing.

## Native installation and original parent payloads

The expanded native wheel SHA256 is
`a950d5c0a3c3806378e8460997dc6c3d5376959bba76cf4b26ab7d819d5eb21b`.
All 1262 product Python source, wheel and clean-install bytes and file sets
match; the formal source retains those bytes, and the final Git commit must
bind the same blobs. Native CPU and SparseBlossom extensions load from the
new installation. Cold catalog loading imports neither Torch nor JAX; the
installed benchmark CLI succeeds. The macOS arm64 artifact does not certify
other native platforms or distributed scalability; pushed-head CI is required.

The parent and expanded installations each pass 156 topology/dynamic scenarios
with no skips, plus thirteen original restoration scenarios after the final
private-alias parameter is synchronized. The expanded installation also passes 151 unchanged QEC tests
with two existing conditional skips, matching the previously qualified parent
installation. These tests run outside any FlagQuantum product/source checkout;
all loaded product modules resolve to their own installed site. The one
source-tree Dynamic architecture inspection remains in source validation.

Three positive QEC scenario groups and the original missing-Stim refusal
(41 complete state/result entries) match parent JSON bytes. Snapshot SHA256:
`ea059d97cbe608d8631a035cf3c8d4b2ee6fd56b5f4ed9752561042874b7e68e`.
The optional refusal is preserved rather than silently replaced by a fallback.

The immutable real-parent QEC pickle SHA256 is
`3895d532c688a2b3b7977a4489bc1869c29a8b41a5a3de68da842af6c349bc1e`.
It restores 34 function references and four public objects: CssCodeMatrices,
MemoryCircuit, DetectorErrorModel and DemSample. All 42 original QEC binding
values/metadata and object fields/shapes/dtypes/tensor-byte hashes match.
The separate real-parent Compiler/Runtime pickle SHA256 is
`64be276afe5c887b823e06ce518cabffebb265cb2c4c561482523337149a2b28`.
It restores sixteen function/class references and the public RoutingCostEstimate,
including equal fields and the recomputed strategy-selection summary. All 25
retired import attributes are absent; all 33 frozen QEC bindings remain. No new
public compatibility is introduced. Original narrow source/artifact evidence and
initial failed compatibility-only previews remain preserved separately.

## Incoming and completion boundary

The current changed-source audit reaches draft/conflicting PR1084 head
`cc74997c2afc4f5beb393a1160175c71e261b4f9`: 3266 changed incoming Python trees,
not every unchanged branch file. The latest nineteen changed paths are blob-verified
with no candidate-path intersection. The 25 new retired aliases have zero raw
scoped import errors. The qualified 545-row parent ledger has sixteen raw
incoming old-import diagnostics: fifteen prior gated obligations and one
AST-identical Runtime instruction donor already retired by the parent. Joint
integration must carry that donor/import deletion and must never restore
forwarding. The raw diagnostic and its structural classification are retained.

The original 98 unused-private-binding review accounts for 29 Simulation
retirements in the parent, these 25 retirements, 33 frozen QEC globals plus the
Observable payload `_PauliTerm` retained, and ten directly dependent frontend/
Ecosystem aliases gated with the full Construction/IR family. This review is
not an exhaustive architecture completion audit. Construction/IR, directly
PR1084-dependent compiler work, native ABI/development distribution and Rust
retain their closeout gates. This child needs its own CI and remains unmerged.
