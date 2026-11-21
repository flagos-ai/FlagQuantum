# Component UI export workflow

## Complete source scope

The source is the complete six-cell `dev_tools/ui-export.ipynb` at Qualtran
v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`19959bb58f2218499ca2b927802ef79fa94cd88d`. It contains a modular-exponentiation
score demonstration, scalar encoder, attribute/hash helpers, navigation export,
class documentation export and complete declared-graph node export. The frozen
notebook is retained unchanged outside the checkout. Adapted workflow code
retains Copyright 2023 Google LLC and Apache-2.0 attribution.

`tools/development/component_ui_export.py` owns only output orchestration and
source-compatible scalar/hash conventions. Existing `NotebookSpec`,
`ComponentDocumentation` and `ComponentExample` supply the catalog. Compiler's
existing declared-graph traversal supplies adjacency. Domain providers supply
component names, metadata, wrapper identity and optional diagrams. Existing
Drawer musical-score dictionaries are JSON-ready. Existing documentation
parsing supplies Markdown. These boundaries avoid a foreign component base
class, duplicated graph traversal or implicit numerical decomposition.

## Complete cell mapping

| Source cell | Native owner and disposition |
| --- | --- |
| 0 | Existing modular-exponentiation Program builder and Drawer layout/rendering |
| 1 | `ComponentUIEncoder` with existing Drawer dictionaries |
| 2 | Domain metadata callback and sorted-parameter `component_filename` |
| 3 | `export_component_ui` navigation over all supplied specifications |
| 4 | Existing class docstring parser and create-if-absent docs |
| 5 | Existing Compiler graph plus all-node record export and optional diagrams |

Run `python -m examples.algorithms.component_ui_export` from the checkout root.
It exports one native WordSwap and its complete declared CX dependency closure
to `component_ui_export/`. Three JSON files and one class documentation file
are expected. `examples/notebooks/component_ui_export.ipynb` adds the original
2048-bit logical modular-exponentiation drawing demonstration using existing
native owners. It exports the same explicit WordSwap catalog and inspects its
navigation, documentation, attributes, score and dependency references.

The exporter emits no quantum instructions. The example independently observes
six physical CX instructions after binding and obtains a six-CX model estimate
from its declared graph. JSON adjacency intentionally omits edge weights, as
the source UI does. Use existing Compiler resource queries for model counts.
The WordSwap has no measurements or auxiliary storage. Its existing bound
operation continues through ordinary `circuit.append(op)`.

## Source defects and explicit adaptations

- Cell 0 imports `qualtran.bloqs.factoring.rsa`, absent in the actual frozen
  source. Its actual class is in `cryptography.rsa.rsa_mod_exp`. The original
  failure is tested. Complete source-cell execution uses an explicit test-only
  import route and records that correction, rather than claiming the untouched
  original runs directly.
- Source filenames hash sorted attribute pairs with non-security MD5. Generic
  adjoint/control prefixes and recursive class names belong to the component
  domain. The native helper accepts the explicit prefix and metadata. It does
  not inspect arbitrary objects to invent decomposition or identity semantics.
- The source unwraps controlled attributes without encoding control polarity.
  Distinct polarities can therefore share a filename. This exact source collision
  is tested and remains an output-format boundary, not a canonical IR identity.
- Original composite/programmed-rotation metadata omissions and unknown-value
  `NOT_SERIALIZABLE` encodings are retained by source-domain providers in the
  reference comparison. Native domain providers explicitly select metadata.
- Symbol spelling, complex values such as `1.0+-2.0i`, three side spellings and
  unsupported scalar sentinels match the source. Native Drawer records are
  dictionaries, so no runtime dependency on a foreign score encoder is added.
- Navigation is refreshed. Existing docs and node records are retained, including
  stale or manually edited content. Source duplicate section/title keys and
  filename collisions retain normal dictionary/first-file behavior.
- A diagram provider's ordinary exception becomes a null diagram. Cancellation
  propagates. The original bare handler also swallows process cancellation.
  Metadata, encoding and filesystem errors propagate. Publication is sequential,
  so a filesystem failure can leave earlier files or a partial current file.
  Native node serialization completes before opening the node destination.
  Unlike the source streaming JSON dump, an encoding failure leaves no new node
  file. This explicit publication adaptation preserves successful output bytes.
- Inputs are trusted internal component catalog metadata. This is a developer
  export tool, not an endpoint accepting untrusted filesystem names.

## Acceptance boundary and remaining native inventory

Q-0080 already owns the complete 110-notebook publication catalog and its 237
source-document mapping. That mapping is reused. It maps authored native assets
and imported owners, not all original example instances to executable native
metadata providers. The generic UI exporter accepts a complete supplied catalog.
Its full reference-catalog test checks the original/native graph and all emitted
files with two supplied diagram profiles. One supplies deterministic test data.
The second replays authenticated actual source diagrams from the completed
source leg. The original node loop, metadata, hashes, navigation and documentation
remain the reference. Fresh all-node diagram construction has a separate bounded
external diagnostic.
The real native runnable example checks the complete supplied WordSwap closure.
These two scopes are distinct.

Q-0120 remains open until every native catalog document has explicit example
factories, parameter/identity providers, score availability and complete output
navigation/dependency coverage. A three-file runnable example does not close
that whole-library inventory. Existing source display, declared-model and native
physical execution evidence is reused where applicable.

## Verification

The concentrated regression passed 108 cases with zero skips and 94 visible
source/parser warnings. Strict typing passed for all three changed Python files.
Black, Ruff, Integration scope, architecture, IR, API and diff checks passed.
The supplied-diagram full-catalog profile produced 3,317 byte-identical files
per exporter. Complete hash/length manifests were retained externally before
removing only the two finished runs' regenerable output copies, reclaiming
about 265 MiB of payload plus filesystem overhead.

The separate real-diagram diagnostic reached its 900-second bound. The source
leg completed 3,317 files. The native leg had 1598 files, and all 1598 common files
were byte-identical. 1719 native files remained absent. This is incomplete evidence,
not a full real-diagram pass. The child and parent processes exited. The
complete diagnostic snapshot, partial-file manifest and process sample remain
in the external reference environment. The original source is untouched.
Do not repeat this full run before isolating the expensive score/decomposition
provider and reusing already generated diagram evidence.

Tests cover
all original cells, including the explicit obsolete-import correction, all
reference-catalog output files, original encoder/hash/control collision,
real native notebook, exact entangled complex append, observed/model counts,
retained-file behavior, diagram failures, cancellation and metadata failures.

## Authenticated actual diagram replay

The completed source leg contains 3,083 unique node JSON records after the
source filename collisions. Class documentation and navigation bring the total
to 3,317 files. The deterministic gzip fixture
`tests/fixtures/component_ui_source_diagrams.jsonl.gz` retains each node path and
its actual diagram dictionary or null. Its adjacent manifest records the frozen
commit, notebook blob, copyright, license, extraction and replay scope. The test
hardcodes SHA256
`d29f2e28abd70f0297c83ea7f3922576780dd9fbdd52ec94b48ba65724b70c58`
and verifies all fixture keys against the actual complete source graph. Only
diagram fields come from this fixture. Metadata, class documentation, navigation,
filenames and adjacency are recomputed from frozen source.

Both profiles produce all 3,317 files with exact source/native byte equality.
The real-diagram replay has 98,815,922 payload bytes per exporter. The supplied
synthetic-diagram profile has 69,467,868 bytes per exporter. This closes exporter
comparison for recorded real diagrams. Fresh full-native diagram execution and
237-reference/234-class/359-example native provider coverage remain open.
The fixture is 837,499 bytes and avoids repeating the 900-second decomposition
run. External manifests retain full output hashes before completed temporary
output copies are reclaimed.

## Native basic-gate instance providers

`examples/algorithms/basic_gate_ui_export.py` composes nine complete native
notebook entries, fourteen documentation groups and all sixteen corresponding
frozen example factories. The entries cover T, Toffoli, H and controlled H,
CNOT, Z/S/CZ, Y/CY, both measurement bases, global phase and identity. Multiple
operations share `PhaseCall`, `CliffordCall` or `ZeroCostCall`. Explicit tool-layer
documentation callbacks give them individual names and owned documentation.
The existing class-documentation defaults remain available to other exporters.

The provider uses existing resource identities, register signatures,
`ComponentProgramBuilder`, `component_register_score` and `register_score_data`.
Every instance produces a fresh native interface diagram. No recorded diagram
fixture or upstream object is needed by this runnable export. The graph contains
sixteen isolated atomic interfaces. Complete decomposition dependency graphs,
source per-gate glyphs and source class-docstring reproduction are separate
coverage gaps. The original diagram replay remains evidence for exporter
serialization, with its previously recorded scope.

All sixteen names, parameter dictionaries, parameter-derived filenames,
ordered signature widths/shapes/sides and independent QEC cost categories are
compared with actual unmodified frozen factories. The thirteen concrete
unitary instances also compare complete complex operators and complex input
states through ordinary `circuit.append(op)`. Symbolic identity retains its
model and signature and requires a concrete width to execute. No auxiliary
quantum storage is allocated by this group.

Both measurement bases compare each unnormalized branch with the original
basis effects and verify that the consumed physical wire is zero. Calling the
source measurement's ordinary tensor contraction raises its existing
information-discard error. This failure is retained. Branch-phase evidence uses
its basis effects instead. Physical clearing retains the classical outcome.
The source logical signature consumes q and produces c.

Emitted IR instruction counts and model categories use separate report rows.
T, H, CNOT, Z, S, CZ, Y and CY each emit one corresponding primitive. Toffoli
emits one CCX and its model counts one Toffoli. Controlled H emits one CH,
while its source QEC estimate is four Cliffords and two rotations. Z measurement
emits measure plus conditioned X. X measurement prepends H. Each model counts
one measurement. Global phase emits two X and two phase instructions on an
arbitrary carrier with zero logical model cost. Concrete identity emits one I
per bit with zero model cost. These are explicit existing execution/model
boundaries, including scalar carrier and physical measurement clearing.

Run `python -m examples.algorithms.basic_gate_ui_export`. Expected artifacts are
seventeen JSON files and fourteen individual documentation files across fourteen
operation folders. All sixteen diagrams are regenerated locally. Retained-file
behavior, fresh navigation and explicit documentation-provider failure are
verified. The remaining six Basic Gates notebook entries contain twenty-one
documentation groups and thirty-five examples. Q0120 stays open for these,
other catalog sections, complete native decomposition providers and fresh
full-catalog diagram execution.

This batch passes 235 concentrated checks. One pre-existing optional PennyLane
identity test is skipped because PennyLane is absent. Ninety-four existing
source/parser warnings remain visible. Strict typing of all three affected
Python files, Black, Ruff, Integration scope, architecture, multi-level IR,
public API baseline and whitespace checks pass. Both full-source catalog
export profiles still produce 3,317 byte-identical files per side after adding
the documentation callbacks. Per-file hash/length manifests are retained in the
external reference environment before reclaiming these completed export copies.
