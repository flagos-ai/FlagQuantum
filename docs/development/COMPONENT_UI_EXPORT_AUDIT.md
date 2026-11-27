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

## Complete Basic Gates native instance catalog

`examples/algorithms/register_gate_ui_export.py` adds all six remaining Basic
Gates entries and joins them with the previous nine in the existing catalog
order. The complete group contains fifteen notebooks, thirty-five documentation
groups and fifty-one actual source factories. It publishes fifty-two JSON files
and thirty-five documentation files. Every instance gets a freshly generated
native typed interface diagram. Large and symbolic examples only construct
metadata and logical interfaces in this workflow.

The six entries reuse `ConjunctionCall`, basis-state/effect resource identities,
canonical integer `FunctionRecord` definitions, `SwapDeclaration`, approximate
swap resource calls, `SwapWithZeroExample`, `MultiplexedSwapExample`,
`PredicateExample` and `ControlledHadamardExample`. Integer ket/effect value and
direction are read with `integer_basis_function_data`. The shared arbitrary
Clifford model key alone would lose those instance parameters. Symbolic ordinary
and controlled swap retain the source's positive-integer symbol assumptions.
Approximate swap retains its distinct unconstrained symbol.

All factory names, documentation grouping/order and ordered signature names,
widths, shapes and sides match the complete frozen group. Forty-three instances
also have exactly matching encoded source metadata and filenames. The remaining
eight are control predicates and ControlledViaAnd examples. Their native metadata
retains actual activation values, typed encoding, shape and register names.
Source ndarray encoding produces `NOT_SERIALIZABLE`, making `ctrl_on_bits` and
`ctrl_on_nd_bits` collide. The test preserves that source result and verifies
that native filenames distinguish the two instances. This native parameter
representation is an explicit adaptation of the example UI artifacts.

The new workflow changes metadata composition only. Existing complete-source
And, swap-network, control-predicate, ControlledViaAnd, integer/basis-vector and
measurement tests are reused for primitive streams, complete complex amplitudes,
measured histories, clean work and resource evidence. A concentrated batch of
149 checks passes with zero skips and fourteen existing parser warnings. Strict
typing of both added files, Black and Ruff pass. The prior full exporter/default
compatibility and real-Jupyter evidence is reused because the exporter has not
changed since that accepted batch.

Q0120 remains open. Native instance metadata and fresh atomic interface drawings
now cover the entire Basic Gates group. The other ten catalog sections contain
202 documentation references and 308 examples. Complete per-instance declared
decomposition graphs, source per-gate glyph parity and fresh native full-catalog
diagrams also remain open. The parent-file acceptance count stays unchanged.

The normal module entry was also executed in an isolated temporary working
directory and produced all expected artifacts with fifty-one non-null diagrams.
Integration scope, architecture, multi-level IR and public API baseline checks
pass for this added provider module. The export workflow performs no numerical
simulation and adds no execution or resource-model behavior.

## Complete Arithmetic native instance catalog

`examples/algorithms/arithmetic_ui_export.py` covers all fourteen Arithmetic
notebook entries, forty-five documentation groups and seventy-four actual
frozen factories. Run `python -m examples.algorithms.arithmetic_ui_export`.
The normal module entry publishes seventy-five JSON files, forty-five
individual documentation files and seventy-four fresh typed interface diagrams.
The SortInPlace group retains its original empty example list and documentation.
Its presence does not manufacture an executable source factory.

Existing addition, negation, subtraction, multiplication, comparison, sorting,
permutation and trigonometric catalogs remain authoritative for their signatures.
Function-owned conversion, bitwise, conditional-arithmetic and triangular-index
signatures follow the existing teaching notebooks. Set difference and sorted
list duplication use their existing compiler signature functions. A private
read-only UI association connects names and parameters with these heterogeneous
owners. It does not introduce a common execution superclass or another IR.

Every factory name, document group, instance order and ordered port matches the
unchanged reference. Port verification includes nominal type, symbolic expression
and assumptions, array shape, LEFT/RIGHT/THRU lifetime, bounded integer iteration
length, and fixed-point fractional width and signedness. The conditional
add/subtract control port retains the actual original QBit type, correcting the
one-bit QUInt interface spelling in its existing teaching notebook for this UI
provider. The earlier notebook execution remains separately recorded.

Seventy-one instances preserve byte-equivalent encoded metadata and filenames.
The three conditional add/subtract instances additionally retain signed versus
unsigned operand encoding. The source encoder discards that nominal type and
produces the same filename for the signed and unsigned eight-bit factories.
The native export keeps them separate. The unmodified source collision is an
explicit test witness. Symbolic permutation cycle shapes retain the original
serializer representation and bound assumptions. The symbolic subtraction
composite remains a CompositeBloq report node inside the Subtract documentation
group, as in the source.

These diagrams show atomic typed interfaces. Complete per-instance decomposition
graphs and source per-gate glyph parity remain open. This metadata workflow emits
no physical gates and does not calculate primitive resources. Existing complete
source-file tests supply the reusable gate-stream, full complex-amplitude,
measurement-history, clean-work and analytical resource evidence. Their measured
primitive counts and source models retain their separate original reports.

The batch regression exposed two stale expectations in the existing conditional
arithmetic foreign-file tests. The original controlled erase still raises a
RecursionError when asked for its unitary. The native importer now diagnoses the
same unsupported leaf with its existing explicit ValueError before that source
recursion. Both original failure and native refusal are asserted independently.
No importer or execution behavior changes in this batch.

Native providers now cover eighty of 237 documentation references and 125 of
359 example factories. The remaining nine sections contain 157 documentation
references and 234 factories. Q0120 remains open for those providers, complete
decomposition drawings and the fresh whole-catalog run. Existing source diagram
replay and prior full-exporter/Jupyter evidence remain applicable.

The concentrated regression passes all 374 checks with zero skips and 37 visible
existing parser, notebook-presentation and source-protobuf warnings. Strict
checking of the provider and both touched test files, Black, Ruff, Integration
scope, architecture, multi-level IR, public API baseline and whitespace checks
pass. The actual module entry also passes in an isolated temporary directory.

## Complete Modular Arithmetic native instance catalog

`examples/algorithms/modular_ui_export.py` covers all seven Modular Arithmetic
notebooks, twenty documentation groups and thirty-five actual source factories.
Run `python -m examples.algorithms.modular_ui_export`. The normal module entry
produces thirty-six JSON files, twenty documentation files and thirty-five fresh
atomic typed interface diagrams. Empty RSA and double-phase ECC root signatures,
2048-bit ModExp output, symbolic words and the four-element ECC control array
are rendered without allocating execution wires.

Modular addition, subtraction, multiplication and inverse examples retain their
existing domain catalog objects. RSA instances use ModularExponentiation and
RSAPhaseEstimation directly. ECC uses the existing signature functions,
EllipticCurveFixedPointAddition, ECPoint and concrete
EllipticCurveWindowAddition. Symbolic ECC interface metadata uses the existing
signature functions instead of passing symbolic widths to int-only executable
class constructors. A small UI tuple associates these heterogeneous signatures
with their operation label, parameter mapping and authoritative native owner.
No execution class hierarchy or program IR is introduced.

Every source factory name, documentation group, example order, exact parameter
object, encoded parameter dictionary and filename matches the original group.
Port comparisons retain nominal encoding, symbolic widths and assumptions,
Montgomery modulus presence or absence, shape and LEFT/RIGHT/THRU lifetime.
Nested point coordinates, modulus, curve coefficient and both window parameters
are included. Concrete modular doubling keeps its original nested dtype metadata.
ECAdd's unbound Montgomery descriptor remains distinct from its algorithm modulus.

This export emits atomic interfaces. Complete per-instance decomposition graphs
and source per-gate glyph parity remain open. The metadata workflow adds no
physical gates or resource estimates. Existing complete modular-source tests and
RSA/ECC tests supply reused primitive-stream, full-complex-amplitude, applicable
measurement-history, clean-work and independent model evidence. Dirty source
lifetimes and out-of-promise ECC domains retain their existing audit boundaries.
The source files and reference defects remain unchanged.

Native instance providers now cover one hundred of 237 documentation references
and 160 of 359 factories. Eight catalog sections still contain 137 documentation
references and 199 factories. Q0120 remains open for these providers, complete
decomposition drawings and the fresh whole-catalog run. Prior complete exporter
and Jupyter evidence is reused because the exporter has not changed.

The concentrated modular batch passes 800 checks with zero skips and fourteen
existing notebook-parser warnings. Strict typing of both added files, Black,
Ruff, Integration scope, architecture, multi-level IR, public API baseline and
whitespace checks pass. The isolated actual module entry also passes.

## Complete field and polynomial native instance catalogs

`examples/algorithms/field_ui_export.py` covers all five GF Arithmetic notebooks
and all three Polynomials over Galois Fields notebooks. The two complete groups
contain fifteen documentation references and twenty-two actual original example
factories. Run `python -m examples.algorithms.field_ui_export`. The normal module
entry produces twenty-two JSON files, fifteen documentation files and twenty-one
fresh diagrams. Both original GF2MulK factories describe the same field and
constant, so they intentionally share their original node file.

Existing field addition, multiplication, square, inverse and polynomial catalogs
remain authoritative for signatures. Binary field codecs supply the actual
irreducible polynomials and field identity. Polynomial codecs retain descending
coefficient order and reconstruct the original concrete constant polynomial.
The original symbolic constant example retains its zero polynomial over GF(2)
independently of the symbolic coefficient field. Exact dictionary comparisons
before encoding preserve symbol assumptions and actual polynomial objects.
Original JSON encoding and filenames match every factory. Unsupported Galois
polynomials retain the source encoder's NOT_SERIALIZABLE spelling, separately
from the full native neutral descriptor identity.

Port checks cover finite-field characteristic, degree, order and irreducible
polynomial, the defining polynomial's own coefficient-field identity, packed
polynomial degree, symbolic bit width, shape and lifetime. GFPolySplit and
GFPolyJoin retain their duplicate logical `reg` names with different LEFT and
RIGHT shapes. Their source register-list order reverses as the view direction
changes. Packed and shaped coefficient representations share fifteen bits in
the concrete degree-four GF(8) example. Existing view tests retain gate-free
wire reuse, complete complex amplitudes and coefficient-order evidence.

Six original symbolic field or polynomial instances are valid display metadata
but are rejected by the existing executable descriptor validator. This UI
provider therefore uses the existing Drawer RegisterStep and layout_register_score
metadata path directly. It preserves nominal port records, array elements,
symbolic widths and disjoint input/output-only lifetimes without constructing
an executable Program. Concrete and symbolic interfaces use the same display
path. This does not change the executable descriptor admission contract.

The diagrams remain atomic interfaces. Complete per-instance decomposition
and source per-gate glyph parity stay open. Existing complete field source-file,
polynomial, packed-view, linear-map and encoding tests supply reused primitive
streams, full complex-amplitude, relevant measurement-history, clean-work and
analytical model evidence. This catalog workflow emits no physical gates and
adds no resource estimator. Source tensor/coefficient-index defects and their
separate accepted/native domains retain the existing file audit records.

Native providers now cover 115 of 237 documentation references and 182 of 359
factories. Six catalog sections retain 122 documentation references and 177
factories. Q0120 remains open for those providers, complete decomposition/glyph
coverage and fresh full-catalog execution. Prior exporter and Jupyter evidence
is reused because the export tool has not changed.

The field/polynomial concentrated regression passes 936 checks with zero skips
and fourteen existing notebook-parser warnings. Strict checking of both added
files, Black, Ruff, Integration scope, architecture, multi-level IR, public API
baseline and whitespace checks pass. The isolated actual module entry passes.
All twenty-two factories have a non-null metadata drawing, with twenty-one
unique node files because the original constant-multiplier factories coincide.


## Complete bookkeeping native instance catalog

`examples/algorithms/bookkeeping_ui_export.py` covers all six Bookkeeping
notebooks, ten documentation references and eleven actual frozen example
instances. Run `python -m examples.algorithms.bookkeeping_ui_export`. The normal
module entry produces twelve JSON files including navigation, ten documentation
files and eleven non-null atomic interface diagrams.

Existing word split/join functions, partition signatures, AutoPartition groups
and cast signatures own the views. Symbolic allocation/free signatures and
Always's AND interface retain the accepted native notebook declarations. The
49-bit partition preserves its 2-by-3 array of two-bit words and 37-bit scalar.
Split and Join preserve duplicate logical names with distinct directional
namespaces. Cast preserves the 32-bit signed source and unsigned fixed-point
target with 32 fractional bits.

Nine factories retain exact original parameter dictionaries after neutral enum
translation, encoded metadata and filenames. Two AutoPartition instances use
explicit native child control values and ordered groups containing unused
segments. Their actual source control scalar is a NumPy array that the source
encoder replaces with NOT_SERIALIZABLE. Native metadata retains the value 1
and unused widths instead. These two filenames deliberately differ. Tests
compare the actual original routing, control polarity, child width and ports
while preserving the source encoding witness.

The existing field-only drawing implementation is extracted into the shared
tool-owned `register_interface_score`. Both catalogs use this helper. All
twenty-two field interface scores are byte-equivalent to accepted commit
c342490df. Export orchestration, score failure handling and executable
descriptor admission retain their existing owners. The complete 3317-file
export-protocol evidence is reused. This batch separately exercises the
exporter regression with both full-fixture profiles deselected. Six remaining
checks pass, including actual Jupyter execution. Its initial local-port bind
was denied by the sandbox, and the same suite passed with escalation.

The concentrated regression passes 940 checks, with zero skips and 25 existing
notebook-parser and protobuf-deprecation warnings. Strict typing of all four
changed Python files, Black, Ruff, Integration scope, architecture, IR and API
checks pass. Actual module execution produces all eleven fresh diagrams.
Existing partition, AutoPartition, split/join, cast, allocation and Always
execution tests retain complex-amplitude, measurement/feedback, clean-work and
primitive resource evidence. View operations emit zero gates in those accepted
execution tests. UI rendering introduces no physical gate stream or analytical
cost estimate. Whole-instance decomposition graphs and per-gate glyph parity
remain separate open work. Native provider coverage reaches 125 of 237
documentation references and 193 of 359 original factories.

## Complete optimization native instance catalog

`examples/algorithms/optimization_ui_export.py` covers all five Optimization
notebooks, six documentation references and twelve frozen example factories.
Existing k-XOR loading and Kikuchi algorithm factories provide every instance.
All twelve actual parameter dictionaries, encoded metadata and filenames equal
the originals. Checks preserve the input constraint multiset, maximum RHS,
symbol assumptions, guiding amplitude, precision, sparsity, bounded index
metadata and packed versus shaped array ports.

Run `python -m examples.algorithms.optimization_ui_export`. The actual isolated
entry produces twelve JSON files including navigation, six documentation files
and eleven unique non-null atomic interface diagrams. The original encoder maps
the two distinct symbolic GuidingState instances to the same filename because
it replaces their ell and amplitude expressions with NOT_SERIALIZABLE. Their
register expressions differ. Existing exporter semantics retain the first file.
Tests preserve this collision and verify both individual Drawer scores are
non-null and distinct. File deduplication therefore does not establish identity
of these two logical instances.

The concentrated regression passes 327 checks, with zero skips and fifteen
existing parser/IPython warnings. Existing complete source-file and notebook
checks exercise measured QROM branches, full complex amplitudes, clean work,
typed child wiring and separate exact/model resources. The source sorting and
matrix-oracle declaration-only boundaries remain explicit. The UI workflow adds
no emitted primitive stream or analytical estimate. Native coverage reaches
131 of 237 documentation references and 205 of 359 source factories.

Two earlier source-file tests still expected the original bounded-register
admission refusal. Current approved bounded storage admits the GuidingState
wiring fixture, so its test now verifies the complete Program and all six child
call multiplicities. Identity child bodies remain a wiring fixture. The packed
scalar to shaped neighbor call reaches the same zero-dimensional IndexError as
the actual frozen source, and the second test checks that exact failure. The
source SortInPlace decomposition refusal remains tested. These assertion and
stale docstring corrections form an independent commit.

Strict typing covers both new provider files, both corrected source-file tests
and the algorithm whose docstring was corrected. Black, Ruff, Integration
scope, architecture, IR, public API baseline and whitespace checks pass. Full
per-instance decomposition graphs, source glyph parity and a fresh whole-catalog
run remain open alongside the four remaining provider groups.

## Complete block-encoding native instance catalog

`examples/algorithms/block_encoding_ui_export.py` covers all ten Block Encoding
notebooks, sixteen documentation references and twenty-six original factories.
It composes existing Unitary, tensor-product, product, phase, linear-combination,
sparse/Hermitian-sparse, Chebyshev and scaled-Chebyshev owners. Existing Hubbard
named/packed catalogs supply SELECT, LCU and packed PREPARE interfaces. The
source BlockEncoding, SelectOracle and PrepareOracle documentation groups retain
empty factory lists and map to the corresponding native structural owners.

Every actual factory signature matches names, nominal dtypes, widths, shapes,
iteration bounds, lifetimes and declaration order. All twenty-six complete
parameter dictionaries, original JSON encodings and filenames match. Recursive
comparisons check explicit matrix arrays before the original encoder replaces
them with its unsupported-object sentinel. Compound blocks preserve child gate
flags, source coefficient tuples, normalization, error and work-width metadata.
Concrete factories retain original integer literal metadata where filenames
would otherwise distinguish equal floating-point values.

Three symbolic composition and sparse-access families use existing native
signature functions, preserving original symbols and assumptions independently
of concrete executable constructors. The symbolic TensorProduct and Product
instances retain independent ancilla widths and source sum/max rules. Native
Hubbard catalog signatures preserve bounded named ports and their packed views.

Run `python -m examples.algorithms.block_encoding_ui_export`. Its actual isolated
entry produces twenty-seven JSON files including navigation, sixteen
documentation files and twenty-six fresh non-null atomic interface diagrams.
Native provider coverage reaches 147 of 237 documentation references and 231 of
359 original factories. Whole-instance decomposition graphs, per-gate source
glyphs and whole-catalog fresh execution remain open.

The concentrated complete component-family regression passes 1822 checks with
zero skips and eighteen existing parser/protobuf warnings. It includes original
gate operands and feedback, LCU full complex amplitudes and measurement branches,
workspace cleanup, explicit sparse entries and Hermitian projections, polynomial
round trips, named/packed oracle replacement and separately recorded emitted
primitive and analytical model counts. Those accepted component owners retain
physical execution and resource semantics. The UI provider supplies interface
metadata. Strict typing of both added Python files, Black, Ruff, Integration
scope, architecture, IR, API and whitespace checks pass. The reference
availability receipt verifies Qualtran 0.7.0 and Kickmix extension 0.1 with its
Simulator present and its compiled binary hash recorded.

A transient filesystem exhaustion prevented a separate environment probe from
starting. The ignored, rebuildable local mypy cache was removed, recovering
approximately 49 MiB. The probe then passed. The running component regression
completed successfully. Subsequent Rotations intake records actual missing
OpenFermion factory dependencies as open evidence.

## Complete Other instance group

`examples.algorithms.other_ui_export` covers all fourteen original Other
notebooks, twenty-one documentation references and thirty-eight actual factory
instances. Existing QromTable and clean-QROAM signature owners supply lookup
metadata and directional retained banks. Native preparation, reflection,
controlled Pauli, rotation, QSP-complement and guided-Hamiltonian declarations
supply the remaining nominal interfaces. Original symbols and assumptions are
compared as actual objects, including the original literal `integers` assumption
in SelectSwapQROM factories. No executable storage admission changes here.

All actual factory port names, nominal encodings, widths, bounds, shapes and
lifetimes match. Declaration order also matches except the two Hubbard GQSP
factories. Frozen QubitizationWalkOperator.selection_registers constructs a set
union, so its order changes across processes. The native provider retains its
existing explicit Hubbard layout and compares complete descriptors by name.
The source is unchanged. This order difference is retained as an adaptation.

Thirty-seven factory metadata encodings and filenames match exactly. Raw nested
arrays, complex coefficients, parameter expressions and symbol assumptions are
checked before encoding. The indexed target callback and the shared-target gate
array use actual native dispatch descriptions. Both original and native encoders
retain their unsupported-object sentinel for those callbacks and arrays. The
callback's four Pauli choices and the original T/H/Z/X sequence are checked.
The Pauli SELECT instance preserves each native Pauli string and its complex
coefficient explicitly. Its source encoder discards the foreign DensePauliString
objects as four sentinels. This one intentional metadata and filename extension
is tested separately, without manufacturing foreign objects in native code.

Actual isolated `python -m examples.algorithms.other_ui_export` produces
thirty-eight JSON files including navigation, nineteen documentation files and
thirty-seven fresh non-null atomic diagrams. Three QROMBase document references
share their first file. Two identical sparse-alias factories share one filename,
matching the original exporter retention behavior. Provider coverage now reaches
168 of 237 documentation references and 269 of 359 factories. Remaining providers
are the complete Rotations and Chemistry groups, with 69 references and 90
factories. Full per-instance decomposition graphs, per-gate source glyphs and
fresh whole-catalog execution remain open.

Concentrated component-family regression passes 1219 checks with fourteen
existing notebook escape warnings and zero skips. Focused actual-factory checks
pass 41 cases. Existing execution evidence covers complete complex amplitudes,
measurement branches, clean workspace, original gate operands and feedback,
lookup and rotation round trips, and separately measured primitive and analytical
resource accounts. This catalog owns metadata and interfaces. Strict typing of
two files, Black, Ruff, Integration scope, architecture, IR, API and whitespace
checks pass. The reference availability check confirms Qualtran 0.7.0 and the
locked Kickmix extension 0.1 with Simulator present and binary hash recorded.

## Complete Rotations instance group

`examples.algorithms.rotations_ui_export` maps all eighteen original notebooks,
thirty documentation references and forty-nine actual factories. Native
rotation and SU(2) primitives, variable/cost phasing catalogs, phase-gradient
addition and constant rotations, Fourier declarations, phase-window owners and
seeded Ising, sparse-chemistry and THC declarations supply its interfaces.
The large THC case regenerates the original RandomState(7) tensors in their
original order. Existing native preprocessing regenerates all 101551 alias rows,
including signs, alternates, keep values, normalization and block exponent.
No reference package is imported by the native provider.

The complete nominal port descriptors match the actual source factories.
All declaration orders match except walk-derived interfaces, where the frozen
walk uses a set union. Native deterministic named order is retained and tested
against the complete source descriptor set. Forty-seven factory metadata
encodings and filenames match exactly, including actual symbolic assumptions,
fixed-point formats, RIGHT phase outputs, complex values and full lookup tuples.
The Ising walk and Ising QPE retain signed Pauli strings and complex coefficients
explicitly. Their source encoder discards foreign Pauli objects as sentinels.
These two intentional native metadata and filename extensions are tested.

The original two square-cost phasing factories collide on one filename despite
different gradient ports. Two symbolic textbook QPE precision factories also
collide because their different expression widths encode as unsupported-object
sentinels. Both collisions retain the source's first-file behavior. Each
individual native interface score is independently non-null, and both colliding
pairs have distinct scores. The test compares each emitted file against the
first retained instance without claiming that file represents both interfaces.

The actual isolated module entry produces forty-eight JSON files including
navigation, thirty documentation files and forty-seven fresh non-null atomic
interface diagrams. Coverage reaches 198 of 237 documentation references and
318 of 359 factory references. Chemistry remains with thirteen notebooks,
thirty-nine document references and forty-one factory references. Full
per-instance decomposition graphs, source gate glyph parity and fresh whole-
catalog execution remain open.

The reference-only restored OpenFermion 1.6.1 environment now runs both original
THC walk and THC QPE factories. Its explicit dependency path is confined to the
source validation lane. The native provider adds no dependency or executable
storage admission. Concentrated component-family regression passes 783 checks,
with seventeen existing parser/deprecation/environment warnings and zero skips.
All 52 focused factory/export checks are included. Existing full complex-state,
measurement-branch, work-zero, source operand/feedback, phase/window/Fourier and
walk evidence is reused through those owners. Observed primitive counts and
analytical resource estimates remain separate. Strict typing of two files,
Black, Ruff, Integration scope, architecture, IR, API and whitespace checks pass.

## Complete Chemistry instance group

`examples.algorithms.chemistry_ui_export` maps all thirteen Chemistry notebooks,
thirty-nine document references and forty-one actual factory references to
existing native sparse, single/double factorization, THC, first-quantization,
grid, Trotter, Hubbard and Givens owners. Forty scoped identities preserve all
thirty-eight original unqualified example names. Projectile PREPARE and SELECT
use separate internal identifiers while retaining their original displayed
names. Repeated SelectHubbard references share one source class, identity and
first-file output. The source's `PolynmomialEvaluationInverseSquareRoot` spelling
is retained as catalog metadata.

Every actual nominal port descriptor, declaration order, recursive parameter
tree, scalar encoding and filename matches the frozen factory. Original symbol
assumptions are retained. Native seeded preprocessing regenerates sparse and
THC alias columns, signs, normalization and block exponents. The explicit THC
block exponent two gives three retained junk cells per target. The PairPotential
example regenerates all four 65536-entry coefficient banks with the existing
native inverse-square-root table owner. The complete arrays are compared before
encoding and hashing. No reference package is imported by this provider.

The actual isolated module entry exports forty-one JSON files including
navigation, thirty-eight documentation files and forty fresh non-null atomic
interfaces. Complete native coverage reaches 237 of 237 document references
and 359 of 359 factory references. Each original factory reference is compared
in document order, including repeated SelectHubbard and scoped projectile names.
Full per-instance decomposition graphs, source per-gate glyph parity and fresh
combined-catalog execution remain open under Q-0120.

The provider emits metadata and atomic interface drawings. Existing execution
owners retain gate operands, measurement feedback, complex amplitudes,
applicable measurement branches and clean work. Observed primitive counts and
analytical resource models remain separate in their existing evidence. Large
catalog declarations allocate no physical simulation storage. No new executable
storage admission or protected API change is introduced.

Concentrated regression across these component families and previously delivered
UI groups passes 1098 checks, with fourteen existing notebook-parser warnings
and zero skips. All forty-three focused Chemistry checks pass. Strict typing
of both new files, Black, Ruff, Integration scope, architecture, IR, public API
baseline and whitespace checks pass. Qualtran 0.7.0 and Kickmix 0.1 Simulator
availability and the accepted binary identity are rechecked.

## Fresh combined native catalog

`examples.algorithms.catalog_ui_export` composes the existing ten domain
exporters in the original eleven-section order. It keeps component ownership,
scoped identities, metadata encoding and score construction with those domains.
Navigation fragments are combined after their exports. Duplicate section
ownership is refused. Sequential failure and existing-file semantics remain
explicit. A rerun refreshes complete navigation while preserving existing node
and documentation files.

The actual isolated module entry exports 354 JSON files including navigation,
233 documentation files and 353 fresh non-null atomic interface diagrams.
Navigation retains all 110 notebooks, 237 document references and 359 factory
references. Every combined node and document is compared byte-for-byte with
the first corresponding output from isolated domain exporters. All source
section, notebook, document and example ordering is checked against the actual
frozen catalog. Previous group-specific factory parameter, type, filename,
physics and resource evidence is reused.

Two original naming boundaries become visible across the complete catalog.
The arithmetic and block-encoding classes both use the unqualified name
`Product`, so two distinct qualified classes share `Product/docs.txt` and the
first arithmetic document is retained. Together with repeated SelectHubbard
and QROMBase references, 237 document references represent 234 qualified
classes but only 233 output documentation folders. The source Subtract group
includes `_sub_symp_decomposition`, whose actual factory returns a
`CompositeBloq`. Original navigation places its filename under `Subtract`,
while node export places it under `CompositeBloq`. The original actual factory,
filename and missing expected navigation path are tested explicitly. All other
358 factory references resolve through their stated documentation folder.
This inherited navigation gap remains open.

Fresh combined root-instance construction is now exercised directly. Complete
per-instance dependency decompositions, source per-gate glyph parity and fresh
native dependency-node coverage remain open under Q-0120. The 353 atomic root
interfaces have no declared child edges. They do not replace the separate
accepted 3083-node source-diagram replay or imply that native decompositions
have been emitted. Observed primitive counts and analytical models remain in
the existing domain execution evidence.

Concentrated catalog/export regression passes 406 checks with fourteen existing
notebook-parser warnings and zero skips. Two full source-catalog diagram
profiles and the real Jupyter notebook check are deselected in this batch;
their previously accepted source replay and actual notebook evidence is reused.
The preceding Chemistry execution-family regression passes 1098 checks.
Strict typing of both new files, Black, Ruff, Integration scope, architecture,
IR, public API baseline and whitespace checks pass. The isolated complete
catalog module entry is also exercised.

## Complete catalog navigation ownership

The combined catalog now routes every one of its 359 factory references to an
existing component record. The existing `filename` string supports a POSIX
relative path from its documentation folder. Shared exporter opt-in
`relative_example_paths=True` obtains that path from the supplied component
name and filename callbacks. Its default keeps source-compatible domain export
bytes. A cross-folder WordSwap workflow exercises this shared routing directly.
Node attributes, filenames, score payloads and dependency references retain
their existing providers and encoding.

Arithmetic and Block Encoding accept the existing documentation-name callback
at their developer export entry. The combined catalog supplies
`arithmetic.Product` and `block_encoding.Product` for the two distinct Product
classes. Both authored domain documents are present. Their examples point to
`../Product/<original-hash>.json`. The Subtract composite example similarly
points to `../CompositeBloq/<original-hash>.json`. All remaining navigation
paths resolve, including the original repeated factory names and scoped
projectile identities. Default standalone exports retain both original naming
witnesses for immutable source comparisons.

The fresh combined entry now produces 354 JSON files, 234 distinct qualified
class documents and 353 non-null atomic interfaces. All 110 notebook titles,
237 document references and 359 factory display names retain source order.
The two Product documentation labels are deliberate native ownership
adaptations. The complete menu also checks original example docstrings.
An actual-factory audit found three missing ModExp descriptions, now supplied
by the native modular catalog. All 359 example descriptions equal the frozen
source. Native records continue to own their lazy factories and descriptions.

Concentrated regression passes 407 checks with fourteen existing parser
warnings and zero skips. Three checks are deselected in that run. The two full
source-catalog export profiles are separately rerun and pass with forty-four
existing reference warnings. They compare all 3317 output files, including the
authenticated actual-source score replay. The earlier actual Jupyter execution
evidence is reused. Strict typing of eight changed Python files, Black, Ruff,
Integration scope, architecture, IR, public API baseline and whitespace checks
pass. The actual isolated combined module and complete description audit pass.
Full native dependency decompositions, per-gate glyph parity and fresh native
dependency-node coverage remain open under Q-0120.


## Native swap and predicate decomposition projections

The complete swap.py, and_bloq.py and ctrl_spec_and.py presentation paths now
select their existing native typed Programs through
`examples/algorithms/basic_gate_decomposition.py`. The Compiler remains the owner
of the concrete child calls, operands, lifetime directions and physical bodies.
The adapter supplies source wire glyphs through the existing Drawer callback.
It adds no compiler graph representation or resource evaluator.

Nine actual source factories match fresh decomposition scores in ordered glyphs,
connection-seeded port order, layout, horizontal lifetimes and vertical labels.
Native SSA identities and boundary dtype text are normalized for comparison.
Those factories include every concrete control pattern and both symbolic-value
patterns, including the nd-array control whose source filename collides. The
comparison executes each source instance directly. The source JSON encoder's
NumPy boolean sentinel is retained as a separate failure witness, while native
JSON stores actual boolean glyphs.

Both 64-bit Swap and CSwap native graphs also render. The original source raises
IndexError for these drawings. This native extension is recorded separately from
the nine comparable successful scores. Symbolic word sizes, symbolic control
counts and atomic And keep the existing atomic interface. Source refusal
evidence and physical execution/resource evidence remain applicable.

Eleven of the 353 unique catalog diagrams now project immediate native calls.
The remaining 342 diagrams retain atomic interfaces. Dependency-node publication
and recursive graph coverage remain open, so Q-0120 remains open. The next
Basic Gates source group comprises states/effects, relative controlled swap,
SwapWithZero, multiplexed swap and ControlledViaAnd. Nine source-success
instances remain in that group. Preserve all 33 recorded original failures and
distinguish their native display extensions from upstream parity.


## Complete integer state and effect child projection

`integer_basis_program` in the existing Compiler basis-vector owner retains
source MSB-first zero/one calls and the Join or Split boundary. The original
canonical loader supplies each child's physical body. Caller outputs retain
ownership until explicitly consumed; an effect requires owned clean release.
Both 8-bit catalog instances now draw these immediate child graphs. Their
ordered fresh source geometry and glyphs match after native identity/boundary
normalization. No source fixture is used for these drawings.

Widths one through four exhaust every integer in both directions. Ordered
child wiring matches the unchanged original. Bound primitive streams and
output mappings equal the canonical implementation exactly. Closed prepare /
effect pairs return all storage to zero with 2 * popcount(value) physical X
gates. Existing tensor and complex-amplitude evidence remains applicable.
The original analytical model retains one ArbitraryClifford call per integer
vector; its logical classification remains separate from physical X counts.

The combined catalog now has thirteen immediate native decompositions and
340 atomic interfaces. Seven original Basic Gates source-success examples
remain in relative swap, SwapWithZero, multiplexed swap and ControlledViaAnd.
Source leaf decomposition refusals, original rendering failures, dependency
node publication and recursive coverage remain separately open.


## Complete relative-phase exchange child graph

`relative_swap_component.py` composes the existing relative-swap physical
segments and the existing typed fanout. The immediate graph retains the
original moment/import ordering, first-use y/x splits, packed MultiTargetCNOT
boundary and final x/y joins. Width one keeps bit ports directly. All storage
remains caller-owned, with no private work or measurement. The existing
physical relative-swap emitter still owns phases and instruction recipes.

Widths one through four match complete ordered original child connections.
Full complex outputs match the existing physical emitter, with independent
original tensor checks at widths one and two. Observed primitive categories
match existing physical emission and observed T/T-dagger count is 4 * width.
The original analytical model also has 4 * width T gates; its opaque Clifford
classification remains separate. Previous complete-operator and controlled
measurement/clean-work evidence is reused and its execution families rerun.

The four-bit catalog graph matches fresh source geometry, ordered glyphs and
MultiTargetCNOT-specific title and symbols after native boundary/identity
normalization. The native 64-bit graph renders, with the original IndexError
retained as an extension witness. There are now fifteen immediate native
decompositions and 338 atomic interfaces. Twelve original successful Basic
Gates instances match fresh drawings. Six source-success instances remain in
SwapWithZero, MultiplexedCSwap and ControlledViaAnd; dependency publication
and recursive coverage remain open.


## Complete typed SwapWithZero traversal

`swap_with_zero_component.py` uses the existing validated network traversal and
`relative_swap_program` child. It retains source split, exchange and join order
for one-dimensional, multidimensional and reverse traversal. All selectors and
targets remain caller-owned. The native selector Join retains the original
bounded descriptor at the Program return boundary; the source child Join uses
QAny. This nominal dtype adaptation is explicit while bits and logical wiring
remain comparable. No additional Cast node or physical gate is introduced.

All three actual source catalog scores match freshly executed source glyphs,
layout and ordered ports, with native SSA/boundary text normalized. Child
CSwapApprox glyphs retain their own `~swap` title and `@(approx)` control.
Small routing fixtures in both directions match original ordered child calls,
physical primitive categories, full complex phases and unchanged owned outputs.
Existing exhaustive branch, operator and resource evidence remains applicable.

The catalog now projects eighteen immediate native decompositions with
335 atomic interfaces. Fifteen original successful Basic Gates instances
match fresh drawings, and three native large-swap displays retain original
IndexError witnesses. MultiplexedCSwap and ControlledViaAnd account for the
remaining three successful instances. Dependency-node publication and recursive
coverage remain open, so Q-0120 remains open.

## Complete controlled-H and multiplexed-exchange immediate graphs

The ControlledViaAnd example now exposes a typed Program through the existing
`control_component_via_predicate` owner. Its immediate calls retain CtrlSpecAnd,
positive CHadamard and the exact measured predicate adjoint. CH uses the existing
controlled-H primitive owner. Both actual source catalog drawings match fresh
ordered glyphs and geometry, including integer and signed control labels.
The bits, integers and nested fixtures bind to the accepted source streams.
Feedback SSA slot names are normalized only after checking each measured wire's
latest live slot. Small bits and nested fixtures verify every measurement history
against the original complex operator, including clean work.

`multiplexed_swap_component.py` composes the shared bounded unary traversal and
exact controlled-word exchange. The unary helper retains scalar X q and CNOT
ctrl/target ports. Selectors split on first use, while one-bit selectors remain
scalar as in the original import. Multi-control reductions use the existing
And/MultiAnd Program and measured adjoint. Caller-owned target arrays and output
are threaded in row-major order. Final joins retain native bounded descriptors,
where original joins return QAny, following the explicit SwapWithZero adaptation.
The physical builder's existing singleton-first-axis refusal is retained.

Nine configurations cover the actual five-row/two-bit catalog instance, padded
widths, one/two/three controls, multiple axes and controlled singleton axes.
Complete ordered source connections match. Bound streams preserve source
operands and feedback lifetimes through the existing physical owner. Small
configurations exhaust all measurement histories, full complex states and clean
work. The actual large catalog instance reuses accepted physical execution
evidence through stream identity instead of allocating a dense large operator.
Observed T/T-dagger count is 7 * rows * word_width + 4 * conjunction_count;
observed measurement count equals conjunction_count. Original CSwap, And and
Clifford analytical classifications remain separate from emitted primitives.

All eighteen originally successful Basic Gates source instances now match fresh
immediate drawings. Three native large-swap displays retain original IndexError
witnesses. The combined catalog has twenty-one immediate decompositions and
332 atomic interfaces, exporting 354 JSON files, 234 documents and 353 diagrams.
All 33 original decomposition/rendering failures remain recorded. Recursive
native graph expansion and dependency-node publication remain open, so Q-0120
is still open. No broader catalog closure is implied by immediate drawings.

## Reachable native Program dependency publication

`component_ui_dependencies.py` projects the existing Compiler call graph into
UI graph entries. Each reachable child function has a file, and all dependency
links resolve through the existing exporter mapping. Multiplicity remains with
the Compiler resource graph. Root metadata, filenames, original ordered glyphs
and the eighteen fresh successful-source comparisons remain unchanged.

Internal nodes are identified by the owning root filename and native function
name. The immutable Program supplies bodies and ports directly. This contextual
identity deliberately preserves separate native helper boundaries across roots.
Descendant pure typed-call bodies use the existing Drawer, while mixed/primitive
bodies retain atomic interfaces. Descendant glyphs use native register defaults.
These nodes have no original Bloq identity, cross-root semantic deduplication or
recursive source-glyph equivalence claim. Further source-identity and recursive
presentation qualification remains in Q-0120.

Every reachable function and edge is checked against the Compiler graph, every
published dependency link is resolved, and all root files remain accounted for.
The combined module now publishes 737 JSON files, 234 documents and 736 non-null
diagrams, including 383 contextual native dependency nodes. The original 353
root diagrams still contain twenty-one immediate decompositions and 332 atomic
interfaces. Concentrated export regression passes 431 cases with fourteen
existing warnings and three deselections. Strict typing of five changed files,
formatting, team scope, architecture, IR and API checks pass. Physical execution
and resource evidence is reused unchanged from the accepted component owners.

A regression first exhausted disk while writing temporary outputs. Only finished
or interrupted temporary outputs created by this continuation were reclaimed,
with source and evidence logs retained. The complete concentrated regression
then passed. Q-0120 remains open for broader catalog dependencies and recursive
source identity/glyph parity, alongside the previously recorded source failures.

## Recursive Basic Gates source presentation

The complete successful Basic Gates source call chains now qualify descendant
CSwap, CSwapApprox, MultiTargetCNOT, MultiAnd and CtrlSpecAnd presentations,
including their measured adjoints. Compiler definition matching ignores only
reachable function names and unreachable definitions. Port metadata, helper
sharing, SSA, call order, operands, phases, feedback, allocation and physical
context remain literal. Only matching stored bodies reuse existing constructor
Programs and source-specific Drawer symbols. Unknown functions retain native
presentation and contextual owner/entry identities.

The unchanged locked source supplies 39 decomposable definitions across eighteen
successful roots, including twenty-one descendants. Every ordered call and
operand connection matches, and all twenty-one recursive drawings match fresh
source geometry and symbols after the previously documented SSA/boundary dtype
normalization. Original atomic decomposition exceptions and large-rendering
IndexError witnesses are retained. Three native large-swap extensions are
checked separately. No cross-root filename equivalence is claimed.

The isolated combined module retains 737 JSON files, 234 documents and 736
diagrams. Twenty-two child files carry qualified operation labels, including
the additional fanout child under a native large-swap extension. The twenty-one
source-success descendant drawings remain the original-success comparison set. Concentrated regression passes 463 cases with fourteen existing
warnings, three deselections and zero skips. Strict typing of five files, Black,
Ruff, team scope, architecture, IR and public API checks pass. Accepted 185-case
execution/control/unary evidence is reused through complete literal stored-body
identity. No new dense execution of large display fixtures is claimed. Drawing
adds no physical instruction, measurement or workspace. Actual primitive counts
and analytical source resource models remain separately owned evidence.

Basic Gates successful recursive presentation is qualified within its native
context. Broader catalog decomposition and dependency coverage remains in
Q-0120, together with original source failure witnesses and the recorded
environment, dashboard and pending contract gaps.

## Word bitwise and negation catalog decomposition

The whole `arithmetic/bitwise.py` and `arithmetic/negate.py` catalog families
now draw their five successful immediate decompositions. Concrete Xor,
BitwiseNot and Negate Programs are the exact existing executable domain
constructions. Symbolic BitwiseNot and Negate retain typed external child
declarations, including the X repetition and AddK constant-one identities.
Declarations have no blocks and cannot bind for execution. Symbolic Xor retains
the original DecomposeTypeError and its atomic interface. Existing XorK
physical/source evidence remains with the complete bitwise owner.

The existing dependency projection now accepts a domain-owned Program supplier.
Every reachable word-arithmetic helper becomes a contextual native child file.
No new component registry or execution entry is introduced. Five ordered source
call/operand traces and fresh drawings match after the established boundary
dtype and SSA text normalization. Child metadata keeps native helper identity
and default glyphs. Recursive arithmetic source-glyph qualification remains
open and is separate from these immediate root comparisons.

The isolated combined exporter produces 798 JSON files, 234 documents and 797
non-null diagrams, including 444 contextual child files. Its 353 root diagrams
now comprise 26 immediate decompositions and 327 atomic interfaces. All 22
previously qualified Basic Gates child labels are retained. Concentrated
catalog regression passes 475 cases with fourteen existing warnings, three
deselections and zero skips. Strict typing of five files and Black, Ruff, team
scope, architecture, IR, public API and whitespace checks pass.

The complete native bitwise and negation physical component regressions pass
189 cases. Concrete Programs are reused literally. These cover ordered original
primitive streams, complete complex amplitudes, applicable measurement histories,
clean workspace, ordinary append entry points and resource formulas. Bitwise
operations allocate no workspace. Negation retains the accepted primitive
T/T-dagger count 4 * (width - 1), with width - 1 measurements and resets,
and independent source analytical models. Existing signed one-bit source
encoding failure remains an explicit native-extension boundary. Drawing and
dependency publication add no physical instructions or workspace.

A fresh intake of all 74 arithmetic factories found 44 successful immediate
drawings, eight decomposition type failures, thirteen missing decompositions,
six ValueErrors and two large-rendering IndexErrors. One controlled-addition
instance has 2008 immediate calls and its fresh layout is deferred under an
explicit 1000-call intake bound. After these five roots, 39 successful root
drawings remain. Q-0120 stays open for those source-file groups, recursive
arithmetic drawings and broader catalog dependencies.

## Complete addition source-family catalog graphs

All ten catalog instances from `arithmetic/addition.py` are now accounted for
through existing Add, OutOfPlaceAdder and AddK owners. Seven concrete instances
retain complete immediate call graphs and typed dependencies. Three symbolic
instances retain the original DecomposeTypeError and atomic interfaces. Five
source-success drawings match fresh geometry, glyphs and ordered connections.
The two large native graphs draw successfully while their original IndexErrors
remain recorded as source witnesses.

The existing `copy_component_program` rebuilds OutOfPlaceAdder in source graph
visitation order. It renumbers independent input XORs and workspace acquisition
using the existing SSA graph traversal. Exact ordered traces now match both
actual concrete factories. The established physical constructor remains the
execution owner. Graph copying reuses the existing scheduling policy and
retains the original primitive category counts.

Source-specific immediate glyphs reuse conjunction, controlled-NOT, bitwise and
partition callbacks. Allocate/Free labels retain original arrow directions.
Always(Add) retains the wrapper's own default a/b labels and Add title. Source
And(uncompute=True) retains its empty title, distinct from a generic Adjoint
wrapper. Child files retain contextual native identities and default symbols.
Recursive arithmetic source-glyph qualification remains open.

The isolated combined exporter publishes 1009 JSON files, 234 documents and
1008 non-null diagrams. Its 353 roots contain 33 immediate decompositions and
320 atomic interfaces, alongside 655 contextual native child nodes. This batch
adds 211 child files. All previously qualified Basic Gates children remain.
Concentrated regression passes 491 cases with fourteen existing warnings, three
deselections and zero skips. Strict typing of three Python files, Black, Ruff,
team scope, architecture, IR, public API and whitespace checks pass.

The complete existing addition, out-of-place addition and constant-addition
physical component tests pass 397 cases. Separate copied-graph checks exhaust
small complete isometries with an entangled spectator, every measured erasure
history, full complex amplitudes and zeroed released work. Copies retain exact
primitive category counts. Existing observed primitives and independent
analytical models remain separately reported. Add/AddK use 4 * (width - 1)
T/T-dagger gates for equal-width words. Forward OutOfPlaceAdder uses 4 * width
with its most-significant carry; its measured adjoint retains the accepted
measurement model. The 64-bit catalog reuses accepted physical stream and
resource evidence. Fresh dense execution remains uncovered.

Ten of the arithmetic intake's 44 successful root drawings are now qualified.
Thirty-four successful roots, the explicitly bounded large CAdd layout, recursive
arithmetic presentation and other catalog dependencies remain in Q-0120.

## Complete subtraction source-family catalog graphs

All eight `arithmetic/subtraction.py` catalog instances are accounted for. Five
concrete Subtract/SubtractFrom Programs reuse their executable domain owners
literally. Two symbolic equal-width instances retain typed external child
declarations and draw the original complete immediate layer. Symbolic Subtract
uses the existing `symbolic_subtraction_steps` catalog sequence for signed casts,
OnEach(X), Add and restoring casts. SubtractFrom uses its source-defined
complement/Add/complement sequence. The original CompositeBloq factory still
raises ValueError when asked to decompose again, and retains its atomic UI
interface. Symbolic declarations have no execution body and fail binding.

All seven successful factories match complete ordered source calls, operands,
shapes, sides and child dtype encodings. Fresh source drawings match geometry,
wire glyphs and titles with only the established SSA/boundary text normalization.
Unequal-width signed extension retains MultiTargetCNOT control/target glyphs,
allocation/free arrows and the distinct Cast versus Split/Join presentations.
Existing Add, bitwise, repeated-X and partition callbacks supply symbols.

The isolated combined exporter publishes 1241 JSON files, 234 documents and
1240 non-null diagrams. Its 353 roots contain 40 immediate decompositions and
313 atomic interfaces, alongside 887 contextual native child files. This batch
adds 232 child files. An additional nested fanout matches the existing qualified
constructor, giving 23 qualified child operation labels. Recursive arithmetic
source-glyph coverage beyond these root drawings remains open.

The complete subtraction physical component regression passes 64 cases,
including original ordered primitives and feedback, full complex states with an
entangled spectator, every small measurement history, zeroed work, output
ownership and source resource models. Exact concrete Program equality connects
the UI to those execution owners. Observed T/T-dagger gates equal
4 * (right_width - 1), with right_width - 1 measurements and resets, while source
analytical formulas retain separate evidence. The measured peak-width model
remains 3 * right_width - 1 for Subtract. Large instances retain accepted stream
and resource evidence; fresh dense large execution remains uncovered.

A concentrated regression exhausted filesystem space while retaining successful
temporary exports. Its diagnostic log is retained. Only this batch's interrupted
temporary output was reclaimed. The repeat enables pytest's failed-only temporary
retention policy so passing fixtures are removed as they finish. The repeated
concentrated suite passes 506 cases, with fourteen existing warnings, three
deselections and zero skips. Strict typing of three Python files, Black, Ruff,
team scope, architecture, IR, public API and whitespace checks pass. Source
files, reference data and evidence logs remain.

Seventeen of the arithmetic intake's 44 successful root drawings are now
qualified. Twenty-seven successful roots, the bounded large CAdd layout,
recursive arithmetic presentation and other catalog dependencies remain open in
Q-0120, alongside the previously registered source failures and external gaps.

## Complete controlled-addition source-family catalog graphs

Both actual `arithmetic/controlled_addition.py` CAdd factories now retain their
complete immediate graph and all native dependencies. Existing executable
`controlled_addition_program` remains the source of truth. The three-bit and
1000-bit factories match original ordered calls and operands, including explicit
control Split/Join, typed Cast, And computation and measured erasure. Both
control polarities match fresh small source drawings using existing Add, And
and partition callbacks. The complete small catalog drawing matches.

The previous 2008-call large layout intake boundary is resolved. Unmodified
Qualtran's actual 1000-bit drawing raises IndexError. The native graph draws
successfully and retains that source failure witness. It is a native display
extension. The isolated combined module produces 1337 JSON files, 234 documents
and 1336 non-null diagrams. The 353 roots now contain 42 immediate decompositions
and 311 atomic interfaces, plus 983 contextual native child files. This batch
adds 96 child files. Previous 23 qualified child operation labels remain.
Concentrated regression passes 510 cases with fourteen existing warnings, three
deselections and zero skips. Strict typing of three Python files, Black, Ruff,
team scope, architecture, IR, public API and whitespace checks pass.

The complete controlled-addition physical component suite passes 149 cases. It
retains both control polarities, exact adjoints, original streams/feedback, full
complex amplitudes, every small measurement history, zeroed work, ordinary
append execution and outer-control callbacks. The actual 1000-bit factory binds
and measures primitive resource categories. T/T-dagger count is 7996,
measurement and reset counts are each 1999, clean work is 1999 wires and total
width is 4000. The independent source analytical model agrees in those
categories. Actual primitive accounting and model queries remain separate
evidence. Fresh dense 1000-bit simulation remains uncovered.

Eighteen of the arithmetic intake's 44 source-success root drawings are now
qualified. Twenty-six source-success roots, recursive arithmetic child glyphs
and broader catalog dependency coverage remain in Q-0120. The complete intake
now has three original rendering IndexErrors and no deferred CAdd layout.
Contextual native child identities and previously registered source failures,
pending contracts and environment/dashboard gaps remain explicit.

## Complete sign-extension source-family catalog graphs

Both actual factories in `arithmetic/conversions/sign_extension.py` now use the
existing executable `sign_extension_program` for catalog drawings and dependency
publication. Each five-call graph matches unchanged source order, operands,
child register encodings, shapes and sides. Fresh original root drawing geometry
and glyphs match for both SignExtend and SignTruncate. Existing workspace,
partition and fanout symbol callbacks are reused. Forward and inverse storage
arguments retain the existing owner contract and MSB-first bit order.

The original truncation ends with an untyped Join whose output is QAny despite
the nominal QInt output signature. The native graph retains this exact view,
with an explicit regression witness. No signed cast is inserted. Physical
execution evidence is reused through exact Program equality. The complete
sign-extension physical/capture/serialization suites and fresh presentation
checks pass 53 cases with six upstream serialization warnings. These cover both
signed encodings, complex amplitudes with an entangled spectator, inverse
restoration, zeroed redundant prefixes and original primitive streams. Actual
8-to-16-bit factories are covered. Their fanout emits 15 CX operations per
direction. Independently, the source analytical query counts one Clifford and
zero T or measurements. Dense state inspection in the component suite is
limited to input widths at most three, with actual-factory composition covered
by the separate file suite.

The isolated combined catalog module produces 1353 JSON files, 234 documents
and 1352 non-null diagrams. There are 44 root immediate decompositions, 309
atomic root interfaces and 999 contextual native dependencies. This batch adds
16 child files. Qualified descendant operation labels increase from 23 to 25.
Twenty of the original 44 successful arithmetic root drawings are qualified,
leaving 24. Recursive arithmetic source glyphs and broader catalog dependencies
remain open in Q-0120. Previously registered source failures, pending contracts
and external environment/dashboard gaps remain recorded separately.

Concentrated catalog regression passes 512 cases with fourteen existing warnings,
three deselections and zero skips. Strict typing of three touched Python files,
Black, Ruff, team scope, architecture, IR, API and whitespace checks pass. The
isolated user-facing `python -m examples.algorithms.catalog_ui_export` entry
succeeds. Existing executable Programs and physical owners remain authoritative.

## Complete sorting source-family catalog graphs

All six actual `arithmetic/sorting.py` factories now retain their immediate
source graphs and contextual native dependencies. Concrete Comparator,
ParallelComparators, BitonicMerge and BitonicSort use the exact existing
executable Programs. Their complete ordered calls, operands and child register
encodings, shapes and sides match unchanged source. Both symbolic-width
factories also match their original graphs. Symbolic Comparator declares
GreaterThan with unsigned operands and CSwap with opaque operands, preserving
the original storage views. Symbolic four-word sort declares its two recursive
sort calls and merge. These leaf declarations have no executable bodies and
physical binding refuses them explicitly.

All six fresh root diagrams match original geometry and glyphs. Existing
workspace, swap and typed port symbols are reused, with source GreaterThan
labels and default sorting titles composed locally. The isolated combined
catalog entry produces 2159 JSON files, 234 documents and 2158 diagrams. The
353 roots contain 50 immediate decompositions and 303 atomic interfaces, plus
1805 contextual native children. This batch adds 806 child files. Qualified
native descendant labels increase from 25 to 36.

Existing physical sorting suites pass 208 cases. They retain complete primitive
streams and measurement feedback, full small complex states with entangled
spectators and branch histories, valid inverse restoration, zeroed temporary
work, explicit logical history outputs, ordinary append entry and original
seeded larger sorting cases. Presentation Programs are checked for literal
equality with those owners. For C comparisons of n-bit words, emitted T/T-dagger
count is C*(15*n-4), and measurement count is C*(2*n-1). Original analytical
queries using the actual seven-T CSwap convention agree. The original model's
default four-T CSwap convention instead gives C*(12*n-4). Both queries are
retained independently from emitted primitive counts. Large dense branch
simulation is outside the small-state evidence scope.

Twenty-six of 44 original-success arithmetic root drawings are now qualified.
Eighteen remain, along with recursive arithmetic source-glyph qualification and
broader catalog dependencies in Q-0120. Original failure witnesses and previously
registered contract/environment/dashboard boundaries remain explicit.

Concentrated catalog regression passes 524 cases with fourteen existing warnings,
three deselections and zero skips. Strict typing of three touched Python files,
Black, Ruff, scope, architecture, IR, API and whitespace pass. A concurrent
export run exhausted local disk space. Its log is retained outside the checkout.
Only generated output from that interrupted run was reclaimed, then the full
concentrated suite passed serially with failed-only temporary retention.

## Complete duplicate-detection source-family catalog graphs

Both documented `arithmetic/lists/has_duplicates.py` factories now retain all
eight immediate calls and publish their native dependencies. Concrete storage
uses the exact existing `sorted_duplicates_program`, retaining the original
inverse half-LT output aliases and known invalid erasure behavior. Symbolic
width keeps declared half-LT, inverse and X children with no executable bodies.
The finite MCX dependency reuses its accepted owner. Symbolic list-length
refusal and the third, undocumented factory remain covered by the existing
whole-file inventory rather than an invented finite catalog graph.

Ordered call identities and complete named operand/output collections match
unchanged source. Input mappings are compared by key and value. Source incoming
traversal groups insertion order by predecessor, while the native traversal
uses the declared signature order. Child signature order, dtype, shape and side
are independently checked. Both fresh root diagrams match source geometry and
glyphs. MCX uses the original nonempty TextBox connector caption without a font
size. The internal Drawer callback now accepts that existing text-only label
shape, including the previously accepted empty caption. Malformed numeric text
remains rejected. Public exports and serialized label fields are preserved.

The combined module entry produces 2311 JSON files, 234 documents and 2310
diagrams. There are 52 root immediate graphs, 301 atomic interfaces and 1957
contextual native children, adding 152 children in this batch. Qualified
recursive operation labels remain 36. Concentrated catalog/Drawer regression
passes 530 cases with fourteen existing warnings, three deselections and zero
skips. Five touched Python files pass strict typing, Black and Ruff. Scope,
architecture, IR, API and whitespace checks pass.

The native duplicate, component, whole-file, resource and classical suites pass
612 cases. These preserve exact source primitive streams, resource/model
queries, valid two-word full complex states and all small measurement histories,
zeroed work and the original three-word erasure counterexample. The optional
foreign-file suite separately passes 19 cases and fails three controlled SDK
cases at widths one, two and three. Its production dependencies and test were
unchanged from published eacdf60f7 when the failures were reproduced. The native
import deliberately prefers source typed-graph expansion. The SDK witness
retains its independent atomic-CCX shortcut. At width one, the native stream has
eight T/T-dagger and two measurements, while that SDK witness has four T/T-dagger,
one measurement and one atomic CCX. Original tests and source remain intact.
This SDK-path primitive parity gap remains open and is separate from the fully
qualified catalog source graphs. Temporary alternate-oracle experiments did
not establish equivalence and provide no acceptance evidence.

Twenty-eight of the original 44 successful arithmetic root drawings are now
qualified, leaving sixteen. Recursive arithmetic glyphs, SDK-path parity and
broader catalog dependencies remain in Q-0120. Full G2 acceptance is pending.

## Complete permutation source-family catalog graphs

The complete `arithmetic/permutation.py` catalog inventory is accounted for.
All four successful original root diagrams now retain their ordered cycle calls.
Concrete dense and partial permutations exactly reuse `permutation_program` and
its cycle, constant equality and controlled-XOR owners. Known symbolic cycle
collections retain separate typed cycle declarations with no executable bodies.
Their ordered data remains in existing root attributes and their bounded word
ports preserve both width and iteration bound. Physical binding refuses those
symbolic graphs. The unknown-cycle-count and unknown-cycle-value examples retain
original decomposition refusal. Both known-cycle factories retain the original
controlled-XorK drawing ValueError and atomic catalog interface. Existing
physical cycle ownership and original source error witnesses remain separate.

All four fresh source-success drawings, complete ordered operands and child
port dtype/width/bound/shape/side agree. Twelve focused module cases pass.
The isolated combined catalog entry produces 2921 JSON files, 234 documents and
2920 diagrams, with 56 root immediate decompositions, 297 atomic interfaces and
2567 contextual native children. This batch adds 610 child files. Qualified
recursive operation labels remain 36.

The existing register/component/declaration/file/foreign permutation batch
passes 112 checks and fails four pre-existing foreign SDK stream checks. Native
owners and those tests are unchanged from baseline 8a4f528bb. Native source-graph
checks cover exact primitive streams, full small complex tensors with spectators,
measurement histories, zeroed work, coherent inverse restoration, actual
factories and normal append entry. Source analytical queries are retained
independently. For n-bit words and E constant-equality calls, emitted T/T-dagger
count is 4*(n-1)*E and measurement count is (n-1)*E, agreeing with the source
model under that same decomposition.

The separate SDK shortcut witness has different primitive counts. Native/source
SDK instruction lengths are respectively 76/15 for width-two cycle, 146/28 for
width-two composition, 139/120 for width-three cycle and 272/238 for width-three
composition. These are explicit unresolved SDK-entry parity cases. The original
SDK tests and reference source stay intact. Their logs and unchanged-baseline
witness are retained. Combined with duplicate detection, seven SDK primitive-path
failures remain open. Catalog presentation completion does not close that gap.

Thirty-two of the original 44 successful arithmetic root drawings are qualified.
The twelve remaining roots belong to the complete comparison source family.
Recursive arithmetic source glyphs, SDK primitive-path parity and broader
catalog dependencies remain open in Q-0120. Full G2 acceptance is pending.

Concentrated existing catalog/Drawer regression passes 530 cases with fourteen
existing warnings, three deselections and zero skips. The twelve focused
permutation presentation cases pass separately, including bounded child-port
checks. Strict typing of three Python files, Black, Ruff, scope, architecture,
IR, API and whitespace checks pass. The isolated user-facing catalog module
entry also succeeds. The four pre-existing SDK failures remain explicitly open.

## Complete comparison catalog disposition and native graphs

The complete `arithmetic/comparison.py` catalog has fifteen instances. Twelve
concrete roots now expose the existing native component Programs and their
contextual dependencies. The symbolic LessThanConstant and LessThanEqual roots
retain original DecomposeTypeError. GreaterThanConstant retains original
DecomposeNotImplementedError. The existing whole-source definition disposition,
physical stream and classical/model evidence remain in COMPARISON_FILE_AUDIT.md.

All twelve concrete roots match complete ordered source calls, named operands
and freshly rendered source geometry and glyphs. Eleven also match complete
child types. EqualsAConstant's original default Join returns QAny, while its
native executable owner retains the nominal unsigned view. Both encodings are
checked separately. This view boundary remains explicit.

The complete LessThanConstant, LessThanEqual, BiQubitsMixer and SingleQubitCompare
owners now use the shared Compiler wire reconstruction. It schedules existing
typed child Programs by earliest physical dependency layers and reproduces the
source's lazy Split/Join, insertion order and allocation-key types without an
SDK dependency. Forward and adjoint tree graphs match complete original traces.
The former opaque primitive wrappers and unused prefix-zero wrapper are removed.
Source circuits and reference code remain unchanged.

| Factory | Native immediate calls | Original immediate calls | Disposition |
| --- | --- | --- | --- |
| lt_k | 44 | 44 | Exact ordered calls, operands, child types and fresh drawing. |
| leq | 43 | 43 | Exact ordered calls, operands, child types and fresh drawing. |
| bi_qubits_mixer | 18 | 18 | Exact shared CX/And layers in both directions. |
| sq_cmp | 7 | 7 | Exact shared And/CX/X layers in both directions. |

All twelve presentation Programs equal the executable owners. Every actual
primitive histogram equals its previously published baseline. Execution checks
cover small complete complex maps, entangled spectators, applicable small
measurement histories, forward flag ownership, clean inverse work, original
gate streams and ordinary circuit.append. Sorting append examples retain their
contiguous logical history layout with the shared components' current pool
order. Eight-bit sorting witnesses explicitly map independent three-wire mixer
allocations from dependency layers to original subtree postorder.
Unequal-width-one physical witnesses
map source direct-generator allocation to the earliest-layer native allocation
order explicitly. Dense state verification remains small-width evidence.

Actual factory primitive counts and analytical model queries are recorded in
separate fields of g2-ui-comparison-resources-20261122.json. Each row below gives
actual T/T-dagger and measurement counts, followed by the independent source
model under the same convention.

| Actual factory | Emitted T family | Emitted measurements | Model T family | Model measurements |
| --- | --- | --- | --- | --- |
| greater_than | 28 | 7 | 28 | 7 |
| equals | 12 | 3 | 12 | 3 |
| eq_k | 12 | 3 | 12 | 3 |
| clineardepthgreaterthan_example | 28 | 7 | 28 | 7 |
| Each of four half comparisons | 12 | 0 | 12 | 0 |
| lt_k | 32 | 8 | 32 | 8 |
| leq | 44 | 11 | 44 | 11 |
| bi_qubits_mixer | 8 | 0 | 8 | 0 |
| sq_cmp | 4 | 0 | 4 | 0 |

All 44 original-success arithmetic root drawings are qualified. The explicit
EqualsAConstant unsigned/QAny Join view and recursive arithmetic glyphs remain
open. The seven duplicate/permutation SDK primitive-path failures are closed by
the concrete SDK gate boundary correction documented in
SOURCE_GRAPH_REGISTER_DECOMPOSITION_AUDIT.md. Q-0120
and full G2 acceptance remain pending. This batch closes the four immediate
graph differences and does not change parent acceptance counts.

The combined module entry succeeds with 3569 JSON files, 234 documents and 3568
non-null diagrams. It publishes 68 immediate root graphs, 285 atomic interfaces,
3215 contextual native children and 38 qualified descendant operation labels.
This batch adds 162 contextual children. Concentrated catalog/Drawer regression
passes 570 cases with fourteen existing warnings, three deselections and zero
skips. Four tree-direction checks were subsequently moved from the generic wire
suite to the comparison suite without changing their assertions. Final physical
checks pass 311 cases. Final downstream checks cover sorting, duplicates,
permutations, symmetric difference and generic wire reconstruction. All 275 downstream
checks pass. Strict typing of
eleven Python files, Black, Ruff, Compiler and Integration scope, architecture,
IR, public API and whitespace checks pass.

The initial concentrated run exhausted temporary storage and exited 120. Its
retry log replaced the first log. Only confirmed reproducible task-generated
exports were reclaimed. The successful retry and isolated module export ran
serially with failed-only temporary retention and first-failure stopping.

## Complete LP root presentation, 2026-11-27

The LP catalogue entry now selects the existing complete native preparation
Program, replacing its former interface-only presentation. Its fifteen source
calls and exact return mapping retain the original constructor order. Two
unnecessary native output views were removed using the already admitted nominal
compatibility rule. Complete primitive IR and outputs are unchanged; allocation
and free order is unchanged, while expanded logical operation positions are
renumbered after removing the views.

The dedicated LP presentation module owns original allocation/free arrows,
interim/adjoint labels, Ry parameters, CZ controls, X and reflection symbols.
The shared QPE window dispatcher delegates to that owner. Fresh root scores at
widths 1, 3, 5 and 7 match source geometry and every glyph after the established
boundary-identity normalization. The actual catalogue exporter publishes the
LP body and its existing native dependencies. Symbolic LP retains the original
unresolved-body boundary. Unqualified native reflection/private descendants
remain explicitly native; this does not certify every recursive source drawing
or complete Q-0120.

The retained full-state, branch, work-cleanup and independent resource evidence
is linked in [the LP audit](LUIS_PERINA_STATE_FILE_AUDIT.md). No protected API,
new scalar opcode, source patch or pending-proposal implementation is involved.
