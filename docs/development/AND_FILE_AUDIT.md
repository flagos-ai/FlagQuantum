# And and MultiAnd whole-file audit

The source files, original tests, and 30-cell notebook are pinned to
Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
Compiler owns the shared four-T And, measured erasure, and literal
MultiAnd ladder. Simulation owns classical values and separately
the five-axis conjunction tensor. Drawer owns the musical score and
the classical dependency graph.

Forward controls are ordered before prefix outputs. The source
signature exposes the middle prefixes as `junk` and the final prefix
as `target`. These are logical outputs, with the adjoint consuming
all of them before releasing their wires. Native examples use
`circuit.append(op)`, caller-owned quantum wires, and explicit
classical feedback slots.

## Notebook and test file acceptance

Q-0544 closes with `examples/notebooks/register_and.ipynb`. All 30
source cell positions and 17 code cells are retained. Both signatures
and call graphs, Clifford+T compute and measured erase, six mixed
controls, state/effect contraction, and both classical examples
execute through existing native owners. The independent native
kernel blocks reference imports and produced six SVGs. The source
notebook also ran independently.

Q-0546 closes with the source test dispositions and corresponding
native evidence:

| Original tests | Native evidence |
| --- | --- |
| And/MultiAnd example and symbolic resource tests | Ordered signatures and symbolic and concrete call graphs; native symbolic T formula and concrete legacy counts |
| Truth tables, unknown target, bad adjoint and input validation | Existing value callbacks and physical predicate admission, plus state/effect contraction compared on all eight notebook inputs |
| Inverse, MultiAnd adjoint and AndIdentity | Existing all-small-history erasure and the complete complex test state through both AndIdentity branches |
| Random classical simulation | All 20 cases from the original seeded fixture, including every prefix output |
| Musical score and connected diagram | Four polarity glyph sets, side and lane geometry, existing MultiAnd score comparisons, and notebook graph |
| T complexity | All four And polarities and three original MultiAnd lengths/polarity cases compared with the original legacy T and Clifford totals |

The unchanged source suite reports 36 passes and three skips. Its
atomic And has no composite decomposition, call-count body, or
decomposition-dependent type check. Native tests compare its
physical Clifford+T stream and its atomic tensor separately.
The mixed six-control example matches both native primitive streams
gate for gate, including every feedback slot. It emits 20 T/T†
during compute and five measurements during erase. All 32 forced
measurement histories reproduce the original complex input with
all five prefix wires zero, at the expected per-branch probability
of 1/32. The source model separately declares five And calls and
six polarity X calls.

The remaining per-control symbolic metadata now has a Drawer owner in
`conjunction_symbols.py`. Atomic And uses the original bool(control) circle
rule. MultiAnd uses equality to one, which can produce a different circle for
an unresolved symbol. Its junk labels retain all supplied index components.
The And circuit labels switch both controls to their original strings when
either control is symbolic, including concrete SymPy integer expressions.
Forward, consumed output and through-port direction boxes follow the source.
An unresolved symbolic control count retains the source refusal to index its
individual values. Concrete count metadata normalizes to positive controls.

The typed register-score consumer accepts these callbacks through its existing
interface and retains complete source line, connector, port and glyph records
apart from native port identifiers. Existing ConjunctionCall, signature and
resource-query owners retain symbolic control elements without introducing
another component representation. Unresolved values remain declaration and
presentation metadata. Native physical execution requires concrete binary
activation values. This is an explicit physical admission boundary.

## Whole implementation definition map

The unchanged implementation blob is
`2e6efeb53938ac43f6f0330d79140f04138861eb`. Its 29 top-level definitions and
direct class methods are recorded in the resource receipt.

| Source surface | Native owner and acceptance evidence |
| --- | --- |
| And declaration, signature, adjoint and atomic decomposition refusal | Existing `and_signature`, `ConjunctionCall`, compute/erase direction and root tensor atomic-provider boundary |
| And classical action | `simulation.classical_and.and_classical_values`, including exact control identity, ignored forward target and dirty erase failure |
| And tensor | `simulation.tensor_network.conjunction`, exact five-axis data and directional target connection |
| And wire symbol and circuit diagram | `drawer.conjunction_symbols`, original symbolic controls, all concrete polarities, both directions and real register-score replacement |
| And string and has-unitary declaration | Existing native names and compute/erase direction. Logical compute is a zero-target isometry. The physical compute circuit has a unitary extension, and erase uses measurement |
| And register decomposition and frozen Clifford+T circuit | `and_compute` and measured erasure owners, gate/operand/feedback reference comparisons already accepted |
| `_and_bloq`, `_multi_and`, `_multi_and_symb` and documentation constants | Existing native notebook and catalog examples, signature and symbolic resource declarations |
| `_to_tuple_or_has_length`, MultiAnd validator, n_ctrls and concrete_cvs | Existing concrete tuple/unknown-count `ConjunctionCall` identity, native count normalization and physical ladder validation. Unknown counts cannot supply concrete values |
| MultiAnd signature, classical action, power and adjoint | Existing `and_signature`, raw `multi_and_classical_values`, forward and exact measured inverse consumers |
| MultiAnd recursive tree and both decomposition entries | Shared `multi_and_stages` and `multi_and_ir`, real typed predicate programs and per-wire original gate comparisons |
| MultiAnd wire symbol and string | New callback plus existing native score title and concrete/symbolic width declarations |
| MultiAnd build_call_graph | Existing `ConjunctionCall` recipes and QEC query, all seven categories compared for symbolic elements and both profiles |

The raw classical helpers retain their source numerical operations, whereas
physical binding separately validates binary activation and wire ownership.
Source private Python classes remain at the reference-test boundary. Native
entry points are placed with their domain owners.

## Resource and verification receipts

Presentation and metadata add zero quantum instructions. Eight numerical
resource witnesses list observed primitive counts separately from original QEC
model categories in `g2-conjunction-symbols-resources.json`. Prefix bits are
logical outputs for compute and consumed inputs for erase. They are not
additional hidden workspace. Existing whole-file evidence compares the mixed
six-control forward and inverse streams, all 32 measured histories, complete
complex amplitudes, clean-work zero and independent T/QEC models.

This follow-up runs the conjunction, raw classical and Drawer regression plus
strict typing of both changed Python files, Black, Ruff, Compiler scope,
architecture and internal IR checks. Dependency policy and previous SDK
integration evidence remain separately recorded. Persistent outputs use the
`g2-conjunction-symbols-` prefix. The original reference implementation and tests
remain unchanged.

The completed follow-up regression passes 643 cases without skips or warnings.
Strict mypy passes both files. Black, Ruff, Compiler scope, architecture and
internal IR checks pass. The dependency scan records 687 existing repository
findings and zero batch findings. The previous 96 SDK integration cases are
reused because SDK and physical lowering production code is untouched in this
presentation batch. The default-tier public docstring example registry failure
remains an existing repository limitation. These callbacks add no executable
docstring examples. Q-0545 closes with the complete definition map above and
its explicit symbolic physical admission boundary. G2 continues.
