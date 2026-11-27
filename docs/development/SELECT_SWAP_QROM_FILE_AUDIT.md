# Select-swap QROM source audit

The implementation, notebook and tests match the pinned Qualtran
v0.7.0 inventory at `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Native evidence |
| --- | --- |
| Q-0483 tutorial | `register_select_swap_qrom.ipynb` covers the two concrete and four symbolic source examples, four independent formula queries, the default and explicit-block call graphs, and both bank policies through `circuit.append(op)`. |
| Q-0484 implementation | The emitter retains the source bank restoration order. The distinct immediate calls and batched QROM data are exposed independently of the aggregate QEC model. Current complete-source closure and retained source limits appear in the November 28 section. |
| Q-0485 tests | The full original test file reported 27 passes and two expected failures for the original unenforced count checks. Native checks reuse the source gate and channel tests, and additionally compare immediate graph children and their parameters. |

## Immediate calls and execution

The two QROM calls share one identity because the source QROM is
self-adjoint. Its shape-only declaration keeps high selector widths,
control count and the bank-shaped targets. Concrete declarations use the
shared high-then-low packing, including zero padding at the final block.
XOR calls retain one target width each. A positive block exponent adds
two distinct route nodes for each target width, forward and adjoint.
Equal-width dataset contributions accumulate. A clean bank uses one
copy and one call in each route direction. A borrowed bank doubles
those three counts to cancel the original bank's contribution.

The direct comparison includes default and explicit blocks, fully blocked
and mixed-axis declarations, symbolic block sizes, repeated widths and
both bank policies. The executor continues to use the existing measured
QROM and signed route streams. Existing tests verify the original
operations, operands and feedback, the complete complex channel across
small measurement histories, arbitrary nonzero destinations, a
spectator correlated with borrowed bank data, and zero work. The new
tutorial exercises enabled and disabled controls in one coherent input.

## Count comparison

Both tutorial modes use the same four-row table, a block length of two
and seven wires. The clean mode emits 16 T/T†, 29 CX and two
measurements. The borrowed mode emits 24 T/T†, 40 CX and two
measurements. The source model reports eight explicit T and two AND
for the clean mode, and 16 explicit T and two AND for the borrowed
mode. Both models declare two measurements and a seven-qubit peak.
The Clifford model counts are 51 and 94 respectively.

## Historical boundaries before November 28

The source can declare mixed or fully blocked call graphs, while its
executable graph loses a required low or high selector in these cases.
This audit verifies the declarations independently and keeps the
existing execution rejection. Shaped target arrays are outside the
original SelectSwapQROM interface. As in the original generic control
path, QROM lowering that would require a controlled non-through AND or
AND adjoint has no supported decomposition. Circuit diagrams use the
native renderer, with the original Cirq spacing covered by the
unchanged reference test.

The 17-cell reference notebook and three native lookup tutorials ran
in independent kernels. The native regression passed 265 tests with
no skips. Strict typing, formatting, team scope, architecture, IR
contracts and public API checks passed.

See [gate and branch validation](REGISTER_SELECT_SWAP_QROM.md).

## Actual graph import through Partition

The live graph importer now sends original `Partition` leaves to the existing
Compiler `partition_component_program` owner. Original Partition's Cirq hook
returns `None` with new register views. Treating that return as a circuit caused
ordinary blocked SelectSwapQROM graph imports to fail before the lookup body.
The adapter now preserves source part order, shapes and direction without
introducing gates or a second partition implementation. A standalone shaped
Partition and its inverse both preserve the exact physical views.

Actual four-row, two-bit target source graphs import and execute through the
owned Program binder, JSON roundtrip and `circuit.append(op)`. Forward and
adjoint cases compare all populated complex amplitudes of coherent addresses,
nonzero targets and a correlated spectator. The dirty forward case includes
all sixteen states of the borrowed four-bit bank in the same coherent input.
The bank is restored and remaining scratch is zero. The original forward
allocates dirty bank words then declares clean Free nodes; its adjoint allocates
clean words and declares dirty Free nodes. The importer retains those original
lifetime declarations, and the binding and numerical checks distinguish them.

| Original graph fixture | Actual imported primitives | Source model account |
| --- | --- | --- |
| Clean bank, either direction | 116 operations including 16 T/T-dagger, 32 CX, 32 H and zero measurements | 16 total T; direct model queried separately |
| Dirty bank, either direction | 220 operations including 32 T/T-dagger, 56 CX, 64 H and zero measurements | 32 total T; direct model queried separately |

These counts belong to the `[0,1,2,3]`, two-bit-target fixture. They do not
replace the earlier one-bit-target, measured-QROM tutorial counts. For the new
fixture there are no measurement histories to enumerate. Existing measured
lookup and channel tests remain in the combined regression.

The actual fully blocked one-dimensional graph still fails finalization for a
missing `selection` Soquet. The mixed `(0,1)` two-dimensional block still fails
with mismatched SwapWithZero register types, and the fully blocked two-axis
graph lacks `selection0`. Tests compare the original decomposition errors with
the importer errors. Those are preserved source boundaries, so Q-0484 remains
open after this dependency repair.

The same import branch covers the original `_PartitionBase` family, including
concrete `Split2` and `Join2` and their adjoints. Four direct view checks prove
that those subclasses retain zero gate and zero workspace behavior through
the same owner. This avoids leaving sibling no-operation hooks broken.

The batch passed 136 concentrated tests, 29 integration tests, strict typing,
Black, Ruff, Ecosystem scope, architecture, IR and public API checks. The
Cirq 1.6.1 lane passed 24 live-graph and partition cases with native Kickmix
present. No source files were modified.


## Complete source and direct dependency closure, November 28

The remaining source-success constructor gap is now implemented. The source
accepts trailing zero block axes when all block exponents are zero. Those axes
remain part of the bank and QROM target shapes even though the source zips them
against fewer data/selection axes. Rejecting the layout early was a native gap.
`SelectSwapDeclaration` now preserves it without weakening `QroamCleanTable`.

| Complete source responsibility | Authoritative native owner |
|---|---|
| `find_optimal_log_block_size`, helper and default blocks | Existing optimal-block and symbolic helpers; `SelectSwapDeclaration` consumes their source objective, including clean-factor-two/dirty-factor-one behavior |
| Base construction, scalar-target check and symbolic state | Composition of the accepted immutable `QromDeclaration` with source block metadata and bank policy |
| `with_data`, `with_log_block_sizes` | Existing base replacement checks, original zipped block bounds and retained block selection; snapshots do not mutate caller data |
| Block sizes, batch shape, high widths and padding | Source zipped arithmetic and fresh `padded_arrays`, preserving original dtype and deferred failures |
| `batched_data_shape`, `batched_data` | Source-shaped integer batch assignment, including original float truncation and broadcast failures |
| `qrom_bloq` | Shared `QromDeclaration.from_shape/with_data`; source-symbolic metadata retains an unknown-data child |
| `swap_with_zero_bloqs`, immediate calls | Shared immutable route/copy identities and `SelectSwapDeclaration.calls`; the existing dense-table call query delegates to this authority |
| QEC and static width | Shared arithmetic, QROM and approximate-swap query owners; modern and legacy profiles remain separate from emitted primitives |
| Allocate, partition/unpartition, load, route, copy and both composite builders | Existing shared blocked lookup stages and `select_swap_qrom_program`, now consuming either a strict clean table or source metadata |
| Actual constructors and adjoints | Shared pickle-free metadata reader and `select_swap_factories`; original inverse allocation/free flags remain unchanged |
| Wire symbols, diagram hook and documentation factories | Existing native glyph/score owner and all six original examples; Cirq spacing remains a separate presentation |

No source code is rewritten. Stored constructor libraries at depths 0, 1 and 2
are resolved without executing SDK constructors. The new native declaration
retains the actual QROM bank shape, instead of discarding extra singleton axes
to make a strict dense table fit. The same Program binder and append protocol
remain in use. Generic QROAM consumers still use their existing strict tables.

Fresh forward and adjoint checks cover one and three trailing zero axes, zero
and one control, both bank policies, actual source primitive operands and
latest-measurement feedback. Comparable enclosing calls and the unreserved
logical QROM recipe are compared separately: the reserved physical QROM frame
has extra zero-cost storage views. Physical primitives are compared again after
binding that frame. Complete complex branch vectors include arbitrary dirty
destinations, a bank-correlated spectator, restored borrowed data and zero
clean work. Every small raw measurement history is enumerated. Larger existing
routing and multidataset certificates are reused rather than relabeled as new
execution work.

### Independent resource accounts

The following `[0,1,2,3]`, two-bit-target fixtures use logs `(0,0)`. The source
static quote excludes the external spectator. The full 12-row receipt
`g2-select-swap-closure-resources-20261128.json` also records both directions,
allocation/free flags, and the scalar-width counterexample.

| Forward policy / controls | Measured T-family / CX / measurements | Peak clean / borrowed allocation | Source AND / Clifford / measurements / static quote |
|---|---|---|---|
| Clean / 0 | 16 / 38 / 4 | 3 / 0 | 4 / 22 / 4 / 7 |
| Borrowed / 0 | 16 / 40 / 4 | 1 / 2 | 4 / 24 / 4 / 7 |
| Clean / 1 | 24 / 52 / 6 | 4 / 0 | 6 / 22 / 6 / 9 |
| Borrowed / 1 | 24 / 54 / 6 | 2 / 2 | 6 / 24 / 6 / 9 |

The inverse of a dirty forward graph allocates a clean bank because the source
forward Free uses its default flag. Tests retain that lifetime distinction;
an inverse is not certified with arbitrary caller-supplied dirty initial banks.
Forward borrowed banks return their original coherent data, not zero.

### Explicit source limits and separate native behavior

- A nonzero leading block followed by extra zero axes, such as `(1,0)` on a
  one-dimensional table, fails the original integer batch broadcast. Both
  protocols retain that failure. It is distinct from the implemented all-zero
  extra-axis case.
- An extra nonzero axis can have defined direct models but a missing low
  selector in its executable graph. Fully blocked layouts and mixed axes also
  retain the previously reproduced missing-selector/type failures. The live
  graph importer propagates original errors; a stored native constructor can
  preserve its header and fail body binding before allocation.
- Stored literal compilation checks source selector retention and exponent
  bounds before bank allocation, including oversized exponents. Existing strict
  table programs remain a separate native path; they do not certify a failed
  literal source graph.
- Boolean target widths can have defined word-count models while original
  integer XOR loading fails. The model quotes CNOT multiplicities, and physical
  loading retains the refusal; Boolean selectors/controls remain separate.
- Negative control widths can have source graph/model declarations but cannot
  become a valid physical port. Scalar-data padding, truncated block lists and
  zero-width array targets retain their original failed physical paths.
  The QROM dependency now preserves the array-target cost failure as well as
  its separate successful zero-width scalar call/model protocol.
- The source one-row uncontrolled model undercounts physical width: a two-bit
  target with its two-bit bank needs four physical bits, while the original
  static quote is three. The empty source component has logical calls and a
  static quote of -1. These quotes are preserved, not used to allocate work.
  This batch adds no public zero-wire Circuit or scalar-carrier contract.
- Source generic control still has no decomposition for non-through controlled
  And/And-adjoint leaves. Original nominal join mismatches, source base array
  identity/cache defects, and unsupported object/shape wire encodings remain
  recorded in their owning dependency audits. No alternative native result is
  presented as a literal source decomposition.
- Native glyphs retain source labels with native score spacing. Default block
  choice follows the implemented source objective; this is not an independent
  proof of the prose or paper optimum. No performance or cross-platform claim
  is made.

The complete batch passed 687 concentrated and 256 independent Cirq 1.6 cases.
The unchanged original file passed 27 cases with its two expected failures.
The local integration lane also passed 84 cases. Strict checking of all ten
changed Python files, Black, Ruff, architecture,
IR, protected API, required checks, Cirq contract, Compiler scope, English,
links and whitespace passed. Ten actual optional/mandatory absence scenarios
preserve the original import error, and native compile/append/JSON/model work
with Qualtran/Cirq/Kickmix blocked. All 119 frozen source/dependency blobs are
unchanged. The existing 570 dependency findings have zero affected consumers.
Final publication and ledger receipts determine whole-parent acceptance.
