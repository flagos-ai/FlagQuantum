# Coherent alias preparation source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The notebook, implementation, and
test blobs are `081add39d9e5a13a0e215483e429b6f0d9e3ff84`,
`4850ef95d9ebf58b0e062505a3482db2f5307443`, and
`c5141c0e4beadb9a890dc3fc035601252a6c0454`.

## Native owners

| Source responsibility | FlagQuantum owner |
| --- | --- |
| Dense probabilities, sparse dictionary, and dense-to-sparse input | The concrete alias table factories preserve input row order, integer alternate and keep data, and normalization. |
| Table-free dense and sparse constructors | `AliasPreparationShape` owns both precision formulas and resource declarations. The dense formula depends on normalization. The sparse formula uses the full domain with normalized tolerance. |
| Signature and logical junk | The alias PREPARE emitter binds selection then sigma, optional sparse row, alternate, keep, and comparison bit. Words are big-endian. The junk is a logical output and can remain entangled with selection. |
| Composite and adjoint | Shared uniform preparation, sigma Hadamards, QROM, measured comparison, and full-word controlled swap. The adjoint reverses their order. |
| Temporary lifecycle | QROM and comparison share cleared temporary wires and classical slots. PREPARE restores temporary work. PREPARE followed by its adjoint restores the selection and logical junk. |
| Independent estimates | Shared uniform, QROM, and comparison models declare their gate categories and live-width peaks. The alias model composes them with logical bystanders. |

The native [tutorial](../../examples/notebooks/state_preparation_alias_sampling.ipynb)
retains both source sections, all five original factories, their signatures,
and the concrete depth-one call graphs. Its final cell appends the dense and
sparse examples with `circuit.append(op)`.

## Verification

The two concrete sparse input forms produce the same table and operation. The
original dense, dictionary-sparse, and diagram fixtures match the source
component order, wire order, and primitive streams. The native interpreter
checks the complete complex output and clean work against the unchanged source
Cirq action, and runs the emitted adjoints to restore all wires. Each larger
fixture uses three seeded measurement histories. Existing small fixtures retain
their exhaustive measurement histories. The original Ising probability grid,
uniform low-precision case, and both sparse probability cases also run through
the native append entry. The original source suite passed 17 cases including
its notebook, with three documented skips and three expected failures.

The two exact symbolic factory declarations match modern and legacy resource
categories and logical width in both directions. The original T-complexity
test expands the model's CSwap and And categories. Its sparse Clifford expression
includes ten additional Cliffords for each declared CSwap and And under that
conversion.

| Original fixture | Emitted T/T† | Emitted CX | Emitted measures | Reserved wires | Source CSwap | Source And | Source Clifford | Source measures | Source logical wires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense, four rows | 42 | 117 | 7 | 19 | 2 | 7 | 80 | 7 | 19 |
| Sparse, four support rows over nine indices | 56 | 144 | 7 | 25 | 4 | 7 | 91 | 7 | 25 |

## Typed preparation ports

The dense and sparse declarations share the native PREPARE contract.
The selection register is bounded unsigned with the original domain
size as its iteration limit. The sparse support word appears between
`sigma_mu` and `alt` only when its width is nonzero. Other empty junk
words are omitted. The comparison result uses a single qubit, and all
declared ports preserve the source through direction and order.

The same declaration accepts a table or symbolic dimensions and
supplies the L1 normalization. The concrete adapter derives its named
layout from the declaration before binding the existing emitter.
The five original factories and the dense and sparse singleton
signatures were checked against the pinned source. The notebook now
renders those same declarations. The gate and resource evidence above
covers the concrete adapter routes.

## Complete implementation definition map

The pinned source has two preparation classes, five factories, two DOC lists and
one array/shape comparison converter. No foreign type enters native algorithms.

| Complete original responsibility | Native owner and evidence |
| --- | --- |
| `_data_or_shape_to_tuple` | Immutable tuple rows in `AliasSamplingTable`, shape-only records in `AliasPreparationShape`; actual source arrays remain at the optional import boundary. |
| Dense constructor, positive mu, one selection register | Table validation and typed one-dimensional declaration. |
| Dense `from_probabilities`, `from_n_coeff` | `alias_sampling_table`, `AliasPreparationShape.from_precision`; independent precision and normalization tests. |
| Dense n/normalization and sigma/alt/keep/selection widths | Native table/shape properties and typed PREPARE signature. |
| Dense junk/QROM/body/call graph | `alias_preparation_ir`, shared uniform/QROM/comparison/swap owners and independent source model. |
| Sparse constructor, dictionary/dense/symbolic factories | Sparse table factories and `AliasPreparationShape.from_sparse_precision`. |
| Sparse n/support/normalization, widths and junk | Native support rows, typed ordered selection and logical junk layout. |
| Sparse QROM and complete body | Three-table lookup, uniform support address, comparison, controlled exchange in `alias_preparation_ir`. |
| All five examples and DOC lists | Existing catalogue and original teaching notebook; both symbolic factories now have actual bound-source execution evidence. |

## Actual source binding and complete logical operators

`test_alias_source_binding.py` resolves the constructor fields of both original
symbolic factories using their actual `attrs` classes. Dense binding uses a
custom `index_word` selection name; sparse binding uses the original `selection`
name. One-bit probability words keep the full source PREPARE signature.

The source is captured in both directions, serialized only as owned native
`CircuitIR`, restored and bound through the existing `BlackBoxPrepare` emitter.
The native comparison uses `circuit.append(AliasPreparationExample(table))`.
Every logical input column is compared, including arbitrary nonzero sigma,
alternate, keep, support and comparison-flag inputs. Full complex output and
actual inverse recovery agree; only private scratch is required to be zero.
Logical junk is retained and can remain entangled with the sampled selection.
It must not be treated as clean temporary workspace.

The existing erasure certificate proves channel identity for every raw history;
separate complete-column checks prove final private cleanup. Both physical
source directions are captured independently, without reversing measured IR.
Independent concrete source models agree with native models. Existing primitive
stage gate, operand and feedback comparisons are reused from
`test_alias_preparation.py`; the new whole-object tests also preserve exact gate
histograms.

| Resolved table, either direction | Instructions | Explicit T/T-dagger | CX | Measures | Physical wires | Source CSwap / And / Clifford / measures / peak |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense, four rows, mu=1 | 118 | 26 | 51 | 3 | 9 | 2 / 3 / 28 / 3 / 9 |
| Sparse, two support rows over four indices, mu=1 | 76 | 18 | 36 | 1 | 10 | 2 / 1 / 22 / 1 / 10 |

The source-capture and standalone native instruction histograms match. Explicit
T counts include the chosen exact controlled exchanges; source models keep
CSwap as a separate category. Runtime/measurement statistics are not substituted
for source model categories.

## Source register admission and remaining boundaries

Q-0700 no longer lacks actual executable symbolic binding. The complete accepted
paths use ordinary existing emitters; there is no new registry or implementation
of alias preparation.

- Dense `BQUInt` selection supports a custom register name. Replacing its dtype
  with `QUInt` or `QAny` makes source `n_coeff` zero and raises the original
  `ValueError: math domain error` during decomposition. This is an original
  domain-query limitation, not a missing native unsigned-gate implementation.
- Sparse source `QUInt` and `QAny` selection also declare `n_coeff == 0`, but their
  actual body uses physical width and does execute. Both are covered by the same
  complete logical-operator, all-history, inverse, cleanup and JSON checks as the
  bounded selection. Their inconsistent declared domain is recorded separately;
  no source model parity is claimed for that zero-domain declaration.
- Sparse source body hardcodes `selection`. A renamed selection register raises
  `KeyError('selection')` in both direct decomposition and actual import. The
  unchanged reference is not silently repaired or renamed during capture.
- Source sparse singleton target/call failures and symbolic two-row width-query
  failure remain separate from the valid concrete two-row execution above.
  Native constant-singleton handling and explicit model rejection retain their
  existing tests.
- Source serialization expected failures and the two original strict typed-edge
  skips remain recorded. Native snapshots retain ordered physical layouts; the
  foreign source object is not stored in native algorithm state.

Q-0700 remains open for those original admission, singleton and symbolic-model
boundaries. This batch adds no parent or child progress item.

## Current batch validation

Seven focused cases passed in 8.59 seconds: four complete bound logical
operators (dense bounded and sparse bounded/unsigned/opaque), two preserved
dense dtype failures and the preserved sparse custom-name failure. All admitted
operator cases include both directions, every logical column, raw-history
channel identity, private cleanup, JSON restoration and exact emitted counts.
The concentrated alias/PREPARE/import/LCU regression passed 125 cases in 28.69
seconds. Strict mypy on the added test, Black, Ruff, Algorithms scope,
architecture, multilevel IR and protected API checks passed. Existing source
notebook and CPU integration evidence is reused; no production implementation
or public contract changes in this batch.


## Complete faithful source-file closure — 2026-11-28

Q-0700 is accepted under the approved G2 faithful-migration rule after reviewing
the complete two-class definition map, all five factories, both documentation
specifications, array/shape conversion and direct preprocessing/quantum owners.
Earlier open statements based solely on original dtype, singleton, codec and
model defects are historical. Two actual native invocation gaps were fixed.

Sparse selection admission now uses the existing source-ordered word consistency
predicate against the QROM's unsigned target. Signed integers, Montgomery words
and pure integer/fractional unsigned fixed point execute upstream but were
previously rejected by the adapter. One-bit signed and GF(2) words also retain
the source single-bit compatibility rule. No Core type, field arithmetic,
control rule or public entry changes. Mixed/signed fixed point, two-bit ones
complement and nonbinary field targets retain the original downstream tensor
failures and adapter refusal. Their outer source roots can exist; this does not
make the failed physical recipe executable. Sparse selection still uses the
source hardcoded name. No missing iteration bound is invented.

Known lookup data with symbolic mu also has a defined original five-call root
but no complete source tensor. The native composer previously attempted physical
QROM binding before returning that root. It now retains a QROM declaration and
the constructor's real mu dispatch. A one-bit signature canonicalizes its type,
so the raw lookup width preserves the distinction between Python 1 and SymPy
Integer(1). Both symbolic cases retain original H/lookup/comparison refusals;
no unresolved or source-refused physical execution is invented. Actual depth-zero
source wires, both directions, ports and producer/operand/result edges agree.

Six executable nominal cases compare complete original complex operators, exact
source forward/inverse root edges and the accepted unsigned primitive/event
streams. Every small forward and inverse raw history preserves a coherent
spectator and returns private work to zero. Selection and all logical junk may
be nonzero and entangled; they remain caller-owned outputs. Representable source
constructor depths zero/one execute identically. Montgomery modulus loss and
field-codec limitations remain separate; direct typed descriptors are retained.
Native construction and append work with source SDK imports blocked.

Four symbolic-mu dense/sparse contexts compare complete stored roots and actual
source serialization while retaining physical refusal. Validated probability
factories reuse the accepted source-ordered precision, cumulative rounding and
donor algorithms; raw preprocessing retains its separately verified source
exceptions and warning order. Prior complete grids, Ising, notebook, symbolic
shape/model and exact primitive/feedback evidence is reused.

Original dense unbounded-domain errors, sparse renamed/singleton failures,
symbolic two-row model and Shaped codec failures, model expansion conventions
and strict source typed-edge skips remain explicit. Historical reflection
inventories do not establish reflection correctness: current inverse preparation
recipes refuse implicit disposal of logical signal junk. Generic reflection,
codec, workflow and unapproved scalar parents do not close here. No reference
source or original assertion was changed.

Fresh two-bit nominal cases emit 18 T/T-dagger, 36 CX and one measurement on
ten wires, with two touched/live private bits. The source model separately
declares two CSwap, one And, 22 Clifford, one measurement and peak ten. One-bit
cases emit 11 T-family, 26 CX and one measurement on eight wires versus one
CSwap, one And, 20 Clifford and peak eight. Extra spectator wires are separate.
Receipts use `g2-uniform-alias-full-closure-*20261128`; all source defects and
unknown-width intake results remain in their own external receipts.

Final batch acceptance passes 463 concentrated cases, 32 relevant local
integration cases, 59 independent Cirq 1.6.1 constructor/nominal cases and the
final 12-case independent symbolic/nominal grid. Three Python files pass strict
typing, Black and Ruff. Compiler scope, architecture, multi-level IR, required
checks, public API baseline, English, links and whitespace pass. Thirty-eight
complete frozen source/asset/dependency blobs and actual Qualtran 0.7.0/Kickmix
0.1 Simulator are verified. Nine fresh inventory rows retain reserved capacity,
touched/live private work and independent source models separately. Ten actual
optional/required reference-absence scenarios preserve skip versus original
exception behavior. Global dependency findings remain 570 in 362 files, with
none in the added consumer; workflow parents remain open. No source patch,
checker waiver, selection exclusion, CI wait, G3 work or merge occurred.
