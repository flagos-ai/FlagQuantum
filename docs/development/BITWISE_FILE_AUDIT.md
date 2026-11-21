# Bitwise source family whole-file audit

The complete reference `arithmetic/bitwise.py` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` defines XorK, Xor and BitwiseNot,
five factories and two documentation specifications. The source and its original
test file are Git-blob checked without reference edits.

## Owners and complete definition disposition

| Source responsibility | Native owner and retained evidence |
| --- | --- |
| XorK dtype, constant, signature and cached bitsize/bit pattern | Existing native register dtype/encoding and ArithmeticCall metadata preserve the source word identity. `test_bitwise_resource_width.py` compares original direct edges, signed patterns, symbolic fallback and cache order. Physical binding consumes the admitted unsigned bit pattern. |
| XorK symbolic predicate, decomposition and self-adjoint | Compiler `registers/bits/bitwise.xor_constant_ir` emits X in original MSB order. The actual eight-bit factory is captured in both directions, rebound to reversed physical wires and round-tripped through IR JSON. Its full coherent action is checked against the original scalar callback, including a spectator. |
| XorK call graph | Existing arithmetic declaration and resource-width owners retain the original popcount for concrete k and full width for symbolic k. These independent models remain separate from emitted primitive counts. |
| XorK classical callback | Simulation `classical_bitwise.xor_constant_classical_value` preserves Python/NumPy values, scalar dtype casts, arrays, broadcasting and symbolic-value errors, as checked by `test_classical_bitwise.py`. |
| XorK glyph and string | Drawer `bitwise_symbols.bitwise_wire_symbol` preserves the empty title and constant label. The actual source factory retains its source title at the external capture boundary. No foreign class name is exported as a native API. |
| Xor signature, decomposition, adjoint and graph | Compiler `xor_ir`, existing metadata and resource owners retain equal-width through words, ordered CNOTs and one CNOT per bit. The complete QAny four-bit factory is captured independently in both directions, JSON restored and checked against the original callback on a coherent state. |
| Xor callback and glyphs | Simulation `xor_classical_values` preserves x and XORs y, including arrays and occupied targets. Drawer preserves x/x-XOR-y labels, including the source fallback for unknown port names. |
| BitwiseNot signature, decomposition and adjoint | Compiler `bitwise_not_ir` reuses the authoritative OnEach implementation and shared X leaf. Both captured directions match its MSB-ordered stream and original full-state callback. |
| BitwiseNot classical callback | Simulation `bitwise_not_classical_value` preserves -x-1, wrapping only unsigned and Montgomery dtypes. Opaque/signed and NumPy promotion/failure behavior remain explicit in existing tests. |
| BitwiseNot glyph | Drawer retains the empty title and complement label. Complete native score geometry is compared with the original score for all three classes, rather than comparing only individual strings. |
| All five factories and both DocSpecs | The three concrete factories pass actual capture, JSON, reordered binding, coherent append/involution and independent category/peak checks. The two symbolic factories retain their original decomposition-depth distinction described below. |

## Direct dependencies and source boundaries

CNOT and X atoms, OnEach, typed split/join bookkeeping and native score layout
retain their accepted owners. Classical execution remains in Simulation and
gate emission remains in Compiler. The drawer callback only formats existing
registers. All concrete operations have through words and no hidden scratch or
measurements. Users continue to append an operation to an ordinary Circuit.

Original symbolic Xor fails at its root decomposition. Symbolic BitwiseNot
successfully decomposes one level to OnEach(n, X, dtype), whose expansion fails
at the symbolic width. Both actual factories are rejected by physical capture
at the concrete-port admission boundary. These are distinct source behaviors.
Existing symbolic resource estimates do not establish symbolic typed Program
execution. Existing zero-width resource/capture/emitter admission differences
remain recorded in BITWISE_RESOURCE_WIDTH.md and the corresponding tests.

The original test file has twelve functions covering factory autotesters,
classical controlled XorK, XOR classical/tensor behavior, BitwiseNot tensor and
classical consistency, and three circuit diagrams. Existing native scenario
suites cover the executable/classical/model portions, and the new complete score
checks cover presentation. Full original autotester replacement and protected
static foreign conversion remain separate admission work. Q-0177 and Q-0178
therefore retain those concrete remaining gaps. The frozen `bitwise.ipynb` has since been restored from its exact Git blob.
Its complete native teaching migration is recorded in BITWISE_NOTEBOOK_AUDIT.md
under Q-0176.

## Resource accounts

The actual eight-bit XorK constant 87 emits five X, QAny four-bit Xor emits four
CX, and four-bit BitwiseNot emits four X. All own zero T, work and measurement
slots. Their independent models report respectively five/four/four Clifford
and peak eight/eight/four. No modeled Toffoli-equivalent total is substituted
for these measured primitive streams. Existing signed, controlled, grouped,
classical, cache and child-replacement evidence is reused and rerun.

Batch logs are `ftqc-reference-env/g2-bitwise-file-*.log`.

The concentrated native regression passed 1765 cases without skips. Eight
existing warnings are paired NumPy uint8/uint64 negation/subtraction overflow
from the original and native classical NOT callbacks. Both changed Python files
passed strict typing, formatting and lint. Compiler scope, architecture,
multilevel IR and protected API checks passed.

The untouched original test file reports 37 passed, four skipped and one xfailed.
The skips retain symbolic XOR decomposition, unavailable count-comparison modes
and dtype checking requiring a decomposition. The original XOR counts mode is
explicitly not enforced by its fixture and remains xfailed. A source unknown
`slow` marker warning is retained. These original modes are distinct from the
native regression and are not counted as native passes. Exact source widths,
operands, controls, feedback and all-small-branch evidence remain in the reused
consumer suites. The new score cases additionally cover negative and symbolic
constant labels without authorizing their physical decomposition.

A minimal native presentation change uses `bitwise_wire_symbol` as the existing
`register_score_data(..., register_wire_symbol=...)` callback. The complete score
test demonstrates this route alongside ordinary append execution. The callback
has no compiler, simulator or foreign-object dependency.

## Typed executable dependency closure

The complete `bits/bitwise_component.py` module now supplies native typed
Program construction for all three physical bitwise operations. It consumes a
neutral `RegisterType` and retains that exact descriptor at its entry and exit.
`xor` has x/y through ports. `xor_constant` and `bitwise_not` have one x port.
Words and their MSB-first bit positions stay caller-owned. The module emits no
allocation, measurement, reset or feedback and introduces no numerical backend
dependency. The user example in the bits package README binds the graph and
executes it with `circuit.append(operation)`.

XOR and constant XOR reuse the authoritative existing primitive emitters to
construct split, ordered leaf calls and join. Complement calls the existing
typed OnEach graph once, retaining the original source's one-level boundary.
Stored calls are counted with `program_component_call_graph`, separately from
emitted primitives and original QEC estimates. The original decomposition's
entire direct call sequence, including split/join, is compared independently.

The constant interface deliberately accepts the encoded unsigned bit pattern.
For signed words the caller encodes the signed scalar before constructing this
graph. In particular, signed -1 uses the all-ones pattern. The Compiler adds
no second dtype value codec. The existing Simulation callbacks continue to own
scalar and array numerical semantics. Positive concrete physical widths are
required, and symbolic metadata/models remain in their existing owners.

Acceptance covers opaque, unsigned, two's-complement and Montgomery words of
widths 1, 3 and 5. Original forward and self-adjoint streams match every ordered
primitive and operand with reversed physical input placement. Full source
matrices independently verify widths 1 and 3 with an entangled spectator.
All tested widths also compare coherent execution against the unchanged source
capture, preserve an additional caller-owned zero wire and restore the state
after a second append. No measurement branches exist. Sparse eight-bit patterns
0, 1, 6, 87, 128 and 255 verify actual leaf multiplicity and MSB order, including
the zero-gate path. Invalid patterns and symbolic/empty widths remain explicit
rejections. These executable graphs close the missing native construction
dependency. General symbolic foreign conversion and the complete source
autotester replacement remain open under Q-0177/Q-0178.

Measured XOR emits one CX per bit and complement emits one X per bit.
Constant XOR emits popcount(pattern) X gates. Their independent source models
agree on Clifford counts and zero T/measurement totals. Stored root NOT has one
OnEach call while its leaf graph and bound stream have width X calls. All three
accounts are retained independently. Batch receipts use
`ftqc-reference-env/g2-bitwise-component-*`.

The final concentrated family regression passes 2,667 cases with eight existing
paired source/native integer overflow warnings. The initial run exposed six
stale OnEach assertions that expected signed, ones'-complement and fixed-point
graph construction to fail. Existing approved logical admission already accepts
these graphs. The repaired tests verify their construction independently,
then bind signed/ones'-complement and retain fixed-point's physical refusal.
The relevant production owners are byte-identical to the batch baseline.
The OnEach correction is committed separately from the new bitwise module.

All three changed Python files and the README example pass strict typing,
formatting and lint. Compiler scope, architecture and multilevel IR checks pass.
The required default run reaches the established public docstring coverage
failure after 434 passes and 51 skips. All nine reported modules and the failed
policy test are byte-identical to the batch baseline. Whole-repository dependency
policy still has 688 findings, with zero findings in the changed paths.

## Existing factory-check integration

The following batch reuses `tests/unit/register_example_checks.py` for the
actual three concrete source factories. Each native factory now passes all
five existing make, decompose, counts, serialize and qtyping modes. Counts use
the independently queried source QEC declaration and compare it with bound
native instructions. Serialization uses the existing circuit codec. The
original source wrapper's XOR counts xfail remains recorded separately.
The full family regression passes 2,682 cases with the same eight overflow
warnings. The original native fixture is reused without another framework.
The two symbolic factories still have only metadata/model owners and explicit
physical refusal. Generic symbolic foreign-object conversion and roundtrip
remain open, including their shared source-testing dependency Q-0937.

## Complete factory interchange and file acceptance

The final closure maps all 29 recursive class/function definitions, all five
factories, both DocSpecs and the twelve original test functions to the owners
above. `test_bitwise_foreign_file.py` exercises every actual source factory
through unchanged source serialization and the existing Compiler library
codec at depths zero, one and two. Deterministic original bytes, constructor
arguments, register dtype metadata, decomposition connections and count rows
roundtrip without reconstruction of foreign Python classes in native code.

Both symbolic factories resolve to native typed declarations through
`resolve_component_program` with explicit caller factories. Width assignment
is explicit caller construction of a concrete dtype and declaration. It is
not a generic symbolic Program specialization engine. Widths one, three and
five then resolve through the unchanged `bitwise_word_program` owner. Every
bound instruction and reversed operand matches original capture. Complete
complex amplitudes including an entangled spectator, output aliases and
normal `circuit.append(operation)` involution agree. No workspace or
measurement branches exist. Unresolved symbolic physical construction retains
its actual rejection. The original symbolic XOR/NOT decomposition-depth
distinction remains checked in the existing audit suite.

This closes Q-0177/Q-0178 using existing owners. The generic arbitrary source
object/type-graph scope of Q-0937 remains open. The preceding sections describe
historical intermediate gaps. Symbolic models and declarations do not imply
physical execution at an unresolved width. Concrete factory five-mode checks,
classical callbacks, controls, complete score geometry and source models reuse
their accepted evidence and are included in concentrated regression.

The concentrated regression passes 2,706 cases with eight existing paired
source/native NumPy integer overflow warnings. Integration passes 17 cases.
The Cirq 1.6.1 lane passes all seven new cases. The untouched original test file
retains 37 passes, four source skips, one source xfail and its unknown slow
marker warning. Two stale existing OnEach S/T assertions still expected
three-bit unsigned fixed-point storage to fail after approved Proposal 115.
Their independent correction checks exact bound instructions and aliases.
All production owners are byte-identical to the batch baseline.

Both changed native test modules pass the established strict type-checking
configuration, Black and Ruff. Explicitly adding the complete unmodified
upstream tree to MYPYPATH separately exposes 36 implicit-reexport diagnostics
in the existing OnEach source imports. No reference annotation or import was
changed to hide them. Scope, architecture, multilevel IR and protected API
checks pass. The existing default-run docstring-policy failure is reused with
all ten implicated files verified byte-identical to the batch baseline.
Dependency policy reports 687 existing findings and zero in changed files.
Receipts and logs use `ftqc-reference-env/g2-bitwise-foreign-*`.

Interchange and explicit construction add zero gates. Actual XOR emits width
CX, NOT emits width X and the eight-bit constant-87 factory emits five X.
All own zero workspace, measurement and T. Original independent QEC models
report corresponding Clifford multiplicities. A symbolic n-Clifford formula
is retained separately from the unavailable unresolved physical stream.

## Recursive bitwise presentation contract, November 22

The next presentation closure covers complete concrete Xor, XorK and BitwiseNot
bodies together with their OnEach(X) direct dependency. The domain owner remains
Compiler bits, while the catalog callback belongs to examples/algorithms. Word
ports retain their supplied dtype, scalar shape and MSB-first Split/Join order.
All logical words are THRU, no clean quantum work is owned, and no measurement
or classical slot is introduced. Declared XorK constants supply their encoded
pattern. Constructor matching must retain the complete reachable body before
reusing a source glyph. Unknown bodies retain the native generic presentation.
Existing physical, coherent, ownership and independent resource acceptance is
reused. Symbolic declarations and original refusal depths remain explicit.

The domain child provider now matches the existing Xor, XorK, BitwiseNot and
OnEach(X) constructors. XorK qualification moves out of the addition provider
into this bitwise owner, eliminating its former cross-domain presentation
placement. AddK's completed recursive checks continue through the same Arithmetic
exporter callback. Concrete scalar words retain their nominal dtypes. The
whole reachable definition is compared before its glyph supplier is selected.
Changing a repeated X to Z rejects both the complement and OnEach labels.

Fresh source graphs and score payloads match for the complete three-class
bitwise file across widths 1, 2, 3 and 8 with opaque, unsigned and signed words.
Signed constants use -1 and the other encodings use the largest represented
unsigned pattern. Symbolic complement still has a successful immediate graph
whose OnEach child refuses decomposition at unresolved width. The existing
symbolic XOR root refusal and zero-width admission differences remain recorded.
The focused combined word/addition checks pass 70 cases. A separate physical
regression passes 411 cases, including source blob checks, full complex actions,
normal append, reversible aliases, child replacement and independent resource
widths. No measured branch is introduced by these bitwise operations.

The combined catalog qualifies 88 contextual descendants, adding 27 matching
word definitions to the preceding addition closure. Its identities and graph
edges retain 3569 JSON files, 234 documentation files, 3568 diagrams and 3215
contextual child nodes. Counts are contextual definitions, not distinct source
classes. The next comparison-file intake retains 62 exact recursive graphs,
of which 29 non-root graph contexts still need glyph qualification. Those
remaining contexts are explicitly inventoried by source type, typed ports and
owner. They comprise tree comparison, conjunction/control and sign-view owners.

Actual constant-87 XorK emits five X, width-four Xor emits four CX and width-four
BitwiseNot emits four X. All own zero clean workspace, measurement slots and T.
Independent source models retain their corresponding Clifford multiplicities
and width-dependent peak storage. Presentation qualification adds zero physical
operations. Symbolic estimates remain separate from unavailable unresolved
physical streams.

The initial complete catalog export failed ENOSPC. Its original log is retained
as g2-word-child-catalog-disk-failure-20261122.log. After removing current failed
outputs, the older pytest native/source export directory was confirmed idle by
lsof and identified from the existing catalog protocol test and durable source
profile log. Only those reproducible generated outputs were removed, releasing
336567580 logical bytes from the older directory. Reference checkouts and
working trees were preserved. Cleanup evidence is recorded separately.

The bounded retry passes the full 636-case concentrated catalog regression,
with three existing deselections and fourteen existing notebook escape warnings.
Strict typing of five changed Python files, Black, Ruff, integration scope,
architecture, multilevel IR, protected API baseline and whitespace checks pass.
The complete bitwise-file source-call and fresh-drawing acceptance is closed.
The explicitly recorded zero-width, symbolic physical and other Arithmetic
presentation boundaries retain their own separate acceptance criteria.

## Symbolic complement dependency graph — 2026-11-22

The existing symbolic BitwiseNot graph is now reused by symbolic Negate and
SubtractFrom. It retains the source-success OnEach(X) call and its original
unresolved-width decomposition refusal. Literal whole-definition matching
qualifies that stored complement before applying bitwise/repeated-X glyphs.
Finite execution rejection for unresolved widths remains with the binder.

The combined complete bitwise/negation/subtraction traversal covers all twelve
source-success catalog roots. It retains 34 exact ordered graphs, 22 source
non-root class labels and 105 original atomic/refusal contexts. Twenty fresh
non-root drawings match exactly. Two Add(width=64) descendant drawings raise
the original frozen IndexError and retain that exception as separate evidence.
The symbolic XOR and malformed source CompositeBloq root refusals retain their
existing dedicated checks. See SUBTRACTION_FILE_AUDIT.md and the durable
`g2-symbolic-word-*20261122` receipts for complete batch acceptance.
