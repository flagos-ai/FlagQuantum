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
