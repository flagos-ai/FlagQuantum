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
