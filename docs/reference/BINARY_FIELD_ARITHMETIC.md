# Binary-field register arithmetic

The internal register compiler accepts big-endian coefficient words: the last
wire is the coefficient of 1. The caller supplies the irreducible polynomial as
an integer including its highest coefficient. For example, `0b1011` denotes
`x^3 + x + 1`. The compiler validates irreducibility without importing a reference
package. Operands and workspace must be disjoint.

`field_multiplication.gf2_multiplication` computes into a clean target. It ports
Qualtran v0.7.0's quadratic `GF2Multiplication` and `GF2MulViaKaratsuba`, including
`BinaryPolynomialMultiplication`, `MultiplyPolyByOnePlusXk`, `GF2MulK`,
`GF2ShiftRight`, and the first-pivot PLU CNOT network. Scratch is caller-owned,
starts in zero and returns to zero. Insufficient scratch raises `ValueError`.

The returned `FieldDecomposition` retains the upstream logical output wire map.
`bind_field_outputs` materializes any virtual permutation for a consumer needing
fixed physical output wires. These extra SWAPs are reported separately. This
preserves the distinction between a free graph edge relabel and a physical gate.

Run the user workflow with:

```bash
python -m examples.algorithms.binary_field_arithmetic
```

The example binds instructions to the existing `circuit.append(op)` protocol and
checks `3 * 5 = 4` in GF(8). It needs only FlagQuantum's normal dependencies.

## Pinned-source adjoint defect

The source pin is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its
[`SynthesizeLRCircuit.build_composite_bloq`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py)
does not read `is_adjoint`, although `adjoint()` sets it. Consequently, its
literal Karatsuba gate decomposition returns **0 for 1 * 1 in GF(4)**, rather than
1. This is covered by `test_pinned_upstream_adjoint_counterexample`, which
executes the unmodified upstream gates.

FlagQuantum applies the mathematical adjoint: reverse the CNOT sequence and
invert the output wire binding. Tests compare the complete Karatsuba gate stream
and output mapping against the pinned source with **only this defect repaired**
in a scoped test patch. This is explicitly different from literal-source parity.
The quadratic path compares against unmodified upstream. Tests preserve operand
order and each wire's ordered gate dependencies; disjoint operations can be
scheduled differently by the upstream DAG exporter. No gate cancellation or
unitary-equivalence substitution is used for this comparison.

## Gate measurements and resource models

The emitted gate set is `{CX, CCX}` with optional boundary SWAPs. These are exact
instruction counts, with no measurements. T gates have not been synthesized in
this module; a seven-T unitary Toffoli substitution is a **model**, not a measured
T count. Qualtran's concrete `QECGatesCost` Toffoli and Clifford estimates equal
the raw CCX and CX counts in every tested case, including the repaired adjoint.

| Degree / modulus | Path | Emitted CCX | Emitted CX | Binding SWAPs | Clean scratch | Qualtran modeled Toffoli / Clifford |
|---|---|---:|---:|---:|---:|---:|
| 3 / `0xB` | Quadratic | 9 | 3 | 0 | 0 | 9 / 3 |
| 3 / `0xB` | Karatsuba, adjoint repaired | 7 | 26 | 2 | 0 | 7 / 26 |
| 4 / `0x13` | Quadratic | 16 | 4 | 0 | 0 | 16 / 4 |
| 4 / `0x13` | Karatsuba, adjoint repaired | 9 | 44 | 2 | 0 | 9 / 44 |
| 8 / `0x11B` | Quadratic | 64 | 24 | 0 | 0 | 64 / 24 |
| 8 / `0x11B` | Karatsuba, adjoint repaired | 27 | 200 | 4 | 0 | 27 / 200 |

The table excludes input and output registers (three degree-sized words).
Other widths can need recursive clean scratch; degree 5 is exercised in tests.
Symbolic field parameters and the plus-equal-product variant remain outside this
concrete slice. Measurement-based product erasure is documented below.

Tests: `tests/unit/test_field_multiplication.py`. Source gate/map comparisons use
degrees 2, 3, 4, 5, 6, and 8; the complex statevector test covers a superposition
of all GF(8) operand pairs, with zeroed output and scratch. The example exercises
the fixed-wire append boundary. These are local CPU correctness checks.

The sparse exact-permutation tests propagate nonuniform complex amplitudes for
all operand pairs at degrees 5 and 8 (65,536 pairs at degree 8), checking every
scratch bit. This covers an active recursive scratch allocation at degree 5.
The multiplication file has 23 passing tests and 98% statement coverage in the
local pinned-reference environment.

## Inversion and workspace ownership

`field_inverse.gf2_inverse_decomposition` ports the pinned `GF2Inverse`
Itoh--Tsujii call chain: addition, repeated squaring through PLU linear networks,
and Karatsuba multiplication. The same adjoint correction described above is
applied. The output mapping contains `(source, result, *junk)`; logical junk
registers are retained exactly as in upstream. They must not be mislabeled as
clean scratch. Raw scratch excludes all logical outputs.

`gf2_inverse_ir(source, target, workspace, modulus=...)` provides a fixed-wire
clean interface. It computes the raw chain, XOR-copies the inverse into target,
then reverses the entire physical gate stream. Source and every workspace bit
are restored; target can contain arbitrary data. Zero maps to zero. The
`gf2_inverse_workspace_size` query gives the exact required workspace for a
concrete width and polynomial; supplying one fewer wire is tested to fail.
The existing append protocol is used in the runnable example, which additionally
checks `inverse(3) = 6` in GF(8) and returns all six workspace wires to zero.

| Degree / modulus | Path | Emitted CCX | Emitted CX | Extra work / logical junk | Qualtran modeled Toffoli / Clifford |
|---|---|---:|---:|---|---:|
| 3 / `0xB` | Raw, adjoint repaired | 7 | 35 | result 3 + junk 3; clean scratch 0 | 7 / 35 |
| 3 / `0xB` | Clean XOR output | 14 | 73 | clean workspace 6 | no matching clean upstream entry |
| 4 / `0x13` | Raw, adjoint repaired | 18 | 104 | result 4 + junk 4; clean scratch 0 | 18 / 104 |
| 4 / `0x13` | Clean XOR output | 36 | 212 | clean workspace 8 | no matching clean upstream entry |
| 8 / `0x11B` | Raw, adjoint repaired | 108 | 943 | result 8 + junk 24; clean scratch 0 | 108 / 943 |
| 8 / `0x11B` | Clean XOR output | 216 | 1,894 | clean workspace 32 | no matching clean upstream entry |

All rows have zero measurements. Clean counts close exactly as
`2 * raw counts + degree CX`; no output-binding SWAP is required because the
copy reads the raw logical output map directly. The seven-T Toffoli model gives
756 T for the degree-8 raw chain and 1,512 T for the clean chain. These are
substitution estimates; this module does not emit or measure T gates. The raw
upstream entry retains junk and therefore has a different output contract from
the clean interface.

`tests/unit/test_field_inverse.py` compares all ordered gate dependencies and
output maps for degrees 1, 2, 3, 4, 5, 6, and 8 against the source with the one
adjoint repair. It checks every field value, including zero, against `galois` and
verifies every workspace bit. Complex128 statevector tests exercise arbitrary
source/target superpositions at degrees 1--4 through `circuit.append(op)`, with
an absolute amplitude tolerance of `1e-12`. Resource counts are compared directly
with the pinned `QECGatesCost` model. Symbolic fields and a measurement-assisted
clean inverse remain unimplemented.

Local acceptance: 19 inversion tests passed with 100% statement coverage of
`field_inverse.py`. The six-file G1/G2 regression run passed all 109 tests with
both pinned references available. Black, Ruff, and the standalone example also
passed. These results do not constitute final whole-PR CI or GPU acceptance.

## Measurement-assisted product erasure

`field_erasure.gf2_product_erasure` ports the **unmodified** v0.7.0 `GF2MulMBUC`
measurement order, all classical parity input lists, and ordered phase correction
calls. It uses no linear adjoint and is unaffected by the defect above. Inputs
and the known product are big-endian words in the same polynomial basis. Each
classical slot corresponds to the product wire at the same position. The product
must contain `left * right`; an arbitrary target is outside the erasure contract.
Every branch preserves the two inputs and clears all product slots to zero.
There is no additional quantum workspace, and the measurement record remains
available to the caller.

The compiler-local result retains measurement order, logical `ParityPhase`
calls, and every upstream parity input list, including an unused final parity.
Its `lower()` method emits existing dynamic IR. Shared `measurement.py` primitives
provide X-basis clearing and XOR-controlled Z/CZ phases; paired QROAM lookup and
erasure use the same helpers. Four representative QROAM instruction streams were
compared with the pre-refactor source and were unchanged. A replacement test
substitutes independent measurement/phase emitters without changing the consumer.
Higher-arity phases are explicitly rejected at this boundary.

For the current target, `(-1)^((a XOR b)*p)` is emitted as the product of two
single-bit-controlled phases. This linear-size expansion uses existing classical
control metadata, avoiding an exponential Boolean expansion or a new runtime
operation. Qualtran has a classical `Parity` operation here, but no gate-level
decomposition for it. Consequently source **logical calls**, target **emitted
instructions**, and source **resource estimates** are separate evidence levels.

| Degree / modulus | Source logical CZ calls | Source modeled CZ | Emitted single-bit-controlled CZ | Target HMR | Extra quantum work | T gates |
|---|---:|---:|---:|---:|---:|---:|
| 2 / `0x7` | 4 | 4 | 5 | 2 | 0 | 0 |
| 3 / `0xB` | 9 | 9 | 12 | 3 | 0 | 0 |
| 4 / `0x13` | 16 | 16 | 22 | 4 | 0 | 0 |
| 5 / `0x25` | 25 | 25 | 36 | 5 | 0 | 0 |
| 8 / `0x11B` | 64 | 64 | 151 | 8 | 0 | 0 |

HMR is the target representation of adjacent H, measurement, and result-controlled
X clearing. The raw dynamic IR has one of each per product bit. These counts
exclude forward multiplication and include no T substitution estimates. Classical
storage is one retained measurement bit per product bit; logical classical parity
is fused into phase feedback instead of allocating additional parity slots.

The runnable example includes compute followed by measured erasure using
`circuit.append(op)` and checks all returned trajectories. The erasure tests
compare source calls for degrees 2, 3, 5, and 8. Exact phase-polynomial evaluation
covers every operand pair and every measurement branch at degrees 1, 2, 3, 5,
and 8, including nonuniform complex input amplitudes. At degree 8 this covers
65,536 input pairs and all 256 outcomes. Independent dynamic complex128 execution
checks degrees 2 and 3; KMX execution on the pinned native reference checks
degrees 3 and 8, all quantum slots, exact zero phase, and observed coverage of
every outcome. The new erasure suite has 15 tests; both new compiler modules have
100% statement coverage. The seven-file G1/G2 regression has 124 passing tests.
