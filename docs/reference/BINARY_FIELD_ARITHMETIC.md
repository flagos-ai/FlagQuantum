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
classical measurement slot corresponds to its product wire. The product must
contain `left * right`; an arbitrary target is outside the erasure contract.
Every branch preserves the inputs and clears the product slots. No extra quantum
workspace is required; measurement records remain available.

The caller supplies a separate, reusable classical result slot:

```python
erase = gf2_product_erasure(left, right, product, measurement_bits, modulus=0x11B)
instructions = erase.lower(parity_bit=8)  # measurement slots are 0 through 7
```

The existing `circuit.append(op)` example binds these instructions without a new
public circuit method. The source's final unused parity computation is retained
as well, so optimization does not obscure the call-chain comparison.

## Shared classical execution

The [internal classical dataflow contract](../development/CLASSICAL_DATAFLOW_CONTRACT.md)
precedes implementation. Core owns `ClassicalXor` and read-before-definition
validation. Its zero-quantum-wire instruction reads old source values and assigns
one result slot. Empty inputs write zero, duplicate sources cancel, and destination
aliases read the old value. Source slots may be measured or previously computed.

Scalar and batched dynamic trajectories execute the operation; density ensembles
track source measurements transitively and compute each branch's XOR. Boolean SSA
records represent `bool_zero` and binary `bool_xor` definitions, with type/def-use
verification and exact public IR round trips. Classical work consumes no quantum
noise or RNG draws and does not increase measurement counts. OpenQASM 3 exports
boolean assignments; unsupported provider profiles explicitly refuse the operation.
There are no added public dataclass fields or root exports.

KMX uses native `BIT_STORE0` and condition-controlled `BIT_INVERT`, including
correct alias handling. This keeps the classical cost executable and visible.
Shared `measurement.py` parity-controlled phases emit one XOR assignment followed
by **one** Z/CZ. Singleton feedback remains direct, so QROAM's existing quantum
instruction streams are unchanged. Multi-bit parity requires a caller-owned
classical result slot; no quantum phase expansion remains.

## Resource comparison at the same layer

The previous target lacked computed classical results. It expanded each parity
into multiple conditional CZ instructions: for degree 8, 64 source logical CZ
calls became 151 emitted CZ instructions. That expansion is superseded by the
shared executable XOR layer. Current source-call and emitted quantum counts are:

| Degree / modulus | Qualtran CZ calls/model | Previous expanded CZ | Current emitted CZ | HMR | Classical XOR assignments | Native bit stores | Native conditional bit inversions | Peak classical slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 / `0x7` | 4 | 5 | 4 | 2 | 2 | 2 | 3 | 3 |
| 3 / `0xB` | 9 | 12 | 9 | 3 | 3 | 3 | 7 | 4 |
| 4 / `0x13` | 16 | 22 | 16 | 4 | 4 | 4 | 9 | 5 |
| 5 / `0x25` | 25 | 36 | 25 | 5 | 5 | 5 | 12 | 6 |
| 8 / `0x11B` | 64 | 151 | **64** | 8 | 8 | 8 | 34 | **9** |

Every erasure row has zero T gates and zero extra quantum workspace. Peak
classical storage is `degree` measurement slots plus one reusable parity slot.
HMR represents H, measurement, and result-controlled X clearing. Counts exclude
forward multiplication and count emitted conditional instructions, rather than
claiming that every condition fires in each trajectory. Qualtran's quantum model
omits classical execution cost; the classical columns therefore remain separate.
The XOR assignments and native bit operations are two levels of the same work,
and must not be added together as independent operations.

## Acceptance evidence

The real unmodified upstream `build_composite_bloq` is bound to a label recorder.
Both sides use the same explicit reusable parity slot and physical X-measurement
cleanup boundary. The **entire ordered measurement/XOR/CZ event stream** matches,
including the unused final parity. Emitted CZ counts match the source model;
native stores/inversions close against the actual parity input lists.

Tests cover every operand pair and every measurement branch at degrees 1, 2, 3,
5, and 8 with nonuniform complex amplitudes; degree 8 covers 65,536 pairs and 256
measurement outcomes. Dynamic complex128 trajectories and KMX execution on the
pinned native reference verify quantum cleanup and phase preservation. QROAM's
existing G1 reference tests and independent emitter replacement test remain in
place. The executable example checks compute/erase trajectories as well.

The classical suite also covers empty, duplicated, aliased and overwritten
sources, undefined reads, both trajectory strategies, density ensembles, SSA
verification, IR/KMX round trips, native execution, preserved RNG streams and
measurement counts, and restricted provider refusal. These are local CPU checks;
final whole-PR CI and GPU acceptance remain separate.
