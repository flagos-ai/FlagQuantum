# Fixed-point and windowed elliptic-curve addition

The frozen reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, files
`qualtran/bloqs/cryptography/ecc/ec_add_r.py` and `ec_add_r_test.py`.

## Fixed point

The unsigned through coordinate words and scalar control match the source.
Control zero preserves the raw affine pair. A set control adds the same
constant point on the original curve, including the infinity sentinel.
Concrete and symbolic signatures are separate from gate binding. The source
reports signature width (2n+1) and raises `DecomposeNotImplementedError`
for decomposition and gate count. The native fixed-point object exposes
the classical map. There is no locked fixed-point gate stream to compare.

## Windowed addition

The ordered address is a big-endian bit array. The native table uses the
locked repeated addition order starting from row 0 (infinity and zero
tangent), converts all three outputs to Montgomery form, and preserves the
source forward bank width. Active cell zero from each bank feeds
`a,b,lam_r` of the reversible six-stage addition. Retained bank cells
remain live until measured erasure. Address, coordinates, and table slots
bind through `circuit.append(op)`. All bound bank outcomes occupy distinct
slots from arithmetic and traversal.

The original one-level call graph contains QROAMClean, ECAdd and
QROAMCleanAdjointWrapper once each. The implementation reuses the
exact source forward decomposition and the verified measurement-assisted
native erasure for its matching banks. The source adjoint leaf has a
resource declaration with no gate decomposition. Its cost is kept
separate from actual measurements and conditional phase gates. Address
widths 1 and 5 verify source data, block size, boundary operands, forward
gates, and feedback slots. Two sparse executions per width compare
complex amplitudes over all 136 address/point combinations at modulus 3.
The entire output support is compared, including the slope, flags, QROAM
banks and shared scratch. The shared erasure regression checks all outcomes
on small signed-bank cases. The modulus-5 address-zero witness preserves
all output data and returns the bank to zero while retaining the underlying
`f1=1`, as previously observed in [six-stage addition](ECC_ADDITION_FILE_AUDIT.md).

| Width, modulus, address width | Native emitted operations | Qualtran model |
|---|---|---|
| 3, 3, 1 | 7,640 T/T†, 19 CCX, 1,331 measurements, 75 wires | 1,322 AND, 19 Toffoli, 336 CSwap, 8,787 Clifford, 1,322 measurements, 59 qubits |
| 3, 3, 5 | 7,824 T/T†, 19 CCX, 1,368 measurements, 88 wires | 36 T, 1,336 AND, 27 Toffoli, 336 CSwap, 9,048 Clifford, 1,336 measurements, 72 qubits |
| 4, 5, 1 | 12,864 T/T†, 24 CCX, 2,164 measurements, 94 wires | 2,152 AND, 24 Toffoli, 608 CSwap, 14,148 Clifford, 2,152 measurements, 74 qubits |

For the first row the lookup adds nine bank measurements over the
six-stage child. The blocked five-bit address has 18 bank outcomes and
uses the same traversal for load and phase correction.

## Test-file disposition

The complete original test file, including its slow selection, ran with
153 passes, seven source skips and three expected failures. Native tests
compare all 132 fast and slow numeric callback inputs, all 16 call-graph
parameter combinations and table block choices, the source's width-16
table fixture, the fixed metadata and the smaller physical windows above.
The numeric callback is independent of construction admission to cover the
source's four-bit and modulus-17 cases. Local regression completed 100
tests plus both tests of the original call-graph parameter matrix.
Q-0459 closes as a test file. Q-0458 retains executable boundaries for the
opaque fixed-point operation and symbolic lookup metadata, and inherits
the coordinate and release gaps from Q-0457.

## Stored constructor and direct dependency closure

The complete `ec_add_r.py` now has owned Ecosystem constructors for ECAddR and
ECWindowAddR. Scalar ECAddR and symbolic windows retain typed declarations when
the source provides no concrete body. Ordinary concrete windows compose the
literal five calls, including three optional junk banks, through existing
Compiler QROAM and six-stage ECAdd owners. Algorithms remains authoritative for
point arithmetic, signatures and table generation; Compiler performs no numerical
evaluation and imports no SDK or Algorithms implementation.

The original source connects nominal QAny/unsigned lookup words to Montgomery
ports without converting their bits. Existing loose source-compatible word
consistency admits these direct links. A follow-up removed the initially added
ten-Cast wrapper and eleven helper definitions after checking complete physical
IR and all returned layouts against published `dec49fcad` at address widths one
and five. No primitive, measurement, output or physical pool changes. The root
now calls the six-stage ECAdd directly, retaining a/b/x/y/lam_r source order.
The root still has five comparable source calls. Original serialized libraries
at depths zero and two resolve without Qualtran, Cirq or Qiskit imports. The
unchanged `_ec_window_add_r_small` example (16 bits, modulus 7, address width 4)
constructs 8,507 definitions and the same root graph; no dense execution is
claimed for that fixture.

Default source resolution retains the original QROAMCleanAdjoint declaration,
so whole physical binding refuses. The execution conformance test explicitly
replaces its wrapper with `qroam_measurement_erasure_program`, leaving every
consumer unchanged. It verifies all 136 address/point columns across address
widths one and five, absolute complex amplitudes, every applicable raw erasure
history, returned coordinate leases and zero private work. Independent sparse
executions use coherent complex weights and two outcome seeds with SDKs blocked.
The forward table child is definition-identical to the existing owned lookup,
whose original gates, operands and feedback are compared on these exact tables.

The shared history checker recognizes a relative-swap block only if every
primitive matches the literal source sequence. It retains the minus sign on
each dirty 111 triple. Complete small complex maps for widths one, two and three,
both directions, and missing-rotation counterexamples ground this lemma.

| Address width, n=3, modulus=3 | Explicit native instructions | T/T-dagger | CCX | Measurements | Live peak / pool | Source default T estimate | Source measurements / peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36,351 | 7,640 | 19 | 1,451 | 59 / 59 | 6,708 | 1,322 / 59 |
| 5 | 37,476 | 7,824 | 19 | 1,488 | 72 / 72 | 6,832 | 1,336 / 72 |

The earlier table describes the existing flat operation. This table describes
the typed root with explicitly selected native bank erasure and exact arithmetic
adjoints. Five inner measured lookup cleanups add 120 H, 120 measurements,
120 conditional X and 60 conditional Z, and remove 60 CX, compared with the
old reverse-forward cleanup. The outer bank erasure is also explicitly native.
Source CSwap/Toffoli model conventions remain distinct from actual T/CCX gates.
No full original primitive stream exists through the model-only erase leaf.

Previously accepted original test-file, 132 classical callback and 16 model-case
receipts are reused. Q-0458 stays open: scalar opaque quantum execution, symbolic
lookup execution, the generic controlled RIGHT-output QROAM refusal and the
inherited mod-5 release, coordinate and wider-window defects are not waived.
This batch does not repair or edit the frozen source and does not claim the
small valid curve domain covers arbitrary coordinates or larger curves.
