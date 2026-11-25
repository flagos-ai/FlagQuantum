# Complete rectangular and Kaiser original test acceptance

Q-0628 and Q-0622 cover complete frozen test files at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both entire files were read and
hash-checked. No original source, test or pytest fixture is edited.

| Complete source file | Git blob | Every top-level test |
|---|---|---|
| qpe_window_state_test.py | 0ca33cf4ab254b52dac9a605e0efe1a1bda98d84 | test_rectangular_window_state_tensor; test_rectangular_window_state |
| kaiser_window_state_test.py | 6a4917cf85cd2f8f50c016a8b930d598680ee775 | test_kaiser_window_state_auto |

The original autotester runs make, decompose, counts, serialize and qtyping.
Each function calls small and symbolic examples sequentially, so a skip or xfail
can prevent its second call. The current intake independently invokes all twenty
raw example/mode pairs. These statuses are preserved before pytest policy:

| Source example | Make | Decompose | Counts | Serialize | Strict types |
|---|---|---|---|---|---|
| Rectangular small | PASS | PASS | UNVERIFIED | PASS | UNVERIFIED |
| Rectangular symbolic | PASS | PASS | UNVERIFIED | PASS | UNVERIFIED |
| Kaiser small | PASS | MISSING | UNVERIFIED | PASS | NA |
| Kaiser symbolic | PASS | MISSING | UNVERIFIED | FAIL | NA |

## Complete native assertion and dependency mapping

The rectangular tensor assertion checks n=4, all sixteen amplitudes equal to
1/4. The existing native window test covers this whole vector, complex spectator,
relocated operands and exact inverse. Both actual constructors and their RIGHT
QFxp signatures, source Allocate -> OnEach(H) calls, numeric and symbolic named
operands, exact wire bytes and depths 0/1/2 resolver behavior retain their
accepted tests. The original count modes lack independent manual annotation.
Original strict typing diagnoses QFxp(n,n) -> QAny(n). Both source diagnostics
remain unchanged; native binding and full physical comparisons are separate.

The sole Kaiser test checks both constructors through those same five modes.
Existing native owners retain bitsize/alpha, fractional RIGHT output, exact
periodic coefficients, symbolic length and declared preparation child. No source
decomposition exists: native source construction stays bodyless and physical
binding refuses it. Only a manual child model exists, so count comparison is
UNVERIFIED and decomposition type checking is NA. Numeric serialization passes;
the symbolic source fails on unsupported HasLength(2**n). Root-only native
declarations remain a separately qualified wire path. No full reconstruction is
claimed for the failing original wire.

Shared dependencies remain Compiler typed calls/OnEach/allocation, Algorithms
window composition, Simulation coefficients and Ecosystem foreign conversion.
The same consumers retain normal circuit.append(op). No numerical kernel,
SDK import, public operation or source algorithm is replaced in this closure.

## Approved empty storage and complex contextual execution

Proposal 116 admits unsigned zero-width fixed-point storage. The old constructor
test incorrectly expected its earlier storage rejection. Standalone original
tensor contraction still fails; native binding with no physical wire reaches
the existing minimum-wire guard. With a borrowed spectator, the empty RIGHT
output binds and returns an empty word without touching the physical wire.

The original H -> RectangularWindowState(0) -> T context has a valid two-by-two
complex operator. Native append agrees on every column at absolute tolerance
2e-15, without phase alignment. Its window body emits zero gates or measurements
and owns no physical work. Existing positive-width window and quantized-child
certificates retain full complex amplitudes, applicable histories and zero work.
The standalone Algorithms wrapper's zero-width refusal remains unchanged.

## Resources and complete-file conclusion

Rectangular width n emits n H and no T, measurements or private work. Its QEC
model independently reports n Clifford. The zero-width context adds only its
caller's H/T. Kaiser root primitives are unavailable. Its separate quantized
two-bit child emits 67 CX, 28 T-family and seven measurements, with five work
wires; its abstract model reports four And, four measurements and peak nine.
The ideal-versus-quantized coefficient errors remain in the Kaiser audit.

Both entire original test files report five passes, five original skips and one
original xfail. Twenty raw observations distinguish actual source assertions
from policy skips. Concentrated native window, stored graph, coefficient and
approved storage regression passes 207 cases. Original skipped assertions are
not promoted to native positive checks. Quantized preparation is not relabeled
as an ideal Kaiser root. The two complete test parents close their actual
assertions and preserved protocol outcomes; the other seven candidate test
files retain their own numerical and dependency obligations.

Strict typing, Black, Ruff, Integration scope, local links, language,
architecture, IR and protected API checks pass for the touched surface.
The applicable Cirq 1.6 suite passes 40 cases. The dependency scan retains the
same 720 existing findings, not a globally clean result. Six frozen source blobs
include the two implementation/test pairs and original testing/pytest helpers.
Earlier 145-case local execution evidence is reused because production quantum
owners are unchanged. Receipts use `g2-window-test-closure-*-20261125`,
`g2-window-original-protocol-matrix-20261125.json` and the full original-test log
`g2-window-complete-original-tests-20261125.log`.
