# GQSP with packed and shaped unitary ports

This finite G2 acceptance connects existing typed register construction to the
same GQSP callback and dynamic append consumer. Reference source is unchanged
Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
No production API, IR schema, control engine, runtime or dependency is added.

## Interface and ownership

The U exposes one THRU register named `word` in three forms: scalar QUInt(2),
QBit with shape (1,2), and QUInt(2) with shape (1,2). Shapes flatten row-major;
each integer word splits MSB first. The native counterparts are existing bit or
uint RegisterTypes. Every bit handle is consumed exactly once and each result
is joined into its original dtype and array position. The callback receives the
existing flat named port, then supplies one physical tuple per typed element to
`bind_register_program`. It verifies that the binder returns the same ordered
physical data and control ports.

Compiler owns typed construction, split/join, control/adjoint transformation,
physical binding and workspace leases. Algorithms owns GQSP composition;
Simulation owns phase recursion; Runtime executes dynamic instructions. Users
still write `circuit.append(op)` with the existing bound GQSP operation. Typed
shape stays inside the stored Program and callback; no second public shape or
register API is introduced.

The existing shared H(a), CNOT(a,b), ZPow(1/4)(b) layer acts on each adjacent pair
of flattened data bits. The four-data-bit case applies two disjoint pairs; it
does not repeat a two-bit gate on the same pair. Scalar QUInt(2) and two shaped
QBits lower to the same physical stream; two shaped QUInt(2) elements lower to
two pair streams. Physical data order is descending wire number, deliberately
unrelated to word significance. Signal is the highest data-adjacent wire, clean
work is wire 0, and the final wire is an idle probe. Clean work is reused only
after measured CZPow cleanup; each U invocation owns one classical slot per
pair, reused by later completed U calls.

## Original call chain

The unchanged source BloqBuilder constructs real Split/Join components around
its CompositeBloq children. Original bookkeeping control calls the original
bookkeeping component unchanged while threading the predicate separately. Its
adjoint exchanges Split and Join. Native code uses the existing split/join
functions and supplies these same neutral control adders and exact adjoint
pairs; it does not infer controlled primitive gates for bookkeeping.

The existing generic control propagates into the shared child: H to atomic CH,
CNOT to atomic Toffoli, and ZPow to measured CZPow/And/phase/And-adjoint. Zero
control wraps the whole U once. Source connections and all resulting linear
Instruction tuples, including angle parameters and feedback metadata, match.
As in the previous slice, SU2/scalar phases use the accepted common physical
normalization; CH and Toffoli remain atomic. This is not a synthesized
Clifford+T comparison. Direct source GQSP Cirq unitary is a separate amplitude
oracle.

The existing independent child-cost query and GQSP call model are reused. The
previous test helper now accepts system width explicitly so the four-data-bit
case accounts for all live bystanders without duplicating the pricing logic.
This is test support only; no production cost model changes.

## Acceptance scenarios

The 39 new tests cover 18 independent U cases (three typed layouts, three
activation modes, two directions), 18 full GQSP cases (three layouts, lengths
2/3, negative powers 0/1/4), and three expansion/binding scenarios. Every logical
basis and one complex coherent input are checked. Together they exhaust 2,304
unnormalized measurement branches and execute 306 real dynamic trajectories.
Absolute complex phase, branch scale `2**(-measurement_count/2)`, exactly zero
work and idle, and complete Instruction equality are required. Reset follows
the preceding measurement projection. Replacing the stored hierarchy with its
expanded Program preserves the same callback consumer and Instructions; short
word bindings and absent clean work are rejected.

An initial run reproduced NumPy BLAS matmul warnings at the four-bit oracle
application. The final oracle applies the unchanged source dense matrix using
`np.einsum(..., optimize=False)`, checks every matrix/vector value is finite,
and independently compares that product against torch matmul before checking
native output. No warning is suppressed and no source or expected amplitude is
modified.

For P=(.25,.1,.2), actual Instructions under the declared atomic normalization:

| Native typed port | k | Native/source Instructions | Atomic ccx | T+T-adjoint | Measurement/reset | Clean / slots |
|---|---|---|---|---|---|---|
| uint2 scalar or QBit(1,2) | 0 | 65/65 | 2 | 8 | 2/2 | 1/1 |
| uint2 scalar or QBit(1,2) | 1 | 63/63 | 2 | 8 | 2/2 | 1/1 |
| uint2 scalar or QBit(1,2) | 4 | 67/67 | 2 | 8 | 2/2 | 1/1 |
| uint2(1,2) | 0 | 105/105 | 4 | 16 | 4/4 | 1/2 |
| uint2(1,2) | 1 | 103/103 | 4 | 16 | 4/4 | 1/2 |
| uint2(1,2) | 4 | 113/113 | 4 | 16 | 4/4 | 1/2 |

Model counts are independently checked against source QECGatesCost:

| Native typed port | k | T | Toffoli | And | Clifford | Rotation | Measurement | Peak |
|---|---|---|---|---|---|---|---|---|
| uint2 scalar or QBit(1,2) | 0 | 2 | 2 | 2 | 14 | 13 | 2 | 4 |
| uint2 scalar or QBit(1,2) | 1 | 2 | 2 | 2 | 12 | 13 | 2 | 4 |
| uint2 scalar or QBit(1,2) | 4 | 4 | 2 | 2 | 14 | 13 | 2 | 4 |
| uint2(1,2) | 0 | 4 | 4 | 4 | 24 | 17 | 4 | 6 |
| uint2(1,2) | 1 | 4 | 4 | 4 | 22 | 17 | 4 | 6 |
| uint2(1,2) | 4 | 8 | 4 | 4 | 28 | 17 | 4 | 6 |

CSWAP model count is zero. The physical idle probe is additional to modeled
peak. Quarter-Z model T and physical phase, logical And and its expanded T
network, and atomic Toffoli versus synthesized Toffoli are separate categories.
No synthesis budget or performance claim is made.

## Reproduction and limits

Run `tests/unit/test_generalized_qsp_shaped_unitary.py` with locked source paths,
preimport torch, and assert Qualtran distribution version 0.7.0 and Kickmix
Simulator exist before pytest. Native dynamic Runtime executes these tests;
availability of Kickmix is not a claim of native Kickmix execution. Verify
reference Git blobs against the inventory before source comparisons.

The previous original CNOT test-module limitation (missing PennyLane) remains
open; no fake optional module or original-test pass is claimed. Multi-control
shaped predicates, empty shapes, zero-width words, arbitrary U decomposition,
symbolic components, synthesis and full-parent/G2 acceptance remain outside this
finite slice. The final checklist records this slice separately.

## Local acceptance

The 39 new cases pass within the full focused selection: 490 passed, 54 warnings in 140.55s (0:02:20).
Related regression: 783 passed, 54 warnings in 118.04s (0:01:58). Integration: 64 passed in 9.09s.
Focused/regression warning counts come from unchanged fixtures: six autoray
matmul warnings in the controlled-NOT tests and the previous 48 GQSP numerical
environment warnings. Strict mypy (two files), Black, Ruff, Compiler ownership,
architecture, IR and public-API checks pass. Twenty-seven source files retain
their inventory Git blobs; selected unchanged original tests report 34 passed, 13 warnings in 1.85s.
The inherited original CNOT module collection limitation remains explicitly open.

Default smoke/unit results: 7 failed, 21507 passed, 300 skipped, 1037 warnings in 321.41s (0:05:21).
All failed IDs match the seven independently recorded baseline failures.
Maxfail stops this run; it is not a fully green suite. New cases have the
complete separate focused evidence above.

External evidence uses the `g2-gqsp-shaped-unitary-` prefix for
focused/regression/integration/original/default logs, counts and default-comparison
JSON, source blob list and the inherited PennyLane collection-error evidence.
