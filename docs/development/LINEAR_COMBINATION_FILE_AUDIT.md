# Linear combination source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0285 | `linear_combination.ipynb` | `e44e694d55cc4d9524756741665b68b38cfce860` | Closed by the full native tutorial. |
| Q-0286 | `linear_combination.py` | `a3d8aa7fafa7a6f3fcaf095826751209427cd255` | Concrete external snapshots, dense leaves and symbolic declarations verified. General symbolic resolution and complete catalogue history certification remain open; returned-adder failures are source boundaries. |
| Q-0287 | `linear_combination_test.py` | `bed60bc7ade281fca1e5b13e341e8b4dd07aabd9` | Closed for the original runnable fixture families. Original fixture-mode outcomes and native branch coverage are recorded below. |

## Binding and decomposition

The block binds system, coefficient-selection plus constituent-signal
ancilla, and constituent-resource plus coefficient-preparation junk, all
in source big-endian order. The temporary PREPARE and indexed SELECT work
and result slots live only for their owning stages. The wrapper emits
PREPARE, signed SELECT, and PREPARE adjoint. The zero-signal preparation
for the resulting block is identity. Structured or repeated outer
controls use the existing selector and predicate owners.

The original tutorial instance is `(T - H + X - Z) / 4`, with one-bit
coefficient precision and port widths `(1, 2, 5)`. Its original formal
fixture and the user-facing native notebook share the same term order,
coefficients, signal metadata and resource model. The reference checks
walk the unchanged source graph using stage-local bindings and compare
the complete ordered primitive stream on each physical wire, with
operands and classical feedback. Both directions match the full complex
reference over all 256 measured histories each for the coherent test
input. All temporary work qubits return to zero in every tested history.

| Count | Physical emission | Independent source model |
| --- | --- | --- |
| T equivalents or T/T-dagger | 60 T/T-dagger | 48 T equivalents at the source default rates: four Fredkins and eight And calls |
| Measurements | 8 | 8 |
| Other model categories | Controlled T/H/X/Z appear as native controlled opcode leaves | 66 Cliffords and 3 rotations |
| Wires | 10 reserved, including two temporary wires | 10 peak logical qubits |

## Original test coverage

| Source tests | Native evidence |
| --- | --- |
| Default fixture, signature, normalization, signal state and tensors | Original four-term catalogue and exact source stream/full-state comparisons. Existing normalized coefficient PREPARE, invalid admission, zero signal and model tests cover the owners. |
| Metadata, error and resource tests | Existing mixed auxiliary-size and nonuniform-normalization tests preserve the implemented maximum normalized error rule. Existing directional, controlled, and legacy-profile resource tests cover the specialized calls. |
| Exact signed coefficients and Hermitian checks | All 35 original coefficient rows across two-, three-, and five-term families, including two-wire targets. The twelve Hermiticity rows also compare coherent full outputs on system, signal and coefficient junk in both directions. The distinct normalization fixture uses child scales two and four. |
| Approximate and random tests | All 13 explicit nine-bit rows and ten original seeded dense-gate cases. One reproducible Python coefficient stream is shared across the native and locked paths since the original seeds only the NumPy gate generator. |
| Notebook | The frozen 10-cell original and the native 12-cell notebook execute independently. The native notebook retains the original signature and call graph and adds a complete complex system projection and work check. |

The unchanged non-notebook source run passes 59 cases with one mode skip
and one expected serialization failure. Source external-object and
returned-adder failures remain documented in the control-owner audits.

## Packed signatures and symbolic widths

The ordered opaque ports are the optional positive control, system,
ancilla, and resource. Ancilla adds the PREPARE selection width to the
largest child ancilla. Resource adds PREPARE junk to the largest child
resource. A zero-width port is omitted. The concrete LCU exposes its
uncontrolled signature, since the outer control is bound at append time.

The declaration accepts symbolic constituent widths with known PREPARE
ports. The locked qubit count falls back to the total signature width.
Tests verify each symbolic port and all three together with and without
control. Locked decomposition and both QEC gate profiles refuse these
symbolic cases. The locked default PREPARE also refuses symbolic precision
and symbolic normalizations in the numerical alias factory. The accepted
supplied-PREPARE path can still keep an unused symbolic precision.

## Coefficient-family execution

The native fixtures append the bound operation and compare every column of
its complex zero-signal projection with the locked tensor contraction.
The source approximation tolerance is checked separately against the
ideal weighted sum. Native-to-source agreement uses a tighter absolute
tolerance of `2e-11`. The branch tests verify private work cleanup, and
keep coefficient junk as a logical through port. They enumerate histories
up to four measurements for exact fixtures, then use all-zero and alternating
histories for longer exact streams. The nine-bit cases execute emitted IR
in the shared sparse branch evaluator at seeds 19 and 73, with each balanced
measurement verified. The dense fixtures provide controlled matrix leaves;
generic synthesis into lower-level gates and exhaustive long histories
remain outside this evidence.

Three representative named-gate fixtures at one and nine bits compare the
locked PREPARE/SELECT/unprepare stream in both directions, including wire
operands and feedback. They also compare each QEC category under both
profiles and the independent logical peak. For the original nine-bit
coefficient row `(-1/3, 1/2)` on T and H, the native IR reserves 48 wires,
emits 150 T/T-dagger, 540 CX and 34 measurements. The modern locked
model charges two Fredkins, 34 And, 423 Cliffords, three rotations and 34
measurements, with a logical peak of 48.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_linear_combination.ipynb
```

## Native and external child snapshots

The existing LCU child factory now accepts the native `CircuitBlockEncoding`
provider. Actual external H/T blocks, including logical ancilla/resource ports,
are converted to four owned directional/control snapshots before composition.
The unchanged LCU consumer replaces its original child factories with imported
`bind_child` methods and retains its original PREPARE/SELECT stages. Both
outer-control modes and directions compare full complex logical outputs on every
small measurement branch. Independent source models remain separate from emitted
instructions and any returned-wire binding swaps. See
[Circuit block snapshots](CIRCUIT_BLOCK_ENCODING.md) for ownership and the append
workflow.

Q-0286 remains open: complete long-history certificates for the entire
coefficient catalogue and resolved general symbolic bindings still need closure.
The original explicit dense matrix families are now converted as described below. The original returned
adder TypeError and grouped extension refusal remain immutable source boundaries;
valid returned roots and the failing adder are separate claims. This batch does
not convert these source failures into successful alternate algorithms.

Snapshot dependency validation: 255 concentrated cases pass. The final scalar
admission and negative-object checks pass 50 focused cases; the strengthened H/T
LCU replacement passes all nine foreign-block cases. The existing native LCU
notebook passes its independent kernel. Seven production files pass strict
typing; formatting, scopes, architecture, IR, API and the documented append
example pass. Primitive inventories and source models are kept separate.


## Declared dense leaf conversion

All ten original seeded MatrixGate families now pass actual optional-boundary
conversion in both LCU directions. The full emitted streams match the accepted
native catalogue, including each explicit controlled matrix, operand and
measurement feedback dependency. Ordinary, adjoint and positive-controlled source
matrix blocks preserve their complete complex matrices as one native matrix leaf.
Classically controlled matrices preserve their operands, conditions and absolute
phase, and the imported tensor owns a copy of external data. Invalid matrices
fail before canonical IR is returned.

The unchanged source returns QubitCount for these leaves but refuses QECGatesCost
with DecomposeNotImplementedError. Native snapshot execution therefore permits
an absent independent model, and resource queries report unavailable instead of
zero. The emitted matrix leaf count, known physical widths and unavailable
Clifford/T synthesis costs remain separate. This is explicit matrix preservation,
not synthesis or contraction of an arbitrary source component into a dense gate.

Dense-leaf validation: 75 focused cases and 281 concentrated cases pass; the
final immutable-source precondition passes 11 focused cases. The MatrixGate,
controlled wrapper and LinearCombination Git blobs match the pinned inventory.
Four changed production files pass strict typing; formatting, scopes, architecture,
IR and API checks pass. The earlier 150-second partial run remains non-acceptance
evidence and was not repeated unchanged.

The existing native LCU notebook also passes its independent kernel. Both Black
and Ruff formatting pass after normalizing the shared test helper assertion;
the final 39 feedback cases pass after that equivalent assertion rewrite.
