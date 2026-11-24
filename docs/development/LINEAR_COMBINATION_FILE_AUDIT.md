# Linear combination source files

The later [typed root audit](TYPED_LINEAR_COMBINATION_AUDIT.md) separates native
Program composition from the earlier Algorithm acceptance below. The shared
root is available, but complete constructor closure remains open under Q-0876:
the original producer drops private payloads and signed Phase storage still
awaits its approved contract. Earlier callback evidence does not close those
constructor gaps.

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0285 | `linear_combination.ipynb` | `e44e694d55cc4d9524756741665b68b38cfce860` | Closed by the full native tutorial. |
| Q-0286 | `linear_combination.py` | `a3d8aa7fafa7a6f3fcaf095826751209427cd255` | Closed for the complete source definitions: owned execution, external/dense leaves, resolved symbolic declarations and complete catalogue history certificates. Missing source classical bodies and returned-adder failures remain explicit boundaries. |
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
generic synthesis into lower-level gates remains outside that numerical evidence.
The complete-history support certificate below supplies the separate long-history
claim without relabeling sampled numerical trajectories.

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

The subsequent closure below completes the catalogue history and resolved-symbol
obligations. The original explicit dense matrix families are converted as described
below. The original returned
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


## Complete source closure

The test-owned support proof now uses a reduced ordered Boolean decision graph.
It replaces the previous expanded ANF representation, which exceeded a 60-second
bound at the nine-bit comparator. The partial run is not acceptance evidence.
No quantum source, emitted gate stream or resource model changed in this proof
replacement. Canonical XOR/AND identities are checked against a complete truth
table, while all prior phase and tamper checks remain.

The literal seven-T Fredkin stream is recognized only when every instruction and
operand equals the accepted compiler macro. Its full eight-state positive complex
permutation is checked independently. Atomic CSwap has the same exact Boolean
update. CH conservatively freshens only its target. For an explicit matrix, every
nonzero row/column transition is inspected: an operand bit is retained only when
all those transitions preserve it. Changed bits become independent variables.
Structural zero means exactly zero, with no threshold; an off-diagonal amplitude
of 5e-21 still invalidates a stale predicate. Matrix dimensions, finiteness and
unitarity (absolute 1e-10) are checked separately. Named opcodes cannot carry
matrix overrides. These rules enlarge support conservatively and never discard a
failed equality obligation.

All 59 original coefficient configurations are covered in both directions,
including every nine-bit approximate and seeded dense configuration: 118 full
streams. Nine structured quantum-control configurations and eight repeated-root
activation vectors cover both supplied/default oracle modes, both directions and
each admitted extra polarity. Every measurement history has the same common
logical operator times the positive factor `2**(-events/2)`, with private work
zero for the whole admitted coherent input subspace. Logical coefficient junk,
signal and external controls remain outputs. Existing unchanged-source gate,
operand, feedback and numerical reference checks identify that common operator;
the support proof does not numerically propagate it.

Unresolved symbolic source widths still refuse decomposition and gate pricing.
The source has no automatic symbol-binding method. Tests resolve system, ancilla
and resource expressions first and rebuild concrete source/native children. All
four width combinations, two directions and outer-control modes compare resolved
ports, independent concrete costs and full complex logical output. Resolved
normalization and precision produce the exact original rescaled probabilities,
alt/keep tables and finite coefficient preparation states. Symbolic estimates
remain estimates, not recomputed physical counts.

| Complete source definition | Native owner / accepted mapping |
| --- | --- |
| Constructor and `of_terms` | LinearCombinationBlockEncoding validation and coefficient/block-pair factory; tutorial now uses the paired factory |
| Signed children and rescaled probabilities | LinearCombinationSelect sign phase and LinearCombinationPreparation; immutable source tables and all original coefficient rows |
| System, ancilla/resource maxima, alpha, epsilon and signature | Native block metadata, packed signature declaration and resolved symbolic tests |
| Signal state and default/custom PREPARE | Existing identity BlackBoxPrepare, alias owner and supplied oracle tests; zero and nonzero logical junk kept distinct from work |
| Padded SELECT and physical composition | Existing AutoPartition/indexed traversal, three ordered PREPARE/SELECT/unprepare stages; complete primitive, operand and feedback comparisons |
| `get_ctrl_system` and specialized adjoint | Explicit native direction/control binding and existing shared control predicates; structured and repeated roots, full branch/phase evidence and full-history certificate |
| Classical-only control dispatch | Existing typed CBit signature owner; both source polarities expose ClassicallyControlled but refuse decomposition/QEC pricing; native measured classical wrapper remains explicitly unavailable |
| String presentation and factory/documentation | Native component graph and complete tutorial retain ordered signed terms and original signature/call graph; native Python object repr is not an upstream compatibility API |

Source-only failures remain failures. The later returned adder raises TypeError
inside `np.concatenate`; grouped extension routed through it is rejected. The
classical root has declaration and tensor semantics but no declared physical
body or QEC price in the pinned source. Its two CBit polarities are probed, and
native typed declaration is checked separately from the measured wrapper refusal.
General typed classical-control capabilities belong to Q-0547 and are not newly
certified here. No fallback from these unavailable source paths is claimed.
The existing zero-ancilla source partition and generic child-cost refusals remain
in their owning audits. Dense leaves are explicit matrices with unavailable
synthesis costs. Arbitrary foreign gate synthesis is not inferred from a tensor.

This closes Q-0286's source-file migration with those stated source/admission
boundaries, without claiming that the upstream missing bodies have been invented
or that Q-0547 or G2 is complete.

Final closure validation: all 318 kernel, original catalogue, complete-history,
resolved-symbol and source-boundary cases pass in 200.87 seconds. The shared
sparse suite passes 66 cases with the replacement proof engine. The native LCU
notebook passes its independent kernel. Strict typing covers the proof engine,
LCU owner and paired-factory example; both formatters, scopes, architecture, IR
and API checks pass. The final two classical-width checks pass separately.
No new child checklist IDs or resource-count conventions were introduced.
