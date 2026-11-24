# Majorana constructor and direct gate dependency disposition

Reference: complete `multiplexers/selected_majorana_fermion.py` and unchanged
tests at Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`. Source
and test Git blobs are `656373799b28c6838bc3528733c26f0fba0adacf` and
`d0b964420757f48f6f600d3d8e555855c1b6843c`. Tests require actual Qualtran
0.7.0 and Kickmix 0.1 Simulator. No reference file is modified.

## Ownership, interface and replacement

Compiler lookup extends the existing majorana_program owner with explicit
controlled-target and accumulator-initializer Program inverse pairs. Existing
Pauli consumers, including the inlined Hubbard path, retain that same owner.
Compiler formats retains source register metadata, constructor defaults and
scoped adjoint recipes. Gate JSON remains opaque there. Ecosystem alone decodes
the gate and selects existing named definitions or its existing typed gate
importer; SDK objects never enter Compiler or canonical IR.

Control words and scalar bounded selections retain original names, dtypes,
order and MSB-first storage. Multiple axes flatten in source C order. Target
is the caller-owned opaque word with the product of iteration extents. The
parity accumulator and unary/conjunction scratch are private, initially zero
and returned zero. A packed positive control word is supported; no correction
is silently applied to the source's separate-control dirty-accumulator case.

For three or more controls, the initial X uses the existing ControlledViaAnd
and CtrlSpecAnd owners with the actual source CNOT adder and port names.
Substituting an SDK-decomposed multi-controlled X would preserve its unitary
but change the source call chain. Unary iteration separately uses its source
And/MultiAnd control reduction. Exact source Adjoints reverse owned calls with
specialized predicate erase bodies and explicit target inverse pairs. No
measurement stream is reversed. The original omitted control and Y defaults
are retained; an explicit invalid null gate is not treated as an omitted value.

## Source and numerical evidence

All six original X/Y constructor-grid cases resolve at depths zero and one in
both directions. Complete comparable recursive calls, operands, views and
dependencies match the unchanged source. Source metadata paths for a three-bit
control word also succeed in both directions at both depths. The previously
recorded flattened-graph uint8 codec refusal is not assigned to these successful
unflattened source libraries.

Small complete complex operators cover zero through three controls and X, Y,
T, Rx and a complex matrix target. Every raw measurement history is additionally
executed on a coherent full-domain input with exact branch normalization and
zero private work. The independent erasure certificate plus complete basis
columns proves the full operator and cleanup. Nested axes and controlled
singletons have complete source call and operator comparisons in both directions.
Ordinary circuit.append(op), emitted instruction payloads and JSON roundtrips
are checked. Source-free compilation and SDK-free binding/execution are verified.

Comparable call chains for X/Y/T are recursively checked through admitted
primitive leaves. Controlled atomic Rx has no further original decomposition;
that refusal is asserted and the native implementation below that boundary is
checked by complete source tensors and every branch. Matrix cases compare the
complete root calls and complete complex channel. The existing deeper matrix
phase-carrier/SDK synthesis comparison boundary remains explicit; no missing
source primitive stream or identical carrier allocation is invented.

The test binder first profiles allocations without constructing a quantum
state, then rebinds with only the physical pool actually used. The full gate
payload and allocation event trace must be identical. This removes untouched
pool wires from dense numerical storage, not from the operator or branch scope.
The initial over-reserved-pool timeout is not acceptance evidence.

## Named numeric literal dependency

SDK decomposition can supply NumPy float64 angle/exponent metadata. Existing
native literal admission requires Python numeric literals; this caused valid
SDK named-gate selection to fail. Ecosystem now converts float64 to the exactly
equivalent Python double before calling the unchanged native factories.
Canonical modules and full complex actions match their Python-literal forms.
Core parameter, serialization and public API contracts are unchanged.

Float32 Rx has a separately reproduced precision boundary: its SDK unitary
differs from the widened Python-angle unitary. Its existing named-angle refusal
is retained rather than silently widening the value or relaxing phase checks.
Other unsupported numeric scalar/storage domains are not claimed by this fix.

## Separate resources and remaining scope

Receipts contain actual primitive inventories and physical allocation events
separately from original QEC and logical peak models. The original four-target
X case emits 12 T-family, 30 CX and three measurements on ten peak wires;
its model retains three And, 19 Clifford, three measurements and ten logical
peak wires. Predicate lease release complements the numerical cleanup proof.

Controlled target synthesis can introduce additional native phase carriers
and measured erasure beyond the original model. Atomic Toffoli, rotations,
controlled rotations, matrix payloads, primitive T-family counts and source
model categories are not interchangeable. The resource receipt records each
direction and any source model-query failure explicitly.

Source separate-control dirty accumulation, symbolic iteration extents,
uncontrolled singleton/empty-selection refusals, unsupported physical register
admission, foreign gate JSON namespace failures and matrix carrier comparison
remain open. Symbolic address width alone retains a logical declaration; it
does not imply physical synthesis. The original SelectPauliLCU factory fails
object-array serialization independently upstream and remains a separate gap.
No generic constructor count or original parent item is incremented for these
unverified domains. Larger exhaustive operators, GPU and performance are unclaimed.

Evidence uses `g2-majorana-constructor-*-20261124` in the external ledger and
reuses the unchanged complete-file evidence in SELECTED_MAJORANA_FILE_AUDIT.md.

Final acceptance: 213 concentrated cases pass without skips, including 60 new
constructor-path cases and three named numeric-literal cases. Local integration
passes 145 cases. The complete unchanged original test file has nine passes.
Strict typing covers all six changed Python files. Black, Ruff, Compiler and
Ecosystem scope, language, architecture, IR and API checks pass. Dependency
output matches exactly the existing 720 findings, with no new finding.
All 36 directional resource rows return zero live private leases. Two-target
three-control matrix cases execute all 2048 raw histories with complete complex
branch outputs and private cleanup; their deeper carrier comparison remains
the explicitly separate source boundary described above.
