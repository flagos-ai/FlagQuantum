# Typed qubitization phase-estimation composition and constructor context

The complete locked source is Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`phase_estimation/qubitization_qpe.py`, blob
`39b608ea5708eba219a7fa4f4365176170c40bf5`; the test blob is
`301228e1f11e677993a2330a1c1dc64f2ec61c6b`.

## Ownership and source order

Compiler phase owns typed composition through the existing wire graph and
explicit child Programs. Formats owns source constructor fields and scoped
inverse recipes; provider functions supply complete native helper closures.
No SDK object, numerical execution, resource-derived body or backend choice
enters Compiler. The normal append entry is unchanged.

The phase word is a scalar RIGHT `qpe_reg`, MSB-first. Targets and logical walk
junk remain caller-owned THRU ports. Preparation transfers phase outputs;
private work belongs to each child. Inverse consumes the phase word under the
prepared-image clean promise, not arbitrary-input projection.

Calls follow the original source: prepare; positive-controlled walk on the
last phase bit; zero-controlled reflection, unconditional walk power, the
same zero-controlled reflection for each remaining bit in reverse order;
then the supplied inverse QFT. Reflection uses `control` and its named subset
of walk targets; controlled walk uses `ctrl` and all targets. Source Power
squaring multiplies the explicit exponent instead of introducing a nested
square tree. The owner accepts those complete powered Programs, never a count
estimate standing in for execution. The source's documented paper discrepancy
is retained.

Every exact inverse computational child is supplied explicitly. Existing
adjoint ownership reverses typed calls and known views; measurements themselves
are not reversed. Unknown controls, missing target ports, missing powers and
inconsistent signatures fail before binding.

## Current evidence and its limits

Widths one through three compare source root calls, named operands and full
complex forward isometries. Small complete basis columns and independent
history certificates retain scratch cleanup. All combined forward/inverse
measurement histories at widths one and two restore coherent target input and
all phase/scratch storage. Width three uses the zero-history roundtrip; no
exhaustive larger inverse-channel claim is made. Constructor providers and
exact inverse dispatch execute with Qualtran/Cirq imports blocked. A native
fixture library additionally goes through the complete resolver naming and
helper-import boundary, physical binding and ordinary circuit.append(op),
with identical emitted payloads to direct typed composition. This fixture is
not represented as successful original Ising serialization. Omitted
Fourier uses the genuine source default; an explicit child invokes its owner.

The child implementation in these replacement tests is the previously accepted
finite SDK snapshot wrapped in a native typed Program with private clean work.
That is a test boundary, not native constructor closure below the walk or a
claim of recursive scalar-node parity. The typed root preserves complete
original call boundaries while this boundary remains explicit. Existing
`QUBITIZATION_PHASE_ESTIMATION_FILE_AUDIT.md` supplies independent production
walk replacement and full branch/phase evidence; those receipts are reused.

The five original constructor examples were inspected without reference edits.
Hubbard small/large and sparse chemistry serialize/parse in both directions
at depths zero and one. Ising object arrays and THC NumPy int64 blocking factors
fail in the original serializer. These are source failures, not native decoder
failures. The initial missing OpenFermion dependency was resolved using the
existing local reference installation before the final intake.

## Remaining complete-constructor dependencies

Original Hubbard/sparse scoped native walk/block-encoding/PREPARE recipes,
LP symbolic/finite construction and ordinary reflection scalar nodes remain
open. Proposal 119 is pending explicit approval. The uncontrolled walk requires
`ReflectionUsingPrepare(global_phase=-1)`, so using a rectangular window alone
does not remove that prerequisite. Formats requires explicit native owners;
it never substitutes same-width children or silently expands the protected
Program opcode vocabulary. Native generic catalogue completion is not claimed.

Actual primitive inventories and source QEC models are reported separately in
the batch resource receipt. Source logical phases have zero logical cost;
existing finite snapshots may emit carrier primitives. Private allocation
release is not numerical evidence of zero amplitude. Whole-repository, GPU,
performance, pending Proposal 116 and broader symbolic domains stay separate.

Local acceptance: 190 concentrated cases, 145 local integration cases, strict
typing on three Python files, Black/Ruff, Compiler scope, repository language,
architecture, IR/API and diff checks. Dependency findings equal the existing
720-line baseline. The original five-factory intake and prior complete-file
execution evidence are reused; original slow numerical/notebook tests are not
rerun as part of this constructor dependency group. Six resource rows retain
all emitted names, including matrix leaves. For the three-bit forward fixture,
16 T-family, 54 CX, 33 opaque matrix leaves and four measurements are emitted
on a seven-wire supplied pool; the independent source model reports one T,
four And, 98 Clifford, four measurements and six logical peak qubits. The
supplied pool width is not reported as a measured logical allocation peak.
