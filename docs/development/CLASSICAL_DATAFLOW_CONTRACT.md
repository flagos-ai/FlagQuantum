# Internal classical dataflow contract, version 1

Implementation authorization: the user requested a shared executable classical
XOR/parity layer in integration PR #1084 on 2026-11-05, before further resource
parity claims. This contract lands with a separate reference fake and conformance
tests before its Runtime and Compiler consumers.

## Ownership and representation

Core owns `ClassicalXor` and its instruction parser. Compiler binds temporary
classical slots and lowers the operation. Runtime executes classical dataflow;
Simulation consumes the same Core record for density ensembles. Neither imports
Compiler. There are no new root exports, public method signatures, dataclass
fields in `Instruction`/`CircuitIR`, quantum gate opcodes, or snapshot updates.
The internal unreleased dynamic instruction vocabulary gains `classical_xor`.

An instruction has no quantum wires or matrix, `params={"sources": tuple[int,
...]}`, and exactly `metadata={"is_dynamic": True, "classical_bit": target}`.
An optional Boolean `constant` parameter XORs a literal into the assignment.
The encoder omits false and uses `constant=True` to complement the parity.
All indices are nonnegative concrete integers, excluding bool. Inputs must have
been defined earlier by measurement or classical computation. Validation reads
the old sources before declaring the destination defined. Repeated inputs cancel
by parity; empty input writes the literal (zero by default). A destination alias
reads its previous value. XOR of a slot with true, assigned to the same slot,
inverts it in place.
Reassignment is allowed, with program-level SSA creating a fresh boolean result.
The v1 baseline rejected conditional classical writes. The internal
[guarded-dataflow addendum](CLASSICAL_GUARDED_DATAFLOW_CONTRACT.md) now admits
a canonical conjunction on already-defined destinations, with inactive-value
preservation and separate Runtime/Simulation conformance. Public schemas and
unconditional encodings remain unchanged.

## Execution and interchange

The operation replaces its destination with XOR of source values and the literal.
The write affects only the destination slot. Source bits are snapshotted before the write. Reading
an undefined source is an error even when duplicated sources would cancel.
Scalar and batched trajectories implement identical semantics, including observed
readout bits. True-bit audit records are transformed separately where present.
Classical work does not consume quantum noise, measurement counts, or RNG draws.
Density ensembles retain measurements feeding XOR transitively and evaluate XOR
per branch. OpenQASM 3 uses boolean XOR assignment; restricted provider dialects
must explicitly refuse unsupported classical computation. Backend package admission
remains blocked until a qualified classical-computation capability is available;
textual OpenQASM export alone is not hardware qualification.

The program graph uses boolean `bool_zero` or `bool_one` seeds and binary `bool_xor` definitions;
conditions reference those definitions through the existing `ConditionRef`.
The public instruction and program graph must round-trip and pass def-use/kind
verification. Existing circuits and serialized payloads remain unchanged.

KMX lowers assignment to `BIT_STORE0 bTarget` followed by
`BIT_INVERT bTarget if bSource` for each non-target source. If the old destination
appears an odd number of times, it is preserved instead of initialized; all
destination occurrences are removed from the inversion list. A true constant
appends one unconditional `BIT_INVERT bTarget`. In-place NOT lowers to just
that operation. This handles
aliasing without an implicit scratch bit. Target validation distinguishes writes
from read/modify/write and checks conditions and initialization before use.
The unreleased tagged KMX subset admits these native classical opcodes while
preserving earlier encodings byte-for-byte. Unsupported opcodes still fail closed.

Positive conjunctions on reversible quantum gates lower to nested
`PUSH_CONDITION` and `POP_CONDITION` scopes plus the last inline condition.
Singleton conditions keep their direct encoding. Scope markers introduce
no work bits.

### Guarded effects in the internal KMX target

The locked modular arithmetic generators use the existing native HMR and R
operations inside condition scopes. HMR measures in the X basis, writes the
result and returns the wire in zero. R discards the X measurement outcome
and returns the wire in zero. With a false guard both operations leave the
wire untouched, and HMR preserves the old result value.

A scoped HMR may define a result for subsequent operations in the same
scope and its descendants. On scope exit only previously defined slots
remain visible. Inline guarded HMR requires a previously defined
destination. Scope entry snapshots its active shot mask. HMR may overwrite
a guard slot, which affects later inline conditions or nested scope entries
that read it. The enclosing scope retains its saved mask. The v1 baseline excluded scoped classical writes. The
[guarded-dataflow addendum](CLASSICAL_GUARDED_DATAFLOW_CONTRACT.md) admits
scoped BIT_INVERT on a defined destination; scoped BIT_STORE0 remains refused. The compiler can bind this target to the local flat circuit. It owns the
classical output block, initializes its slots, and copies each scope guard
to a temporary slot on entry. The copy is unconditional so that every
subsequent predicate is defined, even when an outer scope is inactive.
Scratch slots are reused after scope exit. An inline guard that aliases
the result is also snapshotted for the duration of the effect.

A fragment may declare an ordered set of classical input slots. The
optional `inputs` list in the KMX metadata tag preserves this declaration.
Binding reads these slots from the enclosing circuit and initializes only
fragment-owned slots. A standalone native runner supplies the declared
inputs before executing the fragment.

Only compiler instructions carrying the internal `guarded_effect` marker
may condition measurement or reset. They use one conjunction of defined
bits. A guarded measurement must update an already defined result. Local
trajectories and density ensembles preserve inactive data and result
slots. R binds to a guarded H followed by reset, which discards the X
outcome. The typed program level refuses this marker until it can express
the conditional result merge. Textual OpenQASM can express the effect;
provider qualification stays separate. Existing public builders and KMX
streams keep their encoding.

## Consumer and resource acceptance

Parity-controlled Z/CZ computes a reusable classical result before applying one
phase gate. Single-input conditions remain direct; empty parity applies no phase.
No expansion into extra quantum phase gates or Boolean minterms is permitted.
The finite-field consumer reserves a caller-owned scratch classical bit distinct
from measurement slots, reuses it for each source parity, and preserves source
call order, including its final unused parity computation. QROAM consumes the
same shared parity-phase interface and preserves its singleton-feedback stream.

Report quantum operations, measurements, source parity computations, classical
stores/inversions, peak classical slots, and extra quantum workspace separately.
For GF(256) product erasure the acceptance target is 64 actual conditional CZ,
8 HMR, zero T, zero extra quantum workspace, 8 measurement slots and one reusable
parity slot. Compare unmodified Qualtran v0.7.0 logical measurement/parity/CZ calls
and their executable target forms at the same layer.

Tests must cover all XOR truth tables, empty/duplicate/alias inputs, overwritten
destinations, undefined inputs, dynamic scalar/batched and density execution,
SSA and serialization round trips, KMX/native execution, unchanged singleton QROAM,
all finite-field measurement branch phases, complex amplitudes and quantum cleanup.
Whole-PR CI and GPU acceptance remain separate from these local CPU checks.
