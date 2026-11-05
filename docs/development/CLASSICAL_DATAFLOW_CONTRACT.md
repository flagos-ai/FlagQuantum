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
All indices are nonnegative concrete integers, excluding bool. Inputs must have
been defined earlier by measurement or classical computation. Validation reads
the old sources before declaring the destination defined. Repeated inputs cancel
by parity; empty input writes zero; destination aliases read its previous value.
Reassignment is allowed, with program-level SSA creating a fresh boolean result.
Conditional classical writes are outside this version and are rejected.

## Execution and interchange

The operation replaces its destination with XOR of source values, leaving inputs
and quantum state unchanged. Source bits are snapshotted before the write. Reading
an undefined source is an error even when duplicated sources would cancel.
Scalar and batched trajectories implement identical semantics, including observed
readout bits. True-bit audit records are transformed separately where present.
Classical work does not consume quantum noise, measurement counts, or RNG draws.
Density ensembles retain measurements feeding XOR transitively and evaluate XOR
per branch. OpenQASM 3 uses boolean XOR assignment; restricted provider dialects
must explicitly refuse unsupported classical computation.

The program graph uses boolean `bool_zero` and binary `bool_xor` definitions;
conditions reference those definitions through the existing `ConditionRef`.
The public instruction and program graph must round-trip and pass def-use/kind
verification. Existing circuits and serialized payloads remain unchanged.

KMX lowers assignment to `BIT_STORE0 bTarget` followed by
`BIT_INVERT bTarget if bSource` for each non-target source. If the old destination
appears an odd number of times, it is preserved instead of initialized; all
destination occurrences are removed from the inversion list. This handles
aliasing without an implicit scratch bit. Target validation distinguishes writes
from read/modify/write and checks conditions and initialization before use.
The unreleased tagged KMX subset admits these native classical opcodes while
preserving earlier encodings byte-for-byte. Unsupported opcodes still fail closed.

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
