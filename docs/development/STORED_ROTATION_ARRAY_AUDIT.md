# Stored programmable rotation-array execution audit

## Scope and ownership

The complete frozen `rotations/programmable_rotation_gate_array.py` file defines
a shared batched XOR traversal and the concrete same-target interleaving owner.
The reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. No source file is modified.
Classification is `single_device_fast_path`.

The existing format resolver now reconstructs an unregistered named component
when its library entry contains an explicit stored connection graph. It reuses
the same typed composite builder as an original CompositeBloq. Registered
constructor factories retain precedence. Every executable descendant still needs
its registered factory or an explicit stored graph. Aggregate counts cannot
supply missing instructions or operands.

Compiler owns reconstruction, dependency order and validation. It imports no
external constructor or numerical kernel. The existing QROM, rotation, identity,
fixed-gate and storage-view owners supply the direct implementation dependencies.
Caller-owned selection, load and rotation registers remain THRU. Words are MSB
first. The load bank returns to its original value after the final XOR unload.
Private predicate and controlled-rotation work is measured, erased and restored
to zero through its existing component owner.

## Source serialization and stored execution

The source rotation-array classes are not attrs or dataclass constructors. The
original serializer therefore stores no angles, kappa, rotation gate or
interleaver constructor arguments. Original shallow reconstruction fails for
missing required arguments. Native shallow constructor reconstruction still
refuses that missing body and metadata.

At depth one the original serializer retains the actual connection graph, QROM
data, precise controlled rotation exponents and interleaved gates. That is an
executable body independent of class reconstruction. Native resolution preserves
those calls and operands. A separate actual OnEach fixture proves the same path
also retains named stored graphs with real constructor metadata. Explicit
CompositeBloq argument rejection remains unchanged.

## Verification and boundaries

Stored multi-table roots compare every recursively comparable source call and
operand. Complete dirty-bank complex operators are compared to unchanged source
contraction, with all-history erasure certificates and private work zero. One
8-bit angle table exhausts all 256 measurement histories, including raw branch
weights, complex phase and zero work. Program copy, deterministic library bytes,
CircuitIR round trips and ordinary `circuit.append(op)` are checked.

Arbitrary local rotation/interleaver payloads still require an independently
owned executable leaf adapter when their stored source decomposition is absent.
Existing direct source-binding tests certify arbitrary local matrix leaf
replacement separately. This batch does not promote absent payload ownership to
a source gate decomposition. The two generic rotation-array constructor
identifiers remain partial because their shallow constructor metadata is lost.
The source's existing angle, kappa and custom-target boundaries remain recorded
in the original array file/source-binding audits.

Primitive inventories remain separate from analytical source estimates and local
matrix-atom synthesis estimates. This is a local compiler/execution batch and
carries no GPU or performance claim.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 195 cases. Related local integration passes
145 cases. Strict mypy checks both touched Python files. Black, Ruff, Compiler
scope, language, architecture, multi-level IR and API checks pass. Dependency
analysis exactly matches the 720 existing findings. Four frozen array/source
blobs, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator are verified.

All 67 unchanged original file tests pass in the existing Cirq 1.6 compatibility
environment. In Cirq 1.7, 49 pass and 18 large-ancilla state-extraction checks
fail inside NumPy `eigh`, with LAPACK reporting an illegal workspace parameter.
The affected domain is kappa 9 through 11 for all three original constructors
and both angle tables. A targeted Cirq 1.7 rerun reproduces the complete trace.
Cirq 1.6 uses a direct pure-state factorization for this helper. The first
Cirq 1.7 full log filename was reused during compatibility rerun, so its counts
and diagnosis are retained in a summary alongside the targeted complete trace.
The successful compatibility log has a distinct retained filename. No reference
source, test or numerical routine was patched.

| Stored execution | Actual native primitives and used wires | Independent original model |
| --- | --- | --- |
| Original four-row example | 36 T/Tdag, 70 CX, three phases, nine measurements, six wires | Nine And, 32 Cliffords, one T, one rotation, nine measurements, six logical wires |
| Two-row 8-bit table | 32 T/Tdag, 54 CX, eight phases, eight measurements, five wires | Eight And, 25 Cliffords, one T, six rotations, eight measurements, five logical wires |

Every measurement retains its conditional CZ and reset in the actual primitive
inventory. Logical load-bank outputs remain caller-owned and are separate from
the private work whose amplitudes are certified to vanish outside zero.
