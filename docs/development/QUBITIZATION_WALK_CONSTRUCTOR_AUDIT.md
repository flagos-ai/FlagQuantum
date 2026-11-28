# Qubitization walk composition and signal environment

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The frozen source blob is `117515efb1e42c58b0fab451c454fbd756585f74`.
Reference fixtures assert the actual Kickmix 0.1 Simulator and Qualtran 0.7.0.

## Ownership and executable scope

Compiler preparation owns the literal block-then-reflection calls and exact
reversed calls. Named THRU words, including explicit signal environments,
belong to the caller. Child Programs own their actual clean work. Positive
control uses independently supplied specialized child pairs; negative and
grouped controls reuse predicate computation and measured erasure.
The existing Algorithms signature implementation delegates to the shared
owner without changing its public signature or execution boundary.

Ecosystem captures the actual source Signature and `source.reflect`, replacing
SDK children immediately with native directional Programs. The source union
uses a set: an independently sorted register order would be incorrect. LCU
signal preparation is identity even though `walk.prepare` exposes the original
PREPARE. Formats requires an explicit source-bound child context, preserves
generic inverse/control order, and refuses unknown owners.

Small SELECT and LCU fixtures check recursive comparable source calls and
operands, full complex operators, exact inverses, both scalar control polarities,
grouped controls, small measurement histories, clean work, SDK-free binding,
and the existing `circuit.append(op)` execution entry. A zero-logical-port
SELECT retains its complex scalar through an explicitly charged clean carrier.
Zero-port PREPARE preserves original Cirq moment ordering rather than assuming
the final PREPARE always has the last instance identifier.

## Signal junk is coherent data

The unchanged physical alias reflection can release nonzero signal junk.
Its projected selection-only tensor is not a valid unitary reference. Tests
force the physical root decomposition and intercept the real QubitManager
allocations to retain the exact signal qubits in source order.

Two normalized two-row alias tables, with mu=1, compare every output complex
amplitude and enumerate all small raw measurement histories. Signal output
is demonstrably nonzero. Keeping that environment and applying the full
reflection again restores the joint initial state. Actual private work is
checked separately to be zero; lease release alone is never that proof.

Native prepared reflection therefore accepts explicit caller-owned names for
all signal junk. Without these names, the caller must establish cleanup of
the conjugated reflection: an exact PREPARE inverse by itself is insufficient.
The automatic alias reflection constructor now refuses implicit release.
Alias forward and exact inverse constructors remain executable independently.

The original source Walk ABI omits these retained ports. A native captured
context can compose them and bind normally; the source constructor adapter
explicitly refuses that expanded context instead of discarding the environment
or weakening exact register checks. This source ABI boundary remains open.

## Resource and original-example boundaries

`g2-walk-constructor-resources-20261125.json` contains eight directional rows
for the executable small SELECT/LCU dependency replacements. Actual primitive
inventories, private lease peaks and live peaks are separate from source
QEC/peak models. For ordinary SELECT, emitted instructions include one CX,
three phase instructions, eight X, two H, one CZ, one S and one S-dagger;
private/live peaks are two/four. The source model separately reports twelve
Clifford, one rotation and peak four. These are different accounting layers.
No model rotation is reported as an emitted T count.

The original Ising, 152-spin-orbital/450-dimensional THC and eight-spin-orbital
sparse chemistry examples have captured descriptor/control metadata and
forward/inverse producer-depth evidence. This is not large-example numerical
execution. Original Ising object-array serialization and THC NumPy-integer
serialization failures remain unchanged. The original already-controlled LCU
root's missing control port also remains a source boundary.

The constructor catalogue count must not increase solely from these witnesses.
Large-example exact child ownership, retained-source-ABI disposition and
remaining constructor coverage still require acceptance. G2 remains active;
this batch neither enters G3 nor authorizes merging the integration PR.

## THC source context and the remaining control-direction boundary

`capture_thc_walk_context` now owns the actual THC factory children and narrow
RIGHT-junk reflection rather than substituting a corrected signal extension.
See `THC_CHEMISTRY_WALK_FILE_AUDIT.md` and
`tests/unit/test_thc_walk_source_context.py`. Its positive pair reverses the
stored specialized forward graph. The source's separate
`Controlled(SelectBlockEncoding).adjoint()` constructor changes specialization
to a generic controlled inverse SELECT. That literal constructor path remains
unclaimed; successful ordinary THC factory capture does not close Q-0658 by
itself. Whole physical gates remain unavailable at the original opaque leaves.
