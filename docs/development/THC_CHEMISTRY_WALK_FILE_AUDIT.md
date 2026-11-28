# Tensor hypercontraction tutorial and walk audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/thc`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0416 | `notebook_utils.py` | `a07567e3f8f1d002af00512d9f4a38024edb364f` |
| Q-0421 | `thc.ipynb` | `17e405ec4937ecd571a0ff2389b42f048658546c` |
| Q-0422 | `walk_operator.py` | `30de4297b0027dee7b552ccc31682fd3e5e27e12` |

## Tutorial and diagram utilities

The native tutorial covers all three original examples. It uses
rank ten and four spin orbitals for uniform preparation, the
same seeded integrals and four-row lookup blocks for signed
PREPARE, and rank ten, eight spin orbitals and twelve angle bits
for SELECT. The seeded integral arrays match the reference
element by element. All three full model gate categories and
peaks match the reference.

The two source cost classifications also match. The uniform
arithmetic bin gives 276 T equivalents, and PREPARE data loading
gives 248. The depth-one call graphs expose the grouped local
Clifford and rotation budgets. The weighted flame diagrams include
exclusive local costs as leaves so that the folded stack weights
sum to the full modeled budget. The two register scores retain
the uniform flag lifetimes and the signed preparation's named
lookup outputs. They depict the original composite, including the
source sentinel boundary.

The native view utility uses the shared typed generalizers for
shape bookkeeping, allocation, rotation parameters and conjunction
control values. It groups explicit one- and two-qubit Clifford
nodes and phase S nodes. QROM labels are local graph annotations.
The original Cirq wrapper and process-wide repr overrides belong
to the source environment. The native views use no Cirq dependency
and do not modify global class representations.

The 40-cell original notebook ran independently with the pinned
bundled FlameGraph script. The native 31-cell notebook renders
nine SVGs and two PNG register scores in an independent kernel.
It includes a coefficient-based native append example.

## Walk factory and reflection scope

The native integral factory matches the source alias preprocessing,
default blocking, LCU normalization and gate budget. The source
forms the union of selection registers with a set. The native
source signature uses a deterministic order and matches that
union by name, dtype, shape and side. The original factory
takes the one-body eigenvalues and THC tensors. The physical
binding additionally takes explicit one- and two-body Givens
networks.

Exact and measured eight-bit walks run through
`circuit.append(op)`. Existing regression compares both
directions with the independent full-signal reflection, verifies
complex amplitudes and workspace cleanup on the exercised
histories, and checks disjoint feedback slots. The source cost
model uses the narrower reflection and charges the abstract
QROAM eraser. The unblocked walk retains the source peak failure.
Native emitted counts form separate evidence.

The concentrated suite passed 188 cases. Strict typing,
format, ownership, architecture, IR, and public API checks passed.
Q-0416 and Q-0421 close the native tutorial utilities and complete
tutorial. Q-0422 retains the original erase and narrow-reflection
limits, and the physical rotation networks remain caller-supplied.

## Literal source factory dependency ownership

`ecosystem._thc_walk_component_library.capture_thc_walk_context` captures the
actual coefficient-factory object into owned `WalkContext` records. It preserves
the particular source object's set-derived port order, integer lookup columns,
RIGHT output lifetimes and explicit SELECT direction. No source object remains
in the returned context. Compiler consumes these records after capture without
Qualtran, Cirq or Kickmix. The source factory's original normalization and
blocking remain unchanged. Pure source gate/peak queries now accept the existing
`PrepareTHCChemistry` owner without constructing fictitious Givens networks.

The blocked small factory has one block call, fifteen SELECT calls, seventeen
PREPARE calls and forty ordinary reflection calls. Positive reflection has
thirty-six calls. All comparable operands, producers and output connections
match the unmodified source. The literal reflection consumes allocated RIGHT
junk in inverse PREPARE, produces fresh RIGHT junk in PREPARE, and frees THRU
junk, the phase target and RIGHT junk in source order. This preserves a logical
graph; it does not prove that an invalid source clean Free is executable.

The positive block's forward specialized SELECT leaves its external control as
a bystander. Its owned inverse reverses that actual stored forward graph. The
source's direct `Controlled.adjoint()` recipe instead invokes a generic
controlled inverse SELECT with a `ctrl` port. Those are distinct source paths:
we do not claim the stored inverse equals the latter constructor or that the
source specialization is a correct controlled unitary.

Zero-block source PREPARE fails on `junk_target0_`; its reflection stays an
explicit bodyless header. Rotation, comparator and QROAM eraser leaves have no
original physical decomposition. Physical binding refuses at those leaves;
whole-factory primitive counts and amplitude evidence are unavailable, not
zero. Corrected mathematical eight-bit walk amplitudes and measured erasure
remain separate previously accepted evidence.

The actual 152-spin-orbital, rank-450 source example and its complete native
context passed: two walk calls, one block call, fifteen SELECT calls, seventeen
PREPARE calls and 660 reflection calls. Native capture took 197.68 seconds; the
complete source connection and valid-Program probe took 438.93 seconds (443.68
seconds including process startup/teardown). The native reflection carries
18,360 definitions. No dense statevector or whole physical primitive count was
claimed for this graph.

Two earlier bounded 280-second attempts exceeded their budget. Interruption
localized repeated normalization of the complete QROM inside generic adjoint
matching. PREPARE now records explicit inverse recipes at each actual imported
call; the ordinary forward Program is unchanged. Four entire forward Programs
match the published baseline, and their inverse reachable graphs are compared
up to function names. QROM and successful call-port admission reuse also preserve
24 complete Programs against the old builder and old lookup implementation.
The final source-backed suites exercise every literal PREPARE call and return.
The large receipt precedes the equivalent recipe-recording refactor; the final
small full-graph and baseline comparisons cover that refactor explicitly.

Small source gate/peak categories reuse the accepted THC budget comparisons.
The actual large original resource comparison was interrupted after 269.48
seconds while the source created Soquets; no completed large peak or gate budget
comparison is claimed. This construction limit remains separate from native
emitted counts and from the successful actual large logical context.

Q-0422 closes the complete original coefficient-to-walk factory and direct
source dependencies under faithful-source acceptance. Q-0658 remains open for
the distinct generic positive Controlled.adjoint constructor path. Source
rotation/eraser/comparator absence, invalid sentinel/clean Free and the narrow
signal reflection remain explicit source boundaries, not repaired source code.
