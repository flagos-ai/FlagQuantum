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
