# Dense rotation preparation

The classical rotation tree builds a table per big-endian target prefix and a
full-address phase table. The source quantizes the inverse amplitude angles
before recalculating the phase offsets, so the uncompute stream is built from
its own tables. Empty subtrees use the source zero-probability rule.

The compiler's programmable lookup follows the source QROM, gradient add,
QROM sequence. The angle word occupies the first part of the caller's clean
workspace. The remaining qubits and classical slots are reused by both
children. The gradient register is caller-owned. A negative-exponent gradient
eigenstate gives positive phase kickback and is restored.

The dense preparation uses each earlier target prefix for selection and
appends the current target bit to the all-one controls between two Rx gates.
The full-address phase step borrows a clean final work qubit as its local
control. The source runs this step first on uncompute. A one-coefficient
input has a valid classical angle table, but the locked dense decomposition
fails with a missing target port. The dense emitter requires a nonempty
target register. Symbolic sizing and sparse permutation are separate.

For the four coefficients (1, i, -1, -i) / 2 with eight phase bits and no
external controls:

| Evidence | Forward | Uncompute |
|---|---:|---:|
| Native T/T† | 108 | 108 |
| Native CX | 297 | 301 |
| Native measurements | 27 | 27 |
| Native work qubits | 16 | 16 |
| Native reused classical slots | 7 | 7 |
| Source model Toffoli | 18 | 18 |
| Source model And computes | 6 | 6 |
| Source model measurements | 6 | 6 |
| Source model total qubits | 21 | 21 |

The source `AddIntoPhaseGrad` has fixed-point action and a cost model, with
no executable gate decomposition. The comparison reads the original
component graph, follows each source QROM through its actual decomposition,
and checks the shared measured Add against its own source decomposition.
The eight-bit complex check executes three measurement histories per
direction and compares with the source's two-bit tensor on a gradient
eigenstate after proving that every angle word scales exactly for this
example. The two-bit small cases enumerate all histories when there are at
most four measurements. Full complex outputs include the work qubits.

[Locked Qualtran source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_via_rotation.py)
