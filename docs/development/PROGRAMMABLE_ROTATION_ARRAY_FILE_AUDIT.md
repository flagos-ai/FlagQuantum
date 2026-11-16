# Programmable rotation array source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `programmable_rotation_gate_array.ipynb` | `a08ae58ab5eb1ddb3ba6e5cd21a95defade9f864` | Q-0678 |
| `programmable_rotation_gate_array.py` | `7c6b33d09150c49574a9e5a4fb218a504254a1b4` | Q-0679 |
| `programmable_rotation_gate_array_test.py` | `f3486cab5190ecb4b482f8ff0af11b68c0149794` | Q-0680 |

Run `jupyter notebook examples/notebooks/programmable_rotation_gate_array.ipynb`.
The native notebook preserves the ten original cells and adds a body
compilation example through `circuit.append(op)`. Both source and native
notebooks ran in independent kernels. The native kernel rendered two SVGs
with Qualtran, Cirq and quimb imports forbidden.

## Exact source fixture

The tutorial uses angles `[1, 2, 3, 4]`, a two-bit angle word, and a
Z-power target. The five logical through wires hold the bounded selection,
angle word and target. One extra wire is borrowed sequentially by the
QROM and the measured controlled rotations. The emitted route checks the
scope of the borrowed wire and its measurement slot. The angle word and
scratch return to zero for each selection and measurement history.

The original generator uses XOR tables `(0,1,1,2)`, `(2,1,3,2)`,
and `(2,0,2,0)` and controlled powers `1/8`, `1/4`, and `1/2`.
The tests verify the ordered Cirq operations and matching Bloq children
without the structural Split/Join nodes. They expand the QROM leaves and
the sourced CZPow compute, phase and measurement-feedback sequence. The
direct path retains a controlled-phase instruction per CZ power.
Both routes exhaust their raw measurement histories on one coherent
superposition of all addresses and both target values.

| Resource view | T/T† or modeled T | CX | Uncontrolled phases | Controlled phases | Measurements | Wires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Literal direct phase route | 24 | 52 | 0 | 3 | 6 | 6 reserved |
| Literal measured CZ route | 36 | 70 | 3 | 0 | 9 | 6 reserved |
| Source QEC categories | 1 | — | 1 general rotation | — | 9 | 6 logical |

The source model also declares nine And and 32 Clifford operations. Native
phase instructions carry exact angles before synthesis.

## Original test grid and evidence

The unchanged original suite yielded 49 passes and 18 failures. All 18
failed during Cirq's reduced-state calculation of the auxiliary register
with a NumPy eigensolver nonconvergence error. The native audit exercises
both original angle lists, every kappa from 1 to 11, and all three
constructors, including controlled X/Z interleavers with their own target.
It verifies the bound source gate stream, each QROM XOR map and every
local erasure invariant, followed by two sparse coherent trajectories
against independently assembled per-address target matrices. The sparse
comparison includes relative phases between addresses and zero load and
scratch. Small cases from the existing suite project all raw histories.

## Remaining source boundaries

Q-0678 and Q-0680 are covered. Q-0679 remains open for general import and
synthesis of external arbitrary gate objects. The existing native
callbacks can bind both original custom interleavers and their unrelated
target. Q-0666 remains open for the other rotation exports.
