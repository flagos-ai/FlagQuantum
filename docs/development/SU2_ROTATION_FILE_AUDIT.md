# SU2 rotation source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The code preserves the original
Google LLC 2024 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0256 `su2_rotation.ipynb` | `162c050b6579ba062b242f4391ccbfc4c6a2f6a6` | Closed. All source topics and the three examples are present in the native tutorial. |
| Q-0257 `su2_rotation.py` | `91283ba1822f1cc3c1b1cedc37a4dfa74ec21085` | Exact bound lowering, inverse, recovery and symbolic resource calls verified. The external object protocols remain open. |
| Q-0258 `su2_rotation_test.py` | `61b4d09dbc41dac7590d4655a6b5ecb6c188228f` | Closed. The seven source test functions map to the native scenarios below. |

The source binds one THRU QBit named `q`. Native emission applies the
scalar followed by Rz, Rx and Rz on the caller's wire. The scalar uses that
wire as its physical carrier, so there is no extra work wire or measurement.
The adjoint swaps and negates the phase parameters and negates the scalar
shift. Tests compare source gate operands and full complex spectator states
on both target wire positions. The standard and seeded matrix tests also
preserve absolute phase.

The matrix and ZXZ converters retain the pinned source conventions, including
the near-singular `1e-8` threshold. Within that threshold the source can drop
a phase contribution. The native stream is compared with the recovered
source matrix and the residue against the input matrix is recorded separately.
Symbolic call declarations preserve scalar precision and one-third of the
supplied precision on each rotation. Automatic foreign allocator callbacks,
foreign `_unitary_` and tensor wrapper protocols, and source-specific wire
symbol dispatch remain open under Q-0257.

## Original tests

| Source test | Native evidence |
|---|---|
| `test_decompose_SU2_to_single_qubit_pauli_gates`, `test_tensors` | All twenty original seeded parameter tuples agree across append matrices, original tensors, source formula, and the original Cirq decomposition. |
| `test_su2_rotation_gates`, `test_su2_rotation_gate_example_unitaries_match` | All three factory signatures and executable examples are checked. Exact Hadamard and T are compared as full matrices. The source count mode for the generic example is skipped because its counts derive from the decomposition. |
| `test_from_matrix_on_random_unitaries`, `test_from_matrix_on_standard_gates` | All twenty original seeded unitaries and the five original standard gates are recovered and appended with full complex matrix comparisons. |
| `test_call_graph` | Existing symbolic tests compare the scalar and three rotation children against the original exact parameters. The notebook renders the concrete graph and preserves multiplicity for equal rotations. |

## Emission and resource accounting

Each of the three concrete examples emits two X and two phase instructions
for the nonzero scalar carrier, followed by two Rz and one Rx. The source
graph contains four logical calls, a scalar with zero QEC cost and three
rotations before classification. The notebook prints the individual source
cost categories, which can classify special angles as Clifford or T. All
three examples append without a measurement or other auxiliary wire.

The original 12 cells and the native 14 cells cover the SU2 formula, three
examples, their signatures and the call graph. The added cells check the exact
complex matrices and inverse restoration through the existing append path.

Both notebooks ran in independent kernels. The native kernel rejected
Qualtran, Quimb and Cirq imports. Both native SVGs were inspected. The
original suite had ten passes and one factory count-mode skip. All 168
concentrated and 112 integration cases passed. Strict typing, formatting,
Compiler scope, architecture, multilevel Program IR and public API checks
also passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_su2_rotation.ipynb
```
