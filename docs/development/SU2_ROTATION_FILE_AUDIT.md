# SU2 rotation source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The code preserves the original
Google LLC 2024 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0256 `su2_rotation.ipynb` | `162c050b6579ba062b242f4391ccbfc4c6a2f6a6` | Closed. All source topics and the three examples are present in the native tutorial. |
| Q-0257 `su2_rotation.py` | `91283ba1822f1cc3c1b1cedc37a4dfa74ec21085` | Closed with bound lowering, matrix/ZXZ recovery, actual source capture, symbolic allocation, full display and independent resource calls. |
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
supplied precision on each rotation. The shared native SymbolAllocator owns deterministic symbolic lifetime and
`su2_rotation_calls` consumes its five parameters. Native matrix execution owns
the numerical unitary, while Ecosystem terminates the foreign unitary and
tensor protocols. The dedicated Drawer callback preserves source wire labels.

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


## Complete source method disposition

| Source method | Native authority and evidence |
|---|---|
| `signature`, `rotation_matrix`, `my_tensors`, `_unitary_` | One q THRU port. Native full matrices and tensors reuse the existing 20 seeded examples and matrix-recovery suite. Actual source capture and JSON preserve every complex column and coherent spectator state. Unresolved symbolic unitaries remain unavailable, with the original field-specific matrix refusal retained. |
| `from_matrix`, `from_euler_zxz_angles` | Existing native converters retain exact source angle conventions, numerical singularity thresholds and precision. Original seeded/standard matrices, mixed symbolic ZXZ and near-singular residual checks are reused. |
| `build_composite_bloq`, `adjoint` | Native source-builder lowering retains scalar then Rz/Rx/Rz. Actual named capture independently preserves each source direction and exact exported leaf sequence. Both append paths restore the original full state. |
| `is_symbolic`, `arbitrary` | Existing neutral classifier and SymbolAllocator provide deterministic theta/phi/lambda/alpha/eps placeholders in source order. Native call keys preserve exact source parameters and per-rotation precision. The source symbolic rotation refusal and symbolic graph remain separately verified. |
| `__str__`, `wire_symbol` | Drawer owns concrete two-decimal titles, full mixed/symbolic titles and the constant SU2 register box. Whole-atom musical-score geometry and source title strings agree. Symbolic precision alone does not make angles symbolic. |
| `_su2_rotation_gate`, `_hadamard`, `_t_gate`, documentation declaration | Existing three-factory checks and independently executed native tutorial cover all examples, signatures and call graph. Q-0256/Q-0258 evidence is reused. |

`test_su2_source_binding.py` compares actual named snapshots to a separate direct
export of each original decomposition through the feedback importer. It includes
a scalar shift outside the principal range. Both directions use no workspace
or result slots and bind to the target of an entangled two-wire input. All
physical amplitudes are retained. There are no measurement branches for this
source. The three tested parameter tuples emit seven primitives in each
direction in both native paths. Independent source QEC categories and logical
peak one are checked separately from that inventory.

The source builder's logical sequence is scalar, Rz, Rx, Rz. Cirq can schedule
the zero-wire scalar alongside the first Rz and enumerate Rz before scalar. Its
scalar importer uses the coefficient's principal angle, whereas the direct
native builder retains the source exponent's unwrapped angle. These two
representations keep separate exact-stream evidence and agree as complete
complex operators. No cross-boundary gate-order or raw phase-parameter equality
is claimed. The existing original-builder checks remain unchanged.

Q-0257 closes with all source methods assigned to their native owners. Unknown
foreign leaf protocols still obey Ecosystem admission, and unresolved symbolic
parameters stay in the resource/display paths until numerical binding.

The batch regression passed all 1808 cases without skips. Strict typing of both
changed Python files, Black, Ruff, Compiler scope, architecture, multilevel IR
and public API checks passed in the existing reference environment.
