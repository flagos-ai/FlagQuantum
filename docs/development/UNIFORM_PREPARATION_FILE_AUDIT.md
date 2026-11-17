# Uniform preparation source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `state_preparation/__init__.py` | `024a57c221ae73f95645dbaa4247f423b037a932` | Q-0690 |
| `prepare_uniform_superposition.ipynb` | `90f6b5c08b7834551e2a3a6d091a66f43dbdaff4` | Q-0694 |
| `prepare_uniform_superposition.py` | `e567c913bff5777e899f38afae09e4c861c98d0a` | Q-0695 |
| `prepare_uniform_superposition_test.py` | `25f19b8539880ef9f0ce3ebe954cd885562780d4` | Q-0696 |

Run `jupyter notebook examples/notebooks/prepare_uniform_superposition.ipynb`.
The native notebook preserves the original eleven cells, displays the
two five-state signatures and source call graphs, and appends the
concrete bodies. It also displays the third, symbolic factory model.
Both notebooks ran in independent kernels. The native run rendered
three SVGs with Qualtran, Cirq and quimb imports forbidden.

## Original factories and tests

The active control prepares the first five labels in the tutorial
and leaves the clean workspace available for reuse. Both concrete
factories are compared at source-equivalent lowering depth, including
the comparison, rotation angles, measured predicates and feedback.
The gate stream and full complex state checks include the adjoint and
a coherent spectator. Inactive branches of the controlled body are
checked with their relative phase retained.

The native tests cover all 38 combinations from the source state
grid and both variants of each of the 13 T-complexity bounds. The
unchanged original suite passed 65 tests including its notebook.
The native tests also compare the source wire labels and mixed-control
ordering, full direct-call declarations of all three original
factories, and modern gate categories and widths. Earlier small
cases project all native measurement histories. The larger factory
and grid cases check complete complex trajectories and zero scratch.

| Five-state body | T/T† | CX | Native CH | Native Rz | Measurements | Source categories | Wires |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Plain | 32 | 64 | 0 | 2 | 8 | 8 And, 57 Clifford, 2 rotations, 8 measurements | 8 |
| Controlled on one | 36 | 70 | 3 | 2 | 9 | 9 And, 67 Clifford, 8 rotations, 9 measurements | 9 |

The symbolic count uses the source comparison and conjunction calls,
their adjoints and its declared Hadamard and Rz multiplicities. It
requires a concrete count before append.

## Abstract control dispatch

The port declaration retains the full target width and the abstract
control length `c`. For a concrete state count `n = 2**k * L` and
abstract controls, the source graph applies its comparison and phase
model to the odd factor `L` and its high-word width. The predicate is
the source's all-positive declared MultiAnd. The declared calls omit
the abstract controls, while the peak width still includes those ports.

| States | Odd factor and high width | Source And | Source Clifford | Source measurement | Source rotations | Source peak |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 10 | 5 and 3 | 8 | 45 | 8 | 2 | `Max(8, c + 4)` |
| 18 | 9 and 4 | 11 | 55 | 11 | 2 | `Max(10, c + 5)` |

The original direct call keys and both cost profiles were compared.
The source rejects literal factored predicates shorter than three
controls on the abstract path. The native declaration preserves that
failure. A literal control length resolves to positive controls, and
its append stream matches the explicit concrete control tuple. The
existing primitive counts above describe concrete execution.

## Remaining boundaries

Q-0694 and Q-0696 are covered. Q-0695 remains open for executable
symbolic controls and unresolved source dispatch boundaries. The
source symbolic graph omits control predicates. Its singleton and
one-bit symbolic peak queries exhibit the failures retained by
the existing resource tests. A concrete count with a tuple of symbolic
control values also fails in the locked source. Q-0690 is covered by
the state-preparation package audit.
