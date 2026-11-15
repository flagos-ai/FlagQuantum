# Hubbard qubitization file audit

The frozen sources are under `qualtran/bloqs/chemistry/hubbard_model/qubitization`
at Qualtran v0.7.0, commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
The two original notebooks were recovered by Git blob id and passed in
independent kernels without changing the source files.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0349 | `hubbard_model.ipynb` | `11862cbbb6894da65d4636f06b9e8c654a1d02cc` |
| Q-0350 | `prepare_hubbard.py` | `cfdbf20b0baf5ef1f5fa5a7e21826d7f51f5aec5` |
| Q-0351 | `prepare_hubbard_test.py` | `ef956c7d0b68de1a5805d76f2c6dd4144adc9485` |
| Q-0352 | `select_hubbard.ipynb` | `c9b80d41b44f907f81ae5bfabcac647515e4cb63` |
| Q-0353 | `select_hubbard.py` | `6d912f2ae1f63730cc296cfc35f68eb9ccbabb84` |
| Q-0354 | `select_hubbard_test.py` | `c3307b5f562b8ac01e397ed6a7b1534088e7fbd4` |

## Original examples

The two native notebooks preserve the 4×4 signatures and depth-one
resource queries. The SELECT notebook also shows the controlled
signature, the eleven source calls and operands for 11×11 controlled
SELECT, and the paired symbolic and 3×3 Majorana and spin-up Z signatures.
The coordinate words retain bounded unsigned extents. The system word is
opaque, with spin then y then x ordering (x varies fastest). PREPARE's two
retained direction bits in `temp` are logical words. Compiler-owned
workspace and classical feedback slots are separate.

Run both native examples in independent kernels with
`python -m pytest tests/integration/test_register_graph_notebook.py -k register_hubbard`.
The example resource models do not emit an executable circuit. The
existing executable PREPARE and SELECT owners retain `circuit.append(op)`
and delegate finite-range operations to Compiler.

## Results and unchanged-source differences

- The original source suite reported 35 passed, ten skipped and three
  expected failures. Eight Majorana count comparisons failed. The source
  test generalizer leaves `CirqGateAsBloq(cirq.CY)` in the decomposition.
  The source declarations have matching compute/erase And counts and omit
  exactly `2*n*n` such CY leaves at sides 2, 3, 10 and 11, both without a
  control and with positive control. Tests preserve both count views.
  Source skip modes include unsupported symbolic decomposition and
  strict bounded-to-opaque wire typing.
- Source reference tests cover the 2×2 PREPARE and SELECT gate paths and
  larger size subgraphs. Full complex sparse branch witnesses and scratch
  zero are covered by native PREPARE and SELECT regressions, including
  negative controls and the independent adjoint. PREPARE's 2×2 case
  exercises all eight measurement histories. Larger fixtures and SELECT
  use representative histories.
- The 11×11 controlled SELECT model estimates 603 And operations,
  18 controlled swaps, 603 measurements and a 273-qubit peak. In
  independent executable 3×3 fixtures, PREPARE emits 155 T/T† and
  23 measurements with six work qubits. Controlled SELECT emits
  242 T/T† and 43 measurements with six work qubits.

Q-0349 and Q-0352 close the tutorials. Q-0351 closes the original
PREPARE test file with the source checker modes recorded. Q-0350 stays
open for side-one and source-specific interface boundaries. Q-0353 and
Q-0354 stay open for literal decomposition/count agreement across the
larger sizes, including the eight unchanged-reference discrepancies.
