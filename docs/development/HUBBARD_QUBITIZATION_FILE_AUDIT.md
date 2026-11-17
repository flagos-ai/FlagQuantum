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
open for the locked side-one resource failure and source-specific interface boundaries. Q-0353 and
Q-0354 stay open for literal decomposition/count agreement across the
larger sizes, including the eight unchanged-reference discrepancies.


### Side-one PREPARE

The native authoring path retains all four zero-width bounded coordinate
words and the six physical data bits. Both directions treat empty
uniform preparation, swaps and modulo-one increment as identities.
The reference comparison trims the corresponding five calls from the
frozen generator and compares every retained primitive and feedback
operand. The unchanged Cirq exporter fails to decompose the empty
uniform targets and empty swaps. The source QEC and peak width queries
fail at the zero-width ModAddK connection.

The native forward circuit emits 15 T/T-dagger instructions and two
measurements. It reserves eight physical wires, including two clean
work wires. All four histories match the normalized reference on zero
and coherent inputs, with work cleared. All sixteen forward and
adjoint history pairs return the zero input. The source resource method
reports the unavailable reference model for this size.


### Side-one SELECT

The native bind path keeps the four zero-width coordinate ports and
the full spin address. Its single-orbital Majorana stage selects on that
spin bit. It reuses the existing flat Majorana compiler, including the
zero-feedback case. The spin-up stage retains its predicate and single
Z target. Four empty coordinate swaps are identities.

The locked top-level generator runs. Its graph export fails on the
zero-width y connection, so the source QEC and peak queries fail.
The locked Majorana child fails in its singleton coordinate traversal,
and the spin-up child rejects an empty selection word. The independent
primitive comparison takes the frozen top-level schedule, omits its
empty swaps, invokes the frozen SelectedMajoranaFermion on the spin bit,
and substitutes the single spin-up phase on the retained operands.
The negative control uses the source X sandwich.

The native uncontrolled case emits 14 T/T-dagger instructions with no
measurement and one work wire. Each control polarity emits 26 T/T-dagger
instructions and three measurements with two work wires. These are
physical counts. The tests check the complete complex action on a
coherent state spanning every populated basis value, all small feedback
histories, exact inverse recovery, and work restoration. Source
side-one resource estimation remains unavailable.

### Majorana and spin-up Z score

The native concrete example scores retain the source coordinate, spin
and potential labels, the filled control circles, and both mathematical
target captions. The reference draws a filled circle even when a leaf
is declared with control value zero. The empty connector is a
TextBox without a font-size field. Drawer carries that blank record
through JSON and skips it during rendering.

The frozen and native score records match for the two Majorana Paulis
and spin-up Z at sides one and three, with absent, zero and positive
control. The native scores also render in Matplotlib. Execution and
resource evidence is accounted for separately in the preceding sections.

### Larger SELECT generator streams

At sides two, three, five and eleven, the full forward operation
matches the original stage schedule with no control and both control
polarities. The reference traverses the frozen top-level generator,
then calls the frozen Majorana and spin-up unary generators using
the actual child operands. Each stage uses the same physical work
pool and the feedback comparison tracks the last measured wire
assigned to each logical result slot. The exact small graph also
matches the ordinary whole-graph exporter. For the larger positive
control cases, that exporter has the same opcode counts and a
different order. The stage-generator comparison retains source order.

| Side | Uncontrolled T/T-dagger, measurements, work | Controlled T/T-dagger, measurements, work |
| --- | --- | --- |
| 5 | 578, 120, 7 | 590, 123, 8 |
| 11 | 2526, 600, 9 | 2538, 603, 10 |

The large tests compare all complex amplitudes on coherent inputs
supported at first, middle and last coordinate values. Four
representative records per configuration verify inverse recovery and
zero work. The emitted CY count is exactly twice the site count.
The eight frozen declaration-versus-decomposition comparisons still
exhibit the documented missing CY category. The gate estimates are
recorded independently of the emitted stream.
