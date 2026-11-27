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

### Zero-selection traversal consumer repair

The complete chemistry package-entry regression exposed six existing side-one
SELECT failures after shared unary traversal began requiring an explicit recipe
for empty selectors. The Hubbard spin-up caller now supplies its existing leaf
at index zero with the retained potential predicate. This reuses the original
spin-up CZ behavior and preserves the compute/erase enclosure for controlled
SELECT. Existing tests cover both control polarities, uncontrolled action,
forward/inverse complex amplitudes, every small measurement history and clean
work. Source side-one decomposition and model failures described above remain
unchanged. This repair does not close Q-0353 or Q-0354.


## Standalone SELECT consumers and exact replacement

The complete `select_hubbard.py` file contains 39 class/function/method
definitions and three documentation specifications. The standalone native
consumers now have independent domain owners. Shared register traversal and
measurement erasure remain in Compiler. Source bytes and the frozen module
inventory retain their original hashes.

| Complete source definition group | Native owner and disposition |
| --- | --- |
| SelectHubbard constructor, post-init, log_m and all register/signature properties | Existing `hubbard_select.SelectHubbard` owns square-lattice admission and named physical layout. Original negative control uses the specialized X wrapper. |
| SelectHubbard decompose_bloq/decompose_from_registers, get_ctrl_system and adjoint | Existing SELECT now composes standalone Majorana X/Y and spin-up components with the unchanged swap and phase schedule. All 30 native before/after streams are exactly equal at sides 1/2/3/5/11, all three control modes and both directions. |
| HubbardMajorannaOperator constructor, log_m, N, register/signature properties and target gate | `hubbard_majorana.HubbardMajorana` owns the X/Y choice and address `(x, y, spin)`. Each coordinate word is MSB first. Target index is `(x * side + y) * 2 + spin`. The clean accumulator precedes shared unary predicates. |
| Majorana build_composite_bloq | The actual unchanged source builder supplies SelectedMajoranaFermion. Its generator is compared against the standalone append path at sides 2/3/5 with X/Y and absent/positive control. The native negative control encloses this positive stream in X gates. |
| Majorana build_call_graph | Existing `source_hubbard_majorana_formula` retains the original And-only declaration. Native zero-control adds the two wrapper X gates separately. Full emitted primitive counts remain distinct. |
| HubbardSpinUpZ constructor, log_m and all register/signature properties | `hubbard_spin_up.HubbardSpinUpZ` owns V and the optional external control. It targets the second half at `side**2 + y * side + x`, preserving the source's different coordinate order. |
| SpinUpZ build_composite_bloq | The actual unchanged source builder packs V before external control and selects y before x. Its emitted child generator matches native operands and feedback. Full graph export has identical opcode counts and can recycle physical temporary names differently. |
| SpinUpZ build_call_graph | Existing `source_hubbard_spin_up_formula` supplies the source analytical row. Native zero control adds two X gates. |
| Both child wire_symbol methods and all three __str__ methods | Existing owned notebook score/symbol evidence covers names, blank connectors, filled controls and target captions. Source spelling is provenance. Native Majorana ownership uses the mathematical spelling. |
| Five original examples and three documentation specifications | Existing concrete/symbolic notebook and original autotest evidence is reused. The original source unit suite was independently rerun, including its failures and skips. |

Both standalone components are involutions on valid lattice addresses. Users bind
named ports and caller-owned clean workspace with `make_on`, then use
`circuit.append(op)` on DynamicCircuit. Standalone tests include reversed physical
binding, a coherent spectator, complex amplitudes, JSON, repeated append with fresh
feedback slots and work cleanup. Side-one and side-two fixtures enumerate every
measurement history. Side-three fixtures use two alternating histories, in
addition to existing complete SELECT representative histories and inverse tests.

The actual package-export constructor libraries for PREPARE, SELECT, Majorana X/Y,
SpinUpZ and the walk factory round-trip every protobuf field, reconstruct through
the original resolver and execute owned append/model paths with external SDK
imports blocked. The walk source forms its signature from named sets. The native
handoff preserves each captured named physical port rather than assuming tuple
order, and retains the two persistent direction bits separately.

### Retained source boundaries

- Original standalone Majorana with `control_val=0` raises KeyError for the
  missing control connection. Native negative control is explicitly an X
  sandwich around the positive stream.
- Original SpinUpZ builds exactly the same graph for control values zero and
  one. Native zero polarity uses an X sandwich and accounts for its two X gates.
- Original side-one Majorana construction raises AssertionError, while SpinUpZ
  raises NotImplementedError for empty selection. Native side-one physical
  action passes all small histories. Source resource formulas remain unavailable
  at that size.
- The eight original Majorana count tests fail with Cirq 1.7.0 because its
  CYPowGate is unrecognized by the frozen Clifford classifier. With Cirq 1.6.1,
  ControlledGate is recognized and all eight pass. Fresh unchanged-source runs
  are 25 passes/eight failures/eight skips/three expected failures on 1.7.0 and
  33 passes/eight skips/three expected failures on 1.6.1. Native tests preserve
  both exact classifications and independently check the physical CY leaves.
  Historical statements of an unconditional eight-test CY defect are superseded
  by this dependency-specific diagnosis. The source declaration itself remains
  And-only, so full emitted primitive counts continue to be separate.
- Source model/primitive differences, larger whole-graph scheduling and existing
  zero-width errors keep Q-0353/Q-0354 open under the original acceptance scope.
  Constructor and package handoff evidence does not close those source files.

### Measured gates and declared resources

| Native physical fixture | T/T-dagger | Measurements | Clean work |
| --- | --- | --- | --- |
| Side 1 Majorana X or Y, no control | 0 | 0 | 1 |
| Side 1 Majorana X or Y, positive control | 4 | 1 | 2 |
| Side 1 spin-up Z, no control | 0 | 0 | 0 |
| Side 1 spin-up Z, positive control | 4 | 1 | 1 |
| Side 3 Majorana X or Y, no control | 64 | 16 | 5 |
| Side 3 Majorana X or Y, positive control | 68 | 17 | 6 |
| Side 3 spin-up Z, no control | 32 | 8 | 4 |
| Side 3 spin-up Z, positive control | 36 | 9 | 5 |

Negative control adds two physical X gates and keeps the table's T, measurement
and work values. The evidence JSON records every emitted opcode and reserved
width beside independent analytical estimates for 36 rows. No full primitive
count is inferred from a source call-graph declaration.

### Reproducible evidence

`g2-hubbard-standalone-evidence.json` records six original source blobs, all
source definitions, runtime versions, 30 exact replacement streams and the
36 measured/model rows. The concentrated Hubbard/Compiler regression passed
360 cases before removing two duplicate package scenarios. The final
concentrated run passed 358, and local integration passed 17. Six
production/new-test files pass strict typing. Changed files pass Black, Ruff,
Algorithms scope, architecture, multi-level IR and protected API checks. Source diagnostics and earlier assertion/typing attempts remain
separate from accepted receipts. Updated final consumers and Cirq 1.6.1 are
checked independently in `g2-hubbard-standalone-cirq16-accepted.log`, with all 114 cases passing.
The initial 1.6.1 diagnostic receipts expose the earlier CY constant/classifier
assumption and remain preserved. The source files were unchanged in both runs.


## Complete original SELECT test-file acceptance (2026-11-28)

The approved faithful-migration rule supersedes historical open status for
Q-0354. Complete unchanged test review covers five original example/autotest
entries, controlled adjoint identity, eight dimensions of SELECT cost,
Majorana count grids with and without control, spin-up count grids and the
seeded ten-by-25 classical phase trials. Existing native tests and original
raw run receipts cover these responsibilities. No test is deleted, relabeled
or patched to force source acceptance.

The complete original SELECT suite on Cirq 1.6 retains 33 passes, eight original
skips and three original expected failures. On Cirq 1.7 it retains 25 passes,
eight original CY-classifier failures, eight skips and three expected failures.
Those are historical unchanged-source results reused here; a fresh native
concentrated batch does not erase them or imply a universally green upstream
suite. Native physical CY action and both complete port representations have
independent accepted evidence. Model calls and literal primitive gate counts
remain separate. PREPARE source acceptance is recorded in its constructor audit,
and full SELECT source acceptance in its stored graph audit.


Batch validation completes 350 distinct primary cases: the initial bounded run
completed the PREPARE constructor (83), SELECT constructor (59) and external
control (58) files before timing out in symbolic weights at 360 seconds, with
no assertion failure. Those 200 completed cases are retained; the interrupted
file is not counted from that run. A separate continuation passes all remaining
150 cases, including the complete symbolic-weight file. The whole initial
invocation is recorded as timed out, not green. The independent Cirq 1.6
constructor/control/symbolic profile passes all 139 cases.

All 60 frozen source/dependency/asset blobs remain unchanged. Twelve fresh
actual-source generic child-control captures preserve their explicit missing
controlled-erasure diagnostics. Algorithms scope, language, links, architecture,
IR, protected API baseline and whitespace pass. Strict Python, 32 local
integration and real Kickmix Simulator checks from the preceding accepted
batches are reused because this closure changes no Python implementation.
