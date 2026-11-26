# Complete Kikuchi guiding-state tutorial and original test mapping

The tutorial is frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, source blob
`34e5a813e1b5eaa1f5a9f5774f145be5833c789d`. Its companion test blob is
`c215aad2516a4c61cc745c8909ef90f55fe74e5b`. Copyright 2024 Google LLC,
adapted under Apache-2.0. The reference files are unchanged.

## Ownership and complete cell coverage

The native counterpart is
[kikuchi_guiding_state_tutorial.ipynb](../../examples/notebooks/kikuchi_guiding_state_tutorial.ipynb).
All seventeen source cells retain their identities and code/Markdown roles.
One extra cell separates real `circuit.append(op)` primitive counts from QEC
model estimates. Algorithms composes the instance, guiding declaration and
resource diagram. Compiler owns sort/comparator/half-comparison direct calls,
word views, predicates and expansion. Drawer renders the existing Core Program
and generic call-graph representations. No execution kernel or backend belongs
to the diagram implementation.

| Source cells | Retained native responsibility | Verification |
| --- | --- | --- |
| 0–4 | Title, symbolic instance and theorem context | Source-cell metadata, symbolic parameters and original explanatory content |
| 5 | Symbolic depth-one resource graph | Every child parameter and edge weight matches the original graph |
| 6 | Symbolic word-level root at c=3 | All twelve calls, operands and returned word connections match |
| 7–9 | Seeded concrete instance, standalone signature and root | n=20, m=100, k=4, ell=12, planted advantage 0.8, RNG seed 100; all constraints, ports and twelve root calls match |
| 10 | One-level concrete expansion | All twenty-one source calls and predecessors match, with Partition retained as an original atom |
| 11–14 | QEC query, total T count, rounding substitution and asymptotics | All seven concrete/symbolic categories, unsimplified T expression, simplified expression and multivariate order match |
| 15–16 | Depth-three resource graph and theorem interpretation | Every parameter-bearing node and weighted edge at depths zero through three matches the original |

Every code cell executes with Qualtran, Cirq, OpenFermion and Qiskit imports
blocked. Six figures render and receive visual inspection: two resource graphs,
symbolic and concrete roots, the standalone signature, and one-level expansion.
The symbolic diagram uses nonfloating edge labels so long width expressions
receive layout space. Stored Programs, instance labels and formal ports remain
unchanged by display-name shortening. The original seventeen-cell successful
execution receipt from November 19 is reused, not counted as a new migration.

The resource graph is explicitly bounded to depths zero through three.
Depth one admits symbolic metadata; deeper native diagrams require concrete
instances. These are logical model calls, not a promise that original physical
decompositions exist. Source model leaf identities retain forward/adjoint
direction, no-MSB addition, permutation cycles, exact dense coefficients and
the difference between a packed control word and a QBit array. Existing sort
and duplicate models supply their own multiplicities; the diagram does not
replace their numerical cost owners.

## Complete original test-file disposition

Q-0612 has exactly seven top-level test responsibilities. All are mapped:

| Original test | Native evidence |
| --- | --- |
| test_examples | Five factories: exact fields, signatures, seven QEC categories, typed root admission and all twenty-five independently observed source protocol outcomes |
| test_t_cost_simple | Concrete scope QEC model, including the original phase-width dependence |
| test_t_cost_simple_symb | Symbolic scope model with the original width-roundtrip expression retained |
| test_t_cost | Concrete guiding model, including original CSwap and And categories |
| test_t_cost_symb_c | Both symbolic guiding factories; all seven categories and the original total-T/asymptotic relation |
| test_notebook | Accepted complete Q-0610 notebook, with all twenty-one original cell roles and seven figures |
| test_tutorial | Complete Q-0613 mapping described above |

The unchanged original non-notebook receipt records fourteen passes, nine
original skips, seven original xfails and two notebook deselections. Independent
protocol observations retain the following results instead of hiding them
behind the original skip/xfail policy:

| Example | Make | Decompose | Counts | Serialize | Strict types |
| --- | --- | --- | --- | --- | --- |
| simple_guiding_state | PASS | MISSING | UNVERIFIED | FAIL | NA |
| simple_guiding_state_symb | PASS | NA | UNVERIFIED | FAIL | NA |
| guiding_state | PASS | PASS | FAIL | FAIL | PASS |
| guiding_state_symb | PASS | PASS | FAIL | FAIL | PASS |
| guiding_state_symb_c | PASS | NA | UNVERIFIED | FAIL | NA |

This is nine PASS, seven FAIL, five NA, three UNVERIFIED and one MISSING.
The five serializations fail on unsupported KXorInstance. Two original
count-consistency checks fail; model-only counts remain UNVERIFIED. No failure
is relabeled as native execution or serialization success. Accepting the
complete tutorial and test mapping does not waive the open implementation
parent Q-0611 or its original protocol defects.

## Physical evidence remains separate

The appended two-bit probabilistic-uncompute example emits twenty-eight
instructions: four T/Tdg, one measurement, one reset, and the listed H/X/CX/S/CZ
instructions. Its separate source QEC model is one And, four Clifford and one
measurement. These are different counting conventions, not a full-root total.
The existing complete primitive, feedback, complex-column, applicable branch
and private-work-zero proofs for this component and source-faithful duplicate
detection are reused.

Scope preparation and sorting still lack original physical bodies. Concrete
HasDuplicates does have a body; the existing inverse-alias defect and its
observed branch channels remain retained. Full preparation amplitudes, whole
primitive counts and whole-root work cleanup remain unavailable. Instance
serialization, unknown symbolic child arrays, absent phase-gradient ports and
unapproved contracts remain explicit boundaries. No source repair, new defect
waiver, dependency installation, ReadTheDocs work or Core contract change is
part of this closure.

The batch passes 489 concentrated cases per actual Cirq 1.7.0/Qiskit 2.5.0
and Cirq 1.6.1/Qiskit 2.0.0 profile, with ten visible reference warnings per
profile, plus 145 local integration cases. Seven Python units, including the
complete extracted notebook, pass strict typing, Black and Ruff. Notebook
cells are formatted with the installed Black Python formatter; no Jupyter
formatter dependency is installed. Team scope, English, local links,
architecture, multi-level IR and protected API checks pass. Fourteen source
and dependency blobs match the locked CSV; real Qualtran 0.7.0 and Kickmix
0.1 Simulator are asserted before reference execution.
