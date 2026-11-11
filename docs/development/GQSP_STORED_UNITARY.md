# GQSP with a stored noncommuting unitary

This finite G2 acceptance composes existing Compiler graph transformations with
Algorithms GQSP and Runtime dynamic execution. It adds conformance tests rather
than another control, adjoint, binding or execution implementation. The source
remains Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.

## Ownership and interface

`ComponentProgramBuilder` stores a shared `layer` that applies H then ZPow(1/4)
to one THRU QBit named `q`. A root calls that layer once or twice. This order is
noncommuting: the exact inverse applies ZPow(-1/4) before H, and reverses repeated
calls. Compiler owns the Program, child control dispatch, exact graph adjoint,
control predicate and physical binding. Algorithms owns GQSP composition;
Simulation owns the polynomial phases. Runtime executes the resulting dynamic IR.
No upstream object enters a production boundary; no public export changes.

The existing U callback receives signal, named data ports, workspace, activation
(None/0/1), and direction. The fixture prepares forward/inverse Programs with
`adjoint_component_hierarchy` and both control polarities with
`control_component_via_predicate`. Its specialized children use existing CH and
measured CZPow emitters; private work is allocated and freed inside each CZPow.
`bind_register_program` maps the one supplied clean work wire and named data.
The same `GeneralizedQSP.make_on(...)` operation goes to `circuit.append(op)` on
a `DynamicCircuit`. No numerical statevector execution lives in Compiler.

The scrambled physical binding is signal=1, q=3, clean=0; wire 2 is an idle
probe. One QBit has no word-endian ambiguity. The binder assigns one classical
slot per measured child within a U invocation: slots 0 through repeats-1.
Completed GQSP U invocations reuse those slots. The source test assigns the same
physical work and slot policy explicitly; these labels are binding choices,
not upstream logical register names. Work and idle must finish exactly zero.

## Original source path and comparison boundary

The unchanged original CompositeBloq is constructed from H and ZPow children.
`GeneralizedQSP.decompose_bloq()` reaches its original Controlled graph; generic
control propagates through the composite, dispatching H to atomic CHadamard and
ZPow to CZPow/And/phase/And-adjoint. Zero control wraps the entire composite in
one pair of X gates. The inverse comes from the original CompositeBloq adjoint;
it does not reverse measured primitive instructions. The native graph follows
these same paths through existing components.

The comparison reads actual source connections, using the previously accepted
common atomic normalization for SU2 (including scalar phase on a physical
signal wire), H, CH and ZPow. Original And decomposition supplies measurement,
feedback and reset leaves. Complete Instruction tuples are equal: linear order,
operands, angle parameters and feedback metadata. CH remains atomic on both
sides; this is not a synthesized Clifford+T primitive comparison. Direct source
Cirq unitary is a separate semantic oracle, not the measured instruction trace.
MatrixGate still has no source leaf decomposition; arbitrary U is not accepted.

## Evidence

`tests/unit/test_generalized_qsp_stored_unitary.py` covers:

- 12 standalone cases: one/two shared layers, bare/zero/one control, both
  directions, with full source Instruction equality.
- 18 complete GQSP cases: two layer counts, coefficient lengths 2/3/4, and
  negative powers 0/1/6. Four bases and a complex coherent input give 1,470
  exhaustive unnormalized measurement branches and 90 actual dynamic trajectories.
  Absolute complex phase and branch factor `2**(-measurements/2)` are compared;
  work and idle are exactly zero. Reset is determined by the prior projection.
- Control then exact adjoint equals exact adjoint then control. Expanding stored
  Programs replaces the hierarchy behind the same callback without changing its
  consumer or emitted instructions. Missing clean work is rejected.
- Seven QEC model categories match the original GQSP resource graph. The existing
  CH cost, rotation/CZPow queries and GQSP call model supply child prices;
  physical Instructions are counted separately.

For P=(.25,.1,.2), these are actual emitted counts under the common normalization:

| Shared layer calls | k | Native/source Instructions | T+T-adjoint | Measurement/reset | Clean / classical slots |
|---|---|---|---|---|---|
| 1 | 0 | 63/63 | 8 | 2/2 | 1/1 |
| 1 | 1 | 61/61 | 8 | 2/2 | 1/1 |
| 1 | 6 | 67/67 | 8 | 2/2 | 1/1 |
| 2 | 0 | 101/101 | 16 | 4/4 | 1/2 |
| 2 | 1 | 99/99 | 16 | 4/4 | 1/2 |
| 2 | 6 | 113/113 | 16 | 4/4 | 1/2 |

The model has different categories and is reported separately:

| Shared layer calls | k | T | And | Clifford | Rotation | Measurement |
|---|---|---|---|---|---|---|
| 1 | 0 | 2 | 2 | 14 | 13 | 2 |
| 1 | 1 | 2 | 2 | 12 | 13 | 2 |
| 1 | 6 | 6 | 2 | 14 | 13 | 2 |
| 2 | 0 | 4 | 4 | 24 | 17 | 4 |
| 2 | 1 | 4 | 4 | 22 | 17 | 4 |
| 2 | 6 | 12 | 4 | 28 | 17 | 4 |

Toffoli and CSWAP model counts are zero; modeled peak width is three. The fourth
physical bit is the idle probe. Logical ZPow(1/4) is priced as T by the source
model; the physical IR retains its phase instruction. And model cost and the
expanded four-T measured network are not the same category. No synthesis budget
or performance claim follows from either table.

Run the focused test with the locked source environment and preimport torch;
assert Qualtran distribution version 0.7.0 and Kickmix Simulator before pytest.
The source files must match their inventory Git blobs. Full-parent and G2
acceptance remain open; entangling multi-register U and unbounded/symbolic
components are outside this finite slice.

Local acceptance: 32 new tests, 408 focused tests (20.73s), 701 related regression
tests (26.38s) and 64 integration tests (5.74s) pass. Focused/regression runs each
retain 48 previously reproduced numerical-environment warnings. Strict mypy,
Black, Ruff, Compiler ownership, architecture, IR and public-API checks pass.
Nineteen source files match their inventory Git blobs; 32 original source tests
pass (1.96s), retaining 13 marker/numerical-environment warnings.

Default smoke/unit verification reports 7 failed, 21,507 passed, 300 skipped and
1,037 warnings in 218.08s. All seven failed IDs match the independently recorded
baseline, with no new failed IDs. Maxfail stops this run; it is not a fully green
suite. The new tests have separate complete focused evidence above. External
machine-readable evidence is `g2-gqsp-stored-unitary-{counts,default-comparison}.json`,
`g2-gqsp-stored-unitary-source.txt` and corresponding focused/regression/
integration/original/default logs in the locked reference environment.
