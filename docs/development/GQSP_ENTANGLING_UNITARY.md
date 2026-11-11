# GQSP with an entangling two-register unitary

This finite G2 slice verifies existing shared-register components through an
entangling U and complete GQSP. It follows unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. No new production control engine,
adjoint engine, runtime, public export, dependency or IR schema is introduced.

## Ownership, ports and source chain

The stored U has two ordered THRU QBits, `a` then `b`. Its shared layer applies
H(a), CNOT(a,b), then ZPow(1/4)(b). The root invokes that layer once or twice.
The one-layer output on |00> has Schmidt rank two, independently checked through
the original CompositeBloq's explicit Cirq export. CompositeBloq itself is not
a Cirq Gate; use its existing `to_cirq_circuit(cirq_quregs=...)` conversion.

Compiler owns construction, control propagation, graph adjoint, physical binding
and workspace lifetime. Algorithms owns GQSP composition; Simulation owns phases;
Runtime executes the existing dynamic IR. The existing U callback receives named
ports, signal, clean workspace, activation (None/0/1) and direction. The bound
operation is still appended with `circuit.append(op)` to a DynamicCircuit.

`GeneralizedQSP.decompose_bloq()` reaches the original CompositeBloq control
protocol. H specializes to CHadamard, CNOT specializes to Toffoli (the new signal
precedes the original control), and ZPow specializes to measured CZPow. The
original zero-control wrapper surrounds the entire U with one X pair. Its exact
adjoint reverses child calls and takes each exact child adjoint before control
propagation. Native code reuses ComponentProgramBuilder,
control_component_via_predicate, adjoint_component_hierarchy,
cnot_with_controls_ir and bind_register_program for precisely these roles.

The physical binding is signal=3, a=4, b=1, clean=0, idle=2. Each port is one QBit;
flattening order is a then b and must not follow numerical wire order. CZPow
owns the clean wire only during its call and returns it to zero before reuse.
The existing binder assigns classical slots 0 through repeats-1 within each U;
completed U calls reuse that set. The source comparison explicitly uses the
same physical-work/slot policy; those labels are not upstream logical names.

## What equality means

The source walker follows original connections, control dispatch and adjoints.
Source CNOT and Toffoli stay atomic as native cx and ccx; CH is also atomic.
Original And supplies measured decomposition leaves. SU2 and scalar phases use
the previously accepted common physical normalization; atomic ZPow maps to phase.
Complete linear Instruction tuples, including parameters and feedback metadata,
are compared. This is not synthesized Clifford+T equality. The original GQSP
Cirq unitary supplies a separate semantic oracle for all logical amplitudes.
No source is modified to make native output pass.

The tests cover 12 standalone U cases (one/two repetitions, bare/zero/one control,
both directions), 18 full GQSP cases (one/two repetitions, coefficient lengths
2/3/4, negative powers 0/1/6), and two graph replacement/adjoint cases. Eight
basis inputs plus one complex coherent input exhaust 2,646 unnormalized measured
branches and run 162 actual dynamic trajectories. Absolute phase and the
`2**(-measurement_count/2)` branch factor are checked, with exactly zero work and
idle amplitudes. Reset is fixed by the preceding measurement projection.

Control followed by exact adjoint equals adjoint followed by control. Replacing
the stored hierarchy with an expanded Program behind the same callback preserves
all emitted instructions. Missing clean workspace is rejected. The same source
leaf reader is reused for the previous one-register tests and this two-register
path; it gains only atomic CNOT/Toffoli recognition.

## Actual Instructions and independent model

For P=(.25,.1,.2), actual counts under the stated atomic normalization are:

| Shared layer calls | k | Native/source Instructions | Atomic ccx | T+T-adjoint | Measurement/reset | Clean / slots |
|---|---|---|---|---|---|---|
| 1 | 0 | 65/65 | 2 | 8 | 2/2 | 1/1 |
| 1 | 1 | 63/63 | 2 | 8 | 2/2 | 1/1 |
| 1 | 6 | 73/73 | 2 | 8 | 2/2 | 1/1 |
| 2 | 0 | 105/105 | 4 | 16 | 4/4 | 1/2 |
| 2 | 1 | 103/103 | 4 | 16 | 4/4 | 1/2 |
| 2 | 6 | 125/125 | 4 | 16 | 4/4 | 1/2 |

QEC model categories come independently from existing CH pricing, primitive
rotation/CZPow/Toffoli queries and the GQSP call model. All seven categories match
the original source graph:

| Shared layer calls | k | T | Toffoli | And | Clifford | Rotation | Measurement |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2 | 2 | 2 | 14 | 13 | 2 |
| 1 | 1 | 2 | 2 | 2 | 12 | 13 | 2 |
| 1 | 6 | 6 | 2 | 2 | 18 | 13 | 2 |
| 2 | 0 | 4 | 4 | 4 | 24 | 17 | 4 |
| 2 | 1 | 4 | 4 | 4 | 22 | 17 | 4 |
| 2 | 6 | 12 | 4 | 4 | 36 | 17 | 4 |

Model CSWAP is zero, modeled peak is four, and physical wire 2 is an additional
idle probe. Logical quarter-Z powers are modeled as T while physical IR retains
phase. Atomic ccx and CH have not been synthesized, and And model cost is not
the expanded four-T measured network. Neither table certifies a synthesized T
budget or performance claim.

## Reproduce and limits

Run `tests/unit/test_generalized_qsp_entangling_unitary.py` using the locked
source environment, preimporting torch and checking Qualtran distribution version
0.7.0 and Kickmix Simulator availability before pytest. The conformance tests
execute the existing native dynamic Runtime; availability checks alone are not
Kickmix execution evidence. Source files are checked against inventory Git blobs.

The optional original CNOT test module imports PennyLane at module scope. Its
initial collection fails because the reference environment lacks PennyLane;
retain `g2-gqsp-entangling-unitary-original-pennylane-collection.log`. Do not
replace that dependency with a fake module or report those original CNOT tests
as passed. Native/source CNOT control leaves and full GQSP amplitudes are covered
by the conformance tests above; other selected unchanged original tests are run
separately. This collection limitation does not constitute whole-file acceptance.

Multi-bit/shaped U ports, arbitrary U decompositions, symbolic/unbounded graphs,
Clifford+T synthesis and full-parent/G2 acceptance remain open.

## Local acceptance

The 32 new cases pass within the full focused selection: 451 passed, 54 warnings in 74.99s (0:01:14).
Related regression: 744 passed, 54 warnings in 85.55s (0:01:25). Integration: 64 passed in 10.56s.
Focused/regression warning counts come from unchanged fixtures: six autoray
matmul warnings in the controlled-NOT tests and the previous 48 GQSP numerical
environment warnings. Strict mypy (two files), Black, Ruff, Compiler ownership,
architecture, IR and public-API checks pass. Twenty-four source files retain
their inventory Git blobs; selected unchanged original tests report 34 passed, 13 warnings in 3.08s.
The original CNOT module collection limitation above remains explicitly open.

Default smoke/unit results: 7 failed, 21507 passed, 300 skipped, 1037 warnings in 355.52s (0:05:55).
All failed IDs match the seven independently recorded baseline failures.
Maxfail stops this run; it is not a fully green suite. New cases have the
complete separate focused evidence above.

External evidence uses the `g2-gqsp-entangling-unitary-` prefix for
focused/regression/integration/original/default logs, counts and default-comparison
JSON, source blob list and the preserved PennyLane collection error.
