# AO-05 public workflow trial

This validation slice runs the existing internal Boolean selection policy
through the public native qfunc and Qmod entry points in tests. It exercises
the same checked Runtime planning, run and named readout as ordinary
synthesis. The trial policy is scoped to a pytest fixture.

The fixed trial includes the two-bit sentinel and six further expression/width
configurations: one-bit energy, one-bit general multiply-add, two widths of
square and two widths of constant multiplication. Each frontend is checked
against the independent benchmark oracle for the complete input domain,
nonzero XOR targets and complete complex amplitudes including clean workspace.
The sentinel covers all 256 input/target pairs. Each trial-selected circuit
improves the actual logical operation count relative to its checked legacy
candidate on the same request. Tests also verify basis decomposition, a line
topology and Runtime admission at 8192 bytes with a one-byte-short refusal.

A first unrestricted activation trial passed 852 cases before being stopped
and reported 15 failures. Several assertions require a positive work pool or
pin the previous selected width and operation count. The test suite also
contains ownership checks on candidate rows where a structural refusal has
no circuit. The existing assertions and frozen measurements are preserved
here. The unrestricted activation was reverted in the working tree.

Default activation remains gated on explicitly separating legacy scheduling
checks from default policy checks, completing the wider regression run and
collecting paired compile-stage and peak-memory measurements. This trial
covers a finite subset of the AO-00 corpus.

The [native, Qmod and Circuit examples](SYNTH_PYTHON_PERFORMANCE_PF01_DELIVERY.md#existing-user-workflow)
show the three source forms. The trial sentinel reaches 9 wires, one workspace
wire, 14 operations and depth 9 with input preparation at the logical target.

| Sentinel target | Existing wires / workspace / operations / depth | Trial wires / workspace / operations / depth |
| --- | --- | --- |
| Logical | 16 / 8 / 81 / 59 | 9 / 1 / 14 / 9 |
| Exact decomposed basis | 16 / 8 / 497 / 327 | 9 / 1 / 91 / 54 |
| Exact basis on a line | 16 / 8 / 952 / 477 | 9 / 1 / 140 / 80 |

These are local CPU target-pipeline counts with the same high-level request.
