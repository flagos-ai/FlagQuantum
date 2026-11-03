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

## Default activation follow-up

The native qfunc and Qmod Services entry points request the bounded candidate.
The existing register selector checks it alongside retain and recompute and
applies the same executable constraints. Direct owned-register callers and
the scalar entry retain their existing policy. Regression tests that inspect
legacy allocation, normalization or fixed historical resource counts
explicitly select the legacy policy. The public release matrix runs with
default selection and checks full complex amplitudes, nonzero XOR targets
and actual Runtime memory admission.

A local pilot compared the same source and Runtime path with the candidate
switch off and on. Each policy and workload ran in a separate process with
two warmups and five measured syntheses, fixed single-thread settings, and
independent correctness verification. The table gives median synthesis time
and process-lifetime peak RSS. RSS includes Python imports, verification and
any execution performed by the workload.

| Workload | Synthesis ms, legacy / trial | Peak RSS MiB, legacy / trial |
| --- | ---: | ---: |
| Native sentinel, logical | 19.43 / 14.60 | 223.98 / 211.70 |
| Qmod sentinel, logical | 19.56 / 14.05 | 225.86 / 212.41 |
| Native sentinel, basis | 167.41 / 134.60 | 225.55 / 214.94 |
| Qmod sentinel, line | 490.72 / 309.16 | 251.09 / 235.33 |
| Native energy 8×8, logical | 569.58 / 584.05 | 220.42 / 220.48 |
| Qmod constant product, five bits | 49.97 / 49.63 | 208.33 / 208.27 |

The last two workloads selected retain under both policies. The pilot covers
these six requests and does not establish a cross-host time or memory bound.
