# Signed clauses and construction routes

For each width, the source draws two permutations until each clause has three
different variable indices. The first positive literal uses the clause index.
The second literal is negative and the third is positive. The executable size
formula uses rounding and reaches 68 variables. The frozen 67-variable native
example remains a separate counterpart.

The wide route renders checked integer indices as a source-retaining Python
`qfunc` and asks `synth` for its measured circuit. The benchmark keeps a tight
state-planning allowance. For the three shortest cases and cases where the
bounded synthesis planner refuses the candidate, the walkthrough constructs the
same oracle through the public Circuit API. That route computes the clause flags,
applies a single phase to their conjunction, and uncomputes the flags. The
point markers distinguish the routes, and the timing excludes the checks that
follow construction. A planning refusal is included in that row's elapsed time.

Each generated circuit is interpreted on three basis words to verify the
emitted instructions, phase sign, and every restored work bit. This is a
reversible instruction check. A separate small coherent execution compares every
complex amplitude of both routes with an independent truth-table Circuit.
The complete benchmark does not materialize an exponential statevector.
The source's Qiskit timing loop stays disabled and its published points are
labeled as archived measurements. The two environments and compilation
contracts differ.
