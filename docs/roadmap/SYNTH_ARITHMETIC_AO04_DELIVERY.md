# AO-04 target-aware Boolean selection

The Compiler can opt into one bounded Boolean register-program candidate
alongside the existing retain and recompute schedules. The experiment uses
retain for phase segments and the AO-03 XOR mapper for unsigned value segments.
A program outside the mapper's envelope keeps its existing checked choices.
This slice leaves activation through public synthesis and cross-frontend default
regression evidence to AO-05.

The candidate uses the same exact target conversion, routing, validation,
resource estimation, hard limits and objective ranking as the existing rows.
Each row carries its family, source identities, stage costs and refusals.
A fixed third slot follows the two original schedule checks when the internal
experiment is enabled. Request-budget exhaustion still withholds a selected
program. A structural refusal from the bounded family leaves the legacy
candidates eligible under the same final limits.

For the composed native `x*x+y` program, the logical selector chooses the
9-wire, 1-workspace, 14-operation, depth-9 candidate under the 9/1/24/20 limits.
CPU complex128 execution verifies the entire state with all sixteen outcomes
and zero workspace. Its dense state buffer is 8 KiB. The Runtime memory
admission contract remains authoritative at public program creation.

Tests also verify stable target-stage identities across repeated selection,
refusal under the same gate ceiling after basis decomposition, and selection
of an existing candidate for explicit line topologies that support its width.
The line profiles deliberately omit the nine-wire topology to exercise a real
target admission decision. The test does not establish a reversal on an
identical device topology where both candidates fit.

The [native, Qmod and Circuit examples](SYNTH_PYTHON_PERFORMANCE_PF01_DELIVERY.md#existing-user-workflow)
remain the comparison workflow. The scalar selector and public default use the
existing path. Original AO and PF measurements and oracles remain frozen.
