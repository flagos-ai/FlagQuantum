# Checked pure quantum callbacks in QPE

The native quantum function allocates data and a typed fractional phase counter,
then calls `qpe(lambda: grover(...), phase_result)`. The callback composes existing
phase predicates, preparations and reversible functions. Compiler inserts the
conditional powers and inverse Fourier; users do not choose physical wires or
write those gates. `drop` hides a name without resetting or reusing its state.

The original callback and initial state are preserved. In particular, the
Grover-operator tutorial starts from zero without a separate preparation;
QMC retains all eight decimal probabilities and the complete prefix-conditioned
loader, including its action on nonzero inputs. Quantum counting keeps the
predicate `(a + b) <= 2`. Unscaled unsigned zero tests may use `x <= 0`.
The fractional counter retains the original sign and resolution. Named
`numeric_probabilities` returns exact Fraction labels; raw `probabilities`
returns stored words. Dropped data is marginalized, never assumed clean.

The independent Circuit constructs the Grover reflection, full loading operator,
dense controlled matrix powers and inverse Fourier. Local acceptance compares
all amplitudes without adjusting global phase, the original state and two
additional nonzero coherent data vectors. Clean workspace is checked separately
from the retained environment. The state tolerance is 1e-11 for long gate
sequences; measured errors and physical gate resources are recorded in delivery
evidence. Original and native gate decompositions may differ; no optimality or
performance improvement is claimed.

Callbacks must be a zero-argument lambda containing one positional captured
operation or qfunc call. They cannot allocate, publish Output, mutate Const,
overlap the counter, escape through host callbacks or bypass the 4096-effect
bound and final resource budgets. Explicit nested control states are refused.
Power-dependent Suzuki scheduling remains a separate QPE capability. Full
Notebook workflows are tracked independently from this quantum program.
