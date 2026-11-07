# Numerical measurement outcomes and phases

Simulation owns two independent modules, adapted from unmodified Qualtran
v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`:

- `classical_distributions.py`: the numerical distribution and original random,
  fixed-instance and banned-outcome handler rules from `simulation/classical_sim.py`.
- `classical_measurement.py`: `MeasureX` callbacks from `basic_gates/x_basis.py`,
  the measurement-phase descriptor, modulus validation and bound phase update.

The distribution retains choices and probabilities by identity, without
conversion or validation. A supplied NumPy generator has priority over fixed
values, including an empty fixed mapping. Otherwise, fixed results are retrieved
by instance ID and returned unchanged; missing IDs retain `KeyError`. No handler
means a nondeterminism error. Distribution validation and RNG state consumption
remain exactly those of `Generator.choice`. Distribution objects are immutable
and unhashable, as in the source.

The X-measurement value callback checks membership in `[0, 1]`, retaining original
NumPy and diagnostic equality behavior. It returns a uniform two-choice
distribution without drawing an outcome. The separate phase callback returns
one for input zero, or a deferred measured-output sign for input one.

`MeasurementOutcomePhase` identifies an output register and index. The graph
owner binds those keys to measured values before `apply_classical_phase` runs.
Exactly outcome one multiplies the accumulator by `-1.0`; this path does not
revalidate the accumulator. Ordinary phase contributions use the shared modulus
validator and multiply in original order. None leaves the phase unchanged.
The original modulus comparison permits NaN and retains ambiguous-array errors;
these diagnostics are tested without replacing the reference with stricter math.

This is not a generic classical graph interpreter. It does not own Soquet
propagation, graph traversal, output dtype checking, backend selection or register
signatures. Fixed distribution results are not asserted valid for a particular
register by the sampling helper. The graph binder must perform that validation.
Production modules import no Compiler, Runtime or external SDK types.

Run the complete native user example:

```sh
python -m examples.algorithms.classical_measurement_values
```

It checks numerical outcome phases, then uses `circuit.append(op)` to copy data
to clean work and erase it with the existing X-measurement clearing emitter and
Z feedback. Both unnormalized branches preserve full complex amplitudes up to
the expected `1/sqrt(2)` factor, including a spectator. All trajectory results
restore the initial logical state and return work to zero.

Resource scopes remain separate: the original MeasureX leaf and the existing
native typed resource key both have zero T and one measurement, under both QEC
profiles. The emitted copy/erase operation contains one CX, one H, one
measurement, one conditional X for physical clearing and one conditional Z for
feedback, with one work wire and three declared wires. The original MeasureX
leaf has no primitive decomposition to compare against this entire physical
copy/clear/feedback recipe; its distribution, phase and modeled categories are
compared directly instead.

Forty distribution tests cover exact seeded sequences, NumPy return types,
generator state, fixed aliases, factory precedence and errors. Measurement tests
compare raw callbacks, phase validation, actual source state updates and complete
source leaf-graph execution with both RNG and fixed outcomes. They also exercise
the native phase-validator replacement and every small quantum branch.

An explicit source limitation remains: direct `Controlled(MeasureX())` phase
evaluation extracts all child signature ports before activation. Omitting the
right port `c` raises `KeyError` even when inactive; supplying it reaches an
unexpected-keyword `TypeError` in the active child phase callback. The accepted
native controlled callback retains the same behavior. This slice does not repair
the source or claim that controlled measurement is supported.
