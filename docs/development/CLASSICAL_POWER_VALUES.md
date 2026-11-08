# Numerical repetition of bound components

Simulation owns `classical_power.power_classical_values`, adapted from the
unmodified Qualtran v0.7.0 `Power` constructor and `build_composite_bloq` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler retains the existing
`register_power_ir` emitter and typed resource-call identities.

The owner supplies bound named inputs, the original repetition object, a complete
child numerical evaluator, and output assignment. The child's returned phase is
already resolved, including any child measurement outcomes. This is composition
of evaluated child components, not a replacement graph interpreter.

Guard order follows the source:

1. Reject non-THRU child metadata before inspecting the count.
2. For a nonsymbolic count, require a positive Python or NumPy integer.
3. For actual decomposition, require `isinstance(count, int)` without coercion.

Thus NumPy integer and Sympy integer counts can pass construction yet fail the
concrete build. Python `True` behaves as one in the numerical source path.
A custom integer whose symbolic protocol returns true can bypass the positivity
guard; a nonpositive value then executes an empty source range. These diagnostic
behaviors are verified against the original complete numerical graph. They do
not relax the existing compiler emitter, which requires a positive plain Python
integer and enforces its existing instruction limit. Symbolic resource models
remain separate from executable numerical repetition.

Each child receives a fresh keyword mapping retaining its field objects. Child
outputs are assigned by their owner before the shared phased-update helper
validates and accumulates the returned phase. The next child consumes those
outputs. Binding, validation and measurement inside a child remain its evaluator's
responsibility. No unresolved measurement-phase token is expected to escape that
complete evaluator. Assignment and child failures stop further calls; graph-owned
exception translation remains outside this composition helper.

Run the native user example:

```sh
python -m examples.algorithms.classical_power_values
```

The example repeats the existing word-controlled X/Z component twice and still
uses `circuit.append(op)`. It reuses the existing measured forward and inverse
bodies, work wire and classical slot. The complete emitted sequence has 16 T,
4 measurements, 1 reused work wire and 5 declared wires, including a spectator.
The original model has 16 T equivalents, 4 measurements and 4 peak logical wires;
the live width is not multiplied by the repetition count.

Acceptance compares original construction and full numerical graphs, including
the raw count-object cases above. All eight logical basis inputs, one through
three repetitions and both directions have exact numerical/phase parity. For one
and two repetitions, both original primitive streams, the complete original
tensor, all forward and round-trip measurement histories and clean work agree.
The two-repeat case checks 16 forward and 256 round-trip histories with full
complex amplitudes. Both QEC profiles match the existing native `PowerCall`
resource composition. Actual counts and model categories are compared on their
declared bases. No compiler implementation or source reference was changed.

`Power.__pow__`, wrapper construction of adjoints and diagram formatting are not
new production methods in this slice. The original adjoint is used directly as
the inverse gate-flow reference. Full generic graph traversal and typed wire
propagation remain separate open capabilities.
