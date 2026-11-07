# Classical And predicate values

Simulation owns `classical_and.py`, adapted from the unchanged Qualtran v0.7.0
`And.on_classical_vals` and `MultiAnd.on_classical_vals` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler retains the existing shared
And emitter and prefix ladder; Runtime retains measurement execution.

`and_classical_values` compares the control tuple with the activation tuple and
creates an integer target. Forward evaluation ignores an incoming target.
Erasure accepts only the consistent target and returns the original controls.
The original tuple equality and ambiguous-array failures are retained.

`multi_and_classical_values` performs `equal`, converts to `uint8`, and performs
`bitwise_and.accumulate` in the original order and default axis. It returns the
original controls, the `[1:-1]` prefix slice and the last accumulation entry.
There is no flattening, control-value normalization or inferred workspace reset.
The caller supplies concrete activation metadata; this callback does not replace
the source constructor or signature validation. Prefix junk and target are
logical outputs until a matching quantum erasure consumes them.

Run `python -m examples.algorithms.classical_and_values`. The ordinary
`circuit.append(op)` path verifies coherent outputs, spectator preservation and
measured erasure. A three-control compute/erase emits 8 T gates and two
measurements with two prefix outputs and six declared wires (including a
spectator). The source compute model has 8 T equivalents and zero measurements;
the inverse model has zero T and two measurements. These models are checked
separately from the primitive instruction counts.

The reference suite compares raw output types, mutations, shapes, broadcasting,
ignored forward targets and dirty-erasure failures. Every three-bit activation
pattern is checked with complete complex amplitudes over all controls and a
spectator, then all four unnormalized measurement branches are compared with the
original input divided by two. Work cleanup uses absolute tolerance `2e-12`.
Existing source gate-flow and control-predicate tests remain in the concentrated
regression. No source module is patched and no complete-file acceptance is claimed.
