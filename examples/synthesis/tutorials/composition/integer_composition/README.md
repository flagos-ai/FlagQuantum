# Integer composition

A repeated phase cost in Qmod, native qfunc and an independent Circuit.

Profile: **High-level Python example**.

Run from the repository root.

```sh
python -m examples.synthesis.tutorials.composition.integer_composition.main
```

Start with [main.py](main.py) for the native qfunc program.

The independent [Circuit comparison](comparisons/circuit.py) executes the
[Qmod source](comparisons/source.qmod) and the same imported native program,
then checks complete complex states, absolute phase and clean workspace.

```sh
python -m examples.synthesis.tutorials.composition.integer_composition.comparisons.circuit
```

The native example uses the currently supported scalar qfunc composition.
The Qmod example may use higher-level library calls with a broader source profile.

Qmod provenance and license are retained with the
[Classiq fixtures](../../../../../tests/fixtures/classiq_qmod).
The small inline comparisons are illustrative programs. Copies of original
fixtures retain their exact source text.
