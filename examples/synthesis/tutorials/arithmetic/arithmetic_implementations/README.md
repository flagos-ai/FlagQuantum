# Arithmetic implementations

Prepare, compose and run x*x+y with the shared arithmetic implementation guard.

Profile: **High-level Python example**.

Run from the repository root.

```sh
python -m examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main
```

[main.py](main.py) retains this tutorial's existing preparation, computation,
result checks and supported syntax. Read its module description for limits.

The [Qmod source](comparisons/source.qmod) and [Circuit comparison](comparisons/circuit.py)
verify all 16 inputs, complete amplitudes and clean workspace.

```sh
python -m examples.synthesis.tutorials.arithmetic.arithmetic_implementations.comparisons.circuit
```
