# swap test

One-bit swap test in native qfunc, Qmod and public Circuit syntax.

The pinned three-bit original has non-normalized vectors and remains refused.
Run tools/validate_qmod_swap_test.py for its labelled diagnostic comparison.
This example uses valid one-bit states and existing native phase/CX primitives;
it adds no native array preparation or swap_test export.

## Run

```sh
python -m examples.synthesis.algorithms.state_preparation.swap_test.main
python -m examples.synthesis.algorithms.state_preparation.swap_test.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
