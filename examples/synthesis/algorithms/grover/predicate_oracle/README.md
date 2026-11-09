# predicate oracle

One predicate-driven Grover round in Qmod, native qfunc and public Circuit.

## Run

```sh
python -m examples.synthesis.algorithms.grover.predicate_oracle.main
python -m examples.synthesis.algorithms.grover.predicate_oracle.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
