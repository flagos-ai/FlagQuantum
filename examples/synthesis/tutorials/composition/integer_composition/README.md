# integer composition

A repeated phase cost in Qmod, native qfunc and an independent Circuit.

The Qmod entry binds an exact integer. The equivalent native qfunc uses two
static helper calls; its existing Python parameter contract is unchanged.

## Run

```sh
python -m examples.synthesis.tutorials.composition.integer_composition.main
python -m examples.synthesis.tutorials.composition.integer_composition.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
