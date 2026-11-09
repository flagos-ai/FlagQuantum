# grover

One Grover round in Qmod, native qfunc and an independent public Circuit.

The native form composes existing high-level phase functions; no native Grover
export or Python callback port is claimed by this Qmod compatibility slice.

## Run

```sh
python -m examples.synthesis.algorithms.grover.grover.main
python -m examples.synthesis.algorithms.grover.grover.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
