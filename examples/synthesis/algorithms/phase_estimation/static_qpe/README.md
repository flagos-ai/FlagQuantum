# static qpe

One-bit phase estimation with Qmod, existing scalar qfunc, and Circuit.

The native example composes existing operations. It does not advertise a native
array QPE export; the three unchanged full QPE originals have a separate validator.

## Run

```sh
python -m examples.synthesis.algorithms.phase_estimation.static_qpe.main
python -m examples.synthesis.algorithms.phase_estimation.static_qpe.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
