# amplification iterations

Static amplification in Qmod, existing native scalar qfunc, and Circuit.

Native programs spell out zero/one/two rounds with existing phase/CX primitives.
No native amplification export, mutable helper port or Python loop is added.
Run tools/validate_qmod_iterations.py for the three unchanged counting/QMCI files.

## Run

```sh
python -m examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.main
python -m examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
