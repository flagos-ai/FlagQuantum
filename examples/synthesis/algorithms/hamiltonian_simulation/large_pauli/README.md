# large pauli

Pauli Z evolution as native qfunc phase intent, Qmod and an independent Circuit.

Run this module for the small three-form example. Reproduce the unchanged
12-bit originals with tools/validate_qmod_large_pauli.py; this example does not
claim a native Python qfunc Hamiltonian-array API.

## Run

```sh
python -m examples.synthesis.algorithms.hamiltonian_simulation.large_pauli.main
python -m examples.synthesis.algorithms.hamiltonian_simulation.large_pauli.comparisons.circuit
```

The independent Circuit comparison checks complete complex amplitudes,
absolute phase and hidden workspace. It uses the native program directly.
Original importer comparisons are retained separately as test evidence.
