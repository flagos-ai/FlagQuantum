# Quantum program scope

The migrated program preserves the source's three repeated layers and the
four-by-four complex unitary. Each layer applies that matrix to
`(target[1], target[2])`; local matrix bit zero is the least significant bit,
matching the source's ordered target list. The third qubit remains a spectator.

Run the native program and independent Circuit comparison with:

```sh
python -m examples.synthesis.applications.benchmarking.quantum_volume.main
python -m examples.synthesis.applications.benchmarking.quantum_volume.comparisons.circuit
```

The complete complex statevector comparison checks matrix order, absolute
phase, and clean hidden workspace. At the retained source parameters, both
circuits use 3 qubits, 93 operations, depth 61, and zero workspace. This
validates the quantum-program portion only. The source Notebook's random
ensemble, heavy-output study, plots, and hardware execution remain outside the
migrated workflow and are not counted as a complete Notebook.
