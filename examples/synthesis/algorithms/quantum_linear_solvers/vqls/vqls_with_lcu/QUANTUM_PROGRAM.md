# Retained native variational ansatz

The retained three-qubit ansatz composes three real-angle layers and two ordered CZ layers. Native R(theta, pi/2, q) expresses RY; a local qfunc uses H/CX/H for CZ. The independent Circuit uses direct RY and CZ. Both measured nonzero bindings use 3 qubits, 23 operations, depth 14 and no workspace, versus the frozen original 3 qubits, 15 operations and depth 9. This semantic migration does not claim equal resource efficiency.

Three distinct nonzero angle vectors, including mixed signs and angles beyond one revolution, match the exact original and independent Circuit over complete complex128 statevectors at absolute tolerance 1e-12 without phase alignment, under both width and depth objectives. Shape mismatch and insufficient qubit, gate, depth and memory budgets are rejected.

This covers the retained quantum ansatz, not the complete linear-system solving or chemistry Notebook workflow. No source importer or Qmod file is required by the user example. See [README](README.md) for executable commands.
