# Retained native variational ansatz

The retained four-qubit full_hea is expressed as three bounded native loop iterations. Each layer applies X then a Y rotation to all four qubits before its directed CX chain; native R(theta, pi/2, q) expresses RY. Default binding resources are 4 qubits, 29 operations, depth 13 and no workspace, matching the frozen original row. The second nonzero binding uses 27 operations and depth 13 because the existing optimizer combines numeric rotations; resource counts are binding-specific.

Three distinct nonzero angle vectors, including mixed signs and angles beyond one revolution, match the exact original and independent Circuit over complete complex128 statevectors at absolute tolerance 1e-12 without phase alignment, under both width and depth objectives. Shape mismatch and insufficient qubit, gate, depth and memory budgets are rejected.

This covers the retained quantum ansatz, not the complete linear-system solving or chemistry Notebook workflow. No source importer or Qmod file is required by the user example. See [README](README.md) for executable commands.
