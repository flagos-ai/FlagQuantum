# Native AQC quantum program

Run `python -m examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.main` from the repository root, or run the `comparisons.circuit` module for an independent full-state comparison. User programs import only FlagQuantum Python.

This reproduces the complete retained three-qubit quantum program, not the full multi-scenario linear-solver Notebook. H0 has 10 terms and H1 has 16; all 200 source calls, declared order, floating time literals, and repetitions are retained. Classical binding supplies the schedule to the native program; no physical wire labels or bit-string decoding are required.

The logical product contains 17,000 Pauli rotations. The Compiler uses bounded static products only when expansion would exceed the 4,096-effect capture cap. For complete one-to-three-bit targets, it composes at most 256 consecutive calls, emits one checked unitary per 64-call chunk, and preserves intervening effects and helper ownership boundaries. Existing unitary verification and lowering produce the physical Circuit. This does not enlarge the capture bound or add an execution kernel.

Measured local output is 3 qubits, 4,542 operations, depth 4,053, zero workspace, matching the frozen original resource row. Independent complex128 Circuit products preserve global phase and bit order; the maximum observed complex-amplitude difference is below 1e-11. Reproduction of the declared product formula is not a claim that its physical approximation solves every linear system accurately. The full Notebook workflow remains pending and is not counted complete.
