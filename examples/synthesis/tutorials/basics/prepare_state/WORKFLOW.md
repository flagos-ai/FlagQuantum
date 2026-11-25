# Native probability-state preparation

The user supplies the original 32 probabilities and the probability-vector L2
error bound `0.01`. `prepare_state` infers the five-qubit output array; synthesis
constructs the rotations and entangling gates. The user does not allocate
physical wires or decode measurement bitstrings.

The reviewed Notebook defines the distribution, synthesizes the program, opens
the circuit viewer, samples and displays the first sample rows. `main.py` draws
the native circuit locally, executes it, draws 1000 seeded samples from its named
probabilities, and prints counts plus the first five rows ordered by output value.
This replaces cloud sampling and dataframe display with local sampling and plain
Python rows. It does not claim hardware measurements or identical random counts.

The probability-vector L2 acceptance bound applies to the executed quantum state,
not to a finite sample histogram. The independent dense-reflection Circuit checks
every complex amplitude including absolute phase and output ordering; the runtime
test also checks the nonnegative square-root amplitudes and zero workspace.
The native circuit retains the frozen original cost: five qubits, 61 operations
and depth 57. No approximate gate reduction is claimed.

The [provenance record](provenance.json) binds the original source and all reviewed
Notebook code cells. Run the commands in [README](README.md).
