# Classical Pauli values and basis phases

Simulation owns `classical_pauli_values.py`, ported from the unchanged Qualtran
v0.7.0 `XGate`, `ZGate` and `CZ` value/phase methods in `basic_gates/x_basis.py`
and `basic_gates/z_basis.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.
Compiler retains the existing `pauli_gate_ir` control dispatch and emitted IR.
No numerical executor or external SDK dependency is added.

X evaluation retains `(value + 1) % 2`, including NumPy promotion, overflow and
diagnostic errors. Z and CZ numerical callbacks preserve named values and extra
fields without copying their objects. Their separate basis-phase methods return
minus one only for exact equality with one; CZ checks its first operand before
the second. Non-bit diagnostic values are not promoted into valid qubit states.
Array ambiguous-truth failures and the original short circuits remain visible.

Unchanged classical register values do not mean unchanged complex amplitudes.
The source's phase methods are therefore ported separately from value movement.
Reference tests reconstruct the original X/Z/CZ tensors, then bind the native
value/phase functions to existing emitters over complete complex input states
with an entangled spectator and exact inverse restoration. Existing controlled
Pauli suites retain all small measurement branches and workspace cleanup.

Run `python -m examples.algorithms.classical_pauli_values` for the ordinary
`circuit.append(op)` path. It composes X, Z and CZ and compares the complete
complex state against the numerical bit action plus phase product. The example
emits one X, one Z and one CZ, no T instructions, no measurements and no work.
The source logical model has three Clifford components and zero T equivalents;
both original QEC cost profiles are checked independently from emission.

This slice does not implement generic stochastic classical simulation or fold
measurement-dependent phases into ordinary integers. Raw callback parity,
quantum tensor parity and resource-model parity are separate evidence.
