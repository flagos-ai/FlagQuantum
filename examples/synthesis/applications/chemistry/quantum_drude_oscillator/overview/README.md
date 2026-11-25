# Quantum Drude oscillators with native FlagQuantum VQE

Build both ansatz sizes with native Python quantum functions, regenerate the
oscillator Hamiltonians from occupation operators, and use the returned CPU
statevectors to evaluate the VQE objectives. The notebook runs the original
three-layer, four-qubit test, five two-oscillator separations, and five ten-layer,
six-qubit three-oscillator separations. It starts each optimization with all
angles equal to one and a 500-iteration budget.

Train with Adam and the public differentiable Circuit API using exact
statevector expectation. Synthesize every final binding and compare its full
state with the training circuit. The local learning rate is 0.04. Each plot uses actual optimizer
evaluations or independently regenerated classical curves. The direct
diagonalization is an independent diagnostic for finite bases and remains
separate from the variational result.
