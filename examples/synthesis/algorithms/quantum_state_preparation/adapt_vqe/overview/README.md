# ADAPT-VQE with native FlagQuantum synthesis

Run the original four-qubit Hamiltonian and all 56 ordered operator-pool
entries. Both adaptive loops keep the source's signed gradient selection,
repeat selections, and stopping rules. Each variational objective synthesizes
the active schedule and measures its exact statevector expectation with a
100-iteration local Nelder–Mead budget.

The standard loop preserves the source's one-step delay between selecting an
operator and training it. The modified loop starts with pool index 10 and
trains before screening. The source passes a separate modular schedule to a
helper that reads the modified loop's shared selections. The walkthrough runs
each fixed schedule explicitly and labels its actual native resources, fitted
parameters and energy.

Run the Notebook or invoke
`python -m examples.synthesis.algorithms.quantum_state_preparation.adapt_vqe.overview.main`.
