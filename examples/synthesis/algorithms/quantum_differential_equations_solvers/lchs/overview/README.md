# Linear combination of Hamiltonian simulations

Recompute the original 64-point signed kernel and run the 128-dimensional
classical comparison. The four-dimensional quantum state uses the original
six grid bits, three block flags and one GQSP signal bit. It synthesizes the
signed linear block, generates the fresh Laurent phases and executes the
coherent kernel, followed by its inverse preparation.

Use the source Bessel-tail criterion to select the phase degree. Compare the
absolute zero-flag projection against all 64 independent unitary terms and
the exact nonunitary evolution. The two unseeded source matrix draws use
fixed local seeds for repeatable validation.

Run the Notebook or invoke
`python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.lchs.overview.main`.
