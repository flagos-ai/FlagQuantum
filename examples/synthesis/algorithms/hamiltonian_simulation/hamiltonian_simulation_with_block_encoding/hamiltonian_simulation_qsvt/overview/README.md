# Hamiltonian evolution with quantum singular value transformation

Run the [notebook](walkthrough.ipynb) or the local workflow from the repository
root.

```sh
python -m examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview.main
```

The workflow retains the four Pauli terms, two data bits, block scale 0.75,
evolution time 22, degree 33, and the original response scale 0.9999. It
generates a normalized real state with an explicit seed. In tensor-product
order, the source's logical bit one is the high bit, so its single-qubit
term is `ZI`.

The Jacobi–Anger coefficients are regenerated through Bessel quadrature.
Bounded Newton iterations fit both complementary phase sequences to the even
cosine and odd sine series. The QSVT response then comes from the real and
imaginary projector signals, and an additional LCU bit combines cosine with
`-i` times sine. The two CPU-executed phase streams use exact packed matrix
operations. Reported counts refer to this simulator representation.

A sequential response checks every complex amplitude. The physical block
projection recovers the evolved vector, which is compared against the matrix
exponential using the source's absolute L₂ tolerance of `1e-7`.


The Notebook also executes the visible native `main` in [quantum.py](quantum.py).
Its four-term Pauli block and both full phase streams compose through named
ports, within/apply, controlled borrowed helpers and adjoints. It neither packs
the final response nor admits a dense block matrix. For seeds 132 and 247 and a
signed coherent input, every amplitude agrees with the independent sequential
reference within 1e-10 without global-phase alignment. All 16 signal-block
columns independently verify the source bit order.

Current native resources are ten qubits (six outputs plus four clean work bits),
3077 logical instructions and depth 2729 for seed 132. The explicit budgets are
ten qubits, four work bits, 3500 operations and 1 MiB of statevector memory.
Insufficient workspace, qubit and operation budgets refuse synthesis. Counts
are not directly comparable with the packed six-qubit Circuit representation.
