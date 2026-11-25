# Quantum thermal state preparation

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.main
python -m examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.comparisons.circuit
```

The program retains the original fifty delta cycles at delta 0.1. Each cycle
prepares the sixteen Gaussian amplitudes, applies controlled diagonal energy
phases, selects the four Pauli jump operators, uncomputes energy evolution,
applies inverse QFT and the retained sixteen-entry Y-rotation table, and
conditionally uncomputes the block encoding. It resets seven published block
bits and the hidden delta bit with actual trace-preserving channels. These are
400 real reset operations; replacing them with unitary zero preparation would
change the mixed-state computation.

Named outputs are the two-bit state, Boltzmann flag, four-bit Bohr-frequency
register and two-bit jump register. They flatten the original nested record
without losing fields or changing the register order. The signed Bohr word is
used only for its raw-bit Fourier/table indexing in this program. Native named
arrays expose the same raw storage; no signed arithmetic interpretation is
claimed. The hidden delta bit is explicitly dropped after reset, not asserted
clean without a trace check.

The diagonal Hamiltonian is expressed as the energy polynomial
`2*(1-state[0])*(1-2*state[1])`. Physical jump operators use two bounded three-bit
selection matrices constructed from X, Y and identity. They implement the exact
same four operator branches without a dense nine-bit block matrix. The compiler
shares the ordinary Python reset cycle through its existing checked replay IR;
the public program still uses an ordinary static loop. Default zero preparation
is skipped at capture time so the existing exact stationary-channel execution
path applies. Optional nonzero initial amplitudes/phases use the ordinary density
path; they do not claim the stationary zero-input optimization.

The independent Circuit uses a Householder Gaussian preparation, controlled
analytic energy exponentials, a separate four-branch Pauli matrix, explicit
Fourier gates and independent Y-rotation matrices. Tracing its block/delta
branches constructs 256 Kraus operators on the two retained state bits. Their
completeness relation and the entire final density matrix, including discarded
bits, are acceptance requirements. Encoded-block comparisons additionally check
complex amplitudes and absolute global phase, which density matrices alone
cannot distinguish.

The original default has 11 physical bits, 1 workspace bit, 22,200 operations and depth 19,901.
The native default has 10 physical bits, 0 workspace bits, 57,700 operations and depth 51,051.
Both retain 400 reset channels. These counts use their actual registered gate
bases; the native gate overhead remains an optimization gap. This is a quantum
program counterpart. The Notebook's physical analysis and plotting workflow
remain pending.
