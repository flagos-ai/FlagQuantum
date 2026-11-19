# Hamiltonian workflow acceptance

Retain the original two-qubit Hamiltonian, three initial methods, all 100 time points from 0 to 2, first-order Suzuki formulas with 10 initial and 30 sweep repetitions, and 288 qDRIFT samples. Bind an explicit local seed zero. Exact evolution specializes a static native unitary using a Hermitian spectral decomposition; there is no new synth.exponentiate operation. Each time is bound before synthesis, rather than dynamically binding one compiled program. Native products use declared term order; no Classiq transpilation, search or remote-sampler equivalence is claimed. Diagonal magnetization uses named logical probabilities, not physical wire IDs or bitstring parsing. Torch CPU matrix_exp replaces SciPy for the ideal curve; independent Taylor evolution checks all exact states. Retain all four plotted curves.

Every original code cell is reviewed and hash-bound in provenance. User-facing
examples contain native Python qfunc definitions and public FlagQuantum APIs;
no source-language loader or remote credentials are needed.

Independent Circuit references check absolute complex amplitudes, declared bit
order and zero workspace. LCU checks both construction stages and controller-zero
probability 0.75. Hamiltonian validation covers three initial states and 300 sweep
states, including the original repetition/sample counts, fixed-seed sampled Pauli
sequence, both endpoints and all magnetization values. An independent scaled
Taylor series supplies the exact-evolution oracle rather than reusing the native
spectral construction. Real Jupyter execution and plot inspection complement
headless numeric/plot-content tests. Plot tests skip when optional Matplotlib is
unavailable; numerical tests do not depend on plotting.

The Hamiltonian dense matrices have dimension four. These examples do not claim
scalable dense conversion, hardware execution or resource optimality. Notebook
acceptance is separate from the frozen 216-program quantum corpus.

```sh
python -m pytest tests/integration/test_synthesis_lcu_evolution_notebooks.py -q
```
