# Lcu workflow acceptance

Retain both original construction stages, two controller/data bits, probabilities [0.5,0.25,0.25,0], coherent SELECT of identity/QFT/inverse QFT and within/apply uncomputation. Exact loading satisfies the original 0.01 error allowance. Local resource summaries replace graphical cloud display; named probabilities add a local inspection. No arbitrary-input loading-unitary equivalence or hardware resource comparison is claimed.

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
