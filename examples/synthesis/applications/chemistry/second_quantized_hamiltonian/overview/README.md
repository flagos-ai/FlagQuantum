# Native second-quantized Hamiltonian

Build the source four-mode Hamiltonian with FlagQuantum's fermionic algebra,
map it with Jordan-Wigner and minimize its expectation with a native qfunc.
Retain all three ordered monomials, three X/RY/CX layers and twelve zero initial
angles. Each of 200 objective evaluations executes the actual native program.
The explicit coordinate search replaces a private optimizer with a 200-iteration
limit; these are different stopping contracts and no solver equivalence is claimed.

The last fermion term has positive occupation energy after anticommutation:
-0.3 c†2 c†3 c2 c3 = +0.3 n2 n3. An independent occupation-sign matrix checks
all sixteen basis columns, while a direct Circuit checks full complex amplitudes
for zero and two nonzero parameter profiles and the best visited parameters.
Mode labels in the mapped observable use FlagQuantum's documented qubit order;
aggregate named register words provide particle counts without wire decoding.

The source metadata records particles (1, 1), but its hardware-efficient ansatz
mixes particle sectors. Report actual particle-number populations and full-space
energy. Do not describe this as a fixed-sector variational solver. A 200-evaluation
budget stop does not certify convergence. Spectral enumeration is independent
validation, never an optimizer input. No hardware or speedup claim is made.

Run from the repository root:

```sh
python -m examples.synthesis.applications.chemistry.second_quantized_hamiltonian.overview.main
python -m examples.synthesis.applications.chemistry.second_quantized_hamiltonian.overview.comparisons.circuit
```

Open [the native Notebook](walkthrough.ipynb). It shows the quantum function and
public observable APIs directly. [Acceptance](WORKFLOW.md) and
[provenance](provenance.json) record the independently checked scope. The
[frozen quantum counterpart](../main.py) remains unchanged.
