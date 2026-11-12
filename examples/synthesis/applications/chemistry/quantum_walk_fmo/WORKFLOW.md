# Native continuous-time FMO walk

`physics.py` records the original seven-site physical Hamiltonian and all 35
Pauli terms in their declared product-formula order. Native qfunc allocates
three target bits and applies first-order Suzuki-Trotter evolution with ten
repetitions. Observable indices bind to logical target elements, with zero least
significant. Identity terms retain their global phase.

The full local Notebook workflow executes twenty quantum time points between
zero and 0.1, reads named site populations, records elapsed time, computes one
hundred exact seven-site reference points, and saves the first three sites in
a population comparison plot. The exact reference uses a dense matrix exponential
on the physical site matrix. It does not use the compiler's Pauli decomposition.
The independent Circuit derives each Pauli coefficient by a trace projection of
that same physical matrix, verifies reconstruction and forms the declared dense
product. Raw complex states, including absolute phase, match both the native
program and unchanged source at two nonzero times. All eight basis columns,
including the padding state, are checked separately.

```sh
python -m examples.synthesis.applications.chemistry.quantum_walk_fmo.main
python -m examples.synthesis.applications.chemistry.quantum_walk_fmo.comparisons.circuit
python -m examples.synthesis.applications.chemistry.quantum_walk_fmo.workflow --output fmo-populations.png
```

Each classical real time is specialized at synthesis. Reported timings include
synthesis, execution and named readout; they are not directly comparable to a
cloud service's timing for an already symbolic program. Native Observable
arithmetic replaces source complex projector summation after exact cancellation;
PyTorch's dense matrix exponential replaces SciPy's, and a saved image replaces
interactive display. No source importer, external service or hardware runs.

Both native and original default programs use three qubits, 1472 operations,
depth 1223 and zero workspace. The eighth target state is physical padding,
not compiler scratch. A finite product formula can populate it even though
exact Hamiltonian evolution does not. The workflow reports that population and
probability-vector L2 error separately; it does not claim exact evolution or an
unmeasured approximation bound. Tests check normalization, the full time sweep,
resource refusals and the generated figure.

On the retained twenty-point grid, the measured maximum probability-vector L2
error against exact evolution is 0.95011269; maximum padding-state population is
0.0016898216. These are properties of the original fixed ten-step approximation,
not a compiler accuracy guarantee. Matching the original and an independent
product formula certifies reproduction, not adequate physical accuracy for an
application. Repetition count is intentionally preserved for source acceptance.
