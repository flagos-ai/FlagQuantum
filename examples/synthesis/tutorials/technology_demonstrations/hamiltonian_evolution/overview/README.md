# Water Hamiltonian evolution benchmark

Run the complete example locally from the repository root.

```sh
python -m examples.synthesis.tutorials.technology_demonstrations.hamiltonian_evolution.overview.main
```

[The walkthrough](walkthrough.ipynb) regenerates the STO-3G water model at
the original geometry with one frozen orbital. It retains 12 spin orbitals,
551 Pauli terms, the benchmark's logical label order, and all seven schedules.

| Schedule | Repetitions or draws |
| --- | ---: |
| First-order Trotter | 6 |
| Second-order Trotter | 4 |
| Fourth-order Suzuki | 1 |
| Controlled first-order Trotter | 6 |
| Controlled second-order Trotter | 4 |
| qDRIFT | 1,000 |
| qDRIFT | 2,000 |

The first- and second-order steps and qDRIFT use native qfunc synthesis.
Compose the original number of steps using the public Circuit API. The
fourth-order formula composes five signed second-order steps. For coherent
control, each Pauli factor uses an unconditional basis and parity change
around its controlled diagonal phase. The identity term contributes the
correct phase on the control-one branch.

The independently built Circuit reference checks the full complex state,
including the relative phase of the two control branches. The final basis
converter targets CX and U. A second check allows one overall phase for that
basis conversion. The benchmark reports actual local circuit metrics. The
Qiskit values are the historical comparison data supplied by the source
Notebook and retain their original labels. Local results are not a hardware
or compiler optimization speed comparison.

[Source mapping](provenance.json) · [Third-party license](../../../../CLASSIQ_LICENSE.txt)
