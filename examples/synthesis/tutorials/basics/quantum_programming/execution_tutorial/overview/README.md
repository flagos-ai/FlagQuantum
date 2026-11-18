# Executing quantum programs

Open [the walkthrough](walkthrough.ipynb), or run the complete tutorial from the repository root.

```bash
python -m examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial.overview.main
```

The native Python `qfunc` coherently calculates `x² + 1` on all eight inputs. The synthesized circuit uses three input bits, six result bits, and one clean work bit. The tutorial draws 2,048 and 10,000 samples and checks the complete complex state against an independent Circuit.

The three-qubit GHZ example runs ideal statevector, noisy density-matrix, and MPS execution. The local noise approximation uses the gate and asymmetric assignment errors recorded on Pittsburgh's physical chain 0–1–2 in the public [IBM FakePittsburgh 0.49 snapshot](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/0.49/fake-provider-fake-pittsburgh). Independent dense propagation checks the prepared state, and a classical confusion matrix checks the measured probabilities. The two noisy and MPS sampling paths each draw 2,048 shots. The tutorial lists the backends installed locally.

The chosen approximation models gate errors with a uniform Pauli mixture and includes independent readout errors. It omits time-dependent relaxation, crosstalk, and other details of the original hosted simulator. All results come from local FlagQuantum execution.
