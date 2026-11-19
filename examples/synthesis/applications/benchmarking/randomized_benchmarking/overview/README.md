# Native randomized sequence/inverse workflow

Run all five original-length words and ten local simulator experiments:

```bash
python -m examples.synthesis.applications.benchmarking.randomized_benchmarking.overview.main
```

`walkthrough.ipynb` covers all four source code stages. Native quantum functions
live in `ansatz.py`, independent Circuit/dense references in `reference.py`,
concurrent bounded workflow and actual readout in `workflow.py`, and observed
survival plotting in `plots.py`. All five native programs are retained.

The source hardware list filters to id, rz, sx and x in that order. A local
Python RNG preserves the original seed without altering global RNG state.
RZ(pi) retains its scalar convention and SX includes exp(i*pi/4). Prefix-only
checks on both basis states and a complex coherent state prevent inverse-pair
cancellation from masking incorrect gates or global phase.

The two actual local simulator modes are statevector and density matrix, both
exact CPU complex128. They do not claim to reproduce proprietary hosted backend
implementations. Each job has 1024 explicit seeded samples; the source leaves
its default count implicit. The checked native Program runs directly, and its
public CircuitIR also runs with explicit numerical-mode options. The sealed
Program itself receives no execution override.

The source product order pairs each length with both simulators, while its
analysis assigns a repeated five-length list; indices 1 through 8 are mislabeled.
This counterpart preserves that diagnostic and groups actual results by their
own metadata. Ideal survival curves do not identify noisy-device RB decay or an
error per Clifford. No physical benchmark, remote job or hardware comparison is
claimed. Original quantum-only examples and the frozen corpus are unchanged.
