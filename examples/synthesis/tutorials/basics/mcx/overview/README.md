# Optimizing multi-control X

Open [walkthrough.ipynb](walkthrough.ipynb) or run the complete tutorial.

```bash
python -m examples.synthesis.tutorials.basics.mcx.overview.main
```

The native qfunc allocates 14 controls and a target. It synthesizes a
depth-oriented circuit under the original 20-qubit ceiling. Under the
17-qubit CX/U ceiling, it compares the actual compiler output with an exact
two-clean-bit schedule that temporarily borrows controls from the opposite
group. The smaller emitted CX count wins. Two target values, one disabled
control and complex coherence participate in each full-state check.

For the 21-qubit hardware ceiling, local compilation uses the
[official FakeBoston 0.49.0 snapshot](https://github.com/Qiskit/qiskit-ibm-runtime/blob/0.49.0/qiskit_ibm_runtime/fake_provider/backends/boston/conf_boston.json).
The local [topology data](boston_topology.json) retain all 156 device vertices,
their 176 reciprocal edges and the device basis. The workflow evaluates six
placements and routes within that device, compares emitted depth, and checks
every selected edge and the whole complex state up to one common global
phase. These are offline snapshot results.

The source’s 50-control circuit keeps its complete width. Exact native MPS
checks target zero, a disabled control, and target XOR on both active inputs.
A separate trace over the emitted reversible instructions checks each
control and all clean work bits without a dense statevector.
