# Quantum double-slit wave inversion

Run the original 16-by-16 wave problem from the repository root.

```sh
python -m examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment.overview.main
```

[The walkthrough](walkthrough.ipynb) regenerates both physical source
points, the absorbing boundary matrix, the two slits, and the full
degree-191 inverse. Fresh phases come from the same mapped Chebyshev
response on the measured singular-value range. The complete circuit
retains 191 signal calls with 192 projectors.

The example respects the source Notebook's 18-qubit width budget
with 15 logical and three clean scratch bits. Exact local packed
permutations and projector phases act on no more than seven wires at
once. Reported counts are simulator instructions for this circuit.
The frozen sibling example preserves its separate original Qmod
counterpart and resource contract.

An independent tensor implementation checks every complex logical
amplitude and the circuit checks all scratch sectors for leakage.
The projected field also agrees with the complex singular-value
reference in the prescribed odd-degree phase convention. The figures
show the reciprocal approximation and both physically normalized
wave fields. The animation uses the 60 actual complex-field phases at
the source's nominal 30 frames per second.

[Source mapping](provenance.json) · [Third-party license](../../../../../CLASSIQ_LICENSE.txt)
