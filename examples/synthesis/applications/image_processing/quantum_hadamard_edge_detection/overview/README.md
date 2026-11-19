# Hadamard edge detection

Run the complete 128-by-128 example from the repository root.

```sh
python -m examples.synthesis.applications.image_processing.quantum_hadamard_edge_detection.overview.main
```

[The walkthrough](walkthrough.ipynb) covers all 17 source stages. It loads
the original RGB photo, reproduces the posterized preview and then encodes
the unmodified photo. The 14-bit preparation builds real amplitudes from the
zero state using RY and CX operations. A fast Walsh transform computes the
rotation angles without allocating a dense 16,384-square matrix.

The native qfunc composes both controlled cyclic shifts in the Hadamard basis.
The two direction bits remain in the local simulator state so the complete
amplitudes can be checked before taking the marginal over them. Their OR
sets the readout bit. The circuit stays within the original width budget of
20, including three compiler work qubits. Full-state comparisons use
independent cyclic finite differences and preserve the complex phases.

The reported edge image is reconstructed from 200,000 finite local draws.
Its values are the square roots of the joint pixel and edge frequencies.
The original threshold 0.003 and display transposition are preserved. The
extra exact edge panel provides a comparison with the finite sample.

[Source mapping](provenance.json) · [Source image and license](data/README.md)
