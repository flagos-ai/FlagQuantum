# Hadamard edge detection

Run the complete 128-by-128 example from the repository root.

```sh
python -m examples.synthesis.applications.image_processing.quantum_hadamard_edge_detection.overview.main
```

[The walkthrough](walkthrough.ipynb) covers all 17 source stages. It loads
the original RGB photo, reproduces the posterized preview and then encodes
the unmodified photo. A single native qfunc prepares all 16,384 real amplitudes and performs both
controlled cyclic shifts. The compiler lowers the preparation to RY and CX
operations; users supply the normalized data rather than designing these gates. A fast Walsh transform computes the
rotation angles without allocating a dense 16,384-square matrix.

The high/low seven-bit numeric slices encode row/column; controlled shifts
interfere in the Hadamard basis. The separate Circuit loader is an independent
reference, not the implementation used by the native program.
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
