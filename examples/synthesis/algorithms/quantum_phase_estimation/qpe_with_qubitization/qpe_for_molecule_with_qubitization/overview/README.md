# H₂ phase estimation with qubitization

Run the [notebook](walkthrough.ipynb) or the local workflow from the repository root.

```sh
python -m examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.overview.main
```

The example regenerates the H₂ Hamiltonian at 0.735 Å in STO-3G and runs a
native full CI reference. The walk uses all four data bits, four LCU selector
bits, 15 fresh Pauli terms and five phase bits. The signed selector keeps the
source leaf order and uses zero preparation probability for its final duplicate.

The naïve method synthesizes one controlled walk and composes that exact block
31 times through the public Circuit interface. This keeps each synthesis
request within the compiler's mutable-helper limit. The optimized method
synthesizes the first controlled walk and 15 reflection-conjugated walk steps
as one native qfunc program. Both use all source powers. The four compiler
work bits are shared between successive block applications and return to zero.

Each route draws 1,000 outcomes locally with its own reproducible random seed.
The analysis keeps outcomes whose selector is zero and maps the signed phase
to the energy via the LCU scale times cosine. The plots retain joint frequencies.
The resource comparison converts both complete circuits to the same CX/U3
basis. The independent references check the full complex states of the two
routes separately because the amplitudes after their respective powers differ.
