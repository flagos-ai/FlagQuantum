# Gaussian state preparation with QSVT

Run the [notebook](walkthrough.ipynb) or the local workflow from the repository
root.

```sh
python -m examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.overview.main
```

The workflow keeps all eight fractional input bits, 256 Gaussian points, the
controlled sine block, and the original degree-28 even and degree-27 odd
polynomials. It regenerates the coefficients and symmetric QSVT phases.
The native circuit reuses the original-size sine, projection and reflection
building blocks with the fresh phase arrays.

The source's continuum integral gives a success amplitude of approximately
0.3958. The discrete synthesized circuit measures approximately 0.3919.
The example checks the complete first preparation against an independent
signal-block reference, then uses the measured value to calibrate the two
exact amplification rounds. The complete amplified state is checked against
independent reflections. The probability plot includes the unnormalized
zero-block probabilities, so the success mass remains visible in the report.
Resources are measured in the native instruction sequence and after conversion
to the common CX/U3 basis.
