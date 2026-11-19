# Harmonic oscillator with a quantum Taylor expansion

Run the original 11 time points of the unit-frequency oscillator with native
FlagQuantum synthesis. Each point regenerates the eight Taylor coefficients
and uses them to prepare the three-bit selector. The zero-driving selector
and the one-bit data register preserve the source's five logical qubits.
The circuit applies all eight powers of the evolution generator and reverses
the encoding before postselection.

The table displays signed position and velocity, the original magnitude
readout, and the corresponding analytic and measured energies. Potential
energy uses the position, and kinetic energy uses the velocity. Signed
velocity reaches zero between 0.7 and 0.8, while squaring yields the same
energy curves as the source magnitude readout.

Run the notebook or invoke
`python -m examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.overview.main`
from the repository checkout.
