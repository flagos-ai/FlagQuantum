# Lanchester population dynamics with native HHL

Regenerate the original eight-step finite-difference model and its full
32-dimensional Hermitian embedding. Native Python synthesis implements all
six phase bits and 64 reciprocal rotations. The notebook reads the signed
indicator-one, phase-zero solution and compares all population steps with
the direct solve.

The coherent SWAP comparison uses the entire HHL logical state, including
its indicator and phase branches. After checking the four work bits, it
feeds that state into a second FlagQuantum Circuit together with the
normalized direct solution. Five controlled swaps and two Hadamards produce
the distribution for 30,000 actual seeded local draws. The statevector
overlap and sampled SWAP overlap retain their distinct conditioning.

Run the notebook or invoke
`python -m examples.synthesis.applications.physical_systems.hhl_lanchester.overview.main`.
