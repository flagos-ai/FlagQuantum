# Fixed-point amplification with fresh native QSVT phases

Fit the original odd degree-25 Chebyshev approximation at target height 0.95
on the original overlap interval. A local Remez exchange and symmetric phase
factorization regenerate both the polynomial and its 26 phases. The native
oracle specializes the two arithmetic matches computed over all 64 words.
The predicate work bits return to zero after every projected rotation.

The walkthrough reports the polynomial and phase-response errors, the native
synthesis resources, the expected good count, and 2,048 actual local draws.
It retains the source's five-standard-deviation acceptance rule.

Run the Notebook or invoke
`python -m examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.overview.main`.
