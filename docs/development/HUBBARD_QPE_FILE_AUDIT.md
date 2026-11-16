# Hubbard QPE cost tutorial file audit

Q-0442 covers the pinned Qualtran v0.7.0 Hubbard QPE optimization notebook.
Q-0444 covers the Suzuki test file, including its notebook check. The pinned
notebook at `bcb17bb25360ed68ce3bc084645db3e929799389` matches blob
`3d6f4c68f691a3d5fbdd5de68fc0526df1fb6312`. The unchanged ordinary
Suzuki cost test passed, and the source graph and native circuit checks
are recorded in the Suzuki file audit.

The pinned notebook fails at zero-based cell 22 in the reference
environment. SciPy supplies a one-element vector to the objective;
the pinned scalar logarithm rejects the resulting array. A separate
in-memory diagnostic with only `np.asarray(delta_ts).item()` at that
boundary completed all 36 cells. The source file bytes remained
unchanged. The native tutorial performs the same scalar extraction.

The native independent kernel covers the three error terms, the
initial 8×8 schedule and synthesis precision, a cost surface and
contour, a fixed-total budget sweep, the sampled optimizer, and the
symbolic formula. Its 100-point synthesis sweep uses the same bounded
Suzuki subproblem and rounded per-rotation synthesis cost as the
reference. The plots label the two axes by the actual error arguments.

| Variant | Direct T equivalents per step | Rotations per step | Optimized QPE T estimate |
| --- | ---: | ---: | ---: |
| Ordinary complete | 768 | 320 | 9.226 million |
| HWP complete | 2,008 | 60 | 3.071 million |
| HWP stripped | 1,760 | 48 | 2.565 million |

The initial split gives a timestep of approximately 0.01754777844,
a per-rotation precision of approximately 0.0000008949367, and a T
estimate of 10.49 million. The symbolic count reduces to the numerical
ordinary formula under substitution for both tested synthesis allocations.
These are resource estimates. The reference's separate paper comparison
is 2.420 million under its displayed Toffoli adjustment. Native measured
primitive counts remain in the Suzuki audit.

The replacement test executes the checked-in notebook in a kernel that
rejects Qualtran, Cirq and Quimb imports, validates one SVG and three
plots, and exercises numerical assertions and symbolic substitution.
This closes the native notebook and Suzuki-test migration. The unchanged
source optimizer compatibility finding remains recorded above.
