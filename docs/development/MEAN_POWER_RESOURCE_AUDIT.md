# Mean-estimation signed-power resource models

Reference: Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`bloqs/mean_estimation/mean_estimation_operator.py` and its complex phase and
reflection dependencies.

The existing resource method now accepts uncontrolled nonzero signed powers
already admitted by the operator. Sequential repetitions scale every gate
category by the absolute power and retain one step's maximum live width. Negative
powers retain the same encoder/adjoint child model order: the complete inverse
schedule still begins with encoding and releases phase outputs before decoding.

Tests compare every resource field directly with original source queries for
precisions 1,2,5,32 and powers +/-1,+/-2,+/-3. Asymmetric encoder peak fixtures
independently check both directions, preventing symmetry from hiding a lifetime
error. The direct phase lifetime correction is recorded in
[its audit](PHASE_RESOURCE_LIFETIME_AUDIT.md).

Original controlled estimates fail while decomposing atomic C[And]. Native
controlled estimates retain their existing explicit refusal; neither reference
patches nor invented control formulas are used. These estimates are source
models, separate from physical primitive counts and numerical simulation.

This resource-only change does not publish the prepared shared execution root
or expand Core storage admission. Signed empty fixed-point storage remains
subject to pending Proposal124. Complete parent acceptance and all other gaps
remain in [the mean file audit](MEAN_ESTIMATION_FILE_AUDIT.md).
