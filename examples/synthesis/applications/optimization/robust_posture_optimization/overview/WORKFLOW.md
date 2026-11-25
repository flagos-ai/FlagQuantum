# Posture workflow semantics and acceptance

Run the complete workflow from the repository root:

```bash
python -m examples.synthesis.applications.optimization.robust_posture_optimization.overview.main
```

The adjacent Notebook retains all fourteen source code stages, two qubits, six
angles, the original pi-linear schedule, thirty actual objective evaluations
and one thousand real finite samples for each X/Y/Z basis measurement. The
native ansatz remains unchanged, including the first rotation repeated on the
second qubit. Measurement wrappers compose native quantum functions.

The source Y basis applies S then H; this is retained literally. Its resulting
observable has the opposite sign to the usual inverse-S then H convention.
The source binary-count formula reads the most significant count column first:
that component is native q[1], followed by q[0]. The unit-length links are summed
as in the original. These details are tested independently rather than silently
corrected. All geometry is computed from actual finite counts, never exact means.

Public native COBYLA replaces SciPy COBYLA. Source maxiter limits objective calls,
whereas the native optimizer counts trust-region iterations; the callback stops
before a thirty-first quantum objective. On that stop, select the best actually
evaluated parameters and report nonconvergence, without inventing a solver result.
Trust radii and advancing sampling seeds are explicit. No optimizer trajectory
equivalence is claimed. Every parameter tuple is bound by synthesis, not runtime.

After optimization, retain one thousand final source-Y-basis samples and both
independently resampled position estimates from the original final cells. No
convergence or target-reaching guarantee is made. The convergence plot shows
actual noisy objective queries. The independent Circuit reference checks absolute
amplitudes, bit order and source measurement bases at two nonzero bindings.
No provider, hardware, speedup or scalability claim is made.

Acceptance additionally checks deterministic seeds, nonzero sample noise, the
source Y sign, source marginal geometry, malformed values, width/workspace
budgets and the objective-query cap. Both CLI and Notebook require real execution
and visual inspection; drafts and partial runs earn no acceptance credit.

The Notebook adds two initial three-basis diagnostics outside optimization
(6,000 draws). The workflow draws 3,000 samples per objective query and 7,000
for final readout. At thirty queries these are 97,000 workflow draws and
103,000 total Notebook draws. Both totals are reported explicitly.
