# Source workflow and acceptance

The pinned source is `algorithms/metrology/classical_shadow_tomography.ipynb`
at commit `b69c4893d8264d1e59ce0909df2bd88efed25e00`. Its twenty code cells are
reviewed and hash-bound in `provenance.json`; original source is not bundled here.

| Original cells | Native behavior |
| --- | --- |
| 13-19 | Local imports, seed 552, two-bit Bell state Phi-minus, per-bit RZ/RY basis changes. |
| 21-22 | 400 random Pauli basis pairs and 400 actual single-shot quantum observations. |
| 27-31 | Full inverse-channel matrix reconstruction and Frobenius error against the exact Bell matrix. |
| 36-42 | Measured ZZ median of two block means; compute the weight-two Pauli sample budget for error 0.2, failure probability 0.01. |
| 45-46 | 50 random depth-20 H/S/CX circuits, original gate/seed RNG sequence, one actual shot each, retained gate provenance. The comment mentioning 200 does not override the executed Pauli count 400. |
| 48-54 | Full Clifford inversion, Frobenius error, ZZ median of ten means and ideal-Clifford sample-bound calculation. |

Native R(theta, pi/2) implements the source RY. Every actual circuit passes
through native synthesis and fq.run. Static float parameter binding replaces
the source's remote single-template/batch interface; this example does not
claim generic late runtime binding. NumPy RandomState preserves the original
basis, random-gate and execution-seed stream. Torch sampling explicitly replaces
the remote sampler; identical original remote shot outcomes are not asserted.
All 450 actual single shots are retained; independent references do not replace
quantum execution in the user workflow.

Named q's integer bit i is logical q[i]. Reconstruction tensor factors place
logical index zero first, so matrix basis indices and packed integer values need
an explicit permutation. The Y-basis HS rotation maps plus-y to measured one;
its observable sign is reversed. Median-of-means uses the average of the two
middle block means when the number of blocks is even. Linear-inversion shadow
matrices may be nonpositive and are not silently projected to physical states.

The two calculated error budgets are 81065 Pauli and 108086 ideal-Clifford
samples, each reporting ten divisions. They are computed, not executed. Neither
the actual 400/50 observations nor the finite-depth-20 random Clifford walk is
claimed to satisfy these guarantees. The walk is an approximate ensemble; no
exact design or finite-walk convergence bound is certified here.

Acceptance compares all 450 full absolute complex states, per-circuit resources
and same-seed single-shot outcomes against independent Circuit execution. It
also compares every reconstructed snapshot matrix, full means and ZZ block
statistics using independent Bloch matrices and basis-vector Circuit unitary
construction. An asymmetric complex-state inverse-channel check protects Y
sign and tensor order. Two arbitrary nonzero angle bindings, clean work,
negative input/shape/sequence checks, conventional even median, actual resource
budget refusal, plots, clean Notebook, fresh Jupyter and CLI are required.
Frozen 216+6 corpus replay, quality checks and the full default tier complete
local delivery. Classification: `single_device_fast_path`; no distributed,
hardware, release or quantum-advantage claim is made.

## Recorded local acceptance

On November 13, 2026, all 51 focused cases and 49 catalog/dependency cases passed. Strict typing (including flattened Notebook code), Black, Ruff, architecture, documentation links, repository language and compiler-team scope checks passed. A fresh Jupyter kernel executed all five native cells plus one independent evidence cell; all eight evidence scenarios passed. The actual inline PNG was inspected and the CLI completed.

The workflow executed 450 single-shot circuits:400 Pauli and50 depth-20 Clifford circuits. Each used two physical qubits and zero clean workspace.

| Collection | Operations range | Depth range | Frobenius error | ZZ median of means |
| --- | --- | --- | --- | --- |

| Pauli | 3-7 | 3-5 | 0.229802904681381 | 0.81 |

| Clifford | 7-18 | 5-15 | 0.612372435695793 | 0.999999999999999 |


All 450 full absolute complex states and same-seed observations matched independent Circuit references. Every inverse-channel snapshot, full reconstruction matrix, Clifford matrix and ZZ statistic passed independent checks. Sample budgets81065/108086 are calculations only;the finite-depth Clifford ensemble remains approximate.


The complete216-original+6-approved replay was unchanged:210 original execution passes,6 semantic refusals,and6 approved revision passes. The complete default tier finished with19619 passed,768 skipped,24 failed and6 errors. Its30 unresolved nodes exactly match the preceding accepted baseline,with no added or removed nodes. All4106 Python inputs stayed frozen through the terminal run;only catalog count/validated-entry assertions change afterward and are retested before sealing. This records local acceptance,not a fully passing default tier or remote CI result.
