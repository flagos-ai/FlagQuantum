# Explicit ADCR fixed-point preprocessing

```bash
python examples/algorithms/tensor_fixed_point.py
# ADCR fixed point: 3 -> 1 tensors; 8192 complex outputs
# Explicit preprocessing; no width estimator or physical gates added.
```

Simulation's `full_simplification.simplify_tensor_network` composes the existing
squeeze/A/D/C/R numerical owners. Capture register output groups before this
operation and supply them to the same pure or superoperator contraction consumer
afterward. The example checks complete complex output and phase gradients at 13
output bits. Normal circuit construction remains `circuit.append(op)`; this
numerical helper introduces no physical operation or public export.

## Source control flow and ownership

Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, `_dense.py`
blob `13aa81f50654ccaaa4293e8a63991686723022f5`, delegates wide-network
preprocessing to Quimb. Qualtran does not pin that dependency; comparison locks
unchanged Quimb **1.15.0**. Its `TensorNetwork.full_simplify` SHA256 is
`d2c3e546b58a2af55a33a2035b0cd9632d1ef4416bbf053d46bfde33e67842df`.
The test executes and hashes the original control loop without rewriting it.

Native preprocessing implements the default `ADCR` sequence, tolerance 1e-12 and
no norm balancing. It captures outputs, squeezes once, then always performs a
complete A/D/C/R round. It repeats until the pair `(number of tensors, number of
distinct indices)` is unchanged. Axis occurrence count is not the stopping rule;
there is no arbitrary round cap, intermediate dense reconstruction, SVD or
normalization. Counts cannot increase under these four reductions.

The existing individual owners accept private run-state arguments for reuse by
this loop. Negative structure checks retain occurrence/data identity across
rounds. R retains failed-pair checks, so cached failures do not consume later
search budgets again. Cached values keep the original tensors alive, preventing
Python object-ID reuse during the run; no tensor copies are made for the cache.
Run caches are released on return, including on failure. No global cache or
external persistence API is introduced. Retained references can increase peak
live storage; peak process RAM and GPU performance have not been measured.

D merges ordered neighbor lists by appending removed-edge occurrences to the
retained-edge list. R consumes that order, removes contracted occurrences and
appends the new occurrence. Rebuilding neighbors from global tensor order would
change subsequent operand order and is now avoided. This small run-local side
table owns traversal order only; existing nodes, einsum kernels and register
consumers still own graph data and numerical execution. Standalone R uses the
same ordered-neighbor implementation, replacing its earlier global scan.

## Comparison and limitations

Tests record every source/native flip, reindex, slice, pair operand and retained
axis sequence, plus each pass's tensor/index counts and round order. A fixed
seven-tensor sparse graph needs three rounds, including two productive rounds.
Finite results are compared in full complex form against source and original
contraction. Fixed sparse phase gradients, pure/doubled register groups,
prepared/unitary Bell graphs, input immutability, tolerance refusals and the
executable 13-bit example are covered. Existing physical measurement-branch,
workspace and resource tests remain in concentrated regression.

Inherited [A/D differences](TENSOR_DIAGONAL_REDUCTION.md) and
[C/R boundaries](TENSOR_COLUMN_REDUCTION.md) remain explicit. Native A flips all
repeated occurrences, whereas source A can corrupt repeated-index traces. Native
R absorbs scalars into one surviving leaf rather than spreading fractional powers.
Six real-valued sparse-network cases expose source scalar division by zero:
source returns NaN while native and unsimplified contraction return finite zero.
These are separately asserted defect cases, not a finite-result parity claim.

Positive support tolerance can amplify a discarded 0.5e-12 contribution into a
0.5 final error. Even at zero tolerance, ambient derivatives of discarded zero
entries can change; fixed-support gradients are the supported comparison scope.
Full preprocessing is explicit and is not silently enabled in training or the
default register consumers.

Custom pass sequences, S/L/P splitting, norm balancing and caller-persistent
caches are outside this default-policy slice. The complete Qualtran width > 12
entry is **still open**: source `contraction_width()` uses its contraction planner.
An exact numerical threshold check does not establish equivalent planner width;
substituting native greedy width would be a different policy. This batch does
not claim that automatic trigger or close Q-0887. The next work is the width
planning boundary and its source comparison, followed by explicit integration
with the established tolerance/gradient policy.

## Resources and acceptance

The 13-bit example reduces three numerical tensors to one, returning 8,192
complex128 values (131,072 output bytes, not process peak RAM). Added physical
gates, measurements, work bits and feedback slots are all zero; a quantum gate
cost model is not applicable. No numerical tensor reduction is represented as a
physical gate-count improvement. Q-0887.03 is extended without a duplicate
capability ID, parent closure, G3 expansion or PR merge.

Local acceptance: **60 added scenarios**, **2,352 related tests in 47.14 s**
and **84 CPU/TN tests in 6.07 s** pass. Strict mypy covers all seven changed
Python files. Black/Ruff/scope/architecture/internal IR/protected API and the
executable example pass. New loop plus example cover **57 statements and 12
branches at 100%**, not all existing pass code. Implementation `0f1dbb2b`.
Existing source warnings and the prior default-suite timeout remain open; these
focused results are not whole-repository certification.
