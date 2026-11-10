# Conjugate index trace (GRAPH63)

`simulation/tensor_network/trace_indices.py` adapts unchanged Qualtran v0.7.0
`cbloq_to_superquimb` handling of DiscardInd, pinned at
bcb17bb25360ed68ce3bc084645db3e929799389. It merges numerical labels and reuses
the existing contraction engine. It does not add a second partial-trace executor.

`trace_conjugate_indices(nodes, labels)` consumes existing doubled nodes using
GRAPH62 even forward/odd backward labels. Requests name original undoubled
indices. Each backward edge is renamed to its forward partner, so contraction
sums over a shared index. This is the native integer representation of source
`reindex({(*dind, True): dind, (*dind, False): dind})`. The shared even label is
already unique to that original index, so no third namespace or sentinel class
is required.

The caller identifies exposed edges and omits them from retained output axes.
Each traced direction must occur exactly once with matching dimension. Invalid
labels, duplicate requests, missing/internal/already-traced edges, rank mismatch
and incompatible dimensions raise ValueError before replacements. These native
checks are stricter than blind source reindexing and are documented as such.
Empty requests retain the nodes; an empty graph permits no nonempty trace.
Tensor data, axes, storage, dtype/device, metadata and gradients remain unchanged.
No normalization, zero projection, reset, measurement or physical ownership is
inferred. The helper adds no gates, models, graph types or public API.

## Source and execution evidence

Sixteen tests cover all eight three-bit subsets in complex64 and complex128,
using sparse labels and mixed inputs built from two arbitrary complex joint
states. Every result matches original make_forward/backward tensors followed by
the exact source reindex operation and contraction. Independent environment-axis
summation also agrees, including scalar trace when all bits are discarded.

Three GHZ tests compare each one-bit trace to the existing circuit tensor-network
`tensor_network_discard_qubit` entrypoint. Purity is 1/2 and rank is two, proving
the implementation preserves mixed output rather than projecting or resetting.
Complete MeasureZ and MeasureZ-followed-by-classical-Discard channels match
unchanged source superquimb contractions. The original pure-state path rejects
DiscardInd. A quantum environment initialized to zero, entangled by CNOT and
consumed by DiscardQ reproduces the source dephasing channel; source classical
simulation continues to reject quantum discard.

A single joint tensor supplier and a factorized supplier produce the same
reduced density without consumer changes. A surviving complex phase coherence
retains its analytical gradient. Input nodes and tensor metadata remain intact.
Negative cases cover invalid trace requests and unsupported topology.

For measured two-Add execution, all sixteen actual branch vectors are doubled,
the two clean-work indices are traced with this helper, and the retained full
logical density is compared at branch weight 1/16. The branch sum restores trace
one and all coherences. Work is separately verified zero before trace, since
partial trace alone does not prove cleanup. The established source primitive,
operand and feedback test, all 256 inverse-history pairs and eight append
trajectories are also reverified.

The native helper does not parse foreign DiscardInd objects or discover discard
operations in SSA. Measurement tensors in these scenarios are explicitly
supplied, using the unchanged source local tensor as a reference. A full
superoperator graph/supplier runner and automatic classical-port handling remain
open. Numerical index trace is separate from destructive Program ownership,
which retains its existing owner.

## Resource and user workflow

Index merging creates no tensor data or physical instruction. Measurement
channel models and actual gate streams are reported separately. Measured Add
retains sixteen T/Tdag, four measurements, two work bits and source peak eight;
numerical tracing does not consume additional gates or reset a work bit.
Discarding an entangled environment is a trace operation, not the source
ZeroEffect projection. Full primitive and source-model categories are recorded
in the local resource report.

Existing users continue to append operations, for example:

```bash
python -m examples.algorithms.binary_field_arithmetic
python -m pytest tests/unit/test_tensor_trace_indices.py tests/unit/test_tensor_network_discard.py -q
```

The example prints GF(8) multiplication, inverse work cleanup and sixteen erasure
trajectories. This internal helper changes neither the circuit API nor its
execution backend. Dense density/channel tests are correctness inspection and
do not establish scalable simulation capacity.

## Final local verification

Implementation commit: `23357944`. All 25 new tests passed. Related regression:
979 passed in 19.53 seconds, including the four existing tensor-network discard
scenarios; focused coverage: 280 passed in 24.81 seconds; integration: 64 passed
in 5.87 seconds. The new owner has 28 statements and eighteen branches at 100%
coverage. Ten selected owners total 306 statements and 138 branches at 99%
combined coverage; GRAPH60 retains its two uncovered profile-refusal branches.

Strict typing, Black and Ruff passed for twenty selected files. Simulation scope,
architecture, Program contract and protected API checks passed. Fourteen original
source blobs match the locked inventory. Eleven unchanged upstream tests passed
in 0.73 seconds with two original notebook-mark warnings. Qualtran 0.7.0 and
Kickmix 0.1 with Simulator were checked; the binary-field example passed.
Related regression retains five existing warnings, including three from the
original Add tensor path.

MeasureZ and MeasureZ+Discard source models each report one measurement, zero
T-equivalents and peak one. The entangled environment + DiscardQ composite has
source peak two, but its QEC query raises DecomposeNotImplementedError because
DiscardQ does not declare a decomposition. This is recorded as a model failure,
not a fabricated zero cost. Index merging itself adds no physical instruction.

The default smoke/unit run reports 21,507 passed, 300 skipped and seven failed
in 135.32 seconds. All seven failure IDs exactly match the independent existing
baseline; no new failure ID appeared. Maxfail stops the run early, so this is
neither complete nor green. Exact comparison, resource categories, source hashes
and logs are in the local reference workspace under
`ftqc-reference-env/g2-trace-indices-*`.
