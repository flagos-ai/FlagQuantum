# Forward and conjugate tensor networks (GRAPH62)

`simulation/tensor_network/conjugate_pairs.py` adapts unchanged Qualtran v0.7.0
`make_forward_tensor` and `make_backward_tensor`, pinned at
bcb17bb25360ed68ce3bc084645db3e929799389. It doubles existing numerical nodes;
it adds no quantum operation, physical workspace or execution backend.

## Representation and responsibility

`tensor_conjugate_pairs(nodes)` returns a forward node then a conjugate node
for each input node, matching the source superquimb insertion loop. Both use
existing TensorNetworkNode objects. `conjugate_pair_labels(labels, backward=...)`
maps an original integer label L to 2L for forward and 2L+1 for backward.
This is an injective local representation of source `(connection, bit, direction)`
indices. Repeated labels remain repeated and sparse labels remain collision-free.
The output labels of the consumer must use this same mapping.

Backward data is elementwise `tensor.conj()` with unchanged axes. Quimb Tensor.H
in this protocol does not transpose an operator's output and input axes. Using
PyTorch `.mH` would implement a different network. Forward nodes share original
data storage and the backward conjugate view preserves autograd. Dtype/device,
name and metadata are retained; no metadata is mutated or foreign tag object
is introduced. Source `dag` is a display tag and is not a new native metadata
schema. Invalid integer labels and mismatched rank/label counts fail before
returning a network. Empty input remains empty, without scalar identity.

The caller supplies already selected and wired nodes. Compiler graph selection,
SSA numerical wiring, register binding and contraction retain their existing
owners. This helper imports no Compiler module or external framework. Existing
contraction accepts mapped output axes; tests explicitly request output-forward,
output-backward, input-forward, input-backward order to compare source channels.
No production superoperator grouping facade or alternate contraction engine is
added in this slice.

Discard/trace, Kraus sums, measurement-channel assembly, friendly indices and
full superquimb graph conversion remain open. Doubling a branch vector produces
its unnormalized branch density, not a normalized state or a complete channel.
The conjugate network is not inverse circuit execution.

## Source and execution checks

Sixteen scalar/vector/matrix/asymmetric-rank-three cases cover four real/complex
PyTorch dtypes against actual original forward/backward functions. Every data
entry, unchanged axis position and direction label is compared. Non-Hermitian
complex matrices distinguish conjugation from conjugate transposition. Input
immutability, retained metadata/storage, repeated sparse labels and invalid
inputs are checked.

Nine phase-array/unused-register graphs compare full original superquimb channel
entries and node counts before source dense contraction mutates/simplifies the
reference network. Mixed density inputs include arbitrary complex joint states.
Their channel action equals the mixture of two existing `circuit.append(op)`
executions, including all unused-register coherences and clean work zero.
Each stream has three H and two S gates; doubling tensor nodes adds no gates.

Two initialized phase graphs compare original fully decomposed superquimb output
density matrices. Replacing each supplier node by a globally phased node leaves
the density consumer unchanged. Complex autograd checks an off-diagonal imaginary
coherence and its analytical derivative, detecting missing conjugation or axis
swapping.

The measured two-Add stream is checked across all sixteen measurement histories.
Each unnormalized branch density has trace 1/16 and preserves complete complex
amplitudes, including work axes. Summing the branch densities restores the full
logical output density with trace one. The original primitive/feedback comparison,
all 256 forward/inverse history pairs and eight existing append trajectories
are also reverified. This test supplies actual branch vectors; production does
not claim to build a measurement channel or trace out a register.

## Resource and user workflow

Numerical node duplication, actual primitive counts and source QEC model
categories are reported separately. Existing phase circuits remain three H/two S;
measured Add remains sixteen T/Tdag, four measurements and two clean work bits,
with source peak eight. A numerical backward copy is not an adjoint gate stream
and never doubles those resource counts. Dense channel inspection is exponential
and does not establish scalable simulation capacity.

The existing user workflow remains ordinary appended operations. Run:

```bash
python -m examples.algorithms.binary_field_arithmetic
python -m pytest tests/unit/test_tensor_conjugate_pairs.py -q
```

The example prints GF(8) multiplication and confirms inverse work cleanup and
sixteen erasure trajectories. This internal tensor helper changes no user API.

## Final local verification

Implementation commit: `c188e662`. All 34 new tests passed. Related regression:
950 passed in 17.20 seconds; focused coverage: 255 passed in 20.86 seconds;
integration: 64 passed in 7.86 seconds. The new owner has fifteen statements and
eight branches at 100% coverage. Nine selected owners total 278 statements and
120 branches at 99% combined coverage; the two existing GRAPH60 profile-refusal
branches remain uncovered.

Strict typing, Black and Ruff passed for eighteen selected files. Simulation
scope, architecture, Program contract and protected API checks passed. Twelve
original source blobs match the locked inventory. Seven unchanged upstream
quimb/zero/int-state tests passed in 0.90 seconds with two original notebook-mark
warnings. Qualtran 0.7.0 and Kickmix 0.1 with Simulator were verified; the existing
binary-field example passed. Related tests retain five existing warnings,
including three from the original Add tensor path.

The default smoke/unit run reports 21,507 passed, 300 skipped and seven failed
in 179.52 seconds. All seven failure IDs exactly match the independent existing
baseline. No new failure ID was observed; maxfail stops the run early, so this
is neither complete nor green. Exact comparison, resource categories, source
hashes and logs are in the local reference workspace under
`ftqc-reference-env/g2-conjugate-pairs-*`.

This accepts forward/conjugate node duplication only. Discard/trace and automatic
measurement-channel construction remain open, as does full source-file acceptance.
