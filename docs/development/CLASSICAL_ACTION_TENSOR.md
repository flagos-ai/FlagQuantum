# Dense tensors from classical register actions

Simulation owns `tensor_network/classical_action_tensor.py`. Its enumeration,
output/input layout, 40-total-bit refusal and ValueError wrapping follow the
unchanged Qualtran v0.7.0 `_tensor_from_classical.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

The interface consumes Core `Signature`, the existing `ClassicalValueCallback`,
and two caller-owned codec callbacks. There is no new dtype dispatcher,
contract, registry, backend or graph representation. A decoder receives the
register and its input bits, flattened for scalar ports and reshaped to
`shape + (num_qubits,)` for arrays. An encoder receives the register and
`np.asarray(output_value)`. Existing bit/integer codecs remain their owners;
this module only enumerates and connects their values. The action receives
named input values and returns output values in right-register order.

Enumeration uses signature order, row-major elements and MSB-first bits.
Output bit axes precede input bit axes. The dense result groups into a scalar,
vector or matrix, omitting zero-bit sides. Attach it with the existing
`bind_register_tensor` to replace a leaf supplier without changing contraction
consumers. `circuit.append(op)` and physical lowering remain unchanged; the
existing binary-field arithmetic example and measured Add checks demonstrate
the execution workflow supported by these diagnostics.

The result is a phase-free zero/one table, not proof of unitarity. Non-injective
actions are retained without normalization. The source helper reconstructs Z's
classical value identity rather than its complex quantum phase; the native
helper preserves this distinction. Hadamard's unavailable classical action
remains a failure. There is no inferred phase callback or fallback to this
helper for arbitrary quantum operations. Codec callbacks are not a PyTorch
trainable function: autograd is retained only after explicit conversion and
binding, for tensors applied to a trainable input.

Storage is `8 * 2**(input_bits + output_bits)` bytes before conversion. The
source ceiling of 40 total bits is preserved; it is not a practical allocation
guarantee. Callers must choose small finite inspection problems. Native finite
metadata validation precedes allocation. ValueError retains operation context
and cause; other errors propagate. Original zip truncation, output truthiness
and NumPy indexing are preserved rather than hardened into a different action.

The array reshape deliberately uses quantum bits as the source does. This
means CBit arrays can fail at reshape, while scalar classical bits can work.
The original multidimensional integer codec/validation failures, zero-width
scalar decode failures and empty-array codec errors remain recorded boundaries.
No reference source is modified or silently repaired. Codec ownership and
finite metadata validation are explicit native boundaries, not identical
error-timing claims for every unsupported source dtype.

Run focused tests with the pinned reference environment and torch preimported:

```python
import torch
import pytest

torch.set_num_threads(1)
raise SystemExit(pytest.main([
    "-q", "tests/unit/test_classical_action_tensor.py",
    "tests/unit/test_register_tensor_binding.py",
]))
```

The tests compare X, controlled swap, Toffoli, Add and XOR to both the original
classical-action helper and existing quantum tensors. They compare the actual
leaf binary axes and operands, complete complex input amplitudes and input
phase gradients, scalar/array signed/unsigned inputs, callback order, output
tuple boundaries, source failures and size refusal before allocation. The
measured double-Add consumer checks 16 branch amplitudes and two clean work
bits, then reuses the full source primitive/feedback, 256 inverse-history pairs
and eight append trajectories. Primitive gate counts, numerical tensor sizes
and source QEC estimates are reported separately.

## GRAPH66 local verification

Thirty new tests passed. Concentrated regression: 1,129 passed in 21.26 s
(five existing warnings); coverage: 430 passed in 22.35 s (three existing
warnings); integration: 64 passed in 4.68 s. The new owner has 36 statements and
eight branches, all covered. Thirteen related owners total 380 statements and
164 branches at 99%, retaining the two historical GRAPH60 refusal paths.
Strict mypy passed on 25 related files and separately on both new files with
untyped imports followed for the source Bloq test subclasses. Black, Ruff,
Simulation scope, architecture, Program IR and protected API checks passed.
Eighteen reference blobs match their inventory Git objects. Twenty-two unchanged
source tests passed in 1.67 s (five existing warnings). Qualtran 0.7.0 and
Kickmix 0.1 Simulator were present; the binary-field append example passed.

The table emits no physical instructions. X uses 32 bytes, controlled swap and
Toffoli 512 bytes each, and two-bit Add and XOR 2,048 bytes each (float64).
Separate source models retain their original Clifford/CSwap/Toffoli/AND,
measurement, T-equivalent and peak categories. The measured double-Add path
emits H=12, T=8, CX=40, Tdag=8, S=4, measurement=4, CZ=4, reset=4;
its source model is AND=4, Clifford=22, measurement=4, T-equivalent=16, peak=8,
with two clean work bits. These counts are not inferred from dense entries.
DiscardQ's unrelated existing QEC-model exception remains explicit.

Raw local evidence is under `ftqc-reference-env/g2-classical-action-tensor-*`
(source inventory, regression, coverage JSON, integration, original tests,
environment, example, resources and quality checks). Default-suite baseline
comparison and publication state are recorded separately in the integration
ledger. This finite slice does not certify a complete upstream source file.

The default smoke/unit run recorded 21,507 passed, 300 skipped and seven failures
in 127.10 s. All failure IDs match the independently recorded seven-failure
baseline; no new failure IDs were observed. Maxfail interrupted the run, so this
is not a complete passing suite. See the explicit default comparison JSON.

Publication remains blocked by Git HTTPS timeouts: the implementation is locally
committed, while PR #1084 still points to `1719c3a2`. This also leaves the prior
GRAPH65 commits awaiting push. GitHub API remains available; no code publication
or merge is claimed. Resume by checking the remote SHA before a normal push;
do not reimplement or force-push these batches.
