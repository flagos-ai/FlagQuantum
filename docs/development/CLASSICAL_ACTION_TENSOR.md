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
metadata validation precedes allocation. The dense entry point adds operation
context and cause to ValueError; the leaf entry point preserves the raw error
(as corrected in GRAPH70). Other errors propagate. Original zip truncation, output truthiness
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

## Component leaf attachment (GRAPH67)

`classical_action_tensor_node` implements the numerical attachment responsibility
of source `my_tensors_from_classical_action`: enumerate the explicit action,
convert to the caller's dtype/device prototype, and attach output/input binary
axes through the existing binder. It returns the existing `TensorNetworkNode`.
The graph consumer still receives its original caller-owned leaf callback; no
FunctionRecord-name dispatcher, registry, graph type or automatic substitution
is introduced. Callers pass `Signature(function.registers)` and explicitly
select known phase-free actions. Action enumeration and codec callbacks remain
independent of node construction and component traversal.

Bindings are checked before enumeration. Classical ports are refused before
callbacks because the original leaf supplier counts quantum axes but its dense
table counts all bits. This finite native refusal is explicit; it is not a
claim of matching unsupported-source failure timing. The graph adapter's
existing classical-port rejection also remains. Scalar-only nodes use explicit
scalar data. Caller dtype/device selection is an explicit conversion; the
NumPy action is not trainable, while contraction with trainable inputs retains
PyTorch autograd. No physical instruction, reset, workspace allocation or
resource estimate is emitted by the supplier.

Nineteen new tests cover three nested executable Add component depths and four
floating/complex dtypes. The native shared `register_add_classical_values`
callback replaces reference-backed arithmetic within the supplier. Every leaf's
binary data and integer operands match original `my_tensors_from_classical_action`;
complete native matrices match the unchanged original quantum decomposition.
Replacing the supplier by independently supplied original leaf data leaves the
consumer unchanged. Directional ket/effect/THRU, scalar, pre-callback connection
and classical-port refusals, and callback errors are explicit tests. Complex
input phase gradients, all 16 measured Add branch amplitudes and both clean
work bits are checked before the retained 256 inverse-history and eight append
trajectory comparisons.

GRAPH67 checks: 1,148 related tests passed in 14.93 s (41 upstream/existing
warnings), 449 focused coverage cases passed in 17.74 s (39 warnings), and
64 integration cases passed in 3.25 s. The attachment adds nine statements and
two branches; the whole owner now has 45 statements and ten branches, all
covered. Thirteen related owners total 389 statements/166 branches at 99%,
retaining the two historical GRAPH60 refusals. Strict mypy passed across
26 related files, plus the two source-subclass files with untyped imports
followed; Black/Ruff, scope, architecture, Program IR and public API checks
passed. Eighteen source blobs match the locked inventory; 22 unchanged source
tests passed in 0.61 s (five existing warnings). The locked native Kickmix
Simulator and binary-field example were exercised.

The double-Add example has two action-table leaves (16,384 bytes each after
complex128 conversion) and one direct-through spectator identity node. Supplier
attachment emits zero physical instructions and allocates zero physical work
bits. The existing circuit still emits H12/T8/CX40/Tdag8/S4/measurement4/CZ4/reset4;
its separate source estimate remains AND4/Clifford22/measurement4/T-equivalent16,
peak8. Raw logs and resources are under
`ftqc-reference-env/g2-classical-tensor-supplier-*`. The warning increase is
from twelve comparisons against unchanged source Add decompositions, not
suppressed new native warnings. Default-suite results and publication status
are recorded separately.

GRAPH67 default smoke/unit evidence: 21,507 passed, 300 skipped, seven failures
in 117.59 s. Every failure ID matches the independent seven-failure baseline;
none is new. Maxfail stopped the run, so the suite is not certified as fully
passing. Publication remains blocked: HTTPS push timed out and the official
SSH-over-443 endpoint rejected available public-key authentication. No host-key
verification was disabled. GRAPH65 through GRAPH67 remain local; the verified
remote PR head is `1719c3a2`. Resume with a normal push after checking the remote
when Git connectivity/authentication is restored.


## Directional array conformance (GRAPH68)

The existing action table, register binder and component graph already support
LEFT input and RIGHT output arrays. No additional production abstraction is
needed. Ten conformance cases exercise scalar, `(2,)`, `(1, 2)` and `(2, 1)`
shapes with float64 and complex128. Bitwise NOT and reversed row-major output
words match the unchanged source classical leaf data and integer operands,
then its independently decomposed X-gate dense matrix. Replacing caller codecs
with shared value-array codecs leaves the consumer unchanged. An independent
native executable X-leaf graph contracts to the same matrix.

`circuit.append(op)` preserves the full complex state and the clean spectator.
Directional reversal is represented by the existing bound output words; it
requires no physical SWAP. The action-table input phase derivative is checked
separately. Empty `(0,)` and `(2, 0)` arrays retain the source codec ValueError;
GRAPH68 added operation context, corrected to raw propagation in GRAPH70.
This is finite phase-free array
conformance, not automatic support for arbitrary quantum actions.

Measured resources: one numerical leaf, binary rank two / 64 complex128 bytes
for the scalar and rank four / 256 bytes for each two-element array. Supplier
attachment adds zero instructions and work bits. Executable circuits emit one
or two X instructions, zero added SWAPs, and retain one clean spectator. Separate
source estimates are one or two Clifford gates, zero T-equivalent cost and peak
one or two bits; the caller's spectator is outside that source estimate.

Checks: 1,158 related cases passed in 15.04 s (41 existing/source warnings),
459 focused coverage cases in 17.45 s (39 warnings), and 64 integration cases in
3.20 s. The existing owner remains 45 statements / ten branches at 100%; thirteen
owners total 389 statements / 166 branches at 99%, with the two historical
GRAPH60 refusals unchanged. Strict typing passed for 26 related files and three
source-following files; formatting, scope, architecture, Program IR and public
API checks passed. Eighteen original blobs match the inventory; 22 unchanged
original tests passed in 0.84 s (five existing warnings). Qualtran 0.7.0,
Kickmix 0.1 Simulator and the binary-field example were exercised. Evidence:
`ftqc-reference-env/g2-classical-tensor-arrays-*`.

Publication recovery: the historical GRAPH65–67 Git transport blocker above
was resolved by a normal push, verified at `8b711696` before GRAPH68 publication.

GRAPH68 default smoke/unit: 6 failed, 21485 passed, 300 skipped, 1037 warnings in 143.00s (0:02:22). All 6 observed failure IDs
match the independent seven-failure baseline; 1 baseline IDs were not
reached before maxfail. This is not a fully passing default suite.

## Explicit output discard after a classical leaf (GRAPH69)

A phase-free CNOT action leaf composes with the existing conjugate-pair,
explicit trace and register-superoperator consumers. No production change is
required. Eight cases cover every subset of its two outputs in complex64 and
complex128 against the unchanged original CNOT/DiscardQ composite channel.
The full channel, including all off-diagonal matrix units, is compared exactly;
trace preservation is checked independently. Mixed inputs retain complex
coherences. `circuit.append(op)` executes the original single CX, preserves full
complex amplitudes, and leaves the spectator zero before the caller traces
outputs. A separate input-angle derivative checks retained coherence.

An empty classical-action output is a coherent sum effect, not quantum discard.
For a single input bit its doubled table is an all-ones effect; on a plus-state
density it returns two, whereas the identity trace effect returns one. The
original DiscardQ refuses classical simulation and action-table enumeration.
Keep that refusal: never replace quantum discard with a callback returning no
values, and never normalize a bad map to hide the distinction. Existing explicit
trace composition is sufficient; automatic graph discard discovery and physical
reset remain outside this numerical boundary.

Resources are separate: the complex128 action leaf occupies 256 bytes; two
conjugate nodes share forward storage and a conjugate view. Dense channel
outputs occupy 4096 bytes (keep both), 1024 bytes (keep one), or 256 bytes (keep
none). Numerical doubling/trace adds zero physical instructions and work bits.
The actual append circuit has one CX and one caller-owned clean spectator.
Source peak is two bits for every subset; its no-discard QEC model has one
Clifford and zero T-equivalent cost. With DiscardQ the unchanged source QEC
model raises DecomposeNotImplementedError; that cost is unavailable, not zero.

Ten new cases passed in 1.31 s. Related regression: 1168 passed in 21.12 s
(41 existing/source warnings); focused coverage: 469 passed in 24.23 s
(39 warnings); integration: 64 passed in 4.37 s. Strict typing passed across
27 related files plus three source-following files. Black/Ruff, scope,
architecture, Program IR and public API checks passed. The 13-owner coverage
remains 389 statements / 166 branches at 99%, retaining two historical GRAPH60
refusals. Twenty source blobs match the locked inventory; original tests and
default-suite results are recorded below. Qualtran 0.7.0 / Kickmix 0.1 Simulator
and the binary-field example were exercised. Raw evidence is under
`ftqc-reference-env/g2-classical-tensor-discard-*`.

GRAPH69 original-test evidence: 22 unchanged tests passed in 0.73 s (five
existing warnings). An additional original cnot_test.py run could not collect
because PennyLane is absent; it is not counted as passing. No reference source
was edited. Default smoke/unit: 7 failed, 21507 passed, 300 skipped, 1037 warnings in 124.02s (0:02:04); all observed failure IDs
match the independent baseline. Maxfail stopped the run; it is not fully green.


## Source-function and caller audit (GRAPH70)

The unchanged source file has four functions. An AST inventory and a complete
source-tree reference scan are recorded in
`ftqc-reference-env/g2-classical-tensor-audit-callers.json`. Outside the file,
only `simulation/tensor/__init__.py` re-exports the helpers and the original
`_tensor_from_classical_test.py` calls them. No production source consumer of
these helpers was found at the pinned commit.

| Source function | FlagQuantum responsibility | Result |
| --- | --- | --- |
| `_bits_to_classical_reg_data` | Caller codec plus scalar/array layout in `_classical_action_tensor_data` | Quantum scalar/array ordering and original codec failures verified; no second codec dispatcher |
| `_bloq_to_dense_via_classical_action` | `_classical_action_tensor_data` | Shared binary-axis enumeration, raw errors, original 40-bit limit |
| `bloq_to_dense_via_classical_action` | `classical_action_tensor` | Scalar/vector/matrix grouping and contextual ValueError preserved |
| `my_tensors_from_classical_action` | `classical_action_tensor_node` plus existing binder | Caller-owned integer axes and dtype/device; raw errors corrected in GRAPH70 |

The audit found a real error-path mismatch: GRAPH67–69 routed the leaf through
the contextual dense wrapper, whereas the original leaf calls the raw enumerator.
The loop is now shared by both entry points at its original binary rank. Only
the dense entry point wraps ValueError and groups dimensions; the node binds raw
binary data directly. Four exception types check raw error object identity,
dense message and cause against the unchanged original. A fifth case proves
both size refusals happen before allocation/action, with different raw/contextual
messages. The two retained empty-array tests now require raw source messages.
Operation names still label successful native nodes.

This fixes the verified mismatch without adding a contract, dispatcher or public
export. Parent-file acceptance remains open: native finite-metadata and
connection checks happen earlier, and classical ports are rejected before
callbacks because source leaf axes count quantum bits while dense data counts
all bits. Exact failure timing for these unsupported paths is not established.
Caller codecs own value interpretation; arbitrary phaseful actions remain
ineligible. The original six classical-tensor tests run unchanged as part of
the 22-test reference group; previous scalar/array, output tuple, complete
matrix, measured-Add branch, clean-work and append checks are retained.

GRAPH70: 1173 related tests passed in 19.12 s (41 existing/source warnings),
474 focused coverage cases in 24.83 s (39 warnings), and 64 integration cases
in 4.70 s. The owner has 50 statements / ten branches at 100%; thirteen related
owners total 394 statements / 166 branches at 99%, retaining two historical
GRAPH60 refusals. Strict typing passed for 28 related files plus three
source-following files; Black/Ruff, scope, architecture, Program IR and public
API checks passed. Twenty source blobs match the inventory; 22 unchanged
original tests passed in 0.77 s (five existing warnings). Qualtran 0.7.0,
Kickmix 0.1 Simulator and the binary-field example were exercised.

Resources were rechecked independently: the leaf/storage/channel sizes and
single-CX append counts from GRAPH69 are unchanged. Shared enumeration emits
zero physical instructions and work bits. Source QEC with DiscardQ remains
unavailable, not zero; the source peak remains two bits. Evidence is under
`ftqc-reference-env/g2-classical-tensor-audit-*`.

GRAPH70 default smoke/unit: 5 failed, 21400 passed, 300 skipped, 1037 warnings in 122.28s (0:02:02). All 5 observed
failure IDs belong to the independent seven-failure baseline; 2 were not
reached before maxfail. The default suite is not certified as fully passing.

## Selected action leaves and fully expanded gate leaves (GRAPH71)

The `_flattening.py` audit confirms the existing Compiler selection policy:
full expansion ignores the custom-tensor classifier; partial expansion stops at
explicitly declared custom leaves. The original Python method-qualname heuristic
belongs to its object model. Native callers explicitly classify existing typed
callees; neither Compiler nor Simulation discovers providers by method names.
The existing implementation needs no production change for this verified path.

Fourteen new tests connect the selected graph to the actual
`component_tensor_nodes` consumer. Across three nesting depths, two expansion
modes and float64/complex128, retained source instance labels and leaf order
match unchanged `flatten_for_tensor_contraction`. Partial expansion consumes
native classical identity-action leaves; full expansion consumes decomposed H
gate leaves. Both complete contractions match the original dense path. Replacing
only the supplier by independent dense leaf data leaves the selected graph,
consumer and bindings unchanged. Missing selected suppliers propagate their
original error object on the first requested leaf without mutating the graph.

Complete complex input amplitudes, phase derivatives and an entangled spectator
are verified. `circuit.append(op)` emits the identical original H decomposition
in both selection modes and preserves the clean work wire. The numerical custom
identity never changes physical lowering or removes gates. At nesting depths
one/two/three, actual gate counts are H4/H8/H16, equal to the separate source
Clifford estimates; source peak is one bit and T-equivalent cost zero. The caller
uses three physical bits, including its spectator and clean wire. Numerical
supplier attachment adds zero gates/work. Partial numerical leaf counts are
2/4/8; full counts are 4/8/16. Each 2x2 leaf occupies 32 bytes in float64 or
64 bytes in complex128. These storage counts are distinct from circuit costs.

Verification: 1187 related cases passed in 25.29 s (41 existing/source warnings),
488 focused cases in 26.65 s (39 warnings), and 64 integration cases in 4.58 s.
Strict typing passed across 29 related files and three source-following files;
Black/Ruff, scope, architecture, Program IR and public API checks passed.
Thirteen covered owners remain 394 statements / 166 branches at 99%, with the
two historical GRAPH60 refusals retained. Twenty source blobs match the locked
inventory. Twenty-three unchanged original tests, including the source custom
classifier test, passed in 1.28 s (five existing warnings). The Qualtran 0.7.0,
Kickmix 0.1 Simulator and binary-field example checks passed. Evidence is under
`ftqc-reference-env/g2-tensor-leaf-selection-*`.

This is finite selection-to-supplier composition acceptance. Arbitrary atomic
root wrapping, automatic provider discovery, mixed classical/quantum graph
traversal and full superoperator graph supply remain open. The next numerical
slice should replace manually supplied measurement copy tensors through the
existing doubled-label and explicit-trace consumers, preserving these boundaries.

GRAPH71 default smoke/unit: 5 failed, 21400 passed, 300 skipped, 1037 warnings in 129.86s (0:02:09). All 5 observed
failure IDs belong to the independent seven-failure baseline; 2 were not
reached before maxfail. The default suite is not certified as fully passing.

## Native default codecs

The dense entry now uses [register codec composition](CLASSICAL_REGISTER_CODEC.md)
when decode/encode callbacks are omitted. Explicit callbacks continue to override
the defaults. This connects concrete native metadata to existing codecs; it
does not discover actions or change the original phase-free, classical-axis
or empty-array boundaries. The node entry keeps its explicit codec arguments.
