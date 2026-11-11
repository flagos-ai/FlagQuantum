# Selected component graphs to tensor nodes (GRAPH60)

`simulation/tensor_network/component_tensor_nodes.py` adapts the per-instance
incoming/outgoing connection loop of unchanged Qualtran v0.7.0
`cbloq_to_quimb`, pinned at bcb17bb25360ed68ce3bc084645db3e929799389.
It consumes the existing Core Program SSA and returns existing TensorNetworkNode
objects with external label maps. No foreign graph objects enter production.

## Ownership and interface

`component_tensor_nodes(record, order, supply, like, entry="main")` reads one
verified typed logical return block. Compiler performs expansion first using
`expand_components_for_tensor` and supplies visitation positions from the existing
`iter_component_calls`. Simulation does not import Compiler, traverse child
implementations, select a backend, or interpret foreign tensor methods.

The supplier receives the existing FunctionRecord and two port maps. Each map
contains row-major word elements, each with MSB-first integer bit labels. It
returns an iterable of existing numerical nodes. It can reuse
`bind_register_tensor`, emit multiple tensors, or raise when unsupported. The
supplier owns tensor contents, pure-state projections and their dtype/device;
`like` controls only the automatically inserted direct-through identity nodes.
No callback registry or new contract class is introduced.

SSA producers and consumers share local integer labels. Every external input
and output has distinct labels. A returned untouched input is connected by the
existing per-bit identity helper, including individual untouched array elements
and differently named directional input/output ports. Operated returned values
are wired directly to output labels. Labels express numerical connectivity,
never ownership of physical workspace or a gate operation.

The return value is `(nodes, incoming, outgoing)`, directly consumable by
`contract_register_tensors` with the entry signature. No dense operator is
materialized during wiring. The caller explicitly requests dense contraction
when needed; its memory grows exponentially. Empty graphs remain empty and
retain the existing contraction error unless a supplier emits scalar nodes.

Invalid Program records, unknown entries, unsupported profiles, incomplete or
non-topological orders, and classical ports fail before suppliers run. Only
unconditional typed calls are admitted; primitive, feedback and control-flow
entries need their own numerical admission. Supplier exceptions propagate.

## Source and execution verification

Twelve width/shape layouts compare original `cbloq_to_quimb` tensors, each active
node's data, canonical per-axis internal/external connectivity, identity count,
and the complete dense operator. They include zero-width words, empty arrays,
multiple active array elements and unused ports. Three emitted H gates match the
source Clifford model in each layout. Arbitrary complex joint input states go
through the existing `circuit.append(op)` workflow and match the contracted
operator, with a separate clean work bit remaining zero.

A two-input CNOT graph tests fan-in after independent H calls and a second valid
ordering of the independent calls. The emitted stream is exactly two H and one
CX. Nested custom leaves test both full and selective expansion; a supplier
returning a scaled tensor plus a compensating scalar replaces a single-tensor
supplier without changing the consumer. Autograd traverses supplied weights and
automatically added identity nodes.

Directional ket/bra graphs cover widths zero through two and scalar contraction.
For positive widths their zero-projection agrees with original Allocate/Free.
The original zero-bit source emits no tensors and its empty contraction fails;
native scalar one is explicitly supplied by two scalar nodes. This difference
is recorded, not claimed as an upstream zero-bit contraction match.

A retained Add graph is wired into its native logical permutation tensor and
compared to the complete original composite operator. Its same selected graph's
existing measured append consumer is reverified for exact source primitives,
operands and feedback, all sixteen forward branches, 256 inverse-history pairs,
eight trajectory executions and clean workspace. Full tensor expansion of Add
would require additional primitive suppliers; it is not claimed here.

## Resource accounting and user workflow

Numerical tensor-node counts are separate from emitted primitive counts and
source QEC model categories. Wiring adds no quantum gate, measurement, reset or
physical work. The H layouts use three primitives and three source Clifford
units; the measured two-Add stream uses sixteen T/Tdag, four measurements and
two clean work bits. Source estimates and physical-width evidence are retained
separately in the local resource report.

Users continue to append existing operations. The existing short runnable
workflow supported by this internal work is:

```bash
python -m examples.algorithms.binary_field_arithmetic
```

It prints `GF(8): 3 * 5 = 4`, confirms inverse workspace cleanup, and verifies
all sixteen product-erasure trajectories. No user-facing tensor graph API is
required to execute these circuits.

Run the focused wiring scenarios in the locked reference environment:

```bash
python -m pytest tests/unit/test_component_tensor_nodes.py -q
```

At GRAPH60, superoperator/discard semantics, friendly index names and root
wrapping remained open. Later finite slices are documented below and in the
linked helpers; automatic leaf discovery and full parent-file acceptance remain
open.

## Final local verification

Implementation commit: `f507f3ff`.

- 26 new tests; 883 related tests passed in 22.98 seconds.
- 188 focused tests passed in 20.53 seconds. Six selected owners have 206
  statements and 88 branches, with 99% combined coverage. The new wiring owner
  has 71 statements and 34 branches, with two uncovered statements and two
  uncovered branches (96% combined). The two untested profile-refusal branches
  are not claimed covered.
- 64 integration tests passed in 9.37 seconds.
- Strict typing, Black and Ruff passed for twelve selected files. Simulation
  scope, architecture, Program contract and protected API checks passed.
- Ten original source blobs match the locked inventory; two unchanged upstream
  quimb tests passed in 1.28 seconds. Qualtran 0.7.0, Kickmix 0.1 and its Simulator
  were checked, and the existing binary-field example passed.

The related suite retains three original Add tensor-path autoray matmul warnings
(divide by zero, overflow and invalid value), plus two existing warnings from
other tests. Complete-operator comparisons pass without suppressing warnings.
The default smoke/unit run reports 21,400 passed, 300 skipped and five failed
in 241.88 seconds. Those five failure IDs are all in the independent existing
seven-failure baseline; two baseline cases were not reached before maxfail.
No new failure ID was observed, but the default suite is not green or complete.

The exact comparison, source hashes, resource categories and execution logs live
under `ftqc-reference-env/g2-component-tensor-nodes-*` in the local reference
workspace. Full source-file acceptance remains open.

## Mixed component channels (GRAPH74)

`component_superoperator_nodes(record, order, supply, like, entry="main")`
uses the same private SSA wiring loop as the pure-state helper. Its supplier
receives a fourth argument, `fresh_label()`, sharing the monotonically allocated
numerical namespace with all external and future SSA edges. Labels obtained
from local port maxima are unsafe: they can collide with later calls. Request
hidden environment labels during supply; do not retain the allocator for later
use. These labels allocate no physical qubit or classical slot.

The supplier yields existing `TensorNetworkNode` objects and original integer
labels to discard. This adapts unchanged `cbloq_to_superquimb`'s tensor and
`DiscardInd` protocol. The helper doubles nodes using the existing conjugate
pair mapping, then joins only explicitly requested trace edges. The existing
trace validator requires one exposed leg in each direction and rejects missing,
internal, duplicate and invalid labels. At GRAPH74 all external-port discards were refused. GRAPH77 below admits
explicit trace-only input effects while retaining output-port refusal. Port maps returned by the helper remain
undoubled; pass them to `contract_register_superoperator` with the same entry
signature. Empty graphs still return no nodes and require explicit scalar
supply before contraction.

The pure-state interface and classical-port refusal are unchanged. Both paths
share validation, Compiler-provided call order, SSA connectivity and per-bit
identities for untouched returned ports. No second graph traversal, registry,
serialized contract, public export or foreign graph representation is added.
This is a `single_device_fast_path` numerical inspection slice; no distributed
capacity or accelerator claim is made.

The finite verification graph measures two quantum array elements in Z and X,
retains or explicitly discards their reversed classical outputs, and carries
untouched quantum and classical spectator ports. Four dtypes and scalar/empty
classical layouts compare every tensor's data and canonical per-axis wiring
with original `cbloq_to_superquimb`, then compare the full channel. A supplier
using original source leaf data replaces the native measurement leaves without
modifying the consumer. Repeated graph invocations have independent namespaces;
environment labels never collide with later call ports. A separate two-element
classical permutation checks direct-through array wiring.

All four joint measurement branches preserve full complex amplitudes with an
entangled spectator and a clean work bit. Relative-phase derivatives match
source tensors. Existing `circuit.append(op)` executes the same bound instructions
for both classical input values, retained/discarded outputs and all branches.
The X environment is rotated only for source-amplitude comparison, as documented
in [measurement copy](MEASUREMENT_COPY.md); this is not an emitted gate.

Resource categories remain separate. Numerical wiring/doubling/tracing emits
zero physical instructions or workspace. The bound example emits one H, two
measurements and one classical XOR for external CBit initialization; reset and
T counts are zero. The source QEC model reports two measurements, zero Clifford
and zero T-equivalents. Its X tensor has no matching physical decomposition,
and the initialization is a FlagQuantum adapter cost. The caller has four
physical bits: two consumed inputs, one spectator and one clean bit. Classical
ports occupy immutable classical slots rather than quantum wires. Tensor storage
and dense contraction storage must not be reported as physical resources or
peak construction memory.

Automatic leaf discovery, feedback/conditional call admission, classical parity
leaf supply, nested numerical expansion policy, and full `_quimb.py` parent-file acceptance remain open. Callers
must select/expand components using the existing Compiler boundary first.
The supplier is responsible for genuine classical/quantum semantics; merely
accepting a CBit axis does not infer a stochastic channel or dephase it.

Run the finite scenarios in the locked reference environment:

```bash
python -m pytest tests/unit/test_component_superoperator_nodes.py -q
python -m examples.algorithms.binary_field_arithmetic
```

GRAPH74 verification (`b70fff27`): 25 new cases, 1266 related cases in 21.10 s
(41 existing/source warnings), 541 focused cases in 26.58 s (39 warnings), and
64 integration cases in 5.04 s. The new channel owner has 20 statements and six
branches at 100%; fifteen related owners have 439 statements / 176 branches at
99%. The two historical GRAPH60 profile refusals remain uncovered. Strict typing
passed for 34 related files plus three source-following files; Black/Ruff,
Simulation scope, architecture, Program IR and public API checks passed.
Twenty-two source blobs match the locked inventory, and 24 unchanged original
tests passed in 0.91 s (five existing warnings). Qualtran 0.7.0 and native Kickmix
0.1 Simulator were checked; the existing binary-field append example passed.

Both retained/discarded scalar-idle scenarios return eight doubled nodes. Their
unique tensor storage occupies 96/192/192/384 bytes in
float32/float64/complex64/complex128; logical forward/backward element bytes are
twice that because conjugate views share storage. Dense retained channels occupy
262144/524288/524288/1048576 bytes; discarding both results reduces that to
16384/32768/32768/65536 bytes. These are returned-storage measurements, not peak
construction or contraction memory. The source QubitCount is four including its
classical port; the caller's four physical bits separately include a clean bit.
Equal numerical values do not make these resource categories interchangeable.

Default smoke/unit: seven failed, 21507 passed, 300 skipped and 1037 warnings in
133.07 s. All seven failure IDs exactly match the independently recorded baseline;
there is no new observed failure ID. The maxfail run is not a fully passing
default suite. Raw local logs, source hashes, resources and failure comparison
are under `ftqc-reference-env/g2-component-superoperator-*`.

## Direct input discard effects (GRAPH77)

An explicitly discarded external input with **no numerical tensor leg** now
contributes a binary identity effect between its forward and backward input
axes. The original undoubled input map and declared signature remain intact,
so `contract_register_superoperator` can return the full trace functional or a
partial-trace channel. This effect is delta(i,j), not an all-ones bra, a zero
projection, input normalization, dephasing-only map or a physical reset.

The owner remains `component_superoperator_nodes`; no traversal, registry,
contract, compiler opcode or public export is added. Existing internal edges
still use the unchanged conjugate-pair/trace helpers. The graph validates every
explicit discard label, including duplicates, before splitting internal and
input effects. Discarding declared output ports remains an error. An external
input already present in numerical nodes is also refused: it is not a trace-only
input and silently adding an effect could double-consume it. A graph with no
ports and no nodes still returns an empty graph rather than inventing scalar one.

All direct input effects share one four-element identity tensor with like's
dtype/device. They add one rank-two node per discarded bit, no physical gate,
classical slot, environment wire or work qubit. Node logical bytes therefore
grow linearly; unique effect tensor storage stays four elements. Dense channel
inspection remains exponential and is not a capacity or peak-memory guarantee.

### Unmodified source boundary

Qualtran v0.7.0 Discard and DiscardQ each supply only a `DiscardInd`. A direct
external discard has no producer tensor in `cbloq_to_superquimb`; reindexing that
nonexistent leg does nothing. The resulting graph has missing declared input
indices, so `quimb_to_dense(..., superoperator=True)` raises `KeyError`.
Tests preserve this exact original failure, including missing port/index/bit,
for scalar and shaped classical/quantum inputs. With no other ports, the
original graph contains zero tensors. Reference files are never patched.

The native effect is an explicit extension of that unsupported source case.
For an executable upstream equivalent, tests insert an unchanged same-type Cast
before each original Discard/DiscardQ, without changing source code. The original
Cast supplies identity tensors, making its consumed leg available to the source
trace routine. Native direct effects match the entire resulting channel. This
comparison is **not** raw-tensor or node-count parity with the broken direct
source graph. Each native effect replaces the contracted pair of source Cast
identities in this equivalent reference.

### Verification and existing user workflow

Four dtypes, both input kinds, and scalar/(2,)/(1,2)/(0,) layouts compare complete
channels with the original Cast-wrapped reference. Complex entangled inputs
independently check every spectator density entry and relative-phase derivative
against an explicit partial trace. All-inputs-discarded widths 1/2/8 return the
identity trace functional; source direct failures remain negative cases.
Classical axes are checked as numerical linear maps, not a promise to prepare
physical classical superpositions.

Quantum physical verification uses an existing valid typed **MeasureZ then
classical Discard** body, never Free or a fabricated quantum-discard opcode.
The selected numerical supplier explicitly contracts that effect to a trace.
At widths 1/2/3, all fourteen physical measurement branches preserve complete
complex amplitudes and an unused clean bit remains zero. Summing their
spectator densities matches the numerical channel. Actual instructions remain
one measurement per consumed quantum input, in original operand/slot order;
there is no physical reset and no claim that the consumed measured bits end in
zero. This is the native measure-and-forget adaptation, not a source DiscardQ
gate decomposition.

For two external CBits, all four classical assignments use the existing physical
entry initialization and discard path: two classical XOR assignments, zero
quantum gates/measurements/resets, and no retained classical output. These
initialization costs are separate from zero-cost numerical trace supply.
Users continue to use `circuit.append(op)`. The supported existing example is:

```bash
python -m examples.algorithms.binary_field_arithmetic
python -m pytest tests/unit/test_component_trace_only.py -q
```

This is a finite `single_device_fast_path` slice. Source direct-discard failure,
unsupported source costs, conditional call admission, automatic leaf supply and
full parent-file acceptance remain explicitly open; G3 is not entered.

### GRAPH77 acceptance evidence

Implementation commit: `8d86da2f`. Fifty-three new cases include four coexistence
cases where a direct input effect shares the graph with an internal measurement
environment and discarded classical result. Complete channels match unchanged
Cast-wrapped source graphs for both input kinds and complex64/complex128.

Final concentrated verification: 1372 related tests passed in 16.42 s, 647 focused
tests passed in 22.34 s, and 64 integration tests passed in 3.53 s. The changed
owner has 31 statements and 12 branches, all covered. Sixteen related owners have
494 statements and 200 branches, 99% coverage, retaining two historical GRAPH60
profile-refusal misses. Strict mypy passed for 38 related files plus three
source-following files. Black/Ruff, Simulation scope, architecture, Program IR and
public API checks passed. All 28 locked source blobs match the inventory; 25
unchanged original tests passed in 0.94 s. Qualtran 0.7.0 and native Kickmix 0.1
Simulator were checked, and the existing binary-field append example passed.

Actual numerical supply emits zero physical instructions and allocates zero work
bits. Existing quantum measure-and-forget emits n measurements and uses caller
n+2 physical bits, including the spectator and an unused clean bit. Classical
entry initialization emits n classical XOR assignments and uses two physical
bits, including the spectator and unused clean bit. Source Discard has zero
QEC measurement/Clifford/T-equivalent and QubitCount zero. Source DiscardQ has
QubitCount one, but QEC raises DecomposeNotImplementedError; this unsupported
model is not replaced by zero or by native measure-and-forget costs.

For float32/float64/complex64/complex128, shared effect storage occupies
16/32/32/64 bytes. Eight trace-effect nodes have logical tensor bytes
128/256/256/512; their dense trace functional occupies
262144/524288/524288/1048576 bytes. These measure returned storage, not peak
construction or contraction memory. They do not imply a scalability release.

Default smoke/unit: seven failed, 21507 passed, 300 skipped and 1037 warnings in
148.54 s. All seven failure IDs match the independent baseline, with no new
observed failure ID; this maxfail run is not a green default suite. It ran before
the final four coexistence tests were added, with unchanged production code.
Raw logs, source hashes, original failures, resources and baseline comparison
are under `ftqc-reference-env/g2-component-trace-only-*`.

## Pure supplier discard refusal (GRAPH80)

The pure supplier shares the mixed supplier's integer item type because the
upstream tensor callback can yield a `DiscardInd` token. In the pure path, the
first integer token now raises immediately with guidance to use the explicit
mixed channel entry. No later item is requested, no partial graph is returned,
and dense contraction is never called. The integer is not treated as a tensor,
projection, reset, or implicit trace. The mixed path retains its existing
explicit environment-label trace protocol unchanged.

This follows the relevant behavior in the unchanged Qualtran v0.7.0
`cbloq_to_quimb`: it raises `ValueError` upon the first `DiscardInd`. Tests
compare the error category, information-discard meaning, and stop point; native
guidance names FlagQuantum APIs, so the error message is not byte-identical.
Pure classical-port validation still occurs before supplier invocation, and no
claim is made that its ordering matches every source sentinel path.

The finite tests cover both pure wiring and dense entries, scalar and shaped
discard inputs, all four supported tensor dtypes, and a tensor followed by a
trace token. The latter checks that both unchanged source and native suppliers
stop before their generator tail; a spy proves no contraction occurs. Four
mixed-channel dtypes consume the same node/trace/tail protocol and compare the
entire source channel, a complex spectator density, and a relative-phase
gradient. Existing `circuit.append(op)` measure-and-forget instructions remain
unchanged: widths 1/2/3 execute all 14 branches with full complex amplitudes,
branch-density sum, and an unused clean bit returned to zero. Consumed measured
bits are not claimed reset.

The numerical refusal itself adds zero physical gates and work bits and
performs no contraction. The mixed numerical result is a 4x4x4x4 complex128
array (256 elements, 4096 bytes of returned dense storage), not a peak-memory
measurement. The actual physical measure-and-forget path emits `n` measurements
and uses caller `n+2` bits including spectator and unused clean storage; the
source `DiscardQ` QEC decomposition remains unsupported and is not replaced by
that adapter cost. Source direct superoperator failure remains a separate
negative case from the Cast-wrapped equivalent-channel comparison.

GRAPH80 does not add an automatic provider, graph traversal, tensor recipe,
physical discard decomposition, or public API. The dephasing-shaped supplier
fixture is only a numerical leaf descriptor; its body is not claimed as a
physical lowering. Exact source and local resource records are under
`ftqc-reference-env/g2-pure-tensor-discard-*`.
