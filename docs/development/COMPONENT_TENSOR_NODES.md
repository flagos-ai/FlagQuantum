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
internal, duplicate and invalid labels. External entry ports cannot be discarded
without changing their signature. Port maps returned by the helper remain
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
leaf supply, trace-only inputs without numerical tensor legs, nested numerical
expansion policy, and full `_quimb.py` parent-file acceptance remain open. Callers
must select/expand components using the existing Compiler boundary first.
The supplier is responsible for genuine classical/quantum semantics; merely
accepting a CBit axis does not infer a stochastic channel or dephase it.

Run the finite scenarios in the locked reference environment:

```bash
python -m pytest tests/unit/test_component_superoperator_nodes.py -q
python -m examples.algorithms.binary_field_arithmetic
```
