# Sparse chemistry PREPARE constructors and explicit native erasure

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
Source/test blobs are `02e4b381091a6677d68b299305a9b32c0bf5e04f` and
`4d3fa0ec18ef9abf0aef050231f3147b27bb078f`. Fixtures check both blobs,
Qualtran 0.7.0 and the real Kickmix 0.1 Simulator.

## Ownership and complete source call sequence

Compiler owns the typed eighteen-call PREPARE root, its exact directional
dependency pairs, MSB-first named registers and alias/symmetry swap tables.
Existing Algorithms metadata and execution reuse those tables and register
owners. Algorithms retains integral construction, normalization and its
existing coherent versus measured inverse choices. Existing data construction,
single-orbital extensions and resource evidence are reused by the batch tests.

The root follows uniform(d), QROAMClean, OnEach(H,sigma), LessThanEqual,
CZ(less_than,alt_theta), zero-controlled Z(theta), five alias swaps, three
symmetry H calls and four symmetry swaps. Recursive comparisons preserve
actual named operands, dependency direction and measurement feedback. The
generic OnEach inverse is an actual inverse call graph, not a forward graph
renamed as an adjoint. Independent topological traversal order is not used
as a substitute for the source creation order.

Row/sampling/spin/symmetry words and less_than remain caller-owned THRU ports.
The thirteen selected datasets and retained dataset-major bank arrays are
RIGHT outputs. The inverse consumes them through LEFT ports with explicit
ownership transfer. Only private scratch must be zero at forward return;
prepared selected/bank outputs deliberately retain their quantum state.

Formats binds all fourteen original constructor arguments without importing
SDK objects. Rotation precision and normalization survive library interchange;
the source executable body does not use them, so no quantum effects are
invented. Source depth-zero/depth-one constructors and generic inverses resolve
to the same physical instructions as the independently tested typed owner.
The ordinary `circuit.append(op)` path works after source SDK capture with
Qualtran and Cirq imports unavailable. No public API or Core opcode changes.

## Explicit inverse boundary

The original QROAMCleanAdjointWrapper has one executable reshape/call boundary.
That boundary is compared. Its QROAMCleanAdjoint child has a cost model and
raises DecomposeNotImplementedError; the default source factory still retains
that bodyless declaration. The new explicitly named native assembly supplies
the previously accepted measured-bank erasure recipe behind the same typed
leaf ports. Its primitive body is native, not a claimed upstream decomposition.

The recipe is valid on matching forward lookup outputs, including the routing
relative phases. It does not establish a unitary on arbitrary independent
bank inputs. It must not be used as PREPARE-dagger on arbitrary reflection
inputs, nor may RIGHT ports simply be relabeled THRU. Complete sparse Walk
still needs an explicitly constructed and verified unitary extension or a
correct retained signal environment. THC dependencies remain open too.

## Evidence and precise numerical scope

The final concentrated regression has 196 passes. The separately rerun nine
branch checks add a multi-dataset/multi-bit erasure fixture,
enumerate every small PREPARE forward history and check four distinct joint
roundtrip histories. Local integration has 145 passes. Strict typing covers
eight files; format, scope, language, architecture and IR/API checks are
separate. Dependency findings remain identical to the 720-entry baseline.
The unchanged original suite has thirteen passes, one existing counts xfail
and one source dtype-check skip (QUInt comparator output versus BQUInt sigma).

Four-row one-/two-bit keep tables, unblocked/blocked lookup and coherent
address witnesses compare complete sparse complex forward amplitudes against
the unchanged source and verify private work zero. Forward followed by the
explicit native inverse restores the complete joint input state, selected
banks and work. Each small QROAM component enumerates every raw measurement
history on arbitrary complex selector/control superpositions, including
multidimensional, padded, constant, zero-width and multi-bit datasets.
Existing G1 erasure and Algorithms integral/alias/branch evidence is reused.
Every combined root inverse history is not numerically enumerated; the two
root fixtures have nineteen/thirty-four joint measurements. No generic
all-history certificate for bank-parity feedback is claimed here.

The actual original Walk PREPARE uses eight spin orbitals, sixty-five rows,
twelve keep bits, eight rotation bits and log block size one. Both directions
and depths match every available source call down to the explicit native
erasure leaf, and physical binding succeeds. This large case has call/data/
resource evidence, not a newly evaluated full numerical output operator.
The source single-orbital target0_ failure remains an upstream boundary;
existing native single-orbital execution remains separately verified.

## Actual primitives versus source models

Six directional inventories are in
`g2-sparse-prepare-resources-20261125.json`.

| Case | Actual T/T-dagger | Actual CX | Actual measurements | Source model | Physical wires |
| --- | ---: | ---: | ---: | --- | ---: |
| Four rows, keep 1, unblocked forward | 75 | 127 | 3 | 8 CSwap, 3 And, 55 Clifford, 3 measurements | 25 |
| Same, explicit native inverse | 75 | 101 | 16 | 8 CSwap, 1 And, 23 Clifford, 1 measurement | 25 |
| Four rows, keep 2, blocked forward | 131 | 215 | 3 | 56 T, 8 CSwap, 3 And, 370 Clifford, 3 measurements | 44 |
| Same, explicit native inverse | 131 | 188 | 31 | 8 CSwap, 3 And, 46 Clifford, 3 measurements | 44 |
| Original 65-row forward | 543 | 2134 | 74 | 128 T, 17 CSwap, 74 And, 2247 Clifford, 2 rotations, 74 measurements | 125 |
| Original, explicit native inverse | 543 | 1029 | 138 | 13 Toffoli, 17 CSwap, 43 And, 387 Clifford, 2 rotations, 43 measurements | 125 |

Forward retains respectively 13, 28 and 64 output-bank lease bits. Inverse
consumes those banks; both directions return every truly private lease.
Lease accounting does not replace the independent complex-amplitude cleanup
checks. Source model gates and measurements are never substituted for actual
primitive instructions. Whole chemistry Walk, arbitrary-bank reflection,
repository/GPU/performance acceptance and the upstream boundaries above remain
open. This batch does not enter G3 or merge the integration PR.
