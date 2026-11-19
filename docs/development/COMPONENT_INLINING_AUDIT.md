# Component construction-time inlining

The Q-0155 complete tutorial requires the source `BloqBuilder.add_from` call
chain. Its native entry is `ComponentProgramBuilder.inline_calls(record,
bindings, entry="main")`, returning a dictionary in the declared right-port
order, consistent with the existing `call_dict` interface. It inserts the
selected function's immediate calls into the live builder. Nested children
remain nested. The input record is an explicit, verified Program decomposition;
a foreign Python object or a missing body is not silently interpreted.

`component_binding_map.map_register_bindings` maps scalar and shaped handles
through the supplied old/new pairs. It retains unmapped handles, requires equal
array shapes, preserves scalar/array distinctions and does not consume resources.
The builder's existing call operation owns all consumption and failure effects.
Initial input mapping is prepared before calls, matching source failure timing.
Surplus initial bindings are ignored by the source inlining operation and remain
ignored here. A missing required initial binding raises KeyError before calls.

Core remains the only owner of the Program and SSA. Compiler owns finite graph
mapping and construction. Simulation and resource queries continue to consume
that resulting Program. Equivalent existing definitions are reused; conflicting
local names are deterministically isolated before insertion. Original source
records and caller definitions are not mutated. No public export or schema is
changed by this internal builder operation.

Acceptance requires source/native handle mapping, constructed call order,
shapes and output order, empty and directional signatures, repeated inlining,
name collisions, preserved failure prefixes, and physical/complex-amplitude
checks through the existing binder and `circuit.append(op)`.

The prerequisite regression passes 114 cases. The new mapping module has 19
statements with 100% measured coverage. Strict typing passes all three changed
Python files. Scope, formatting, lint and architecture checks pass. The source
comparison includes scalar, shaped and zero-width words, overwritten mappings,
missing inputs, failed duplicate consumption, two crossed CNOT calls, nested
calls, definition collisions, allocation/release and repeated inlining.

Actual primitives are two CX instructions for the crossed CNOT example and six
X instructions for two three-bit parallel components. There are no measurements
or workspace events in either gate example. Complete complex states include an
untouched spectator and compare exactly. The allocation/release example emits
zero primitives and returns its three clean work wires. These measurements do
not substitute for source model costs: source TestAtom has a deliberately
synthetic 100-T static model while its tensor is X.

One source defect remains explicit. For shape (0, 2), source `final_soqs`
omits the empty output, so `add_from` raises KeyError. The native typed signature
retains and returns that empty array. This accepted existing signature behavior
is tested separately and is not reported as source parity. The 42-cell tutorial
Q-0155 still awaits full integration and is not closed by this prerequisite.
