# Default adjoint protocol audit

This batch audits the complete frozen `qualtran/_infra/adjoint.py` and its
`adjoint_test.py` at v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Their Git blobs are
`0601a6248eec488301b31dd7c800a6ba14a7fb27` and
`f349c5dafbdc5ed7764d4aa39b0689fd14b9e4f7`. Reference files remain unchanged.
The complete native tutorial is already accepted separately as Q-0138.

## Complete definition and protocol ownership

| Frozen source definition or protocol | Native owner and evidence | Remaining boundary |
| --- | --- | --- |
| `_adjoint_final_soqs` | `component_adjoint._adjoint_function` starts from explicit returned SSA values and reconnects original inputs | Source empty-array return failure is retained as a recorded source difference |
| `_adjoint_cbloq` | `adjoint_component_program` and `adjoint_component_hierarchy`; existing graph, shaped ports, lifetimes, complex states and all measured Add branches | Exact specialized atomic definitions remain explicit |
| `Adjoint.signature` | Core `Signature.adjoint`, used by both graph reversal and private Cirq gate inversion | No missing finite directional metadata behavior found |
| `Adjoint.decompose_bloq` | Stored-body hierarchy reversal with shared definitions and explicit leaves | Native atomic bodies need a supplied exact inverse; arbitrary Python method discovery is outside the current owned graph interface |
| `Adjoint._circuit_diagram_info_` | Private Ecosystem `RegisterCirqGate` retains labels and multiplies the diagram exponent by its power | Source missing diagram protocols return `NotImplemented`; native declared registers supply default labels |
| `Adjoint.adjoint` | Private Cirq inverse returns the exact original object on double inversion; graph reversal has existing structural double-inverse evidence | Program transformations return new immutable graphs |
| `Adjoint.build_call_graph` | `AdjointCall.callee_counts` wraps arbitrary hashable forward identities by default, applies specialized inverse mappings, and reuses `component_callee_counts`; actual stored calls use `program_component_call_graph` | Declarations and specialized inverses stay with their native owners |
| `Adjoint.__str__` | `AdjointCall.__str__` appends a dagger to the original display name without changing node identity | Program function names remain stable executable identifiers |
| `Adjoint.wire_symbol` | Drawer `adjoint_wire_symbol` has source comparisons for root labels and directional register symbols; register callback inversion uses the original direction | Presentation callbacks remain in Drawer or Ecosystem |
| Classical and tensor decomposition-only semantics | Existing component graph and tensor owners; full complex operator and explicit leaf execution evidence | Generic wrapper admission and dispatch still require explicit owned interfaces |

The source class is distributed across the established native ownership boundaries.
`AdjointCall` is the lazy resource identity. Core owns signature inversion,
Compiler owns executable graph derivation, Drawer owns glyph inversion,
Ecosystem owns SDK diagrams, and Simulation owns decomposition evaluation.
No foreign class or alternate executable graph is stored in Core. Exact leaf
inverses remain explicit, including measurement-assisted erasure. The listed
source failures and API adaptations are retained in the acceptance scope.

## Ecosystem repair

Previously, `RegisterCirqGate.__pow__(-1)` inverted the physical operation while
leaving its diagram exponent at positive one. A direction-sensitive label
supplier also received the inverse register instead of the original register
it described. The private adapter now retains the signed display power,
delegates labels against the original direction, and returns the original gate
on a second inverse. Display power participates in value equality and hashing.
Each diagram query creates independent metadata. Positive and negative repeated
powers retain the original labels and their corresponding signed exponent.

This repair uses private adapter state and the existing constructor. Public
constructor arguments, Core records and Compiler transforms need no changes.
The materialized native body remains the authority for execution. Existing
feedback and workspace gate restrictions continue through their normal checks.

`test_register_adjoint_protocols.py` compares unchanged source diagram metadata,
actual primitive order and operands, the full complex matrix and an entangled
spectator through `circuit.append(op)`. The H/T example emits two primitives,
uses zero workspace and zero measurements, and has an independent source model
of one T plus one Clifford. Repeated powers emit twice their magnitude in
primitives. Existing complete Add and AND branch evidence is rerun in the same
concentrated suite. Source set/dict declarations, symbolic counts and multiple
children mapped to one inverse are checked through the existing shared count
interface. No extra resource graph representation is introduced.

## Verification

The nine unchanged source tests pass, including their actual Jupyter execution.
The source emits five collection warnings. The native concentrated suite passes
1409 cases and local integration passes 96 cases. Strict typing covers both
changed Python files. Black, Ruff, Ecosystem scope, architecture and IR checks
pass. Dependency scanning retains 689 existing repository findings and zero
findings in the changed files. Persistent logs and the exact prior mismatch are stored as
`ftqc-reference-env/g2-adjoint-protocol-*` and `g2-adjoint-metadata-gap.json`.
The prior default-suite public-docstring registry failure remains recorded by
Q-0138's acceptance run. This batch does not claim a full default-suite pass.


## Declared identity and whole-file completion

The native resource module `compiler/_component_trace/adjoint.py` adds the
missing default wrapper for arbitrary hashable declaration nodes. Names are
computed from the child, while equality and hashing retain its full identity.
Two distinct children with the same printed name remain distinct. Calling the
inverse of this wrapper returns the original child object. The common helper
also unwraps an already inverse node when traversing declarations.

`callee_counts` first invokes the original forward declaration with the same
symbol allocator. It delegates accumulation to the existing count engine and
maps every child to its default or explicitly specialized inverse. Both set and
dictionary declarations retain symbolic multiplicities. Unsupported direct
queries propagate their error. The existing resource graph policy may retain a
capability refusal as a terminal node. No numerical primitive is emitted by
resource traversal, and declared costs are independent of executable IR counts.

The complete source TestBloqWithCallGraph hierarchy matches every directed edge
and weighted leaf count, including specialized Split/Join inversion. The native
H/T executable witness still emits two primitives with one T and one Clifford
in its independent model. The three-X decomposition witness ignores source
specialized classical/tensor overrides exactly as the default source wrapper
does, evaluates both basis values and matches the full complex matrix.

| Original test | Native whole-workflow evidence |
| --- | --- |
| `test_serial_combo_adjoint` | Tutorial connection maps and existing reversed serial graph tests |
| `test_cbloq_adjoint_function` | Existing validated graph, shape, ownership and workspace adjoint tests |
| `test_adjoint_signature` | Core direction inversion and one-sided Cirq label callback tests |
| `test_adjoint_adjoint` | Resource original-object unwrap, private gate original-object return and graph double inversion |
| `test_bloq_counts` | Default and specialized set/dict symbolic declaration tests |
| `test_call_graph` | Complete TestBloqWithCallGraph edge and weighted-leaf comparison |
| `test_names` | Arbitrary original labels, trailing daggers, identity/hash separation and existing root glyph tests |
| `test_wire_symbol` | Existing IntState/IntEffect/T source callback comparisons and both directional sides |
| `test_notebook` | Complete 17-cell native tutorial and unchanged Jupyter execution |

The original source's empty-array final-output omission, raw CompositeBloq
wrapped decomposition refusal and static TestAtom inverse-cost refusal retain
their previously documented source behavior. Native explicit empty-array
metadata is preserved. Unsupported measured primitive inversion still requires
its dedicated inverse body. This avoids inferring a unitary inverse for erasure.

Final declared-wrapper acceptance passes 1459 concentrated cases, strict typing
of both new Python files, Black, Ruff, Compiler scope, architecture and IR
checks. The resource module has 100% statement coverage. The dependency scan
retains 689 repository findings with zero changed-file findings. Frozen source
and test blobs remain unchanged, so the previous nine original test results
remain applicable. The local integration result is recorded in
`g2-declared-adjoint-integration.log`. This closes Q-0147 and Q-0148 through
the complete ownership map above, including explicit source differences.
