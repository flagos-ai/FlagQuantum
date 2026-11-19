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
| `Adjoint.build_call_graph` | Existing `component_callee_counts` with a supplied inverse-key generalizer, plus `program_component_call_graph` for actual stored calls | No generic wrapper infers arbitrary declared node inverses |
| `Adjoint.__str__` | Existing caller-selected graph names and Drawer `adjoint_wire_symbol` preserve explicit labels | Automatic generic wrapper display-name derivation remains open |
| `Adjoint.wire_symbol` | Drawer `adjoint_wire_symbol` has source comparisons for root labels and directional register symbols | Automatic coupling from a generic wrapper to its label supplier remains open |
| Classical and tensor decomposition-only semantics | Existing component graph and tensor owners; full complex operator and explicit leaf execution evidence | Generic wrapper admission and dispatch still require explicit owned interfaces |

Q-0147 and Q-0148 remain open. The table records whole-file coverage and the
remaining interface obligations instead of treating individual passing protocol
checks as complete generic-wrapper acceptance.

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
