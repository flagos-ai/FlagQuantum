# Rotation library constructor dependencies

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete `rotation.py` and
`rotation_test.py` source files were read. Their previously accepted source
method, drawing, control, tensor and original-test evidence in
`ROTATION_FILE_AUDIT.md` remains applicable. This constructor batch reuses those
native implementation owners through the explicit library resolver.

## Interfaces and dependency ownership

| Source constructors | Serialized arguments | Existing native owner |
| --- | --- | --- |
| Rx, Ry, Rz | required angle, eps default 1e-11 | `registers/primitives/pauli_rotation.py` |
| XPowGate, YPowGate | exponent default 1, global_shift default 0, eps | `registers/primitives/pauli_rotation.py` |
| ZPowGate | exponent default 1, eps | `registers/primitives/pauli_rotation_component.py` |
| CRz | required angle, eps | `registers/primitives/controlled_rotation_component.py` |
| CZPowGate | exponent default 1, eps | `registers/primitives/controlled_rotation_component.py` |

`formats/legacy_rotation_factories.py` supplies eight explicit constructor
identifiers to `resolve_component_program`. Six classes have one QBit q THRU
port. CRz has scalar ctrl/q THRU ports. CZPowGate has q[2] in MSB order. Its
conjunction owns one clean temporary, one measurement slot and measured erasure.
Other constructors need zero private work or measurement. Existing namespace
rewriting isolates all root-local helper definitions and calls.

Unknown fields, conflicting signatures, missing required angles, non-real
numerical payloads and nonfinite angles fail before binding. Source numeric
precision remains in the serialized library and independent resource keys.
Precision does not affect numerical tensor action. Symbolic precision remains
model metadata. X/Y global_shift admits finite concrete real values, including
constant symbolic rationals. A parameterized shift retains the existing concrete
storage requirement from the source float-typed constructor.

### Shared parameter conversion

`formats/symbolic_parameters.py` owns the supported symbolic-scalar conversion
previously embedded in the Cirq adapter. The adapter retains its original SDK
symbol-discovery and identity diagnostics and calls this shared owner. Compiler
imports no SDK constructor. The existing owned Parameter/ParameterExpression
vocabulary and approved Proposal 107 supply sin, cos, exp, log, powers, division,
addition and multiplication. Unsupported symbolic functions and nonfinite or
complex constants fail explicitly.

The Pauli primitive owner now accepts the same owned expressions as its existing
Z-power component. Existing concrete angle, overflow and wire validation remains
in place. Symbolic global phase uses the existing exact scalar-phase recipe.
The Z-power component delegates to the shared primitive, eliminating a duplicate
angle-conversion path. A bound symbolic recipe matches its concrete stream,
full complex state and an independent analytic Torch derivative.

## Execution and resource accounts

All eight original factory examples resolve at depths zero, one and two.
Mixed CRz/CZPow composites with different angles resolve at three depths,
without helper collisions or feedback-slot aliasing. The normal `circuit.append(op)`
consumer is retained. Measured CZPow execution uses the existing DynamicCircuit
append and trajectory runtime. Symbolic source expressions survive library
serialization, native Program lowering and CircuitIR JSON before explicit binding.

The concrete grid covers zero, generic positive and negative values, half powers
and full powers, in complex64 and complex128. Source-comparable controlled gate
streams retain operands and measurement feedback. Full operators and complete
complex spectator states retain absolute phase. Every forward/inverse erasure
history is enumerated and restores work to zero with its original probability
factor. The shared branch-inspection helper evaluates branches in complex128.
Actual Circuit and trajectory execution separately exercise the requested dtype.

| Account | Emitted primitive inventory | Independent source/native QEC model |
| --- | --- | --- |
| Rx/Ry/Rz | One radian rotation | Existing angle-sensitive Clifford/T/rotation category |
| ZPow | One phase | Existing angle-sensitive Clifford/T/rotation category |
| XPow at zero shift | H, phase, H | One source atom classified by exponent |
| YPow at zero shift | Sdg, H, phase, H, S | One source atom classified by exponent |
| Shifted X/Y power | Corresponding basis recipe plus four scalar-phase primitives when shift angle is nonzero | Source atom cost remains independent of scalar recipe |
| CRz | Two phase, two CX | Two half-angle Z-power classifications plus two Clifford |
| CZPow | Three H, six CX, two T, two Tdg, one S, one phase, one measurement, one conditional CZ, one reset | One And, one Clifford, one angle-sensitive phase category, one measurement |

For general source X/Y power atoms the upstream Cirq decomposition hook returns
NotImplemented. The native basis implementation and upstream atomic leaf have
different primitive inventories. Exact complex operator equivalence is checked
across that boundary. The controlled decompositions
have source-comparable instruction and feedback evidence. Synthesis precision
and model estimates remain separate from every measured primitive account.

## Remaining boundaries

Generic serialized Adjoint/Controlled/Power constructor wrappers remain part of
open Q-0876. In particular CRz's source Adjoint wrapper is not a new identifier
in this group. Existing exact-adjoint component evidence remains applicable.
Negating CRz's angle restores its unitary, while reversing/inverting its original
stream is a distinct previously accepted path. Original source hashes are verified.

An already-decomposed CZPow composite whose root directly contains serialized
And leaves still needs explicit And constructor mappings. A named CZPowGate root
uses its complete native And/phase/erase dependency graph. Constructor resolution
requires an explicit factory for each referenced leaf. The next
coherent group should map the complete `and_bloq.py` constructor family and its
source Adjoint dependency through existing conjunction owners.

The built-in groups now cover 26 of the frozen resolver's 303 identifiers,
leaving 277 other identifiers and generic reflection open. Q-0876 and the G2
parent totals remain open. Pending Proposals 116–119, whole-repository/GPU and
performance boundaries retain their existing status.

## Batch acceptance

The affected native regression passes 482 cases. Twelve Qiskit interop cases
skip because the optional Qiskit dependency is absent. All 36 unchanged original
rotation tests pass, with 14 original atomic/count/dtype-mode skips. The affected
local integration selection passes 145 cases with 18 existing warnings. Coverage
acceptance passes 145 cases and measures 100% statement/branch coverage for the
new factory and the changed Pauli primitive owner, 92% for the extracted symbolic
converter and 97% combined. The converter's defensive unknown-symbol and empty
Add/Mul branches are retained from the original adapter and remain unexercised.

Strict typing passes eight Python files. Black, Ruff, both team scopes,
repository language, architecture, multilevel IR and protected API checks pass.
Dependency policy retains the identical 720 baseline findings. Three complete
source/test/resolver Git blobs and the locked native Kickmix 0.1 Simulator are
verified separately. Source protobuf deprecation warnings remain visible. Whole-repository and GPU
acceptance remain open.
