# Fixed gate library resolution

The reference is the unchanged Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch follows the complete
`serialization/resolver_dict.py` inventory and the complete `cnot.py`,
`hadamard.py`, `s_gate.py`, `t_gate.py`, `toffoli.py` and `y_gate.py` sources.
Their existing implementation audits retain ownership of classical callbacks,
tensor factors, controls, drawing and external SDK behavior. This delivery
closes their native library constructor mapping dependency. Q-0876 remains open
for the other catalogue groups.

## Interface and ownership

`compiler/formats/legacy_fixed_gate_factories.py` owns explicit interchange
names mapped to existing native typed gate lowering. Each invocation returns
an independent mapping suitable for `resolve_component_program`. Compiler
imports no foreign constructors. The existing resolver owns subcomponent
references, composite connections, signature checks and Program verification.
Simulation owns matrix execution through the ordinary append path.

| Source class | Ordered THRU registers | Native instruction |
| --- | --- | --- |
| CNOT | ctrl, target | CX |
| Hadamard | q | H |
| CHadamard | ctrl, target | CH |
| SGate | q | S or SDG |
| TGate | q | T or TDG |
| Toffoli | ctrl[0], ctrl[1], target | CCX |
| YGate | q | Y |
| CYGate | ctrl, target | CY |

Register elements retain MSB-first source order independently of physical wire
assignment. All eight classes are atomic at this boundary. They introduce no
work, allocation, measurement or feedback. Serialized S/T `is_adjoint` values
are binary integers because the original codec writes booleans into int fields.
Native construction validates this flag and rejects unknown constructor fields
and conflicting signatures. Absent serialized signatures are derived from the
known source constructor. The mapping remains an explicit caller choice.

## Existing user workflow

The data boundary accepts serialized library bytes without importing executable
objects. A caller supplies native factories and binds the reconstructed Program.

```python
from flagquantum import Circuit
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_library import deserialize_component_library
from flagquantum.compiler.formats.legacy_component_resolution import resolve_component_program
from flagquantum.compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from flagquantum.compiler.registers.register_binding import bind_register_program

library = deserialize_component_library(library_bytes)
program = resolve_component_program(library, 0, fixed_gate_factories())
bound = bind_register_program(program, ((0,), (1,)))
circuit = Circuit(2)
op = FieldOperation(2, bound.circuit.instructions)
circuit.append(op)
print(circuit.state())
```

For an atomic CNOT library, the default input remains the zero basis state.

The executable end-to-end example is
`tests/unit/test_legacy_fixed_gate_factories.py`. It constructs source libraries,
resolves native definitions, binds reversed noncontiguous operands and appends
the existing `FieldOperation`. The same consumer also accepts a replacement
factory returning a hierarchical Program through the established interface.

## Separate resource accounts

| Source class | Observed native primitives | Independent source QEC model |
| --- | --- | --- |
| CNOT, H, S/SDG, Y, CY | One corresponding atomic instruction | One Clifford |
| T/TDG | One corresponding atomic instruction | One T |
| CH | One CH | Four Clifford and two rotations |
| Toffoli | One CCX | One Toffoli |

Resolution and typed Split/Join add zero quantum instructions. No synthesis
expansion is inferred from atomic instruction counts. Measurement branches are
inapplicable to these unitary leaves. Existing measured composite resolver
evidence remains in `PROTO_COMPONENT_FILE_AUDIT.md` and its regression tests.

## Acceptance and remaining boundaries

The focused checks compare complete unitaries to original tensor contractions,
forward/adjoint restoration, complex64/complex128 spectator amplitudes, untouched
clean work, exact ordered gate/operand streams, signatures, deterministic wire
round trips and every original QEC category. Frozen inventory fingerprints
include all six complete source files and the complete 303-entry resolver.

The concentrated native regression passes 87 cases, including the existing
serialized composite, predicate and all eight measured-erasure histories.
The affected local integration run passes 145 cases with 18 existing emulation
deprecation warnings. Strict typing passes both changed Python files. Black,
Ruff, Compiler scope, repository language, architecture, IR and API checks pass.
Dependency policy reports the identical existing 720 findings.

The six unchanged original test files pass 56 cases in the isolated Cirq 1.6
and PennyLane 0.44.1 environment. Their 21 skips are original atomic-decomposition,
call-count and decomposition-dependent dtype modes. Cirq 1.7 retains the original
Y/CY reverse-conversion failure documented by `Y_GATE_FILE_AUDIT.md`. The initial
combined run also lacked the source `notebook` marker registration. The final
native runner registers that original marker and passes with strict markers
enabled. The default reference environment initially lacked PennyLane, so the
original run explicitly reuses the existing isolated PennyLane installation.

This mapping covers eight source catalogue identifiers and ten gate directions.
The other 295 identifiers, generic reflection and arbitrary foreign payload
execution remain open. The source codec's boolean-to-int protobuf deprecation
is retained in the reference run. Native constructor validation deliberately
refuses nonbinary flags that the unvalidated source dataclass may accept.
Protected static SDK round trips remain in their existing audits. No complete
resolver catalogue, whole-repository, GPU or performance acceptance is claimed.
