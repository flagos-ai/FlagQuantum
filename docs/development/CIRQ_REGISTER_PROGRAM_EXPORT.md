# Bound Program export and output ownership

`ecosystem.cirq._register_program_export.export_bound_register_cirq` exports
an already compiler-bound Program and returns its Cirq circuit plus named output
arrays. Compiler remains responsible for physical binding, capacity, linear
ownership and component traversal. The ecosystem adapter translates the existing
instruction stream and allocation events. It shares the workspace replay used
by `RegisterCirqGate`.

This closes the finite executable Program workflow behind frozen Qualtran
v0.7.0 `CompositeBloq.to_cirq_circuit_and_quregs`, following
`cirq_interop/_bloq_to_cirq.py::_cbloq_to_cirq_circuit`. The original file's Git
blob is `d49ab63f9f74104dc5f2856a9e1c997635e21e5b`. Reference tests verify that
blob, the corresponding test/notebook/init files, Qualtran 0.7.0 and the locked
Kickmix 0.1 native Simulator before comparison.

## Interface and lifetime

Bind input register elements consecutively from physical wire zero, in signature
order and row-major element order. Supply each element's bit tuple in its existing
MSB-first order. `input_quregs` maps the same register names to arrays of shape
`register.shape + (register.bitsize,)`; omitted mappings use the existing native
register naming convention. Cirq qubits can have any physical names or order.

The caller supplies a Cirq qubit manager for allocation and release. An explicitly
owned LEFT input can be released and immediately reallocated as a RIGHT output.
The adapter returns the actual final physical qubits, including bit permutations.
New output leases remain allocated and belong to the caller. Intermediate leases
are released at their recorded instruction boundaries. On replay failure, remaining
newly allocated leases are returned to the manager. Released inputs cannot be
restored by rollback.

The direct conversion retains every primitive's instruction order using explicit
single-operation moments. Cirq's default earliest scheduling can move independent
operations and would invalidate positional gate comparisons. The earlier
fixed-gate export continues to use its existing framing behavior.

```python
import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_program_export import export_bound_register_cirq

signature = Signature.build(q=1)
leaf = bit_gate_function("flip", tuple(signature), CircuitIR(1, (Instruction("x", (0,)),)))
builder = ComponentProgramBuilder(signature, (leaf,))
record = builder.finalize(builder.call_dict("flip", builder.inputs))
bound = bind_register_program(record, ((0,),))
operation = FieldOperation(1, bound.circuit.instructions)
circuit = fq.Circuit(1).append(operation)
exported, outputs = export_bound_register_cirq(bound, signature)
assert tuple(outputs["q"])[0].name == "q"
assert circuit.to_ir().instructions == bound.circuit.instructions
assert len(tuple(exported.circuit.all_operations())) == 1
```

## Differential evidence

The source and native builders construct the same connected workflows. Tests
compare clean/borrowed allocation and release calls, reordered shaped inputs,
returned RIGHT leases, reversed three-bit LEFT release/reallocation, empty
outputs, primitive operands and complete complex amplitudes with an entangled
spectator. A repeated AND/measurement-uncompute child is tested with both
positive and negative control polarity. All four two-measurement histories are
projected independently using Cirq and native execution; their unnormalized
complex amplitudes match and the work wire returns to zero. The same native
instructions run through `circuit.append(operation)`.

| Workflow | Emitted primitives | Source model | Work |
| --- | --- | --- | --- |
| Two CNOT compute/uncompute pairs and H | 4 CX + 1 H | Model omitted by this fixture | One reused clean or borrowed bit |
| Two positive AND/erase cycles | 34 primitives, including 8 T/T-dagger and 2 measurements | 2 AND, 2 Clifford, 2 measurement | One clean bit, zero in every branch |
| Two negative-polarity AND/erase cycles | 42 primitives, including 8 T/T-dagger and 2 measurements | 2 AND, 2 Clifford, 2 measurement | One clean bit, zero in every branch |
| Three-bit release/reallocation | 0 | 0 gates | Caller receives the three final live bits |

The source QEC model records the same AND/Clifford counts for both polarities,
although the negative-control decomposition emits additional X gates. Both the
unchanged source model and measured primitive stream are recorded above.

The source measurement key is based on the work qubit's name and repeats when
that qubit is reused. Native keys use instruction occurrence indices. Gate and
feedback comparisons rename those keys explicitly; each condition still reads
its own preceding measurement.

## Remaining boundaries

The bound input layout is an explicit precondition because `BoundRegisterProgram`
does not store the original input register mapping. Classical output registers,
non-binary Qids and symbolic physical widths are rejected. Existing compiler
binding requires at least one physical carrier even for zero-width-only Programs.
Empty arrays and zero-width outputs retain their exact shapes in the returned
mapping. The output arrays carry idle extent that a bare Cirq circuit cannot
represent by itself.

This module exports finite executable native instructions. Arbitrary source SDK
wrappers, undecomposed source Python callbacks, symbolic typed Program admission
and the broader Q-0156 source-file closure remain open. This evidence does not
close those requirements or the separate Q-0154 PennyLane adapter.

## Local acceptance

The concentrated regression passes 129 cases without skips, including the
existing gate exporter, feedback importer, input pool and output-transfer suites.
Strict typing passes for all three changed Python files. Black, Ruff, ecosystem
scope and architecture checks pass. The complete dependency scan reports 706
existing findings and zero findings in this batch. The documented workflow runs.
The new exporter has 52 statements and 20 branches fully covered. The shared
workspace replay measures 85% under this regression; remaining uncovered paths
are recorded as validation/error paths rather than a whole-file coverage claim.
The source tests' notebook marker is registered only in the external regression
runner; repository pytest configuration is unchanged.
