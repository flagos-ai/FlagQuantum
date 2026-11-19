# Register component graphs

FlagQuantum represents hierarchical quantum programs in the immutable Core
`ProgramRecord`. Named `FunctionRecord` definitions declare registers and
contain calls to other definitions. Compiler's `ComponentProgramBuilder`
constructs and verifies the graph, and `bind_register_program` assigns physical
wires. Simulation and Drawer consume those representations at their own
boundaries. A programmer follows returned bindings through each call.

## Build a graph

This example records allocation, Hadamard calls and release using existing
owners. A binding identifies a current logical value. Physical wire assignment
occurs after the builder freezes the program.

```python
from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def hadamard_graph(repetitions: int) -> ProgramRecord:
    bit = quantum_bit_type()
    h = bit_gate_function(
        "hadamard", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    builder = ComponentProgramBuilder(Signature(()), (h,))
    q = builder.allocate(dtype=bit)
    for _ in range(repetitions):
        output = builder.call_dict("hadamard", {"q": q})["q"]
        assert isinstance(output, RegisterBinding)
        q = output
    builder.free(q)
    return builder.finalize({})


single_h_graph = hadamard_graph(1)
assert verify_program_record(single_h_graph) == ()
```

`allocate` and `free` record logical lifetime operations. The graph above
contains a single H between allocation and release. Construction checks linear
ownership. It does not prove a released clean qubit has returned to zero. The
source guide also stops at graph construction. Its abbreviated `H`, `qubit=q`
and tuple assignment are pseudocode placeholders. The source-backed acceptance
uses the actual source Hadamard `q` port and a QBit allocation.

The following executable echo uses two H calls, supplies a clean physical work
wire and returns it to zero. Additional circuit wires remain caller-owned.

```python
@dataclass(frozen=True)
class HadamardEcho:
    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        bound = bind_register_program(hadamard_graph(2), (), clean=(0,))
        return circuit.compose(bound.circuit)


circuit = fq.Circuit(1).append(HadamardEcho())
assert [gate.name for gate in circuit.to_ir().instructions] == ["h", "h"]
```

## Definitions, calls and signatures

Definitions follow value equality over their stored immutable fields. Two
references to an equal definition describe the same operation. Two calls to
that definition have separate positions and independent output SSA bindings.
In the echo, the two H calls share one definition but occur in sequence. A
call's ordinal and its operand/result values distinguish its occurrence.

A `Signature` declares names, element dtypes, shapes and direction. `Register`
is metadata. `RegisterBinding` is one owned occurrence of a register element.
For a scalar, the builder returns one binding. For an array, it returns an
ndarray of bindings in row-major element order. `BindingCollection` describes
that union. Named dictionaries such as `builder.inputs` and `call_dict` results
contain one entry per register, with scalar or shaped values as declared.
Individual indexed elements retain separate ownership.

Bindings are opaque to ordinary component code. Obtain them from builder
inputs or returned calls and reuse the returned objects in subsequent calls.
Using one binding twice or reusing a consumed binding fails. Finalization must
account for every live output exactly once. Failed construction can consume an
input prefix, so rebuild after an invalid call. These are the existing builder
failure semantics.

## Word size, arrays and gates

An n-bit word is one binding carrying width metadata. A shape of `(n,)` is an
array of n bindings. A 3-by-3 array of 32-bit words therefore contains nine
bindings and 288 bits. Use `split` and `join` to expose or reassemble the
MSB-first bits inside a word. This preserves high-level word signatures until a
caller requests their bit decomposition.

THRU registers consume and return values. LEFT registers consume inputs only,
and RIGHT registers produce outputs only. Such asymmetric signatures can
express preparation and effects, with numerical interpretation supplied by
Simulation. Naming a function or declaring a signature does not supply its
numerical action. High-level modular exponentiation and phase estimation can
remain hierarchical while lower-level gate definitions expose one-bit elements
and through-registers. Their implementations and models remain with their
existing algorithm and Compiler owners.

## Source mapping and evidence

This guide replaces the complete conceptual content of Q-0145,
`qualtran/_infra/Readme.md`, Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`a6c706ec0d97a999b66da4e51b0c772eaca2514b`.
Copyright Google LLC. SPDX-License-Identifier Apache-2.0.

| Original section | Native responsibility |
| --- | --- |
| Data structure | Core immutable Program, Compiler construction and the executable lifetime examples above |
| Components | Value-equal definitions, occurrence positions, signatures, linear scalar/shaped bindings and meaningful variable names |
| Gates | Arbitrary widths and shapes, asymmetric direction, high-level hierarchy and through-bit gate bodies |

`tests/unit/test_register_component_graph_guide.py` executes the actual Markdown
code, verifies its frozen source hash, compares the normalized single-H source
call chain and complete two-H source primitive stream, and checks full complex
spectator amplitudes and clean work for the appended echo. The single-H example
claims graph equivalence only. The echo checks full amplitudes and zero-work leakage at an absolute tolerance
of 3e-16, retaining finite-precision Hadamard cancellation. It emits two actual H primitives and uses
one clean work wire, with zero measurements. The source guide supplies no QEC
resource model. The complete 70-cell component tutorial Q-0144 is tracked
separately.

The batch passes 310 concentrated cases, including workspace, builder,
control-extension and basis-recipe consumers. Strict typing passes for the
extracted guide, its test and signed lifetime tests. Black and Ruff pass.
Architecture, machine-contract and 74 IR/API evidence from the preceding
signed-addressing batch is reused because production code and contracts are
unchanged. The additional workspace refusal witness is updated to a malformed
parameterized signed type, and source/native signed allocation/release is
explicitly tested for widths 1, 3 and 8. This repairs a missed test witness from
the preceding batch. Whole default-tier acceptance is not claimed.
