# Circuit control of a carried operand

## Decision and authorization

Status: **implemented on this branch, pending review.** This proposal amends the
`[control]` table of `contracts/circuit-composition-contract.toml` and retires one
of its refusals. It adds no method, no keyword argument name, no field, no opcode,
and no root export, and it leaves `IR_VERSION` at `1.0`.

What it changes is a refusal that was frozen by name. Until now,
`Circuit.control` refused **every** instruction that carried its own matrix, on
`flagquantum/core/_composition.py`, with the phrase *"it carries its own matrix,
which no opcode describes controlling"*. That refusal was contracted as
`control_of_matrix`, one of the 32 rows the composition gate pins. This document
authorizes replacing that route with a construction, keeping the refusal count at
32 by narrowing the refusal to the operand a construction cannot read.

Authorization for the base method is
[`FQ-CIRCUIT-CONTROL-20261022.md`](FQ-CIRCUIT-CONTROL-20261022.md), which owns the
signature, the added-qubit rule, the width rule, the ancilla decision and the ladder.
This document does not replace it; the machine-checked `[control]` table simply names
the most recent record that changed its shape, and `authorization` at the top of the
contract now lists both. Rule 8 of `AGENTS.md` ("Treat the Stable Core public API as
protected") is why a refusal phrase change is a proposal rather than an edit:
`[issue_codes].vocabulary_frozen = true`, so a caller may rely on the phrase, and
`contracts/circuit-composition-contract.toml` is a protected integration surface
rather than a team-owned file.

The change is confined to the operand route. A noise channel, a dynamic operation, a
classically conditioned instruction, an instruction carrying execution metadata, and a
request past the ladder ceiling are refused exactly as before, by the same classes and
the same phrases.

## Problem and affected user journey

### The one route a custom operation has was the one route control refused

A custom operation in FlagQuantum is an instruction that carries its own matrix, built
through `Circuit.any` or through `Circuit.gate` with a `matrix=`. That matrix is what
executes; the name is a label. Before this change, controlling such an instruction
raised:

```python
import torch
import flagquantum as fq

s_gate = torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
fq.Circuit(1).any(0, matrix=s_gate).control(1, ctrl_qubits=(1,))
# CapabilityError: Cannot control instruction 'any' on qubits (0,): it carries its own
# matrix, which no opcode describes controlling.
```

So a user who had defined an operation as a matrix — the only way to define one that
is not a registered opcode — could not make it conditional, while a user of the
registered gates could. The gap was not an approximation decision: the refusal named
the mechanism ("no opcode describes controlling") rather than a missing capability.

### The construction the refusal denied is the definition itself

`Circuit.control` is defined by one equation: the receiver runs when every added
control qubit is set and is skipped otherwise. For an instruction that carries an
operand, that equation has a direct matrix form. Write the operand as `U` on `m`
qubits and the control set as `k` qubits; the controlled operand is the block
diagonal that leaves every branch with a clear control untouched and applies `U` on
the single branch where all of them are set:

```text
C^k(U) = diag(I, U)      I is 2^(k+m) - 2^m wide, U is the trailing block
```

The wrap is a block diagonal rather than a permutation because of the orderings
`Instruction` already fixes: the last wire named is the least significant one, and
the controls are written ahead of the operand's own wires. Measured, against the
declared gate the same matrix equals:

```python
import math
import torch
import flagquantum as fq

s_gate = torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
by_matrix = fq.Circuit(1).any(0, matrix=s_gate).control(1, ctrl_qubits=(1,))
print([(item.name, item.wires) for item in by_matrix.to_ir().instructions])
# [('any', (1, 0))]
print(tuple(by_matrix.to_ir().instructions[0].matrix.shape))
# (4, 4)

prepare = lambda: fq.Circuit(2).x(0).h(1)
amplitudes = lambda circuit: [
    complex(round(float(value.real), 3) or 0.0, round(float(value.imag), 3) or 0.0)
    for value in circuit.state().reshape(-1)
]
print(amplitudes(prepare().compose(by_matrix)))
# [0j, 0j, (0.707+0j), 0.707j]
print(amplitudes(prepare().cphase(1, 0, theta=math.pi / 2)))
# [0j, 0j, (0.707+0j), 0.707j]
```

`diag(I, S)` is the declared `cphase` at `pi/2`, so the wrap has a second route to
agree with rather than only its own definition. The in-repo test compares the two
operators at `atol=1e-12`.

### The price is different from the ladder's, so it is stated separately

A declared gate under `k` controls is emitted as a phase ladder whose instruction
count grows as `4 * 3 ** (level - 1) - 3`. A carried operand is not: it is one
instruction carrying a dense `2 ** (k + m)` square matrix. Measured over control
counts one through three:

| Control qubits | Carried operand | Declared one-qubit gate |
|---|---|---|
| 1 | 1 instruction, 4 x 4 | 1 instruction |
| 2 | 1 instruction, 8 x 8 | 9 instructions |
| 3 | 1 instruction, 16 x 16 | 33 instructions |

That is the trade this document records: the operand route spends matrix memory where
the ladder spends depth. It is not a faster ladder, and it is not differentiable
through a symbolic angle — the angle was already frozen into the matrix by the
caller, which is what carrying a matrix means.

### A declared name cannot survive the wrap, and that is the second decision

`Instruction` validates a known opcode's arity, so an `x` may not carry two qubits.
A matrix recorded under a declared name — `Circuit.gate("x", (0,), matrix=...)` is
accepted — therefore cannot keep that name once the operand owns more wires. The
name it takes is `any`, which is the name `Circuit.any` already gives an instruction
whose only description is its matrix. A name the registry does not declare is kept,
so a custom operation stays recognisable:

```python
import torch
import flagquantum as fq

s_gate = torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
renamed = fq.Circuit(1).gate("x", (0,), matrix=s_gate).control(1, ctrl_qubits=(1,))
print([item.name for item in renamed.to_ir().instructions])
# ['any']
```

The alternative — keeping the declared name — was declined because the IR refuses it
at the next stage, so the failure would move from the construction that caused it to
`to_ir()` or to the backend, which is the latest-knowable-stage failure that rule 9 of
`AGENTS.md` forbids.

### What is still refused

A matrix whose shape does not say how many entries one qubit spans cannot be wrapped.
`Instruction` counts entries rather than reading a shape, so a nested Python sequence
is a well-formed operand here; a shape such as `(2, 8)` holds the right number of
entries without being a matrix. Wrapping either would decide which qubit each entry
belongs to, and that decision is a guess:

```python
import flagquantum as fq
from flagquantum.errors import CapabilityError

try:
    fq.Circuit(1).any(0, matrix=[[1, 0], [0, 1j]]).control(1, ctrl_qubits=(1,))
except CapabilityError as error:
    print(error)
# Cannot control instruction 'any' on qubits (0,): its matrix exposes no square
# operand a control can wrap around.
```

This is the row that replaces `control_of_matrix` in the frozen vocabulary, under the
code `control_of_operand_without_a_square_matrix`. The refusal count stays at 32 and
the reachable count stays at 31, so the gate that pins them is a check on this change
rather than something this change relaxes.

## Decision candidates

**Wrapping the operand, rather than decomposing it into registered gates.** A
decomposition would need a synthesis routine for an arbitrary unitary, which is a new
compiler capability and a different product decision; it would also lose the exactness
the caller's matrix already has. Declined.

**Keeping the refusal and closing the gap in the compiler instead.** That would leave
`Circuit.control` unable to express a controlled custom operation while the compiler
could, which is two answers to one question. Declined.

**Lenient shape reading — treat a flat sequence as a square matrix by taking its
square root.** Declined: it would make `(2, 8)` silently mean an 8-qubit operand.

**A new opcode for a controlled matrix.** Declined: the IR already has a field for a
matrix on an instruction, and a new opcode would be an `IR_VERSION` change for a case
the existing field covers.

## The contract change

Three things in `contracts/circuit-composition-contract.toml`:

1. The refusal row `control_of_matrix` becomes
   `control_of_operand_without_a_square_matrix`, with the new phrase and trigger.
2. The `[control]` table gains five keys — `operand_control`,
   `operand_control_matrix`, `operand_control_wires`, `operand_control_name`,
   `operand_control_cost` — so the wrap, the wire order, the rename rule and the cost
   are contractual rather than prose.
3. `[control].authorization` names this document, and `authorization` lists it
   alongside the four existing records.

`tools/check_circuit_composition_contract.py` is extended to measure the wrap rather
than read it back: it wraps `diag(1, i)` under one control, compares the recorded
instruction with a `diag(1, 1, 1, i)` written out in the gate itself, and checks the
wire order and the rename rule. Editing the contract's prose without editing the
measurement now fails in both directions.

## Evidence

| Claim | Command | Result |
| --- | --- | --- |
| The wrap is `diag(I, U)`, entry by entry | `pytest tests/unit/test_circuit_control.py -k wrapped_into_the_block_diagonal` | one instruction, wires `(1, 0)`, matrix equal to `diag(1, 1, 1, i)` |
| The wrap is the definition, at one through four controls | `pytest tests/unit/test_circuit_control.py -k matches_its_definition` | two operands x 4 control counts against the definition oracle, `atol=1e-12` |
| The wrap agrees with the declared gate it equals | `pytest tests/unit/test_circuit_control.py -k declared_gate_it_equals` | `diag(I, S)` against `cphase(pi/2)`, `atol=1e-12` |
| The operand's own arity, not the control count, sets the size | `pytest tests/unit/test_circuit_control.py -k own_arity` | a two-qubit operand under one control records `(8, 8)` on wires `(2, 0, 1)` |
| The cost is one instruction at any control count | `pytest tests/unit/test_circuit_control.py -k costs_one_instruction` | 1 at `k=1,2,3`; the declared one-qubit gate emits 9 at `k=2` |
| A declared name is renamed, an undeclared one is kept | `pytest tests/unit/test_circuit_control.py -k renamed_and_an_undeclared` | `['any']` and `['custom']` |
| Metadata is still refused | `pytest tests/unit/test_circuit_control.py -k execution_metadata_is_refused` | `CapabilityError` naming the dropped metadata |
| An unreadable shape is refused, not guessed | `pytest tests/unit/test_circuit_control.py -k no_square_shape` | `CapabilityError` naming the shape |
| The frozen vocabulary still counts 32 rows, 31 reachable | `python tools/check_circuit_composition_contract.py` | `Circuit composition contract passed` |
| The new bytes are load-bearing | `pytest tests/unit/test_circuit_composition_contract.py -k mutated_contract` | 35 mutations, including `operand_control_matrix` and `operand_control_wires`, each caught by name |
| The guide's quoted transcripts are real | `pytest tests/unit/test_circuit_composition_guide.py` | every block executes and matches its quoted output |

## What this does not change

No root export — `fq.__all__` is unchanged. No new method, argument, field, enum
value, or opcode. `IR_VERSION` stays at `1.0`. `docs/operator_manifest.json` is
unchanged, because the operand route adds no operator declaration. The ladder, the
ancilla decision, `MAX_LADDER_LEVEL`, the one-control partner table and the
`not_available` absence rule are untouched.

The remaining owned gaps in the family are unchanged and still refused by name: a
fractional exponent, because the IR has no instruction form for a matrix root, and a
declared gate beyond the published ladder ceiling.
