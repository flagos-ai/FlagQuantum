# Construction-time circuit composition

`fq.Circuit.compose` continues one circuit with another program placed on chosen
qubits. `fq.Circuit.adjoint` returns a new circuit that undoes one. `fq.Circuit.control`
returns a new circuit that runs one only when a set of added control qubits is all set.
All three are **construction-time** operations: they emit instructions the existing IR
already describes, so `IR_VERSION` stays `"1.0"` and none of them adds a root export. This
guide is about writing programs that use them; the guarantees and every refusal are frozen
in [`contracts/circuit-composition-contract.toml`](../../contracts/circuit-composition-contract.toml).

Two things this guide does not do. It does not describe a second program
representation — a composed circuit *is* a `fq.Circuit`, and `to_ir()` on it is the same
IR as before. And it does not present `power`: that one is recorded as absent in the
contract, see [What is not provided](#what-is-not-provided).

## Placing a program on chosen qubits

A block is an ordinary circuit. Composing it onto a larger circuit, without naming
targets, gives it the identity placement:

```python
import flagquantum as fq

block = fq.Circuit(2).h(0).cx(0, 1)

program = fq.Circuit(4).x(0).compose(block)
print(program.n_qubits)
# 4  -- composition does not resize the receiver
print([item.name for item in program.to_ir().instructions])
# ['x', 'h', 'cx']  -- the receiver's own gate first, then the block's, in order
```

`qubits=` names the target of each qubit of the block, in the block's own order, so a
two-qubit block on a four-qubit circuit moves up. `qubit_map=` is the same statement
written with the block's own labels as keys, which is what you want when the block was
built with a numbering you are not going to count:

```python
import flagquantum as fq

block = fq.Circuit(2).h(0).cx(0, 1)

print(fq.Circuit(4).x(0).compose(block, qubits=(1, 2)).to_ir().instructions[-1].wires)
# (1, 2)  -- the block placed on the two targets named, in the block's own order
print(fq.Circuit(4).compose(block, qubit_map={0: 2, 1: 3}).to_ir().instructions[-1].wires)
# (2, 3)  -- the same statement, keyed by the block's own labels
print(fq.Circuit(4).compose(fq.Circuit(1).h(0), qubits=3).to_ir().instructions[-1].wires)
# (3,)  -- a bare label is a length-one placement, accepted for a one-qubit block
```

`qubits` and `qubit_map` are mutually exclusive, and a map may only name qubits the
block actually has:

```python
import flagquantum as fq
from flagquantum.errors import ValidationError

block = fq.Circuit(2).h(0).cx(0, 1)

try:
    fq.Circuit(4).compose(block, qubits=(1, 2), qubit_map={0: 1, 1: 2})
except TypeError as error:
    print(error)
# Circuit.compose accepts either qubits or qubit_map, not both

try:
    fq.Circuit(4).compose(block, qubit_map={0: 2, 1: 3, 2: 0})
except ValidationError as error:
    print(error)
# Circuit.compose qubit_map names local qubit 2, which a 2-qubit program does not
# have
```

### The map has to be total

A map that omits a qubit of the block is refused too, and this is the refusal worth
understanding rather than merely reading:

```python
import flagquantum as fq
from flagquantum.errors import ValidationError

block = fq.Circuit(2).h(0).cx(0, 1)

try:
    fq.Circuit(4).compose(block, qubit_map={0: 2})
except ValidationError as error:
    print(error)
# Circuit.compose qubit_map must name every local qubit; missing (1,)

try:
    fq.Circuit(4).compose(block, qubits=(1,))
except ValidationError as error:
    print(error)
# Circuit.compose qubits must name all 2 qubit(s) of the composed program, got 1

try:
    fq.Circuit(4).compose(block, qubits=(1, 1))
except ValidationError as error:
    print(error)
# Circuit.compose cannot place two local qubits on qubit 1: (1, 1)

try:
    fq.Circuit(1).compose(fq.Circuit(1, bsz=2).h(0))
except ValidationError as error:
    print(error)
# Circuit.compose cannot mix batch sizes: this circuit has bsz=1, the composed
# program has bsz=2.
```

The reason for the first of those is that the obvious alternative — leaving an unnamed
qubit where it was — turns a short or mistyped map into a **silently misplaced gate** on
a circuit that still runs and still returns a number. The second is its other half: a
`qubits` sequence of the wrong length is the same mistake written as a list. The third
and fourth are the two ways placement could be quietly made to work anyway, by stacking
two block qubits on one target or by broadcasting a batched block into an unbatched
circuit, and neither is done.

Occupied targets, by contrast, are legal. Composition appends, so placing a block on
qubits that already carry instructions adds to them; it does not overwrite and it does
not cancel:

```python
import flagquantum as fq

program = fq.Circuit(1).h(0).compose(fq.Circuit(1).h(0))
print([item.name for item in program.to_ir().instructions])
# ['h', 'h']  -- two Hadamards on one qubit are two instructions, not an identity
```

## Why not a hand-written loop

The result is the program you would have written by hand, instruction by instruction —
that equivalence is what `tests/unit/test_circuit_composition_contract.py` asserts. But
"by hand" is the part that goes wrong, and the first way it goes wrong is a custom
operation:

```python
import flagquantum as fq
import torch

gate = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
block = fq.Circuit(1).any(0, unitary=gate)

placed = fq.Circuit(2).compose(block, qubits=(1,))
print(placed.to_ir().instructions[0].wires)
# (1,)  -- the block's qubit, on the target named
print(type(placed.to_ir().instructions[0].matrix).__name__)
# Tensor  -- the matrix travelled with the instruction, so it still executes

# A loop that re-issues each instruction by name cannot reconstruct that matrix.
try:
    target = fq.Circuit(2)
    for item in block.to_ir().instructions:
        getattr(target, item.name)(*item.wires)
except TypeError as error:
    print(error)
# Circuit.any() missing 1 required keyword-only argument: 'unitary'
```

The second way is a noise channel, and it is quieter. `Circuit.depolarizing` records a
channel, and `is_channel` is the free metadata that says so. Rebuilding each instruction
without that metadata — the field a first draft leaves out — drops it:

```python
import flagquantum as fq

noisy = fq.Circuit(1).depolarizing(0, 0.1)
print(noisy.to_ir().instructions[0].metadata)
# {'is_channel': True}

# `to_ir()` and `Circuit.from_ir` are the public route between a circuit and its IR, so
# the loss is reachable without touching a private attribute: an IR whose channel
# instruction carries no metadata rebuilds into a circuit whose channel is unnamed.
payload = noisy.to_ir().to_dict()
payload["instructions"][0].pop("metadata")
unnamed = fq.Circuit.from_ir(fq.CircuitIR.from_dict(payload))
print(unnamed.to_ir().instructions[0].metadata)
# {}  -- the channel is still in the program, and nothing says it is one

print(fq.Circuit(1).compose(noisy).to_ir().instructions[0].metadata)
# {'is_channel': True}  -- compose copies the whole payload, metadata included
```

The loss is not cosmetic. `is_channel` is read at **40 sites in 24 modules** — the
planner, noise selection, the statevector, MPS and tensor-network simulators, the MPS
and JAX executors, and compiler routing — so the unnamed circuit plans as if it were
noise-free:

```python
import flagquantum as fq

noisy = fq.Circuit(1).depolarizing(0, 0.1)
payload = noisy.to_ir().to_dict()
payload["instructions"][0].pop("metadata")
unnamed = fq.Circuit.from_ir(fq.CircuitIR.from_dict(payload))

composed = fq.Circuit(1).compose(noisy)
print(fq.plan(composed).mode, fq.plan(composed).state_bytes)
# density_matrix 32
print(fq.plan(unnamed).mode, fq.plan(unnamed).state_bytes)
# statevector 16  -- half the memory, and a mode that cannot hold the channel
```

The difference is visible a second time in `adjoint`, which refuses the channel for
being a channel and refuses the unnamed circuit for a different reason entirely:

```python
import flagquantum as fq
from flagquantum.errors import CapabilityError

noisy = fq.Circuit(1).depolarizing(0, 0.1)
payload = noisy.to_ir().to_dict()
payload["instructions"][0].pop("metadata")
unnamed = fq.Circuit.from_ir(fq.CircuitIR.from_dict(payload))

try:
    fq.Circuit(1).compose(noisy).adjoint()
except CapabilityError as error:
    print(error)
# Cannot invert instruction 'depolarizing' on qubits (0,): it is a noise channel,
# which has no unitary inverse.

try:
    unnamed.adjoint()
except CapabilityError as error:
    print(error)
# Cannot invert instruction 'depolarizing' on qubits (0,): its matrix does not
# expose a conjugate transpose.
```

So the rule for hand-written placement is not "be careful". It is that the instruction
payload is `name`, `params`, `matrix`, and `metadata`, all four, and that compose is the
thing that copies all four.

## Undoing a program

`adjoint()` returns a **new** circuit rather than mutating the receiver, so a reusable
block can be inverted and then used a second time without disturbing the first use:

```python
import flagquantum as fq

block = fq.Circuit(2).h(0).s(0).cx(0, 1)
inverse = block.adjoint()

print([item.name for item in block.to_ir().instructions])
# ['h', 's', 'cx']  -- the block is unchanged by having been inverted
print([item.name for item in inverse.to_ir().instructions])
# ['cx', 'sdg', 'h']  -- reverse order, one inverting gate each

program = fq.Circuit(4).compose(block, qubits=(0, 1)).compose(inverse, qubits=(0, 1))
print(len(program.to_ir().instructions))
# 6  -- a block and its inverse are both in the program, ready to cancel when run
```

The qubits of each instruction are unchanged, because the inverse of a gate acts on the
same qubits in the same order. What inverts a gate is declared once, per opcode, in
`flagquantum/core/operator_schema.py`, and read from there: a self-inverse gate is kept,
an angle is negated, and `s`/`t`/`sx` become `sdg`/`tdg`/`sxdg`. 31 of the 35 registered
opcodes invert this way.

A custom operation is inverted through its own matrix rather than through an opcode rule,
because that matrix is what executes. The four opcodes that do not invert are the noise
channels, and a channel is refused as a channel rather than as an unreadable
declaration:

```python
import flagquantum as fq
import torch

gate = torch.tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex64)
print([item.name for item in fq.Circuit(1).any(0, unitary=gate).adjoint().to_ir().instructions])
# ['any']  -- inverted through the matrix, not through an opcode rule

channels = [schema for schema in fq.core.OPERATOR_SCHEMAS.values() if schema.semantic_kind == "channel"]
print(len(fq.core.OPERATOR_SCHEMAS), len(channels))
# 35 4  -- registered opcodes, and the four that cannot be inverted by a rule
```

Three further refusals cover instructions whose inverse does not exist as a fixed gate at
all. The first is a mid-circuit measurement, reachable through the same public IR route:

```python
import flagquantum as fq
from flagquantum.errors import CapabilityError

payload = fq.Circuit(2).h(0).to_ir().to_dict()
payload["instructions"].append(
    {"opcode": "measure", "wires": [0], "params": {}, "matrix": None, "metadata": {"is_dynamic": True}}
)
measured = fq.Circuit.from_ir(fq.CircuitIR.from_dict(payload))

try:
    measured.adjoint()
except CapabilityError as error:
    print(error)
# Cannot invert instruction 'measure' on qubits (0,): it is a dynamic operation,
# which has no fixed inverse.
```

The other two are a reset (`it is a dynamic operation, which has no fixed inverse`) and a
classically conditioned gate (`it is classically conditioned, which has no fixed
inverse`). Each refusal names the reason, so `except CapabilityError` tells you what
happened instead of where.

## Controlling a block

`Circuit.control` turns a block into a conditional one: the receiver runs when every
control qubit is set and is skipped otherwise. The control qubits are **added** to the
circuit rather than taken from it, so their labels must sit outside the receiver's own
range:

```python
import flagquantum as fq

# Run `h` on qubit 0 only when qubits 2 and 3 are both set.
conditional = fq.Circuit(1).h(0).control(2, ctrl_qubits=(2, 3))
print(conditional.n_qubits)
# 4
print([(item.name, item.wires) for item in conditional.to_ir().instructions])
# [('ry', (0,)), ('h', (3,)), ('cphase', (2, 3)), ('h', (3,)), ('cphase', (3, 0)),
#  ('h', (3,)), ('cphase', (2, 3)), ('h', (3,)), ('cphase', (3, 0)), ('cphase', (2, 0)),
#  ('ry', (0,))]
```

The result is a new circuit and the receiver is unchanged, so a block can be controlled
once per call without being consumed. Where the registry declares a controlled form of the
receiver's gate, one control uses it directly, so the common cases stay a single
instruction:

```python
import flagquantum as fq

print([(item.name, item.wires) for item in fq.Circuit(1).x(0).control(1, ctrl_qubits=(1,)).to_ir().instructions])
# [('cx', (1, 0))]
print([(item.name, item.wires) for item in fq.Circuit(2).swap(0, 1).control(1, ctrl_qubits=(2,)).to_ir().instructions])
# [('cswap', (2, 0, 1))]
```

Every emitted instruction is a registered opcode, so a controlled program is still a
program the compiler, the drawer, and the simulator understand, and a symbolic angle stays
symbolic rather than being frozen into a matrix:

```python
import flagquantum as fq

parameter = fq.Parameter("th")
conditional = fq.Circuit(1).rz(0, theta=parameter).control(2, ctrl_qubits=(1, 2))
print(sum(type(item.params["theta"]).__name__ == "ParameterExpression"
          for item in conditional.to_ir().instructions if item.params))
# 4
```

The price is depth, and it is published rather than hidden. A `w`-qubit block under `k`
control qubits is built from a phase ladder of level `k + w - 1`, whose instruction count
grows as `4 * 3 ** (level - 1) - 3`. For a one-qubit block:

| Control qubits | Level | Emitted instructions |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 2 | 9 |
| 3 | 3 | 33 |
| 4 | 4 | 105 |
| 5 | 5 | 321 |
| 6 | 6 | 969 |
| 10 | 10 | 78729 |

The growth is `O(3**level)`, so this is a correctness-first construction: a deep control is
exact and differentiable, and it is not cheap. No ancilla qubit is used, which is the
deliberate trade for that depth. The deepest supported level is 10, and a wider block
spends the budget faster — a three-qubit block reaches level 10 with eight controls, and
`fq.Circuit(3).cswap(0, 1, 2).control(8, ctrl_qubits=tuple(range(3, 11)))` emits 236193
instructions. Beyond that the call refuses by name rather than running for an unbounded
time:

```python
import flagquantum as fq

try:
    fq.Circuit(1).t(0).control(11, ctrl_qubits=tuple(range(1, 12)))
except ValueError as error:
    print(error)
# 11 control qubit(s) on a 1-qubit gate needs a phase ladder of level 11; the deepest
# supported ladder is 10
```

A channel is not a unitary, so conditioning it has no meaning here and the call is refused
rather than approximated:

```python
import flagquantum as fq
from flagquantum.errors import CapabilityError

try:
    fq.Circuit(1).depolarizing(0, 0.1).control(1, ctrl_qubits=(1,))
except CapabilityError as error:
    print(error)
# Cannot control instruction 'depolarizing' on qubits (0,): it is a noise channel,
# which has no controlled form.
```

## What is not provided

`Circuit.power` belongs to the same construction-time family and is **not** provided on
this revision. It is named in the contract as `not_provided`, with the reason, so that its
absence is a recorded gap rather than something you discover by trying:

```python
import flagquantum as fq

print(hasattr(fq.Circuit, "power"))
# False
```

Adding it changes the contract, and the contract names the authority it needs — an
approved API change proposal. Until one exists, a repeated block is written by composing
the block with itself.

## Where the guarantees are written down

| Question | Answered by |
|---|---|
| What does `compose` guarantee about placement? | `[placement]` in `contracts/circuit-composition-contract.toml` |
| What does `adjoint` guarantee about order and qubits? | `[adjoint]` in the same contract |
| What does `control` guarantee about the controls, and what does it cost? | `[control]` in the same contract |
| Which refusals exist, and are they reachable? | `[[refusals]]`, 30 rows, 29 reachable |
| Does a composed circuit equal the hand-built one? | `contract["verification"].expansion_tests`, three named tests |
| Does the contract still describe the code? | `python tools/check_circuit_composition_contract.py` |

The refusal vocabulary is deliberately message-phrase based, because the operations raise
the exception classes the package already exports (`TypeError`, `ValidationError`,
`CapabilityError`). A caller may rely on the class and the phrase. Changing a phrase
changes a documented refusal, and therefore changes the contract.

## Related

- [API change: construction-time composition](../api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md)
- [API change: circuit adjoint](../api-changes/FQ-CIRCUIT-ADJOINT-20261003.md)
- [API change: circuit control](../api-changes/FQ-CIRCUIT-CONTROL-20261022.md)
- [Algorithms at demonstration scale](ALGORITHMS.md)
- [Local simulation, measurement, and training](LOCAL_WORKFLOWS.md)
