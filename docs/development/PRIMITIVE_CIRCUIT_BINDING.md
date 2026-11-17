# Bind owned primitive snapshots

`compiler/registers/primitives/circuit_binding.py` binds a finite, self-contained
CircuitIR instruction fragment to explicit caller quantum wires and classical
slots. The concrete consumers are imported column and square-root entry oracle
snapshots. Conversion of dependency objects stays in Ecosystem; this helper
accepts only owned IR and allocates no storage.

```python
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.compiler.registers.primitives.circuit_binding import bind_primitive_circuit

source = CircuitIR(2, (
    Instruction("measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}),
    Instruction("x", (1,), metadata={"conditions": ((0, 1),)}),
))
bound = bind_primitive_circuit(source, (5, 2), classical_bits=(7,))
assert bound[0].wires == (5,)
assert bound[1].metadata["conditions"] == ((7, 1),)
```

Quantum and classical maps are separate injective tuples. They cover every
source quantum wire and the source classical slot prefix, including unused
positions in a sparse slot range. Caller slot numbers may be sparse and may
overlap quantum numbers because they are different storage spaces. Maps must
contain distinct nonnegative built-in integers. No slot is silently allocated,
inferred from ambient caller state, or merged with another result.

The binder preserves gate order, parameters, explicit matrices and metadata.
It relocates measurements, classical XOR sources and targets, conjunctions,
disjunctions and existing guarded effects. Conditions are canonically reordered
after arbitrary slot permutations. Classical validation runs before and after
binding, so unresolved external reads are rejected. The source fragment is
unchanged. The helper emits no added gates and makes no resource estimate.

Program-level observables and static result declarations are rejected rather
than dropped from the instruction tuple. Typed Programs with lifetimes,
returned words or classical inputs still use the existing register Program
binder. `Circuit.compose` retains its established quantum-only placement
contract.

Tests compare every raw measurement branch before and after quantum and slot
permutations on a coherent state entangled with spectator wires. A previous
caller measurement stays live at slot zero and drives a later spectator gate.
The fragment exercises classical XOR, AND and OR guards, rotation parameters
and conditional reset. Separate matrix execution checks the exact complex phase
and untouched spectator state. Invalid maps, external classical reads and
program-level results have explicit rejection evidence.

```bash
pytest -q tests/unit/test_primitive_circuit_binding.py
```
