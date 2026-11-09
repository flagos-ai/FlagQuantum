# Locked modular doubling dispatch

`kickmix_modular_doubling_program` ports the space/tolerance dispatch and
fallback bodies from Kickmix `gen_imul2_mod.cc` at commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. The source blob remains
`f2c07c7825e05cd44bd2f793c0c8a826cf426cb5`.

## Ownership and dispatch

The target/modulus/enable contract is shared with the exact doubling module.
Words are big-endian, the literal modulus endpoints are `True`, and arithmetic
inputs are in `[0, modulus)`. Clean work starts and ends at zero; borrowed dirty
work is restored. Classical modulus slots are declared inputs in the returned
compiler artifact. Local execution binds that artifact without changing the
existing append pattern.

The source increments forward tolerance before selecting its exact path.
The inverse first clamps tolerance at zero and makes its own exact-path
selection. Both dispatches honor the space preference. The shared exact
implementation is selected without changing its emitted stream. A fallback
uses the original half-modulus complement, high-zero clipping, mixed adder,
comparison and measured erase calls. A fallback needs its callees' actual
workspace in addition to one forward flag or two inverse flags.

Forward tolerance is capped at the word width. The inverse retains the
source's additional `+1` only when its leading-one test is at most the
requested tolerance plus twenty. A 32-bit, near-power-of-two modulus probe
covers both outcomes of that source heuristic. No tolerance or workspace
policy is inferred from a desired gate count.

The inverse's final phase repair is inside the saved outcome of its first
flag. That outcome occupies one native result slot; the nested comparison
starts at the next slot. The compiled artifact preserves both the saved
condition and the distinct inner feedback slots. No classical guard is
silently overwritten by its nested comparison.

## Evidence and scope

Native C++ generation against the unchanged source matches 1,068 effective
streams: 1,036 small cases and 32 large-word cases. These cover both
directions and enable modes, constant/classical/quantum/mixed modulus
coefficients, several workspace counts, finite and infinite tolerances, and
both space preferences. Register declarations and tags are omitted; known-zero
leaf resets and native implicit classical initialization are normalized.
All remaining operations, operands and effective nested guards agree.

The same matrix also rejects 278 insufficient-work configurations before
lowering: 186 missing the mixed adder's two clean wires, 54 missing the
low-space comparator's three clean wires, 28 missing full quantum-comparison
work, and ten missing the exact controlled body's extra flag. These are not
counted as successful source-stream matches. Existing comparator bounds and
source extra-clean mixed-adder defects remain dependency limitations.

At full tolerance, the space-fallback tests certify every raw HMR history for
valid small inputs, including borrowed work in arbitrary basis values. Full
complex states through five target bits preserve coherent enable, dirty work
and classical modulus inputs. A coherent quantum modulus is tested through
three bits, including a runtime all-ones value represented by a quantum bit.
KMX round trips retain the declared input slots and nested scopes.

Finite tolerance is an approximation. One committed native primitive fixture
has four target bits, modulus 13, a quantum enable, five clean wires,
`btol=0` and the space preference enabled. Both native Kickmix and FlagQuantum
map input 7 to 14 with zero residual phase and zero work; exact modular
doubling would give 1. The complete low-precision signed map is certified in
all measurement histories. The basis-state vector error against the exact
map is `sqrt(2)`. This native approximation is preserved, not called an exact
arithmetic failure introduced by the port.

A literal all-ones constant makes the complemented half-modulus empty in the
fallback. The source reads the end of that empty span/vector: the tested
forward generator returned a stream despite the undefined access, while the
inverse process exited with signal 10. The compiler rejects this configuration
explicitly. The exact entry point remains usable for that constant modulus.
The unmodified source is not patched to obtain an agreement claim.

## Counts and model distinction

For the four-bit modulus-13 low-precision fixture, the native source emits two
CCX, five CX, three CSWAP, two CZ, two X, two HMR and two known-zero R.
The normalized artifact keeps all of those except the two known-zero resets.
This is a primitive count, not a Clifford+T count or an execution-time measure.
The locked routine supplies no independent analytical formula; no model
estimate is invented from the emitted stream. The exact, fallback and
Clifford+T conversion bases must remain separate.

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.modular.modular_doubling_dispatch import (
    kickmix_modular_doubling_program,
)
from flagquantum.dynamic import DynamicCircuit

program = kickmix_modular_doubling_program(
    (0, 1, 2), (True, False, True), control=3,
    clean=(4, 5, 6, 7, 8), minimize_qubits=True,
)
circuit = DynamicCircuit(9)
circuit.append(FieldOperation(9, bind_reversible_phase_execution(program)))
```

This internal compiler slice introduces no stable API change.
