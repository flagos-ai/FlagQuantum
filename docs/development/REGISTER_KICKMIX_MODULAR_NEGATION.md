# Locked modular negation

The compiler-owned `mixed_modular_negation_program` follows
`gen_ineg_mod.cc` at Kickmix commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453` in the Qualtran repository.
The unmodified source Git blob is
`73017205939c5ca0803f216fa91c9252047af90d`.

## Ports and ownership

Target and modulus coefficients are big-endian. A target value must be in
`[0, modulus)`. The highest modulus coefficient must be literal `True`.
The remaining coefficients may be booleans, quantum wires, or caller-defined
`ClassicalFieldCoefficient` slots. Quantum modulus inputs are preserved.
All target, quantum modulus, clean work and dirty work wires are disjoint.
Borrowed dirty work may be entangled with the logical input and is restored.
Clean work is initially zero and returns to zero. For widths above one, the comparator requires
one clean wire per target bit after reserving the zero-repair flag.

The sequence complements the target, calls the existing mixed adder with
carry-in, computes equality with the modulus, and conditionally XORs the
modulus to repair the zero input. Measuring the flag opens a saved condition
scope around the equality-to-zero phase repair. The existing precise
classical scope and input-slot contracts are reused.

Known-zero resets inside mixed arithmetic leaves are omitted under their
clean-input contract. The explicit zero-repair flag reset is retained. No
other gate, feedback condition or phase is optimized during the port.
The mixed adder's previously recorded extra-clean-prefix source defect
remains a dependency limitation; this port does not repair or alter that
source branch. Values outside the stated modular range have no arithmetic
or cleanup guarantee.

## Gate-flow and numerical evidence

The native C++ oracle compiled against the locked source agrees on 234
normalized effective streams. Cases cover widths 1, 2, 3, 4, 5, 6 and 8;
full-capacity constant moduli and classical/quantum low modulus coefficients;
one to ten clean wires and zero, three or eight dirty wires where the
comparison workspace is sufficient. Tags and register declarations are
excluded, known-zero leaf resets are normalized, and explicit classical zero
initialization is omitted on both sides. Effective nested classical guards,
operation order, operands and phase operations are compared.

The tests certify every measurement/reset history for two-bit modulus 3 and
three-bit modulus 5, including disabled effects and their prior result bits.
Their signed basis map is exactly `x -> -x mod modulus`; all clean work is
zero and spectator data is unchanged. Constant and caller-bound classical
moduli have full complex-state tests through width four. Two- and three-bit
quantum moduli are tested in coherent superposition with entangled borrowed
work. KMX round trips preserve declared classical inputs and the full program.

| Width / modulus / clean work | Native CCX | Native CX | Native CZ | Native HMR | Native R | Artifact R | Logical + clean qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / 3 / 3 | 3 | 5 | 3 | 6 | 6 | 1 | 5 |
| 3 / 5 / 4 | 6 | 6 | 5 | 9 | 9 | 1 | 7 |
| 4 / 11 / 6 | 9 | 10 | 9 | 12 | 12 | 1 | 10 |

These are emitted primitive counts, not executed branch counts or a T
estimate. The locked routine supplies no independent analytical model.
Binding expands HMR into H, measurement and conditional X, the retained R
into X-basis discard, and each scalar minus into its exact XZXZ realization.
For example, the three-bit bound IR has six CCX, six CX, five CZ, ten H,
nine measurements, one reset, 34 X, five Z and three classical XOR operations.
No Clifford+T conversion is selected here, so an unexpanded CCX must not be
reported as zero T cost or silently assigned a four-T conversion.

## Existing append interface

The existing example operation wrapper can bind this compiler program:

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.modular.mixed_modular_negation import (
    mixed_modular_negation_program,
)
from flagquantum.dynamic import DynamicCircuit

program = mixed_modular_negation_program(
    (0, 1), (True, True), clean=(2, 3, 4)
)
circuit = DynamicCircuit(5)
circuit.append(FieldOperation(5, bind_reversible_phase_execution(program)))
```

This is an internal compiler slice and introduces no stable public API.

## Whole-file acceptance

K-0124 through K-0126 cover the implementation above, header blob
`7bcbdd3e5615880b41aeb722eda81b56a55878a2`, and test blob
`39358fd2f847549d874cc7880c1400ca28ea3baf`. The unchanged
`ineg_mod.fuzz` passed at the locked revision. All 17 focused native tests
and 5,420 concentrated regressions passed, along with strict typing, format,
scope, architecture, internal IR and public API checks.

A separately generated fixture adds 30 admitted layouts through width 17,
covering all three modulus operand kinds and zero or eight borrowed wires.
It retains raw source counts and compares every effective quantum
operation, ordered operand, and saved feedback predicate over independent
classical and measurement variables. Known-zero leaf resets and block
marks are counted separately. The fixture was independently reproduced
byte for byte. For width 17 and eight borrowed wires:

| Modulus kind | Source records including marks | Matched effective quantum operations | HMR on both sides | CCX on both sides |
| --- | ---: | ---: | ---: | ---: |
| Constant | 396 | 305 | 51 | 48 |
| Classical | 478 | 387 | 51 | 48 |
| Quantum | 479 | 388 | 51 | 65 |

The locked Simulator ran 64 deterministic basis cases for each operand
kind at widths 8 and 17 on both implementations. All sampled arithmetic
outputs and phases agreed with the modular negation, clean work returned
to zero, and borrowed work and modulus inputs were restored. Exhaustive
complex-amplitude and branch checks remain scoped to the smaller
instances above. The source has no standalone negation benchmark.

Regenerate the source fixture with:

```sh
python tests/unit/fixtures/generate_kickmix_modular_negation_file.py \
  --source-root <locked-kickmix-root> \
  --output tests/unit/fixtures/kickmix_modular_negation_file.json
```
