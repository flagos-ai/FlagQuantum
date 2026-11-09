# Locked exact modular doubling

`kickmix_exact_modular_doubling_ir` ports the exact forward path and
`gen_imul2_inv_mod_with_subcmp_merged_by_inlining` from Kickmix
`gen_imul2_mod.cc` at commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
The unmodified source blob is `f2c07c7825e05cd44bd2f793c0c8a826cf426cb5`.

## Interface and decomposition

Target and modulus have the same nonzero width in big-endian order. The
highest and lowest modulus coefficients must be literal `True`, fixing an
odd full-capacity modulus. Interior coefficients may be constants, quantum
wires or caller-defined classical slots. Target inputs must satisfy
`0 <= x < modulus`. The forward map is `2*x mod modulus`; `inverse=True`
selects the native inline inverse, not an automatically reversed gate list.
An optional quantum enable preserves the input when zero.

At width one the only valid input is zero and no gates are emitted. At width
two the source emits one SWAP or CSWAP. Above width two, the carry workspace
uses `n-2` initially zero wires; a quantum enable needs one more clean flag.
An additional clean wire selects the native measured triple-control leaf.
Without that wire, the source uses four CCX gates and borrows the lowest
available wire ID. This borrowed wire may be live or entangled and is restored.
The fresh result slot must not overlap caller-defined modulus slots.

The forward body calls the shared source-ordered left rotation, ripples
subtraction carries, fixes underflow, then measures and clears carries and
the enable flag. The inline inverse builds its own carries in source order
and ends with the shared right rotation. Known-zero leaf resets are omitted
under the caller's clean-input contract. All remaining gates, operands,
classical feedback and scalar phases follow the source sequence.

The source forward dispatch only checks `clean.size()+2 >= n` before entering
the exact body. A controlled body also takes a flag and can then reject the
workspace. This interface validates the full actual requirement in advance.
Finite-tolerance and minimum-space fallback schedules are not represented by
this exact entry point and remain separate planned slices. No source
approximation is relabeled as an exact result.

## Acceptance evidence

The native C++ oracle built against the unchanged source matches 396
normalized effective streams: widths 1, 2, 3, 4, 5, 6 and 8, odd full-capacity
constant moduli, and quantum, classical or mixed interior modulus bits.
Both enable modes, both directions, and minimum or additional clean work are
covered. Tags, register declarations, explicit zero initialization and
known-zero leaf resets are normalized; all remaining ordered leaves and
feedback conditions match.

Tests enumerate every raw HMR history over valid small inputs. The signed
basis permutation is checked before comparing complete complex states through
the local batched executor. Tested constants reach five bits. Quantum moduli
are in coherent superposition through four bits, with an entangled enable and
spectator. The constant/classical tests also compose the two independently
ported directions and recover the input state. A noncontiguous layout probes
the implicit borrowed helper while it is entangled with input data. KMX round
trips preserve the entire normalized program.

| Width / modulus / enable / clean | Native CCX (each direction) | Native CSWAP | Native HMR | Native R forward / inverse | Emitted R |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 / 5 / none / 1 | 2 | 0 (2 SWAP) | 1 | 1 / 0 | 0 |
| 3 / 5 / quantum / 2 | 6 | 2 | 2 | 2 / 1 | 0 |
| 3 / 5 / quantum / 3 | 4 | 2 | 3 | 3 / 2 | 0 |
| 5 / 19 / quantum / 4 | 12 | 4 | 4 | 4 / 3 | 0 |
| 5 / 19 / quantum / 5 | 10 | 4 | 5 | 5 / 4 | 0 |

These are source primitive and emitted primitive counts. They do not select
a Clifford+T decomposition or claim executed branch counts. The locked
routine has no separate analytical resource formula. A four-T CCX/CSWAP
model, a seven-T exact decomposition and unexpanded primitives are distinct
bases and are not substituted for one another in this table.

## Existing append pattern

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.modular.exact_modular_doubling import (
    kickmix_exact_modular_doubling_ir,
)
from flagquantum.dynamic import DynamicCircuit

ports = (0, 1, 2)
modulus = (True, False, True)  # Modulus five.
circuit = DynamicCircuit(6)
circuit.append(FieldOperation(6, kickmix_exact_modular_doubling_ir(
    ports, modulus, control=3, clean=(4, 5)
)))
```

This is an internal register compiler primitive with no stable API change.
