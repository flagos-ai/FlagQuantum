# Locked near-power-of-two doubling and halving

`kickmix_near_power_doubling_ir` ports both routines in
`gen_idouble_mod_approx.cc` at Kickmix commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. The unmodified source blob is
`dd0c9047465a74b422aae35957dadde24317b33c`.

## Interface and source order

Words use big-endian ports. The modulus is nonempty and its lowest coefficient
is literal `True`. Other coefficients may be booleans, quantum wires or
caller-defined classical slots. Remove only leading literal `True` values
when computing `modulus_cutoff`; runtime one-valued inputs do not reduce it.
The source then sets `bottom_span = cutoff + ceil(btol)`. It requires
`cutoff + btol <= modulus_width` and at least `bottom_span` clean wires.

The forward body performs adjacent SWAPs in the original order, then subtracts
from target bits one through `bottom_span-1` in source little-endian indexing.
The target's low bit controls the mixed subtraction, with `carry_in=True`.
The halving body performs the corresponding mixed addition before the reverse
adjacent-SWAP sequence. The full `modulus.skip(1)` offset is passed through,
even when it is wider than the clipped arithmetic target. The source's
adjacent SWAPs are not replaced by the shared two-reversal rotation, whose
permutation agrees but whose primitive order differs.

The caller owns clean work, borrowed dirty wires and fresh result slots.
Quantum ports are disjoint. Classical modulus inputs are supplied by the caller
and restored. The mixed adder's actual workspace and slot requirements also
apply; supplying the top-level span requirement does not bypass those checks.
Known-zero leaf resets use the existing clean-input convention.

Empty target/modulus words, nonfinite tolerance, and zero/negative or
out-of-target clipped spans are rejected before invoking the source's unsafe
span operations. A finite negative tolerance is retained when the resulting
span is valid, following the source's checks. Arithmetic input/output range
claims require the caller's appropriate near-power-of-two problem; the port
itself retains the precise clipped permutation.

## Numerical and gate-flow evidence

The native C++ oracle matches 1,088 normalized effective streams: 1,056 small
cases and 32 large cases. The latter include 256 target bits, ten mixed low
modulus coefficients, thirty bits of tolerance and the source test's large
clean workspace. Both directions and space preferences, constant, classical,
quantum and mixed modulus inputs, fractional tolerances, and different clean
counts are included. Tags/register declarations and implicit zero
initialization are excluded; known-zero source resets are normalized. All
remaining ordered gates, operands and feedback agree.

Tests certify every raw HMR history for all basis inputs in the selected
small layouts, including arbitrary borrowed work. Full complex-state tests
cover coherent modulus bits and entangled work. Independent clipped-word
arithmetic predicts the permutation; the two separately emitted directions
restore the original full complex state. The eight-bit vented case includes
an offset wider than its four-bit arithmetic window. It preserves both
borrowed carry wires and clears the clean work. KMX round trips preserve the
normalized stream and classical feedback.

The clipped-word rule tested independently is: rotate left once, then update
bits 1 through `bottom_span-1` by subtracting
`((modulus >> 1) + 1) * low_bit` modulo that window's capacity. The inverse
adds the same value before rotating right. This describes the source's
approximate permutation, not a replacement algorithm used by the compiler.

A native primitive fixture with four target bits, modulus 13 and tolerance
zero maps input 7 to 14 with zero phase. FlagQuantum emits exactly the same
six primitives and returns the same state. Exact modular doubling would
return 1; the vector error is `sqrt(2)`. This source approximation is retained.
The existing mixed-adder extra-clean-prefix source defect remains a dependency
boundary; gate-flow matches alone do not certify correct arithmetic for every
such configuration.

## Counts and model basis

| Width / modulus / tolerance / direction | SWAP | CCX | CX | X | HMR | Native R / emitted R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 / 15 / 1 / forward | 3 | 0 | 0 | 0 | 0 | 0 / 0 |
| 4 / 15 / 2.5 / forward | 3 | 1 | 0 | 8 | 1 | 1 / 0 |
| 4 / 13 / 0 / forward | 3 | 0 | 1 | 2 | 0 | 0 / 0 |
| 8 / 251 / 2 / forward, vented | 7 | 8 | 19 | 16 | 2 | 2 / 0 |
| 8 / 251 / 2 / inverse, vented | 7 | 8 | 19 | 8 | 2 | 2 / 0 |

The vented cases additionally emit one CZ, three Z, a classical inversion,
and one explicit zero initialization for the native implicitly zero vent
slot. These are primitive counts before branch selection or Clifford+T
conversion. The locked routine supplies no independent analytical formula;
no T-equivalent estimate or executed-gate measurement is inferred from these
undecomposed CCX/SWAP counts.

## Existing append interface

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.modular.near_power_doubling import (
    kickmix_near_power_doubling_ir,
)
from flagquantum.dynamic import DynamicCircuit

operation = FieldOperation(8, kickmix_near_power_doubling_ir(
    (0, 1, 2, 3), (True, True, False, True),
    btol=2, clean=(4, 5, 6, 7), classical_bits=(0, 1, 2, 3),
))
circuit = DynamicCircuit(8)
circuit.append(operation)
```

This compiler-owned primitive changes no stable public API.
