# Locked register bit rotation

`kickmix_bit_rotation_ir` ports `gen_bit_rotate.cc` at Kickmix commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453` in the Qualtran repository.
Its unchanged source blob is `a01c261ed496876753769feaa93ec36e13fc12b4`.
This shared register primitive supplies the source rotation sequence needed
by modular doubling; it is independent of modular arithmetic and execution.

## Interface and ownership

Target and variable shift words are big-endian. A constant signed 64-bit
shift is reduced modulo target width. `right=True` reverses the direction.
Right rotation of the minimum signed 64-bit value is rejected because the
source negates that value with signed overflow.
Variable shift coefficients are quantum wires, caller-defined classical
slots, or literal booleans; the source limits a shift word to 60 bits.
The optional enable is quantum. All quantum ports are distinct. A variable
quantum shift with a quantum enable borrows one initially zero clean wire
and one measurement slot distinct from the input classical slots. It returns
that work wire to zero. Constant and classical shifts need no workspace.
The emitted result slot contains the final erase result, not a logical output.

The constant path retains the two shifted reversals and the native operand
order inside each broadcast SWAP/CSWAP. The variable path visits adjacent
shift coefficients from low to high, temporarily XORs their parity into the
same operand selected by the source, performs the shifted reversal, and
restores the parity carrier. Two quantum controls merge into a clean AND and
use measured AND erasure. Classical shift parity reuses Core's existing XOR
assignment with both old source and old destination values; no classical
execution semantics are implemented in Compiler.

Source resets on the already-clean AND wire are omitted under the clean-input
contract. No other gates or phase corrections are optimized during the port.
The source's empty-target variable path still performs any required parity
and merged-control computation and cleanup, despite having no swaps.

## Verification and resource counts

The native C++ oracle built against the locked source matches 384 effective
streams. Target widths include 0, 1, 2, 3, 4, 5, 8 and 11. Constant shifts
include negative, zero and oversized values; variable words include quantum,
classical, mixed and literal/bit/quantum inputs. Both directions and enable modes are covered.
Register declarations, tags and explicit zero initialization are excluded;
known-zero source resets are normalized, and nested guards are compared as
effective conditions. Every remaining operation and operand is compared in
source order.

The tests enumerate every computational input and every raw HMR history for
the small layouts, certify the signed permutation and zero work, then verify
full complex amplitudes through the local batched executor. Coherent shift
and enable inputs remain entangled with target data. Independent classical shift slots
are restored. Aliased classical coefficients are not covered by the arithmetic
acceptance claim; their in-place parity follows the source. KMX round trips retain the full normalized gate stream.

| Native source layout | SWAP | CSWAP | CCX | CX | CZ | HMR | Native R / emitted R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Four target bits, constant left shift 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| Four target bits, two quantum shift bits and quantum enable | 0 | 4 | 3 | 2 | 3 | 3 | 3 / 0 |

These are emitted primitive counts before any Clifford+T conversion or
measurement-branch selection. The source routine has no separate analytical
cost function. A T-equivalent conversion model must be named explicitly; it
cannot be inferred from an IR that still contains CCX and CSWAP.

The existing example wrapper retains `circuit.append(op)`:

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.bits.bit_rotation import kickmix_bit_rotation_ir
from flagquantum.dynamic import DynamicCircuit

operation = FieldOperation(4, kickmix_bit_rotation_ir((0, 1, 2, 3), 1))
circuit = DynamicCircuit(4)
circuit.append(operation)
```

The port is internal and changes no stable public interface. The modular
doubling consumer remains the next integration step.
