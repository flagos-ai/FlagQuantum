# Conditional add/subtract through shared register primitives

This slice retains Qualtran v0.7.0's original
[`ControlledAddOrSubtract`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py)
program, commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
Google copyright and Apache-2.0 source attribution remain in the compiler.

## Interface and ownership

`compiler/registers/arithmetic/addition.py::measurement_add_or_subtract_ir` owns native IR
emission only. Inputs are a coherent control and big-endian bit-pattern registers
`a`, `b`, where `1 <= len(a) <= len(b)`. The caller supplies `len(b)-1` clean
carry wires and distinct nonnegative classical slots. Quantum wires must be
distinct. Inputs `a` and the control are preserved; `b` may be arbitrary. All
carry work returns to zero. No runtime, dtype conversion or hidden allocation
is introduced; no stable root API changes.

The existing default `add_when_ctrl_is_on=False` is preserved for THC consumers.
Explicit `True` adds on control one and subtracts on zero; `False` does the
opposite. Flipping this flag implements the **original adjoint**, which is
another complement/Add/complement program, not a substituted Add-adjoint
program. The two controlled complement shells now call shared
`controlled_xor_constant_ir` with an all-ones mask. The existing fully unitary
shell uses the same helper, retaining its existing equal-width ripple kernel.
Their gate streams are unchanged on valid layouts.

The source call chain is controlled BitwiseNot, ordinary Add, controlled
BitwiseNot, with outer control X gates only for the add-on-one variant. There
are no control-load Ands. Original composite call counts and full primitive
streams, operands, output ports and measurement feedback are independently
checked for both variants and their source adjoints.

## Numeric domains and reference exceptions

QUInt, QInt and QMontgomeryUInt source dtypes are compared without changing the
reference. Equal-width two's-complement words implement modular signed addition
and subtraction correctly. Montgomery words here are ordinary bit-pattern
arithmetic modulo the register width, not reduction modulo a Montgomery modulus.

For **narrow signed `a`**, the pinned Add decomposition zero-extends its bit
pattern. This differs from signed extension in the original high-level classical
Add model. Example: `QInt(1)` a=-1, `QInt(2)` b=0, ctrl=1, add-on-one gives -1
in the source composite classical model, but both original expanded gates and
native IR produce pattern 01 (+1). The original gates match exactly; mathematical
signed widening is not claimed. This source discrepancy is recorded separately
in the integration ledger. Do not silently repair it in a source parity test.

At one bit the original QEC model undercounts Clifford gates by four, inherited
from Add's `6*n-9` logical cost formula. T and measurement counts remain zero;
measured gate counts are preserved separately.

## Correctness and resources

Tests compare all three source dtypes and eight width pairs, both control
variants and original adjoints. Complete bit-pattern inputs are checked for the
listed cases through five target bits; eight-bit checks use explicit boundaries.
All these inputs have a compositional certificate for every measurement history.
Five small width pairs additionally execute every raw measurement branch on
complete complex superpositions of control, both words and an entangled
spectator. They check unnormalized probabilities, phases, zero work and recovery
under the opposite-polarity original adjoint.

| Equal-width eight-bit variant | Measured T | Measured CX | Measured X | Measurements / resets | Clean quantum work | Total qubits | Classical slots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Add on zero | 28 | 97 | 0 | 7 / 7 | 7 | 24 | 7 |
| Add on one | 28 | 97 | 2 | 7 / 7 | 7 | 24 | 7 |

Separately, source QEC estimates 28 T and 7 measurements for either variant;
source QubitCount estimates 24 qubits. Component records contain one Add and
two controlled BitwiseNots. Source logical Clifford costs and expanded primitive
CX/H/S/feedback CZ counts are distinct quantities. The 28 T is ordinary Add's
cost; a fully controlled Add has a different contract and resource cost.

Run the native user example, without Qualtran or NumPy:

```bash
python -m examples.algorithms.register_controlled_add_or_subtract
```

It uses `circuit.append(ConditionalArithmetic())`, with a coherent control and
arbitrary three-bit words, then appends the original opposite-polarity inverse.
It checks complex amplitudes, two clean carry wires, restored inputs and separate
measured primitive counts and source estimates.

Local acceptance: 83 new tests and 2,103 combined regressions pass without
skips, including eight direct THC QROM/phase-gradient consumer cases. The final
83 focused cases pass after the shared-helper replacement test was added.
Production arithmetic statement coverage is 100%. Strict mypy passes on all
962 production files and the example; minimum/current Black, Ruff, scope,
architecture and multi-level IR checks pass. All three original source Git
blobs match the immutable inventory.
