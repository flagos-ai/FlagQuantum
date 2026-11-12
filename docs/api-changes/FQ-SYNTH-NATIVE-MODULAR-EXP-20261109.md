# Native bounded modular exponentiation

The user authorized native capabilities required for migration of all five
example categories. This unreleased addition exposes
`modular_exp(n, a, x, power)` with the original positional argument order.
Classification is `single_device_fast_path`; `release_authorized=false`.
Stable Core, serialized schemas and numerical execution are unchanged.

For initialized distinct unsigned numeric registers, the operation maps
`|power,x>` to `|power, a**power*x mod n>` when `x<n`; it acts as identity on
`x>=n`. The valid-residue map is a permutation because the base is positive and
coprime to the modulus. Preparing `x=1` therefore computes modular powers without
a second result register. It preserves absolute phase and restores scratch bits.

`n` and `a` use the existing closed static-integer syntax: built-in integer
literals, immutable integer globals and bounded integer expressions, each value
in -4096..4096. No host calls, numeric hooks, real parameter coercion or quantum
constants are executed. The shared register validator requires target width
1..5, exponent width 1..10, `2<=n<=2**target_width`, a positive coprime base,
initialized storage and a writable target. Full mutable numeric helpers can
compose the operation under existing plain-port borrowing rules. Mixed mutable/
Const helpers, partial views and new controlled/adjoint forms are not admitted.

Capture emits existing `qreg.modular_exp`; the shared validator, modular
permutation lowering, workspace accounting and resource ceilings retain ownership.
No implementation is copied from the Qmod adapter and no synthesis/runtime path
loads source files. The new export has a narrow candidate contract, preserving
the existing API baselines.

```python
from typing import Annotated

import flagquantum as fq
from flagquantum.synth import (
    Output, QNum, allocate, hadamard_transform, modular_exp, qfunc, synthesize,
)


@qfunc
def main(power: Output[Annotated[QNum, 3]], x: Output[Annotated[QNum, 3]]) -> None:
    x |= 1
    allocate(power)
    hadamard_transform(power)
    modular_exp(5, 4, x, power)


program = synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("power", "x"))))
```

Acceptance covers the unchanged original, native program and full three-cell
Notebook workflow, including seeded local sampling and the source modular
relation assertion. Independent dense permutation Circuits use ordinary integer
`pow` and arithmetic rather than the compiler's adjacent-swap construction.
Checks cover every small basis column, padding states, several coprime bases,
complex coherent inputs, absolute phase, bit order, clean workspace, both
objectives, mutable helper composition, maximum supported widths and resource
refusals. Invalid calls, host effects, overlap, Const/uninitialized targets and
unsupported widths fail before execution. All 216 source identities and frozen
resources remain separately checked.
