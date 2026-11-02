"""Independent integer mathematics and sparse expected amplitudes.

No compiler IR, arithmetic kernel, source evaluation, or production lowering is
used here. All admitted formulas are nonnegative and coordinatewise monotone;
the maximum at the upper corner therefore gives the complete-domain width.
"""

import cmath
import math
from collections.abc import Mapping
from itertools import product

from .expressions import Case


def evaluate(name: str, values: tuple[int, ...]) -> int:
    """Evaluate the benchmark specification without calling its source function."""
    x = values[0]
    y = values[1] if len(values) > 1 else 0
    z = values[2] if len(values) > 2 else 0
    if name == "add":
        return sum(values)
    if name == "double":
        return 2 * x
    if name in {"square", "power_square"}:
        return x**2
    if name == "multiply":
        return math.prod(values)
    if name == "energy":
        return pow(x, 2) + y
    if name == "multiply_add":
        return math.prod((x, y)) + z
    if name == "shared_square":
        return x**2 + 2 * x * y + y**2
    if name == "mixed_products":
        return x * (x + y)
    if name == "linear":
        return sum((3 * x, 5 * y))
    if name == "overlapping":
        return x**2 + x * y + x * z + y * z
    if name == "zero_product":
        return 0
    if name in {"identity_product", "zero_add"}:
        return x
    if name == "shift_product":
        return x << 3
    if name == "constant_product":
        return sum((x, x, x))
    if name == "disjoint_add":
        return 256 + x
    raise ValueError(f"unknown oracle specification: {name}")


def output_width(case: Case) -> int:
    maximum = evaluate(case.expression.name, tuple((1 << w) - 1 for w in case.widths))
    return max(1, maximum.bit_length())


def expected_amplitudes(
    case: Case,
    registers: Mapping[str, tuple[int, ...]],
    n_wires: int,
    *,
    phased: bool = False,
) -> dict[int, complex]:
    """MSB-first exact support, with every unlisted workspace wire zero.

    The deterministic nonuniform phase pattern is a test input, not an optimizer
    assumption. Refuse exponential oracle construction above six input bits.
    """
    bits = sum(case.widths)
    if bits > 6:
        raise ValueError("complete amplitude oracle is limited to six input bits")
    amplitudes: dict[int, complex] = {}
    for values in product(*(range(1 << w) for w in case.widths)):
        encoded = (*values, evaluate(case.expression.name, values))
        names = (*case.expression.arguments, "result")
        index = 0
        phase = 0.0
        for name, value in zip(names, encoded, strict=True):
            wires = registers[name]
            if value >= 1 << len(wires):
                raise ValueError("declared output truncates the oracle")
            for position, wire in enumerate(wires):
                if value & (1 << (len(wires) - position - 1)):
                    index |= 1 << (n_wires - wire - 1)
                    if phased and name != "result":
                        phase += (wire + 1) * 0.173
        amplitudes[index] = cmath.exp(1j * phase) / math.sqrt(1 << bits)
    return amplitudes
