# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_spin.cc.

"""Format a packed spin hash and the measured operation rate."""

_PREFIX = "pnum kMGT"


def _si2(value: float, *, truncate_hundreds: bool = False) -> str:
    scale = 4
    while value < 1 and scale > 0:
        value *= 1000
        scale -= 1
    while value > 1000 and scale < len(_PREFIX) - 1:
        value /= 1000
        scale += 1
    if 1 <= value < 10:
        digits = f"{int(value)}.{int(value * 10) % 10}"
    elif 10 <= value < 100:
        digits = f" {int(value)}"
    elif 100 <= value < 1000:
        digits = str(int(value / 10) * 10 if truncate_hundreds else int(value))
    else:
        digits = f"{value:g}"
    return f"{digits} {_PREFIX[scale]}"


def format_spin_report(
    packed_hash: int, width: int, operations: int, micros: int
) -> str:
    """Report the source bit order, nominal operation count and local timing."""
    hashed_bits = "".join("_1"[(packed_hash >> bit) & 1] for bit in range(width))
    micros = max(1, micros)
    return (
        f"hash: {hashed_bits}\n"
        f"total simulated operations: {operations}\n"
        f"              time elapsed: {_si2(micros / 1_000_000)}s\n"
        f"{_si2(float(operations * 1_000_000 // micros))}ops/sec\n"
    )
