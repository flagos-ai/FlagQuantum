"""Pure reversible arithmetic instructions shared by compiler and algorithms.

Registers are MSB first. The caller supplies disjoint physical qubits and zero
carry/carry-out. The Cuccaro sweep restores carry, preserves a, and leaves the
low sum in b and its high bit in carry_out. No execution objects enter here.
"""

from collections.abc import Iterator

from ..core.ir import Instruction


def ripple_add(
    a: tuple[int, ...], b: tuple[int, ...], carry: int, carry_out: int
) -> tuple[Instruction, ...]:
    """Emit the existing exact 8*n+1 gate adder on explicit disjoint qubit maps."""
    if not a or len(a) != len(b):
        raise ValueError("ripple_add requires nonempty equal-width registers")
    wires = (*a, *b, carry, carry_out)
    if any(isinstance(w, bool) or not isinstance(w, int) for w in wires):
        raise TypeError("ripple_add wires must be Python integers")
    if min(wires) < 0 or len(set(wires)) != len(wires):
        raise ValueError("ripple_add wires must be nonnegative and disjoint")
    instructions: list[Instruction] = []
    for left, right in zip(reversed(a), reversed(b), strict=True):
        instructions.extend(
            (
                Instruction("cx", (carry, left)),
                Instruction("cx", (carry, right)),
                Instruction("ccx", (left, right, carry)),
            )
        )
    instructions.append(Instruction("cx", (carry, carry_out)))
    for left, right in zip(a, b, strict=True):
        instructions.extend(
            (
                Instruction("ccx", (left, right, carry)),
                Instruction("cx", (carry, right)),
                Instruction("cx", (carry, left)),
                Instruction("cx", (left, right)),
                Instruction("cx", (carry, right)),
            )
        )
    return tuple(instructions)


def multiply_accumulate(
    a: tuple[int, ...] | int,
    b: tuple[int, ...] | int,
    product: tuple[int, ...],
    scratch: tuple[int, ...],
) -> Iterator[Instruction]:
    """Add a*b modulo 2**len(product), returning zero-initialized scratch clean.

    Binary partial products control increments of the accumulator. Repeated
    operand qubits are legal across a/b; identical Boolean controls collapse to
    one control. Product and scratch must be disjoint from every operand and each
    other. Scratch has one qubit per product bit. Emission is streamed so the
    caller can stop at its operation budget without materializing a wide circuit.
    """

    def bits(value: tuple[int, ...] | int) -> tuple[tuple[int, int | None], ...]:
        if isinstance(value, bool):
            raise TypeError("multiplication constants must be Python integers")
        if isinstance(value, int):
            if value < 0:
                raise ValueError("multiplication constants must be nonnegative")
            return tuple(
                (i, None) for i in range(value.bit_length()) if value & (1 << i)
            )
        if any(isinstance(q, bool) or not isinstance(q, int) for q in value):
            raise TypeError("multiplication operand labels must be Python integers")
        if not value or len(set(value)) != len(value):
            raise ValueError(
                "multiplication operand maps must be nonempty and injective"
            )
        return tuple(enumerate(reversed(value)))

    left, right = bits(a), bits(b)
    operands = tuple(q for _, q in (*left, *right) if q is not None)
    if not product or len(scratch) != len(product):
        raise ValueError(
            "multiplication requires equally wide product and scratch maps"
        )
    allocated = (*product, *scratch)
    if any(isinstance(q, bool) or not isinstance(q, int) for q in allocated):
        raise TypeError("multiplication qubit labels must be Python integers")
    if min((*operands, *allocated)) < 0 or len(set(allocated)) != len(allocated):
        raise ValueError("multiplication allocations must be nonnegative and disjoint")
    if set(operands).intersection(allocated):
        raise ValueError("multiplication allocations overlap an operand")
    for i, control_a in left:
        for j, control_b in right:
            shift = i + j
            accumulator = tuple(reversed(product))[shift:]
            if not accumulator:
                continue  # Modular kernel; the caller proves full precision.
            controls = tuple(
                dict.fromkeys(q for q in (control_a, control_b) if q is not None)
            )
            prepare = Instruction(
                ("x", "cx", "ccx")[len(controls)], (*controls, scratch[0])
            )
            yield prepare
            # c[k+1] = c[k] & old_product[k]. Overflow is modular, not measured.
            for k in range(1, len(accumulator)):
                yield Instruction(
                    "ccx", (scratch[k - 1], accumulator[k - 1], scratch[k])
                )
            for k, target in enumerate(accumulator):
                yield Instruction("cx", (scratch[k], target))
            # old_bit = new_bit XOR carry. The two gates erase each saved carry
            # after the accumulator changes, without erasing its input controls.
            for k in reversed(range(1, len(accumulator))):
                yield Instruction(
                    "ccx", (scratch[k - 1], accumulator[k - 1], scratch[k])
                )
                yield Instruction("cx", (scratch[k - 1], scratch[k]))
            yield prepare
