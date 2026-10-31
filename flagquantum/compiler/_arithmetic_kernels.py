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


def comparison_xor(
    a: tuple[int, ...] | int,
    b: tuple[int, ...] | int,
    target: int,
    equality: tuple[int, ...],
    scratch: int,
    *,
    operation: str,
) -> tuple[Instruction, ...]:
    """XOR an exact unsigned predicate, restoring operands and zero scratch.

    The shared prefix-equality comparator supports eq/lt/gt. Numeric operands
    are MSB first and zero extended; constants need no physical register.
    Operand aliases are legal, but each register has distinct qubits and every
    target/flag/scratch qubit is disjoint. This emits no relative phase.
    """
    if operation not in {"eq", "lt", "gt"}:
        raise ValueError("comparison operation must be eq, lt or gt")

    def bits(value: tuple[int, ...] | int) -> tuple[tuple[int, int | None], ...]:
        if isinstance(value, bool):
            raise TypeError("comparison constants must be Python integers")
        if isinstance(value, int):
            if value < 0:
                raise ValueError("comparison constants must be nonnegative")
            return tuple((int(bit), None) for bit in f"{value:b}")
        if not value or len(set(value)) != len(value):
            raise ValueError("comparison registers must be nonempty and distinct")
        return tuple((1, wire) for wire in value)

    left, right = bits(a), bits(b)
    width = max(len(left), len(right))
    if len(equality) != width + 1:
        raise ValueError("comparison needs width + 1 zero equality flags")
    operands = tuple(wire for _, wire in (*left, *right) if wire is not None)
    temporary = (*equality, scratch, target)
    wires = (*operands, *temporary)
    if any(isinstance(w, bool) or not isinstance(w, int) for w in wires):
        raise TypeError("comparison qubits must be Python integers")
    if min(wires) < 0 or len(set(temporary)) != len(temporary):
        raise ValueError("comparison temporary qubits must be nonnegative and distinct")
    if set(operands) & set(temporary):
        raise ValueError("comparison operands and temporary qubits must be disjoint")
    left = ((0, None),) * (width - len(left)) + left
    right = ((0, None),) * (width - len(right)) + right
    gates: list[Instruction] = []

    def condition(
        terms: tuple[tuple[tuple[int, int | None], int], ...], dest: int
    ) -> None:
        controls: dict[int, int] = {}
        for (bit, wire), expected in terms:
            if wire is None:
                if bit != expected:
                    return
            elif wire in controls and controls[wire] != expected:
                return
            else:
                controls[wire] = expected
        ordered = tuple(controls)
        negative = tuple(q for q in ordered if controls[q] == 0)
        gates.extend(Instruction("x", (q,)) for q in negative)
        if len(ordered) <= 2:
            gates.append(
                Instruction(("x", "cx", "ccx")[len(ordered)], (*ordered, dest))
            )
        else:
            first = Instruction("ccx", (*ordered[:2], scratch))
            gates.extend(
                (first, Instruction("ccx", (scratch, ordered[2], dest)), first)
            )
        gates.extend(Instruction("x", (q,)) for q in reversed(negative))

    def prefix(index: int) -> None:
        lbit, lwire = left[index]
        rbit, rwire = right[index]
        previous, following = equality[index : index + 2]
        if lwire is not None and rwire is not None and lwire != rwire:
            # Extracted from Algorithms' existing comparator equality step.
            gates.extend(
                (
                    Instruction("cx", (lwire, scratch)),
                    Instruction("cx", (rwire, scratch)),
                    Instruction("x", (scratch,)),
                    Instruction("ccx", (previous, scratch, following)),
                    Instruction("x", (scratch,)),
                    Instruction("cx", (rwire, scratch)),
                    Instruction("cx", (lwire, scratch)),
                )
            )
        elif lwire is not None and rwire is not None:
            condition((((1, previous), 1),), following)
        elif lwire is not None:
            condition((((1, previous), 1), ((1, lwire), rbit)), following)
        elif rwire is not None:
            condition((((1, previous), 1), ((1, rwire), lbit)), following)
        elif lbit == rbit:
            condition((((1, previous), 1),), following)

    gates.append(Instruction("x", (equality[0],)))
    for index in range(width):
        prefix(index)
    if operation == "eq":
        gates.append(Instruction("cx", (equality[-1], target)))
    else:
        expected = (0, 1) if operation == "lt" else (1, 0)
        for index in range(width):
            condition(
                (
                    ((1, equality[index]), 1),
                    (left[index], expected[0]),
                    (right[index], expected[1]),
                ),
                target,
            )
    for index in reversed(range(width)):
        prefix(index)
    gates.append(Instruction("x", (equality[0],)))
    return tuple(gates)
