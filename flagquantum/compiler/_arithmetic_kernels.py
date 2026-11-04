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
    if isinstance(carry_out, bool) or not isinstance(carry_out, int):
        raise TypeError("ripple_add wires must be Python integers")
    return _ripple_add(a, b, carry, carry_out)


def ripple_add_modular(
    a: tuple[int, ...], b: tuple[int, ...], carry: int
) -> tuple[Instruction, ...]:
    """Add a into b modulo its width and restore the clean carry."""
    return _ripple_add(a, b, carry, None)


def _ripple_add(
    a: tuple[int, ...], b: tuple[int, ...], carry: int, carry_out: int | None
) -> tuple[Instruction, ...]:
    if not a or len(a) != len(b):
        raise ValueError("ripple_add requires nonempty equal-width registers")
    wires = (*a, *b, carry, *((carry_out,) if carry_out is not None else ()))
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
    if carry_out is not None:
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

    left, right = _multiplication_bits(a), _multiplication_bits(b)
    _validate_product_maps((*left, *right), product, scratch, len(product))
    for i, control_a in left:
        for j, control_b in right:
            shift = i + j
            accumulator = tuple(reversed(product))[shift:]
            if not accumulator:
                continue  # Modular kernel; the caller proves full precision.
            controls = tuple(
                dict.fromkeys(q for q in (control_a, control_b) if q is not None)
            )
            yield from _controlled_increment(controls, accumulator, scratch)


def _multiplication_bits(
    value: tuple[int, ...] | int,
) -> tuple[tuple[int, int | None], ...]:
    if isinstance(value, bool):
        raise TypeError("multiplication constants must be Python integers")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("multiplication constants must be nonnegative")
        return tuple((i, None) for i in range(value.bit_length()) if value & (1 << i))
    if any(isinstance(q, bool) or not isinstance(q, int) for q in value):
        raise TypeError("multiplication operand labels must be Python integers")
    if not value or len(set(value)) != len(value):
        raise ValueError("multiplication operand maps must be nonempty and injective")
    return tuple(enumerate(reversed(value)))


def square_accumulate(
    operand: tuple[int, ...], product: tuple[int, ...], scratch: tuple[int, ...]
) -> Iterator[Instruction]:
    """Add x*x modulo 2**width using diagonal and doubled cross terms.

    x_i**2=x_i and the two ordered x_i*x_j terms contribute at i+j+1.
    Product may be nonzero; operands survive and initially zero scratch is clean.
    """
    if not isinstance(operand, tuple):
        raise TypeError("square requires a physical operand map")
    bits = _multiplication_bits(operand)
    _validate_product_maps(bits, product, scratch, len(product))
    for i, control in bits:
        assert control is not None
        yield from _controlled_increment(
            (control,), tuple(reversed(product))[2 * i :], scratch
        )
    for position, (i, a) in enumerate(bits):
        for j, b in bits[position + 1 :]:
            assert a is not None and b is not None
            yield from _controlled_increment(
                (a, b), tuple(reversed(product))[i + j + 1 :], scratch
            )


def _validate_product_maps(
    bits: tuple[tuple[int, int | None], ...],
    product: tuple[int, ...],
    scratch: tuple[int, ...],
    required: int,
) -> None:
    operands = tuple(q for _, q in bits if q is not None)
    if not product or len(scratch) != required:
        raise ValueError("arithmetic output/scratch widths disagree")
    allocated = (*product, *scratch)
    if any(
        isinstance(q, bool) or not isinstance(q, int) for q in (*operands, *allocated)
    ):
        raise TypeError("arithmetic qubit labels must be Python integers")
    if min((*operands, *allocated)) < 0 or len(set(allocated)) != len(allocated):
        raise ValueError("arithmetic allocations must be nonnegative and disjoint")
    if set(operands).intersection(allocated):
        raise ValueError("arithmetic allocations overlap an operand")


def _controlled_increment(
    controls: tuple[int, ...], accumulator: tuple[int, ...], scratch: tuple[int, ...]
) -> Iterator[Instruction]:
    if not accumulator:
        return
    prepare = Instruction(("x", "cx", "ccx")[len(controls)], (*controls, scratch[0]))
    yield prepare
    for k in range(1, len(accumulator)):
        yield Instruction("ccx", (scratch[k - 1], accumulator[k - 1], scratch[k]))
    for k, target in enumerate(accumulator):
        yield Instruction("cx", (scratch[k], target))
    for k in reversed(range(1, len(accumulator))):
        yield Instruction("ccx", (scratch[k - 1], accumulator[k - 1], scratch[k]))
        yield Instruction("cx", (scratch[k - 1], scratch[k]))
    yield prepare


def constant_multiply_accumulate(
    operand: tuple[int, ...],
    constant: int,
    product: tuple[int, ...],
    scratch: tuple[int, ...],
) -> Iterator[Instruction]:
    """Shift/add a nonnegative constant with shared Cuccaro sweeps.

    Only zero padding and one carry are needed. Omitting carry-out implements
    modular addition; the Cuccaro inverse sweep restores input and carry.
    This is addition into an accumulator, never an arbitrary-target XOR writer.
    """
    if not isinstance(operand, tuple):
        raise TypeError("constant multiplication requires a physical operand map")
    bits = _multiplication_bits(operand)
    terms = _multiplication_bits(constant)
    required = max(0, len(product) - len(operand)) + 1
    _validate_product_maps(bits, product, scratch, required)
    for shift, _ in terms:
        accumulator = product[: len(product) - shift] if shift < len(product) else ()
        if not accumulator:
            continue
        variable = operand[-len(accumulator) :]
        padding = scratch[: len(accumulator) - len(variable)]
        yield from _ripple_add((*padding, *variable), accumulator, scratch[-1], None)


def xor_add_parts(
    a: tuple[int, ...] | int,
    b: tuple[int, ...] | int,
    target: tuple[int, ...],
    scratch: tuple[int, ...],
) -> tuple[tuple[Instruction, ...], tuple[Instruction, ...]]:
    """Return clean-carry prefix and XOR writes; invert the prefix afterwards.

    c[i+1]=a[i]b[i] XOR a[i]c[i] XOR b[i]c[i]. Target is arbitrary,
    disjoint from operands/zero carries, and must not enter the prefix.
    Equal operand maps implement x+x as a shift with no carry allocation.
    """
    left, right = _multiplication_bits(a), _multiplication_bits(b)
    required = 0 if a == b else len(target) - 1
    _validate_product_maps((*left, *right), target, scratch, required)
    left_bits, right_bits = dict(left), dict(right)
    prefix: list[Instruction] = []
    writes: list[Instruction] = []

    def toggle(
        terms: tuple[tuple[int, int | None], ...], output: int
    ) -> tuple[Instruction, ...]:
        if any(bit == 0 for bit, wire in terms if wire is None):
            return ()
        controls = tuple(dict.fromkeys(wire for _, wire in terms if wire is not None))
        return (Instruction(("x", "cx", "ccx")[len(controls)], (*controls, output)),)

    def bit(bits: dict[int, int | None], index: int) -> tuple[int, int | None]:
        return (1, bits[index]) if index in bits else (0, None)

    for i, output in enumerate(reversed(target)):
        if a == b:
            if i:
                writes.extend(toggle((bit(left_bits, i - 1),), output))
            continue
        x, y = bit(left_bits, i), bit(right_bits, i)
        carry = (1, scratch[i - 1]) if i else (0, None)
        writes.extend(toggle((x,), output))
        writes.extend(toggle((y,), output))
        writes.extend(toggle((carry,), output))
        if i < len(scratch):
            for terms in ((x, y), (x, carry), (y, carry)):
                prefix.extend(toggle(terms, scratch[i]))
    return tuple(prefix), tuple(writes)


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
