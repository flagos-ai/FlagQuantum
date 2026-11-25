"""Sound finite-width facts for exact unsigned expression values.

Masks describe the complete declared domain, never a selected input state.
Multiplication deliberately uses cheap conservative rules rather than expanding
an unbounded Boolean network; general bit-level optimization belongs downstream.
"""

from dataclasses import dataclass

from .ranges import unsigned_width


@dataclass(frozen=True)
class BitFacts:
    width: int
    upper_bound: int
    known_zero: int
    known_one: int

    @property
    def effective_width(self) -> int:
        return unsigned_width(self.upper_bound)

    @property
    def constant(self) -> int | None:
        mask = (1 << self.width) - 1
        return self.known_one if self.known_zero | self.known_one == mask else None

    @property
    def trailing_zeros(self) -> int:
        count = 0
        while count < self.width and self.known_zero & (1 << count):
            count += 1
        return count


def facts(width: int, upper: int, zero: int = 0, one: int = 0) -> BitFacts:
    mask = (1 << width) - 1
    zero |= mask ^ ((1 << unsigned_width(upper)) - 1)
    if upper == 0:
        zero = mask
    if zero & one or (zero | one) & ~mask or upper > mask or one > upper:
        raise ValueError("inconsistent unsigned bit facts")
    return BitFacts(width, upper, zero, one)


def constant_facts(value: int) -> BitFacts:
    width = unsigned_width(value)
    return facts(width, value, ((1 << width) - 1) ^ value, value)


def _choices(value: BitFacts, bit: int) -> tuple[int, ...]:
    if bit >= value.width or value.known_zero & (1 << bit):
        return (0,)
    return (1,) if value.known_one & (1 << bit) else (0, 1)


def add_facts(left: BitFacts, right: BitFacts, width: int, upper: int) -> BitFacts:
    carry = {0}
    zero = one = 0
    for bit in range(width):
        sums = {
            a + b + c
            for a in _choices(left, bit)
            for b in _choices(right, bit)
            for c in carry
        }
        outputs = {value & 1 for value in sums}
        if outputs == {0}:
            zero |= 1 << bit
        elif outputs == {1}:
            one |= 1 << bit
        carry = {value >> 1 for value in sums}
    return facts(width, upper, zero, one)


def multiply_facts(
    left: BitFacts, right: BitFacts, width: int, upper: int, *, square: bool
) -> BitFacts:
    for variable, constant in ((left, right.constant), (right, left.constant)):
        if constant == 0:
            return facts(width, 0)
        if constant is not None and constant > 0 and constant & (constant - 1) == 0:
            shift = constant.bit_length() - 1
            zero = (variable.known_zero << shift) | ((1 << shift) - 1)
            # Zero-extend a narrow known operand before shifting it.
            zero |= ((1 << width) - 1) ^ ((1 << min(width, variable.width + shift)) - 1)
            return facts(
                width, upper, zero & ((1 << width) - 1), variable.known_one << shift
            )
    zero = (1 << min(width, left.trailing_zeros + right.trailing_zeros)) - 1
    one = left.known_one & right.known_one & 1
    if square:
        # n^2 mod 4 is 0 or 1; an odd square is 1 mod 8, for every integer n.
        zero |= (1 << 1) if width > 1 else 0
        if left.known_one & 1 and width > 2:
            zero |= 1 << 2
        zero |= left.known_zero & 1
        one |= left.known_one & 1
    return facts(width, upper, zero, one)
