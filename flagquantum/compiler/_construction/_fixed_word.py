"""Source-neutral exact dyadic arithmetic and bounded fixed-word XOR tables."""

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from operator import add, eq, ge, gt, le, lt, mul, ne, sub
from typing import NoReturn, TypeVar

_Node = TypeVar("_Node")
_Location = TypeVar("_Location")


def exact_expression(
    expression: _Node,
    *,
    name: Callable[[_Node], str | None],
    constant: Callable[[_Node], Fraction | None],
    operator: Callable[[_Node], str],
    children: Callable[[_Node], tuple[_Node, ...]],
    fail: Callable[[_Node, str], NoReturn],
) -> Callable[[Mapping[str, Fraction]], Fraction]:
    """Compile pure dyadic arithmetic without evaluating a host function."""
    count = 0
    comparisons = {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}
    binary: dict[str, Callable[[Fraction, Fraction], Fraction]] = {
        "+": add,
        "-": sub,
        "*": mul,
        "min": min,
        "max": max,
    }

    def checked(value: Fraction, node: _Node) -> Fraction:
        if value.denominator & (value.denominator - 1):
            fail(node, "fixed-point coefficients must be exact binary rationals")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
            fail(node, "fixed-point value exceeds 256 bits")
        return value

    def compile_node(
        node: _Node, depth: int
    ) -> Callable[[Mapping[str, Fraction]], Fraction]:
        nonlocal count
        count += 1
        if count > 512 or depth >= 128:
            fail(node, "fixed-point expression exceeds 512 nodes/128 depth")
        label = name(node)
        if label is not None:
            return lambda inputs: inputs[label]
        literal = constant(node)
        if literal is not None:
            literal = checked(literal, node)
            return lambda inputs: literal
        operands = children(node)
        symbol = operator(node)
        if symbol == "select" and len(operands) == 3:
            condition, yes, no = (compile_node(child, depth + 1) for child in operands)

            def selected(inputs: Mapping[str, Fraction]) -> Fraction:
                decision = condition(inputs)
                if decision not in {0, 1}:
                    fail(node, "fixed-point condition must be Boolean")
                return yes(inputs) if decision else no(inputs)

            return selected
        if symbol in comparisons and len(operands) == 2:
            left, right = (compile_node(child, depth + 1) for child in operands)
            compare = comparisons[symbol]
            return lambda inputs: Fraction(int(compare(left(inputs), right(inputs))))
        if symbol == "-" and len(operands) == 1:
            operand = compile_node(operands[0], depth + 1)
            return lambda inputs: -operand(inputs)
        if symbol in binary and len(operands) == 2:
            left, right = (compile_node(child, depth + 1) for child in operands)
            operation = binary[symbol]
            return lambda inputs: checked(operation(left(inputs), right(inputs)), node)
        fail(node, "unsupported exact fixed-point expression")

    return compile_node(expression, 0)


def fixed_xor_table(
    operands: Mapping[str, tuple[Sequence[_Location], int]],
    targets: Sequence[_Location],
    *,
    target_fraction: int,
    evaluate: Callable[[Mapping[str, Fraction]], Fraction],
    guard: tuple[Sequence[_Location], int] | None = None,
) -> tuple[tuple[_Location, ...], tuple[int, ...]]:
    """Floor at destination significance, mask and XOR into arbitrary target bits."""
    if not 1 <= len(targets) <= 8 or not 0 <= target_fraction <= len(targets):
        raise ValueError(
            "fixed-point XOR target requires 1..8 bits and a fitting scale"
        )
    if len(set(targets)) != len(targets):
        raise ValueError("fixed-point XOR target locations must be distinct")
    reads: list[_Location] = []
    for view, scale in operands.values():
        if not view or len(set(view)) != len(view) or not 0 <= scale <= len(view):
            raise ValueError(
                "fixed-point XOR input requires distinct bits and a fitting scale"
            )
        for location in view:
            if location not in reads:
                reads.append(location)
    if guard is not None:
        view, expected = guard
        if (
            not view
            or len(set(view)) != len(view)
            or not 0 <= expected < 1 << len(view)
        ):
            raise ValueError("fixed-point XOR guard value or locations do not fit")
        for location in view:
            if location not in reads:
                reads.append(location)
    if len(reads) > 7:
        raise ValueError("fixed-point XOR needs at most seven read bits")
    if set(reads).intersection(targets):
        raise ValueError("fixed-point XOR reads must be disjoint from the target")
    positions = {location: bit for bit, location in enumerate(reads)}

    def word(basis: int, view: Sequence[_Location]) -> int:
        return sum(
            (basis >> positions[location] & 1) << bit
            for bit, location in enumerate(view)
        )

    table: list[int] = []
    for basis in range(1 << len(reads)):
        if guard is not None and word(basis, guard[0]) != guard[1]:
            table.append(0)
            continue
        inputs = {
            label: Fraction(word(basis, view), 1 << scale)
            for label, (view, scale) in operands.items()
        }
        value = evaluate(inputs)
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
            raise ValueError("fixed-point intermediate exceeds 256 bits")
        scaled = value * (1 << target_fraction)
        table.append(
            (scaled.numerator // scaled.denominator) & ((1 << len(targets)) - 1)
        )
    return tuple(reads), tuple(table)


def truth_cubes(
    values: Sequence[int], width: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Partition a bounded Boolean table into disjoint constant cofactors."""
    if (
        not 0 <= width <= 7
        or len(values) != 1 << width
        or any(v not in {0, 1} for v in values)
    ):
        raise ValueError("truth XOR requires a Boolean table of at most seven bits")
    cubes: list[tuple[tuple[int, int], ...]] = []

    def walk(
        table: Sequence[int], depth: int, fixed: tuple[tuple[int, int], ...]
    ) -> None:
        if not any(table):
            return
        if all(table):
            cubes.append(fixed)
            return
        half = len(table) // 2
        zero, one = table[:half], table[half:]
        if zero == one:
            walk(zero, depth - 1, fixed)
        else:
            walk(zero, depth - 1, (*fixed, (depth - 1, 0)))
            walk(one, depth - 1, (*fixed, (depth - 1, 1)))

    walk(tuple(values), width, ())
    return tuple(cubes)
