"""Exact bounded truth evaluation for sparse unsigned phase predicates."""

from collections.abc import Callable
from operator import and_, eq, ge, gt, le, lt, ne, or_, xor
from typing import NoReturn, TypeVar

from ._phase_polynomial import Location

_Node = TypeVar("_Node")


def sparse_predicate(
    expression: _Node,
    quantum_word: Callable[[_Node], tuple[Location, ...] | None],
    number: Callable[[_Node], float],
    *,
    operator_of: Callable[[_Node], str],
    operands_of: Callable[[_Node], tuple[_Node, ...]],
    fail: Callable[[_Node, str, str], NoReturn],
) -> tuple[tuple[int, ...], tuple[Location, ...]]:
    """Return at most 32 exact minterms, or one all-true sentinel (-1).

    Compile the syntax once to a bounded postorder integer tape. Every assignment
    evaluates the same tape, including cancelled reads and both Boolean branches.
    No host function, dynamic indexing or Python evaluation is invoked.
    """
    reads: list[Location] = []
    tape: list[tuple[str, tuple[int, ...]]] = []
    count = 0

    def compile_node(node: _Node, depth: int) -> int:
        nonlocal count
        count += 1
        if count > 512 or depth >= 128:
            fail(
                node,
                "budget.phase",
                "sparse predicate syntax exceeds 512 nodes/128 depth",
            )
        word = quantum_word(node)
        if word is not None:
            for location in word:
                if location not in reads:
                    reads.append(location)
            if len(reads) > 12:
                fail(node, "budget.phase", "sparse predicate exceeds twelve input bits")
            item = "word", tuple(reads.index(location) for location in word)
        else:
            children = operands_of(node)
            operator = operator_of(node)
            if not children:
                value = number(node)
                if not value.is_integer() or abs(value) > 2**31:
                    fail(
                        node,
                        "phase",
                        "sparse predicate constants must be bounded integers",
                    )
                item = "constant", (int(value),)
            else:
                if operator not in {
                    "+",
                    "-",
                    "*",
                    "**",
                    "%",
                    "&",
                    "|",
                    "^",
                    "~",
                    "==",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                }:
                    fail(node, "phase", "unsupported sparse predicate operator")
                if (
                    len(children) not in {1, 2}
                    or len(children) == 1
                    and operator not in {"-", "~"}
                ):
                    fail(node, "phase", "invalid sparse predicate operands")
                if operator in {"**", "%"}:
                    if (
                        operands_of(children[1])
                        or quantum_word(children[1]) is not None
                    ):
                        fail(node, "phase", "use a literal exponent or modulo divisor")
                    literal = operator_of(children[1])
                    if not literal.isdecimal():
                        fail(node, "phase", "use a literal exponent or modulo divisor")
                    value = int(literal)
                    if operator == "**" and not 0 <= value <= 8:
                        fail(node, "phase", "sparse predicate exponent must fit 0..8")
                    if operator == "%" and (value < 1 or value & (value - 1)):
                        fail(
                            node,
                            "phase",
                            "modulo divisor must be a positive power of two",
                        )
                item = operator, tuple(
                    compile_node(child, depth + 1) for child in children
                )
        tape.append(item)
        return len(tape) - 1

    compile_node(expression, 0)
    if not reads:
        fail(expression, "phase", "sparse predicate requires quantum inputs")
    comparisons = {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}
    bitwise = {"&": and_, "|": or_, "^": xor}
    truth: list[int] = []
    for assignment in range(1 << len(reads)):
        values: list[int] = []
        for operator, operands in tape:
            if operator == "word":
                result = sum(
                    ((assignment >> bit) & 1) << i for i, bit in enumerate(operands)
                )
            elif operator == "constant":
                result = operands[0]
            else:
                a = values[operands[0]]
                if len(operands) == 1:
                    if operator == "~" and a not in {0, 1}:
                        fail(
                            expression,
                            "phase",
                            "predicate complement requires a Boolean",
                        )
                    result = -a if operator == "-" else 1 - a
                else:
                    b = values[operands[1]]
                    if operator in comparisons:
                        result = int(comparisons[operator](a, b))
                    elif operator in bitwise:
                        if a < 0 or b < 0:
                            fail(
                                expression, "phase", "bitwise operands must be unsigned"
                            )
                        result = bitwise[operator](a, b)
                    elif operator == "+":
                        result = a + b
                    elif operator == "-":
                        result = a - b
                    elif operator == "*":
                        result = a * b
                    elif operator == "**":
                        # Check the size before constructing a large Python integer.
                        if a and abs(a).bit_length() * b > 54:
                            fail(
                                expression,
                                "budget.phase",
                                "sparse predicate integer exceeds 53 bits",
                            )
                        result = a**b
                    else:
                        if a < 0:
                            fail(expression, "phase", "modulo operand must be unsigned")
                        result = a % b
            if abs(result) > 2**53:
                fail(
                    expression,
                    "budget.phase",
                    "sparse predicate integer exceeds 53 bits",
                )
            values.append(result)
        if values[-1] not in {0, 1}:
            fail(expression, "phase", "phase condition must evaluate to zero or one")
        if values[-1]:
            truth.append(assignment)
    if len(truth) == 1 << len(reads):
        return (-1,), tuple(reads)
    if len(truth) > 32:
        fail(
            expression,
            "budget.phase",
            "sparse predicate exceeds 32 satisfying assignments",
        )
    return tuple(truth), tuple(reads)
