"""Source-retaining arithmetic corpus, independent of the oracle implementation."""

from collections.abc import Callable
from dataclasses import dataclass


def add(x: int, y: int) -> int:
    return x + y


def double(x: int) -> int:
    return x + x


def square(x: int) -> int:
    return x * x


def multiply(x: int, y: int) -> int:
    return x * y


def energy(x: int, y: int) -> int:
    return x * x + y


def multiply_add(x: int, y: int, z: int) -> int:
    return x * y + z


def shared_square(x: int, y: int) -> int:
    return (x + y) * (x + y)


def mixed_products(x: int, y: int) -> int:
    return x * x + x * y


def linear(x: int, y: int) -> int:
    return 3 * x + 5 * y


def overlapping(x: int, y: int, z: int) -> int:
    return (x + y) * (x + z)


def zero_product(x: int) -> int:
    return x * 0


def identity_product(x: int) -> int:
    return x * 1


def shift_product(x: int) -> int:
    return x * 8


def constant_product(x: int) -> int:
    return x * 3


def zero_add(x: int) -> int:
    return x + 0


def disjoint_add(x: int) -> int:
    return x + 256


def power_square(x: int) -> int:
    return x**2


@dataclass(frozen=True)
class Expression:
    name: str
    text: str
    arguments: tuple[str, ...]
    function: Callable[..., int]


EXPRESSIONS = (
    Expression("add", "x+y", ("x", "y"), add),
    Expression("double", "x+x", ("x",), double),
    Expression("square", "x*x", ("x",), square),
    Expression("multiply", "x*y", ("x", "y"), multiply),
    Expression("energy", "x*x+y", ("x", "y"), energy),
    Expression("multiply_add", "x*y+z", ("x", "y", "z"), multiply_add),
    Expression("shared_square", "(x+y)*(x+y)", ("x", "y"), shared_square),
    Expression("mixed_products", "x*x+x*y", ("x", "y"), mixed_products),
    Expression("linear", "3*x+5*y", ("x", "y"), linear),
    Expression("overlapping", "(x+y)*(x+z)", ("x", "y", "z"), overlapping),
)
EDGES = (
    Expression("zero_product", "x*0", ("x",), zero_product),
    Expression("identity_product", "x*1", ("x",), identity_product),
    Expression("shift_product", "x*8", ("x",), shift_product),
    Expression("constant_product", "x*3", ("x",), constant_product),
    Expression("zero_add", "x+0", ("x",), zero_add),
    Expression("disjoint_add", "x+256", ("x",), disjoint_add),
    Expression("power_square", "x*x", ("x",), power_square),
)


@dataclass(frozen=True)
class Case:
    expression: Expression
    widths: tuple[int, ...]

    @property
    def identifier(self) -> str:
        return self.expression.name + "-" + "x".join(map(str, self.widths))


def corpus() -> tuple[Case, ...]:
    """Six uniform domains, two unequal domains, and two edge domains."""
    rows = [
        Case(expression, (width,) * len(expression.arguments))
        for expression in EXPRESSIONS
        for width in (1, 2, 3, 4, 8, 16)
    ]
    rows.extend(
        Case(expression, widths[: len(expression.arguments)])
        for expression in EXPRESSIONS
        if len(expression.arguments) >= 2
        for widths in ((1, 3, 2), (2, 4, 1))
    )
    rows.extend(Case(expression, (width,)) for expression in EDGES for width in (1, 3))
    return tuple(rows)


def square_helper(x: int) -> int:
    return x * x


def helper_energy(x: int, y: int) -> int:
    return square_helper(x) + y


def exchanged_energy(y: int, x: int) -> int:
    return y + x * x
