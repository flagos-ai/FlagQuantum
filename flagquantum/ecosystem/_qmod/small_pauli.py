"""Exact compact products for narrow bounded Pauli evolutions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compiler import _pauli_products as products
from .syntax import Expression, Statement, Token, reject

if TYPE_CHECKING:
    from .context import _Adapt


def lower_narrow_pauli_sequence(owner: _Adapt, body: tuple[Statement, ...]) -> bool:
    """Compose only a bounded, consecutive, same-view Suzuki schedule."""
    if (
        not 2 <= len(body) <= 256
        or not owner.pauli_bindings
        or any(
            statement.token.text != "suzuki_trotter"
            or len(statement.arguments) != 5
            or statement != Statement(statement.token, statement.arguments)
            or statement.arguments[0].children
            or not 1
            <= len(
                owner.pauli_bindings.get(
                    statement.arguments[0].token.text, statement.arguments[0]
                ).children
            )
            <= 256
            or statement.arguments[-1].children
            for statement in body
        )
    ):
        return False
    target = body[0].arguments[-1]
    if any(
        statement.arguments[-1].token.text != target.token.text for statement in body
    ):
        return False
    if owner.widths.get(target.token.text) not in {1, 2, 3}:
        return False
    view = owner.array_view(target)
    positions = {wire: index for index, (_, wire) in enumerate(view)}
    matrix = products.identity(1 << len(view))

    def record(
        token: Token,
        operator: Expression,
        time: float,
        destination: Expression,
        order: int,
        repetitions: int,
    ) -> None:
        nonlocal matrix
        _, terms = owner.pauli_terms(token, operator, destination)
        try:
            step = products.pauli_step(
                positions, terms, time / repetitions / order, order
            )
        except ValueError as error:
            reject(token, "angle", str(error))
        matrix = products.product(products.matrix_power(step, repetitions), matrix)

    wires = tuple(reversed(view))
    for start in range(0, len(body), 64):
        matrix = products.identity(1 << len(view))
        for statement in body[start : start + 64]:
            owner.suzuki_trotter(statement.token, statement.arguments, emit=record)
        try:
            matrix = products.roundoff_unitary(matrix)
        except ValueError as error:
            reject(body[start].token, "pauli", str(error))
        entries = tuple(
            tuple((value.real, value.imag) for value in row) for row in matrix
        )
        owner.effect_op(
            "qreg.unitary",
            body[start].token,
            tuple(value for value, _ in wires),
            indices=tuple(index for _, index in wires),
            matrix=entries,
        )
    return True
