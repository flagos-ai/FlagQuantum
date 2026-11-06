"""Compose the checked Qmod QAOA ansatz from existing register effects."""

from typing import TYPE_CHECKING

from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def lower_qaoa_penalty(
    owner: "_Adapt", token: Token, arguments: tuple[Expression, ...]
) -> None:
    if len(arguments) != 4:
        reject(
            token,
            "signature",
            "qaoa_penalty requires width, parameters, Hamiltonian and target",
        )
    size, parameters, operator, target = arguments
    register, terms = owner.pauli_terms(token, operator, target)
    view = owner.array_view(target)
    if owner.static_integer(size) != len(view):
        reject(size.token, "width", "QAOA width must match the target view")
    angles = owner.real_array(parameters)
    for _, wire in view:
        owner.effect_op("qreg.gate", token, (register,), gate="h", index=wire)
    for layer in range(len(angles) // 2):
        gamma, beta = angles[2 * layer : 2 * layer + 2]
        for support, coefficient in terms:
            owner.pauli_exponential(token, register, support, gamma * coefficient)
        for _, wire in view:
            owner.effect_op(
                "qreg.gate", token, (register,), gate="rx", index=wire, angle=beta
            )
