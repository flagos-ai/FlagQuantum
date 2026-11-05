"""Conservative basis-preservation checks at the Qmod source boundary."""

from collections.abc import Mapping

from .syntax import Function, Statement, Token, reject

_BASIS_OPERATIONS = frozenset(
    {
        "I",
        "X",
        "Z",
        "S",
        "RZ",
        "CX",
        "CZ",
        "SWAP",
        "allocate",
        "=",
        "^=",
        "+=",
        "->",
        "phase",
    }
)


def verify_permutations(definitions: Mapping[str, Function]) -> None:
    """Keep qperm distinct from qfunc and prove the admitted call graph subset.

    Both branches and inverse/within bodies are checked, without algebraic
    cancellation assumptions. Ownership, initialization and widths remain the
    responsibility of the existing adapter and Compiler verifiers.
    """
    checked: set[str] = set()
    active: set[str] = set()

    def check_parameters(function: Function) -> None:
        if any(port.role == "callable" for port in function.parameters):
            reject(
                function.name,
                "permutation",
                "qperm proof does not admit dynamic callback ports",
            )

    def call(token: Token) -> None:
        name = token.text
        if name in _BASIS_OPERATIONS and name not in definitions:
            return
        if name in active:
            reject(token, "recursion", "qperm call graph must be acyclic")
        if name in checked:
            return
        function = definitions.get(name)
        if function is None or name == "main":
            reject(
                token,
                "permutation",
                f"qperm cannot prove basis preservation of {name!r}",
            )
        check_parameters(function)
        active.add(name)
        body(function.statements)
        active.remove(name)
        checked.add(name)

    def body(statements: tuple[Statement, ...]) -> None:
        for statement in statements:
            if statement.declaration is not None:
                continue
            scopes = (
                statement.within,
                statement.apply,
                statement.control_body,
                statement.control_else,
                statement.repeat_body,
                statement.inverse_body,
                statement.static_if_body,
            )
            if statement.map_callback is not None:
                body(statement.map_callback[1])
            elif statement.token.text == "apply_to_all":
                if not statement.arguments or statement.arguments[0].children:
                    reject(
                        statement.token,
                        "permutation",
                        "qperm map requires a checked callback",
                    )
                call(statement.arguments[0].token)
            elif statement.token.text not in {
                "within",
                "control",
                "repeat",
                "invert",
                "if",
            }:
                call(statement.token)
            if statement.power_callback is not None:
                reject(
                    statement.token,
                    "permutation",
                    "qperm does not admit phase-estimation callbacks",
                )
            for scope in scopes:
                if scope:
                    body(scope)

    for name, function in definitions.items():
        if function.permutation:
            check_parameters(function)
            active.add(name)
            body(function.statements)
            active.remove(name)
            checked.add(name)
