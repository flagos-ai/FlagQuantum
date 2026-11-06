"""Conservative basis-preservation checks at the Qmod source boundary."""

from collections.abc import Mapping

from .syntax import Function, Statement, Token, reject

_BASIS_OPERATIONS = frozenset(
    {
        "I",
        "IDENTITY",
        "X",
        "Y",
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
        "$kickback_phase",
    }
)


def is_const_cx_permutation(function: Function) -> bool:
    """Recognize static CX networks that read a borrowed array as controls."""
    reads = [port for port in function.parameters if port.role == "const"]
    writes = [port for port in function.parameters if port.role == "inout"]
    if (
        not function.permutation
        or len(reads) != 1
        or reads[0].kind != "qbit[]"
        or reads[0].record is not None
        or len(writes) != 1
        or writes[0].kind != "qbit"
        or any(
            port not in reads + writes and port.role != "integer"
            for port in function.parameters
        )
    ):
        return False
    read, write = reads[0].name.text, writes[0].name.text

    def checked(statement: Statement) -> bool:
        if statement.repeat_index is not None:
            return all(checked(child) for child in statement.repeat_body)
        if statement.static_if_body is not None:
            return all(checked(child) for child in statement.static_if_body)
        if statement.token.text != "CX" or len(statement.arguments) != 2:
            return False
        control, target = statement.arguments
        return (
            control.token.text == "["
            and not control.children[0].children
            and control.children[0].token.text == read
            and not target.children
            and target.token.text == write
        )

    return bool(function.statements) and all(checked(s) for s in function.statements)


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
                "power",
                "invert",
                "if",
            }:
                call(statement.token)
            if statement.power_callback is not None or statement.qpe_body is not None:
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
