"""Defer private computational measurements whose registers are dropped on return."""

from collections.abc import Mapping
from dataclasses import replace

from .syntax import Expression, Function, Statement, reject

_UNARY = frozenset({"H", "X", "Y", "Z", "S", "T"})
_BINARY = frozenset({"CX", "CZ", "SWAP"})


def _plain(statement: Statement) -> bool:
    return statement == Statement(statement.token, statement.arguments)


def _dynamic(statement: Statement) -> bool:
    return statement.token.text in {"$measure", "$measured_if"} or (
        statement.declaration is not None and statement.declaration.kind == "bool"
    )


def _defer_helper(
    function: Function,
) -> tuple[tuple[Statement, ...], frozenset[str]]:
    ports = {port.name.text for port in function.parameters}
    if (
        function.permutation
        or function.disable_const_checks is not None
        or len(ports) != len(function.parameters)
        or any(
            port.role != "inout" or port.kind != "qbit" for port in function.parameters
        )
    ):
        reject(
            function.name,
            "measurement",
            "feedback helper requires distinct writable qbits",
        )

    gates: list[Statement] = []
    flags: set[str] = set()
    measured: dict[str, Expression] = {}
    corrected = False
    for statement in function.statements:
        token = statement.token
        declaration = statement.declaration
        if declaration is not None and declaration.kind == "bool":
            name = declaration.name.text
            if name in ports or name in flags or corrected:
                reject(
                    declaration.name, "measurement", "feedback Boolean must be fresh"
                )
            flags.add(name)
        elif token.text == "$measure":
            flag, qubit = statement.arguments
            if (
                flag.token.text not in flags
                or flag.token.text in measured
                or qubit.token.text not in ports
                or any(qubit.token.text == bit.token.text for bit in measured.values())
                or corrected
            ):
                reject(
                    token,
                    "measurement",
                    "measure one fresh Boolean per distinct qbit before feedback",
                )
            measured[flag.token.text] = qubit
        elif token.text == "$measured_if":
            (condition,) = statement.arguments
            control = measured.get(condition.token.text)
            body = statement.control_body
            if (
                control is None
                or flags != measured.keys()
                or body is None
                or len(body) != 1
            ):
                reject(
                    token,
                    "measurement",
                    "feedback requires one gate and an initialized measurement flag",
                )
            (gate,) = body
            if (
                not _plain(gate)
                or gate.token.text not in {"X", "Z"}
                or len(gate.arguments) != 1
                or gate.arguments[0].children
                or gate.arguments[0].token.text not in ports
                or any(
                    gate.arguments[0].token.text == bit.token.text
                    for bit in measured.values()
                )
            ):
                reject(
                    gate.token,
                    "measurement",
                    "feedback requires X or Z on a disjoint unmeasured port",
                )
            corrected = True
            gates.append(
                Statement(
                    replace(gate.token, text="C" + gate.token.text),
                    (control, *gate.arguments),
                )
            )
        else:
            size = 1 if token.text in _UNARY else 2 if token.text in _BINARY else 0
            if (
                not size
                or measured
                or not _plain(statement)
                or len(statement.arguments) != size
                or any(
                    arg.children or arg.token.text not in ports
                    for arg in statement.arguments
                )
                or len({arg.token.text for arg in statement.arguments}) != size
            ):
                reject(
                    token,
                    "measurement",
                    "feedback helper requires a static gate prefix",
                )
            gates.append(statement)
    if not corrected or flags != measured.keys():
        reject(
            function.name,
            "measurement",
            "feedback helper must use private measurement results",
        )
    return tuple(gates), frozenset(qubit.token.text for qubit in measured.values())


def defer_private_measurements(
    definitions: Mapping[str, Function],
) -> tuple[dict[str, Function], bool]:
    """Inline checked feedback only at a call followed by all required drops."""
    rewritten = dict(definitions)
    helpers = {
        name: _defer_helper(function)
        for name, function in definitions.items()
        if name != "main" and any(_dynamic(item) for item in function.statements)
    }
    main = rewritten["main"]
    if any(_dynamic(item) for item in main.statements):
        reject(
            main.name,
            "measurement",
            "runtime Booleans must stay inside a checked helper",
        )
    if not helpers:
        return rewritten, False
    statements: list[Statement] = []
    consumed: set[str] = set()
    kinds = {port.name.text: port.kind for port in main.parameters}
    locals_: set[str] = set()
    for index, statement in enumerate(main.statements):
        name = statement.token.text
        if name not in helpers:
            if statement.declaration is not None:
                local = statement.declaration
                kinds[local.name.text] = local.kind
                locals_.add(local.name.text)
            statements.append(statement)
            continue
        gates, measured = helpers[name]
        formals = definitions[name].parameters
        if (
            not _plain(statement)
            or name in kinds
            or len(statement.arguments) != len(formals)
            or any(
                arg.children or kinds.get(arg.token.text) != "qbit"
                for arg in statement.arguments
            )
            or len({arg.token.text for arg in statement.arguments}) != len(formals)
        ):
            reject(
                statement.token,
                "measurement",
                "feedback helper needs distinct scalar actuals",
            )
        actuals = {
            port.name.text: arg
            for port, arg in zip(formals, statement.arguments, strict=True)
        }
        drop_names = {actuals[formal].token.text for formal in measured}
        following = main.statements[index + 1 : index + 1 + len(drop_names)]
        drops = {
            item.arguments[0].token.text
            for item in following
            if item.token.text == "drop"
            and _plain(item)
            and len(item.arguments) == 1
            and not item.arguments[0].children
        }
        if drop_names != drops or not drop_names <= locals_:
            reject(
                statement.token,
                "measurement",
                "every measured local must be immediately dropped",
            )
        statements.extend(
            replace(
                gate, arguments=tuple(actuals[arg.token.text] for arg in gate.arguments)
            )
            for gate in gates
        )
        consumed.add(name)
    for name in helpers:
        if name not in consumed:
            reject(
                definitions[name].name,
                "measurement",
                "feedback requires a verified main call",
            )
        rewritten.pop(name)
    rewritten["main"] = replace(main, statements=tuple(statements))
    return rewritten, True
