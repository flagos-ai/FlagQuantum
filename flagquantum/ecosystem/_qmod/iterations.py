"""Static amplitude-amplification composition in the existing statement language."""

from collections.abc import Mapping
from dataclasses import replace

from .syntax import Expression, Function, Statement, Token, reject


def expand_iterations(definitions: Mapping[str, Function]) -> dict[str, Function]:
    def callback(node: Expression, oracle: bool) -> Function:
        function = definitions.get(node.token.text)
        if node.children or function is None or len(function.parameters) != 1:
            reject(
                node.token,
                "amplification.signature",
                "amplification needs a named single-array callback",
            )
        port = function.parameters[0]
        if (
            port.kind != "qbit[]"
            or port.record is not None
            or port.role != ("const" if oracle else "inout")
        ):
            reject(
                node.token,
                "amplification.signature",
                "oracle needs a Const array and preparation an inplace array",
            )
        return function

    def sign_oracle(function: Function, active: tuple[str, ...] = ()) -> None:
        name = function.name.text
        if name in active or len(active) >= 32:
            reject(
                function.name,
                "amplification.recursion",
                "sign oracle call graph must be acyclic and bounded",
            )
        const = [
            p for p in function.parameters if p.role == "const" and p.kind == "qbit[]"
        ]
        if len(const) != 1 or any(
            p not in const and p.role != "integer" for p in function.parameters
        ):
            reject(
                function.name,
                "amplification.signature",
                "sign oracle cannot capture or export storage",
            )

        def check(body: tuple[Statement, ...]) -> None:
            for statement in body:
                text, args = statement.token.text, statement.arguments
                indexed_z = text == "Z" and len(args) == 1 and args[0].token.text == "["
                phase = text == "phase" and (
                    (
                        len(args) == 1
                        and args[0].token.text == "pi"
                        and not args[0].children
                    )
                    or (
                        len(args) == 2
                        and args[0].token.text == "["
                        and args[1].token.text == "pi"
                        and not args[1].children
                    )
                )
                if statement.control_body is not None:
                    if statement.control_else is not None:
                        reject(
                            statement.token,
                            "amplification.oracle",
                            "sign oracle else branches are outside this profile",
                        )
                    check(statement.control_body)
                elif indexed_z or phase:
                    continue
                elif text in definitions:
                    sign_oracle(definitions[text], (*active, name))
                else:
                    reject(
                        statement.token,
                        "amplification.oracle",
                        "oracle must contain checked pi sign phases; no fractional phase, writes or Output escape",
                    )

        check(function.statements)

    def scope(statements: tuple[Statement, ...]) -> tuple[Statement, ...]:
        result: list[Statement] = []
        for statement in statements:
            if statement.token.text == "amplitude_amplification":
                args = statement.arguments
                if len(args) != 4:
                    reject(
                        statement.token,
                        "amplification.signature",
                        "amplitude_amplification requires repetitions, oracle, preparation and target",
                    )
                reps, oracle, preparation, target = args
                callback(preparation, False)
                sign_oracle(callback(oracle, True))
                if target.children:
                    reject(
                        target.token,
                        "amplification.signature",
                        "amplification target requires a whole named array or bound view",
                    )

                def token(text: str, source: Token = statement.token) -> Token:
                    return Token(text, source.location)

                def call(text: str, *values: Expression) -> Statement:
                    return Statement(token(text), tuple(values))

                a = call(preparation.token.text, target)
                mark = call(oracle.token.text, target)
                x = call("apply_to_all", Expression(token("X")), target)
                pi = Expression(token("pi"))
                flip = Statement(
                    token("control"), (target,), control_body=(call("phase", pi),)
                )
                step = (
                    mark,
                    Statement(token("invert"), (), inverse_body=(a,)),
                    x,
                    flip,
                    x,
                    a,
                    call("phase", pi),
                )
                loop = Statement(
                    token("repeat"),
                    (reps,),
                    repeat_index=token(
                        f"$amplification_{statement.token.location.line}_{statement.token.location.column}"
                    ),
                    repeat_body=step,
                )
                result.extend((a, loop))
            else:

                def optional(
                    body: tuple[Statement, ...] | None,
                ) -> tuple[Statement, ...] | None:
                    return None if body is None else scope(body)

                result.append(
                    replace(
                        statement,
                        qpe_body=optional(statement.qpe_body),
                        within=optional(statement.within),
                        apply=scope(statement.apply),
                        control_body=optional(statement.control_body),
                        control_else=optional(statement.control_else),
                        repeat_body=scope(statement.repeat_body),
                        inverse_body=optional(statement.inverse_body),
                        static_if_body=optional(statement.static_if_body),
                    )
                )
        return tuple(result)

    return {
        name: replace(function, statements=scope(function.statements))
        for name, function in definitions.items()
    }
