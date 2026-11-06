"""In-place parity and bounded constant XOR over disjoint borrowed bits."""

from typing import TYPE_CHECKING

from .syntax import Expression, Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt
    from .phase import Location


def lower_indexed_parity(
    owner: "_Adapt", token: Token, output: Expression, rhs: Expression
) -> bool:
    if output.token.text != "[" or output.children[0].children:
        return False
    target = owner.gate_target(output)
    parity = 0
    pending = [rhs]
    locations: dict[Location, bool] = {}
    while pending:
        node = pending.pop()
        if node.token.text == "^":
            pending.extend(reversed(node.children))
        elif not node.children and node.token.text in {"0", "1"}:
            parity ^= int(node.token.text)
        elif node.token.text == "[":
            source = owner.gate_target(node)
            if source == target:
                reject(
                    node.token, "alias", "parity target and sources must be disjoint"
                )
            locations[source] = not locations.get(source, False)
        else:
            reject(
                node.token,
                "expression",
                "indexed XOR requires a parity of bits and Boolean literals",
            )

    value, bit = target
    owner.effect_op(
        "qreg.gate", token, (value,), gate="x" if parity else "i", index=bit
    )
    for source, enabled in locations.items():
        if enabled:
            owner.emit_indexed_x(token, (source, target))
        else:
            owner.effect_op(
                "qreg.control_phase",
                token,
                (source[0],),
                indices=(source[1],),
                angle=0.0,
            )
    return True


def lower_masked_word_xor(owner: "_Adapt", statement: Statement) -> bool:
    body = statement.control_body
    guard = statement.arguments[0]
    if (
        body is None
        or len(body) != 1
        or statement.control_else is not None
        or guard.token.text != "=="
        or body[0].token.text != "^="
        or len(body[0].arguments) != 2
    ):
        return False
    word, pattern = guard.children
    target, constant = body[0].arguments
    if any(node.children for node in (word, pattern, target, constant)):
        return False
    if not pattern.token.text.isdecimal() or not constant.token.text.isdecimal():
        return False
    selector_type = owner.parameters.get(word.token.text)
    target_type = owner.parameters.get(target.token.text)
    if (
        selector_type is None
        or selector_type.kind != "qnum"
        or target_type is None
        or target_type.kind != "qnum"
        or selector_type.name.text in owner.signed
        or target_type.name.text in owner.signed
    ):
        return False
    selector, outputs = owner.array_view(word), owner.array_view(target)
    pattern_value, xor_value = int(pattern.token.text), int(constant.token.text)
    if not 1 <= len(selector) <= 7 or not 0 <= pattern_value < 1 << len(selector):
        reject(guard.token, "control", "word XOR guard must fit at most seven bits")
    if not 0 <= xor_value < 1 << len(outputs):
        reject(constant.token, "width", "word XOR constant must fit the target")
    if set(selector).intersection(outputs):
        reject(
            statement.token,
            "alias",
            "word XOR selector must be disjoint from the target",
        )
    token = body[0].token
    values, indices = zip(*selector, strict=True)
    if not xor_value:
        owner.effect_op(
            "qreg.control_phase",
            token,
            values,
            indices=indices,
            control_state=pattern_value,
            angle=0.0,
        )
        value, bit = outputs[0]
        owner.effect_op("qreg.gate", token, (value,), gate="i", index=bit)
    for ordinal, (value, bit) in enumerate(outputs):
        if xor_value >> ordinal & 1:
            owner.effect_op(
                "qreg.control_x",
                token,
                (*values, value),
                indices=(*indices, bit),
                control_state=pattern_value,
            )
    return True
