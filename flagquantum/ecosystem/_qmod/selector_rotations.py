"""Checked uniformly controlled rotations on a statically resolved bit view."""

from operator import eq, ge, gt, le, lt, ne
from typing import TYPE_CHECKING

from .syntax import Expression, Function, Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def is_selector_helper(function: Function) -> bool:
    """Recognize a borrowed array helper with a single checked rotation table."""
    if function.permutation or len(function.statements) != 1:
        return False
    operation = function.statements[0]
    if operation.token.text != "select_rotation" or len(operation.arguments) != 4:
        return False
    selector, target = operation.arguments[2:]
    ports = {port.name.text: port for port in function.parameters}
    read, write = ports.get(selector.token.text), ports.get(target.token.text)
    return (
        read is not None
        and read.role == "const"
        and read.kind in {"qbit[]", "qphase", "qnum"}
        and read.record is None
        and not selector.children
        and write is not None
        and write.role == "inout"
        and write.kind == "qbit"
        and not target.children
        and all(port in (read, write) or port.role == "real" for port in ports.values())
    )


_GUARDS = {"==": eq, "!=": ne, ">=": ge, ">": gt, "<=": le, "<": lt}


def lower_guarded_selector(owner: "_Adapt", statement: Statement, index: str) -> bool:
    body = statement.control_body
    condition = statement.arguments[0]
    if (
        body is None
        or len(body) != 1
        or statement.control_else is not None
        or condition.token.text not in _GUARDS
        or len(condition.children) != 2
    ):
        return False
    call = body[0]
    cursor = call
    seen: set[str] = set()
    while cursor == Statement(cursor.token, cursor.arguments):
        name = cursor.token.text
        helper = owner.definitions.get(name)
        if (
            helper is None
            or name in seen
            or len(seen) >= 16
            or len(helper.statements) != 1
        ):
            return False
        if is_selector_helper(helper):
            break
        if any(
            port.role not in {"const", "inout", "real"} for port in helper.parameters
        ):
            return False
        seen.add(name)
        (cursor,) = helper.statements
    else:
        return False
    word, threshold = condition.children
    if word.children or word.token.text not in owner.parameters:
        return False
    port = owner.parameters[word.token.text]
    if port.kind != "qnum" or port.record is not None:
        return False
    guard = owner.selector_guard
    if guard is not None:
        return False
    owner.selector_guard = (
        owner.array_view(word),
        condition.token.text,
        owner.static_integer(threshold),
    )
    try:
        owner.statement(call, index)
    finally:
        owner.selector_guard = guard
    return True


def lower_select_rotation(
    owner: "_Adapt", token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 4:
        reject(
            token,
            "signature",
            "select_rotation requires an axis, angles, selector and target",
        )
    axis, table, source, destination = args
    if axis.token.text not in {"X", "Y", "Z"}:
        reject(axis.token, "rotation", "select_rotation supports Pauli::X, Y or Z")
    selector = owner.array_view(source, allow_phase=True)
    target, wire = owner.gate_target(destination)
    if not 1 <= len(selector) <= 8:
        reject(
            source.token,
            "rotation",
            "select_rotation requires one to eight selector bits",
        )
    if len(set(selector)) != len(selector) or (target, wire) in selector:
        reject(
            destination.token,
            "alias",
            "rotation selector and target locations must be distinct",
        )
    constant = owner.constants.get(table.token.text) if not table.children else None
    angles = (
        tuple(owner.classical_number(item) for item in constant.children)
        if constant is not None and constant.token.text == "["
        else owner.real_array(table)
    )
    size = 1 << len(selector)
    if len(angles) != size:
        reject(
            table.token,
            "rotation",
            "rotation table length must equal 2**selector width",
        )

    guard = owner.selector_guard
    if guard is not None:
        guarded_locations, comparison, threshold = guard
        if guarded_locations != selector:
            reject(token, "rotation", "guard must read exactly the rotation selector")
        compare = _GUARDS[comparison]
        angles = tuple(
            angle if compare(i, threshold) else 0.0 for i, angle in enumerate(angles)
        )

    # The Walsh coefficients in cyclic Gray order give one CX transition per
    # step. Selector bits are controls throughout, including the closing edge.
    coefficients = [angle / size for angle in angles]
    stride = 1
    while stride < size:
        for start in range(0, size, 2 * stride):
            for left in range(start, start + stride):
                right = left + stride
                a, b = coefficients[left], coefficients[right]
                coefficients[left], coefficients[right] = a + b, a - b
        stride *= 2

    def gate(name: str, **attrs: object) -> None:
        owner.effect_op("qreg.gate", token, (target,), gate=name, index=wire, **attrs)

    if axis.token.text == "X":
        gate("h")
    rotation = "ry" if axis.token.text == "Y" else "rz"
    for step in range(size):
        gray = step ^ (step >> 1)
        gate(rotation, angle=coefficients[gray])
        following = (step + 1) % size
        edge = gray ^ following ^ (following >> 1)
        control, index = selector[edge.bit_length() - 1]
        owner.effect_op(
            "qreg.pair_gate", token, (control, target), gate="cx", indices=(index, wire)
        )
    if axis.token.text == "X":
        gate("h")
