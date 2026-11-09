"""Qmod register views and static arguments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import (
    register_width,
)
from ...compiler._hybrid.model import (
    Value,
)
from .adapter_model import (
    _INTRINSICS,
    _STATIC_GATES,
    MAX_LOOP_ITERATIONS,
    _bound_helper,
    _literal_angle,
)
from .indexed_pauli import indexed_pauli_member
from .integers import checked_integer, checked_integer_array, integer_operation
from .qpe import (
    lower_modular_loop,
)
from .real_matrices import matrix_cell
from .repeated_effects import lower_repeated_effects
from .syntax import (
    Expression,
    Parameter,
    Statement,
    Token,
    reject,
)
from .unitary_effects import (
    is_reversible_scope,
)

if TYPE_CHECKING:
    from .context import _Adapt


def record_field(
    self: _Adapt, node: Expression, *, allow_fractional: bool = False
) -> tuple[Parameter, tuple[tuple[Value, int], ...]] | None:
    if node.token.text != "." or len(node.children) != 2:
        return None
    root, member = node.children
    parent = self.record_field(root) if root.children else None
    parameter = (
        parent[0] if parent is not None else self.parameters.get(root.token.text)
    )
    if parameter is None or parameter.record is None:
        reject(root.token, "field", "field access requires a declared record")
    record = self.record_types[parameter.record]
    offset = 0
    for field in record.fields:
        assert field.width is not None
        size = field.width * (field.length or 1)
        if field.name.text == member.token.text:
            if (
                field.fractional_bits or field.kind == "qphase"
            ) and not allow_fractional:
                reject(
                    member.token,
                    "precision",
                    "fractional record fields require checked fixed-point lowering",
                )
            view = parent[1] if parent is not None else self.array_view(root)
            return field, view[offset : offset + size]
        offset += size
    reject(member.token, "field", "unknown record field")


def record_bit(self: _Adapt, node: Expression) -> tuple[Value, int] | None:
    """Resolve one checked row/column access without erasing field rank."""
    if node.token.text != "[" or node.children[0].token.text != "[":
        return None
    row, column = node.children
    resolved = self.record_field(row.children[0])
    if resolved is None:
        return None
    field, locations = resolved
    if field.kind != "qbit[][]":
        reject(
            node.token,
            "field",
            "double indexing requires a two-dimensional bit field",
        )
    assert field.width is not None and field.length is not None
    row_index = self.static_integer(row.children[1])
    column_index = self.static_integer(column)
    if not 0 <= row_index < field.length:
        reject(row.token, "width", "field row index is out of bounds")
    if not 0 <= column_index < field.width:
        reject(column.token, "width", "field column index is out of bounds")
    return locations[row_index * field.width + column_index]


def static_integer(self: _Adapt, node: Expression) -> int:
    text = node.token.text
    if text == "$exact_log2":
        size = self.static_integer(node.children[0])
        if size < 1 or size & (size - 1):
            reject(
                node.token,
                "width",
                "static log base two requires a positive power of two",
            )
        return size.bit_length() - 1
    if text == "$ceil_log2":
        size = self.static_integer(node.children[0])
        if size < 2:
            reject(
                node.token,
                "width",
                "binary log width requires a static integer of at least two",
            )
        return (size - 1).bit_length()
    if text.isdecimal():
        return checked_integer(int(text), node.token)
    if not node.children and text in self.classical_integers:
        return self.classical_integers[text]
    if text == "[" and not node.children[0].children:
        values = self.integer_arrays.get(node.children[0].token.text)
        if values is None:
            reject(
                node.token,
                "expression",
                "static indices require an integer array, not quantum or real data",
            )
        offset = self.static_integer(node.children[1])
        if not 0 <= offset < len(values):
            reject(node.token, "binding", "integer array index is out of bounds")
        return values[offset]
    if text == "-" and len(node.children) == 1:
        return checked_integer(-self.static_integer(node.children[0]), node.token)
    if text in {"+", "-", "*", "%", "**"}:
        left, right = (self.static_integer(child) for child in node.children)
        return integer_operation(node.token, left, right)
    if text == "floor" and node.children[0].token.text == "/":
        numerator, denominator = (
            self.static_integer(child) for child in node.children[0].children
        )
        if denominator <= 0:
            reject(node.token, "expression", "static floor requires a positive divisor")
        return numerator // denominator
    if text == ".":
        root = node.children[0]
        member = node.children[1].token.text if len(node.children) == 2 else "len"
        if (entry := indexed_pauli_member(self, node, root, member)) is not None:
            return entry
        if member == "size" and root.children:
            field = self.record_field(root, allow_fractional=True)
            if field is not None:
                return len(field[1])
        if member == "size" and not root.children:
            parameter = self.parameters.get(root.token.text)
            width = self.widths.get(root.token.text)
            if (
                parameter is not None
                and parameter.kind in {"qbit", "qbit[]", "qnum"}
                and width is not None
            ):
                return width
        if member != "len":
            reject(node.token, "field", "expected array len or record size")
        if root.children:
            if root.token.text == "[":
                resolved = self.record_field(root.children[0])
                if resolved is not None and resolved[0].kind == "qbit[][]":
                    return len(self.array_view(root))
            field = self.record_field(root)
            if field is not None and field[0].kind in {
                "qbit[]",
                "qnum[]",
                "qbit[][]",
            }:
                return field[0].length or len(field[1])
            reject(node.token, "field", "len requires an array field")
        name = root.token.text
        numeric = self.parameters.get(name)
        if (
            numeric is not None
            and numeric.length is not None
            and numeric.record is None
        ):
            return numeric.length
        if name in self.callback_arrays:
            return len(self.callback_arrays[name])
        if name in self.classical_arrays:
            return len(self.classical_arrays[name])
        if name in self.classical_matrices:
            return len(self.classical_matrices[name])
        if name in self.integer_arrays:
            return len(self.integer_arrays[name])
        if name in self.indexed_pauli:
            return len(self.indexed_pauli[name])
        width = self.widths.get(name)
        if (
            width is None
            or self.parameters[name].kind != "qbit[]"
            or self.parameters[name].record is not None
        ):
            reject(node.token, "width", "len requires a qubit array with a known width")
        return width
    if text in self.loop_indices:
        return self.loop_indices[text]
    reject(node.token, "expression", "expected a static integer or loop index")


def gate_target(
    self: _Adapt, node: Expression, *, allow_numeric: bool = False
) -> tuple[Value, int]:
    cell = self.record_bit(node)
    if cell is not None:
        return cell
    source = node.children[0] if node.token.text == "[" else node
    if source.token.text == ".":
        resolved = self.record_field(source)
        assert resolved is not None
        field, field_view = resolved
        if node.token.text == "[":
            if field.kind != "qbit[]":
                reject(node.token, "field", "indexing requires an array field")
            index = self.static_integer(node.children[1])
            if not 0 <= index < len(field_view):
                reject(node.token, "width", "field index is out of bounds")
            return field_view[index]
        if field.kind != "qbit":
            reject(node.token, "field", "gate requires a scalar field or element")
        return field_view[0]
    indexed = node.token.text == "["
    target = self.atom(node.children[0] if indexed else node)
    if target.text in self.suspended:
        reject(target, "binding", "a bound source name is suspended")
    parameter = self.parameters.get(target.text)
    view = self.views.get(target.text)
    if indexed and (parameter is None or parameter.kind != "qbit[]"):
        reject(target, "operator", "indexing requires a qubit array")
    index = self.static_integer(node.children[1]) if indexed else 0
    if indexed and parameter is not None and parameter.length is not None:
        if self.widths[target.text] != parameter.length:
            reject(
                target,
                "type",
                "a scalar gate requires a one-bit numeric array element",
            )
    if view is not None:
        width = len(view)
    else:
        value = self.register(target, allow_qubit_array=True)
        width = register_width(value)
    scalar_kinds = {"qbit", "qbit[]"}
    if (
        allow_numeric
        and target.text not in self.signed
        and target.text not in self.fractional_bits
    ):
        scalar_kinds.add("qnum")
    if not indexed and (
        parameter is None or parameter.kind not in scalar_kinds or width != 1
    ):
        reject(target, "type", "gate operand requires a scalar qubit or array element")
    if not 0 <= index < width:
        reject(node.token, "width", "qubit array index is out of bounds")
    if view is not None:
        return view[index]
    return value, index


def array_view(
    self: _Adapt,
    node: Expression,
    *,
    allow_phase: bool = False,
    allow_signed: bool = False,
) -> tuple[tuple[Value, int], ...]:
    if node.token.text == "[" and not node.children[0].children:
        source, index_expression = node.children
        port = self.parameters.get(source.token.text)
        if port is not None and port.length is not None and port.record is None:
            offset = self.static_integer(index_expression)
            if not 0 <= offset < port.length:
                reject(
                    index_expression.token,
                    "width",
                    "numeric array index is out of bounds",
                )
            array = self.array_view(source)
            width = len(array) // port.length
            return array[offset * width : (offset + 1) * width]
    cell = self.record_bit(node)
    if cell is not None:
        return (cell,)
    if node.token.text == "[" and node.children[0].token.text == ".":
        resolved = self.record_field(node.children[0])
        assert resolved is not None
        field, word_locations = resolved
        if field.kind not in {"qnum[]", "qbit[][]"}:
            return (self.gate_target(node),)
        index = self.static_integer(node.children[1])
        assert field.length is not None and field.width is not None
        if not 0 <= index < field.length:
            reject(node.token, "width", "field index is out of bounds")
        return word_locations[index * field.width : (index + 1) * field.width]
    if node.token.text == ".":
        resolved = self.record_field(node, allow_fractional=allow_phase)
        assert resolved is not None
        if resolved[0].fractional_bits:
            reject(
                node.token,
                "precision",
                "only fully fractional unsigned fields admit this bit view",
            )
        if resolved[0].kind == "qnum_signed" and not allow_signed:
            reject(
                node.token,
                "type",
                "signed field requires an explicit signed or raw-bit view",
            )
        return resolved[1]
    if node.token.text == "{":
        return tuple(
            location
            for item in node.children
            for location in (
                (self.gate_target(item),)
                if item.token.text == "["
                else self.array_view(item)
            )
        )
    if node.token.text == ":":
        source, start_node, stop_node = node.children
        source_view = self.array_view(
            source, allow_phase=allow_phase, allow_signed=allow_signed
        )
        start, stop = self.static_integer(start_node), self.static_integer(stop_node)
        if not 0 <= start < stop <= len(source_view):
            reject(node.token, "width", "qubit array slice is out of bounds or empty")
        return source_view[start:stop]
    target = self.atom(node)
    if target.text in self.suspended:
        reject(target, "binding", "a bound source name is suspended")
    parameter = self.parameters.get(target.text)
    phase = (
        allow_phase
        and parameter is not None
        and (
            parameter.kind == "qphase"
            or (allow_signed and parameter.kind == "qphase_signed")
        )
    )
    if parameter is None or (
        parameter.kind not in {"qbit[]", "qnum", "qbit"} and not phase
    ):
        reject(target, "type", "expected an unsigned quantum register or view")
    if (target.text in self.signed and not allow_signed) or (
        target.text in self.fractional_bits and not phase
    ):
        reject(target, "type", "views require an unsigned integer register")
    view = self.views.get(target.text)
    if view is None:
        value = self.register(
            target, allow_qubit_array=True, allow_fractional_output=phase
        )
        view = tuple((value, index) for index in range(register_width(value)))
    return view


def integer_array(self: _Adapt, node: Expression) -> tuple[int, ...]:
    if node.token.text == "[]":
        return checked_integer_array(
            tuple(self.static_integer(child) for child in node.children), node.token
        )
    if not node.children and node.token.text in self.integer_arrays:
        return self.integer_arrays[node.token.text]
    reject(
        node.token,
        "binding",
        "expected a classical integer array literal or binding",
    )


def real_array(self: _Adapt, node: Expression) -> tuple[float, ...]:
    if node.token.text == "[]":
        return tuple(self.classical_number(item) for item in node.children)
    source = node.children[0] if node.token.text == ":" else node
    values = self.classical_arrays.get(source.token.text)
    if values is None or source.children:
        reject(source.token, "binding", "helper requires a bound real array")
    if node.token.text != ":":
        return values
    start, stop = (self.static_integer(item) for item in node.children[1:])
    if not 0 <= start < stop <= len(values):
        reject(node.token, "binding", "real array slice is out of bounds or empty")
    return values[start:stop]


def repeat(self: _Adapt, statement: Statement, index: str) -> None:
    variable = statement.repeat_index
    assert variable is not None
    name = variable.text
    if (
        name in self.parameters
        or name in _INTRINSICS
        or name in self.constants
        or name in self.loop_indices
    ):
        reject(variable, "name", "loop index cannot shadow a register or intrinsic")
    count = self.static_integer(statement.arguments[0])
    if not 0 <= count <= 256:
        reject(statement.token, "budget.expansion", "repeat count must be in 0..256")
    self.loop_iterations += count
    if self.loop_iterations > MAX_LOOP_ITERATIONS:
        reject(
            statement.token,
            "budget.expansion",
            "static loops exceed 4096 iterations",
        )
    if lower_modular_loop(self, statement, count) or lower_repeated_effects(
        self, statement, count
    ):
        return
    for item in statement.repeat_body:
        callee = self.callback_bindings.get(item.token.text) or self.definitions.get(
            item.token.text
        )
        bound_call = (
            callee is not None and callee.name.text != "main" and _bound_helper(callee)
        )
        if item.declaration is not None or (
            item.token.text
            not in _STATIC_GATES
            | {
                "phase",
                "suzuki_trotter",
                "multi_suzuki_trotter",
                "unitary",
                "apply_to_all",
            }
            and not bound_call
            and item.control_body is None
            and item.callback_index is None
            and item.map_callback is None
            and item.static_if_body is None
            and item.repeat_index is None
            and item.inverse_body is None
            and not is_reversible_scope(item)
        ):
            reject(
                item.token,
                "scope",
                "repeat bodies admit only gates and bound helper calls",
            )
    for iteration in range(count):
        self.loop_indices[name] = iteration
        for ordinal, item in enumerate(statement.repeat_body):
            self.statement(item, f"{index}.{iteration}.{ordinal}")
    self.loop_indices.pop(name, None)


def classical_number(self: _Adapt, node: Expression) -> float:
    text = node.token.text
    if not node.children:
        if text in self.classical_integers or text in self.loop_indices:
            value = self.static_integer(node)
            if abs(value) > 2**53:
                reject(
                    node.token,
                    "precision",
                    "integer angle conversion exceeds exact real range",
                )
            return float(value)
        if text in self.classical_values:
            return self.classical_values[text]
        if text == "pi" and text not in self.parameters:
            return math.pi
        constant = self.constants.get(text)
        if constant is not None and text not in self.parameters:
            return _literal_angle(constant, node.token)
        return _literal_angle(node, node.token)
    if text == "[":
        if (cell := matrix_cell(self, node)) is not None:
            return cell
        source, offset = node.children
        if not source.children and source.token.text in self.integer_arrays:
            value = self.static_integer(node)
            if abs(value) > 2**53:
                reject(
                    node.token,
                    "precision",
                    "integer angle conversion exceeds exact real range",
                )
            return float(value)
        values = (
            self.real_array(source)
            if source.token.text == ":"
            else self.classical_arrays.get(source.token.text)
        )
        if (source.children and source.token.text != ":") or values is None:
            reject(source.token, "binding", "expected a bound classical array")
        index = self.static_integer(offset)
        if not 0 <= index < len(values):
            reject(offset.token, "binding", "real array index is out of bounds")
        return values[index]
    if text == ".":
        return float(self.static_integer(node))
    if text == "**":
        base, power = (self.static_integer(child) for child in node.children)
        if power < 0:
            if power < -255 or not 0 < abs(base) <= 2**53:
                reject(
                    node.token,
                    "budget.number",
                    "negative real powers require a nonzero exact integer base and exponent in -255..-1",
                )
            result = math.pow(base, power)
            if result == 0:
                reject(node.token, "precision", "negative real power underflows")
        else:
            value = self.static_integer(node)
            if abs(value) > 2**53:
                reject(
                    node.token,
                    "precision",
                    "integer angle conversion exceeds exact real range",
                )
            result = float(value)
    elif text in {"acos", "asin", "sqrt", "exp"}:
        value_real = self.classical_number(node.children[0])
        if text in {"acos", "asin"}:
            if not -1 <= value_real <= 1:
                reject(node.token, "angle", f"{text} requires an argument in [-1, 1]")
            result = math.acos(value_real) if text == "acos" else math.asin(value_real)
        elif text == "sqrt":
            if value_real < 0:
                reject(node.token, "angle", "sqrt requires a nonnegative argument")
            result = math.sqrt(value_real)
        else:
            if value_real > 53 * math.log(2):
                reject(node.token, "angle", "exponential exceeds finite angle range")
            result = math.exp(value_real)
    elif text == "-" and len(node.children) == 1:
        result = -self.classical_number(node.children[0])
    elif text in {"+", "-", "*", "/"}:
        left, right = (self.classical_number(child) for child in node.children)
        if text == "/" and right == 0:
            reject(node.token, "angle", "classical division by zero")
        if text == "+":
            result = left + right
        elif text == "-":
            result = left - right
        elif text == "*":
            result = left * right
        else:
            result = left / right
    else:
        reject(node.token, "angle", "expected a finite classical real expression")
    if not math.isfinite(result) or abs(result) > 2**53:
        reject(node.token, "angle", "classical real exceeds finite ±2**53 range")
    return result


def angle(self: _Adapt, node: Expression, gate: Token) -> float:
    if node.token.text in {
        "+",
        "-",
        "*",
        "/",
        "**",
        "pi",
        "acos",
        "asin",
        "sqrt",
        "exp",
    } and (node.children or node.token.text == "pi"):
        return self.classical_number(node)
    if (
        not node.children
        and (
            node.token.text in self.classical_integers
            or node.token.text in self.loop_indices
        )
    ) or (
        node.token.text == "[" and node.children[0].token.text in self.integer_arrays
    ):
        return self.classical_number(node)
    if node.token.text == "[" and node.children[0].token.text == ":":
        return self.classical_number(node)
    if not node.children and node.token.text in self.classical_values:
        return self.classical_values[node.token.text]
    if node.token.text == "[":
        if (cell := matrix_cell(self, node)) is not None:
            return cell
        array, index = node.children
        values = self.classical_arrays.get(array.token.text)
        if values is not None and not array.children:
            if (
                not self.helper_stack
                and not self.loop_indices
                and (index.children or not index.token.text.isdecimal())
            ):
                reject(
                    index.token,
                    "binding",
                    "real array index requires an integer literal",
                )
            offset = self.static_integer(index)
            if not 0 <= offset < len(values):
                reject(index.token, "binding", "real array index is out of bounds")
            return values[offset]
    return _literal_angle(node, gate)


def encode_data(self: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
    if len(args) != 2:
        reject(token, "signature", "data encoding requires a real array and output")
    source, output = (self.atom(arg) for arg in args)
    values = self.classical_arrays.get(source.text)
    if values is None:
        reject(source, "binding", "encoding requires a bound real array")
    inplace = token.text == "inplace_encode_on_bloch"
    bloch = token.text != "encode_in_angle"
    width = (len(values) + 1) // 2 if bloch else len(values)
    parameter = self.parameters.get(output.text)
    if parameter is None or (
        parameter.kind != "qbit[]"
        and not (inplace and parameter.kind == "qnum" and width == 1)
    ):
        reject(
            output,
            "type",
            "encoding requires a qubit array or one-bit numeric in-place target",
        )
    if inplace and len(values) % 2:
        reject(
            source,
            "binding",
            "in-place Bloch encoding requires complete data pairs",
        )
    target = self.register(output, None if inplace else width, allow_qubit_array=True)
    if register_width(target) != width:
        reject(output, "width", "encoding output width must match its data")
    if not inplace:
        self.effect_op("qreg.allocate", token, (target,))
    for index, value in enumerate(values):
        if bloch:
            gate = "rz" if index % 2 else "rx"
            scale = (
                math.tau if index == len(values) - 1 and len(values) % 2 else math.pi
            )
            wire = index // 2
        else:
            gate, scale, wire = "ry", math.pi, index
        self.effect_op(
            "qreg.gate",
            token,
            (target,),
            gate=gate,
            index=wire,
            angle=scale * value,
        )
