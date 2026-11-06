"""Bound Qmod helper and callback composition."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import (
    register_width,
)
from ...compiler._hybrid.model import (
    Value,
)
from .adapter_model import (
    MAX_HELPER_EXPANSIONS,
    _bound_helper,
)
from .affine import lower_snapshot, snapshot_start
from .bindings import (
    record_view,
)
from .callback_arrays import (
    capture_callbacks,
    capture_effects,
)
from .computed_rotation import (
    lower_computed_rotation,
)
from .copied_selector import lower_copied_selector
from .fixed_payoff import lower_fixed_payoff
from .indexed_pauli import bind_indexed_pauli
from .integers import checked_integer, checked_integer_array
from .qpe import (
    is_modular_power,
)
from .quantum_callbacks import (
    BORROWED_EFFECT_SIGNATURES,
    CALLBACK_ARRAY,
    Lambda,
)
from .real_matrices import bound_matrix
from .scoped_amplification import declare_workspace, known_zero
from .small_pauli import lower_narrow_pauli_sequence
from .syntax import (
    Expression,
    Function,
    Parameter,
    Token,
    reject,
)
from .unitary_effects import (
    PURE_EFFECTS,
    preserves_basis_labels,
)

if TYPE_CHECKING:
    from .adapter import _Adapt


def borrowed_effect_callback(
    self: _Adapt, argument: Expression, kinds: tuple[str, ...]
) -> Function:
    if isinstance(argument, Lambda):
        if len(argument.parameters) != len(kinds) or len(
            {p.text for p in argument.parameters}
        ) != len(kinds):
            reject(
                argument.token,
                "callback.signature",
                (
                    "callback requires two distinct array ports"
                    if len(kinds) == 2
                    else "callback requires one numeric port"
                ),
            )
        location = argument.token.location
        return Function(
            Token(f"$callback_{location.line}_{location.column}", location),
            tuple(
                Parameter(port, kind, None, "inout")
                for port, kind in zip(argument.parameters, kinds, strict=True)
            ),
            argument.body,
        )
    name = self.atom(argument)
    callback = self.callback_bindings.get(name.text) or self.definitions.get(name.text)
    if (
        callback is None
        or tuple(p.kind for p in callback.parameters) != kinds
        or any(p.record is not None or p.role != "inout" for p in callback.parameters)
    ):
        reject(
            name,
            "callback.signature",
            "callback requires matching borrowed quantum ports",
        )
    return callback


def call_bound_helper(
    self: _Adapt, syntax: Function, token: Token, arguments: tuple[Expression, ...]
) -> None:
    if len(arguments) != len(syntax.parameters):
        reject(token, "signature", "helper argument count disagrees")
    if any(port.fractional_bits for port in syntax.parameters):
        reject(
            token,
            "precision",
            "fixed-point helpers require a checked computed rotation",
        )
    if any(p.kind == "SparsePauliOp" for p in syntax.parameters):
        reject(
            token,
            "qpe",
            "typed power helpers require a lexical qpe_flexible callback",
        )
    if syntax.name.text in self.helper_stack or len(self.helper_stack) >= 16:
        reject(
            token,
            "recursion",
            "bound helper expansion must be acyclic and at most 16 levels",
        )
    if self.helper_expansions >= MAX_HELPER_EXPANSIONS:
        reject(
            token,
            "budget.expansion",
            "native expansion exceeds 4096 bound helper calls",
        )
    child = type(self)(
        syntax, self.resolve, self.definitions, self.constants, self.record_types
    )
    child.helper_stack = (*self.helper_stack, syntax.name.text)
    child.helper_expansions = self.helper_expansions + 1
    child.loop_iterations = self.loop_iterations
    child.selector_guard = self.selector_guard
    child.classical_integers.update(
        (name, value)
        for name, value in self.classical_integers.items()
        if name in self.constants
    )
    actuals: set[tuple[Value, int]] = set()
    outputs: set[Value] = set()
    phase_borrows: set[Value] = set()
    readonly: set[tuple[Value, int]] = set()
    modular_phase_ports = {
        statement.arguments[0].token.text
        for statement in syntax.statements
        if is_modular_power(statement)
    }
    phase_exports: dict[str, str] = {}
    signed_bit_exports: dict[str, int] = {}
    deferred: dict[str, Token] = {}
    for argument, parameter in zip(arguments, syntax.parameters, strict=True):
        name = parameter.name.text
        if parameter.kind == "qfunc()":
            if isinstance(argument, Lambda):
                if argument.parameters or not argument.body:
                    reject(
                        argument.token,
                        "callback.signature",
                        "expected a nonempty zero-argument callback",
                    )
                effects = capture_effects(self, argument.token, argument.body)
            else:
                captured = self.zero_callbacks.get(argument.token.text)
                if argument.children or captured is None:
                    reject(
                        argument.token,
                        "callback.signature",
                        "expected a bound zero-argument callback",
                    )
                effects = captured
            child.zero_callbacks[name] = effects
            continue
        if parameter.kind in BORROWED_EFFECT_SIGNATURES:
            child.callback_bindings[name] = self.borrowed_effect_callback(
                argument, BORROWED_EFFECT_SIGNATURES[parameter.kind]
            )
            continue
        if parameter.kind == CALLBACK_ARRAY:
            child.callback_arrays[name] = capture_callbacks(self, argument)
            continue
        if parameter.role == "callable":
            actual = self.atom(argument)
            callback = self.callback_bindings.get(actual.text) or self.definitions.get(
                actual.text
            )
            if (
                callback is None
                or callback.name.text == "main"
                or len(callback.parameters) != 2
            ):
                reject(
                    actual,
                    "signature",
                    "callback requires a named (real, qbit[]) helper",
                )
            first, second = callback.parameters
            if (
                first.role != "real"
                or first.kind != "real"
                or second.role != "inout"
                or second.kind != "qbit[]"
                or second.record is not None
            ):
                reject(
                    actual,
                    "signature",
                    "callback requires a named (real, qbit[]) helper",
                )
            child.callback_bindings[name] = callback
            continue
        if parameter.kind == "IndexedPauli[]":
            child.indexed_pauli[name] = bind_indexed_pauli(self, argument)
            continue
        if parameter.kind == "PauliTerm[]":
            terms = argument
            if not terms.children:
                symbol = terms.token.text
                terms = (
                    self.pauli_bindings.get(symbol, terms)
                    if symbol in self.parameters
                    else self.constants.get(symbol, terms)
                )
            if (
                terms.token.text != "PauliTerm[]"
                or not 1 <= len(terms.children) <= 1024
            ):
                reject(
                    argument.token,
                    "pauli",
                    "helper requires a static PauliTerm array",
                )
            child.pauli_bindings[name] = terms
            continue
        if parameter.role == "integer":
            if parameter.kind in {"int[]", "Pauli[]"}:
                if parameter.kind == "Pauli[]" and argument.token.text == "$pauli_axes":
                    values_int = tuple(
                        "IXYZ".index(item.token.text) for item in argument.children
                    )
                else:
                    values_int = self.integer_array(argument)
                if parameter.kind == "Pauli[]" and any(
                    item not in range(4) for item in values_int
                ):
                    reject(
                        argument.token,
                        "pauli",
                        "Pauli array contains an invalid axis",
                    )
                child.integer_arrays[name] = checked_integer_array(
                    values_int, argument.token, parameter.width
                )
            else:
                child.classical_integers[name] = checked_integer(
                    self.static_integer(argument), argument.token
                )
            continue
        if parameter.role == "real":
            if parameter.kind == "real[][]":
                matrix = bound_matrix(self, argument)
                if not matrix or any(len(row) != parameter.width for row in matrix):
                    reject(
                        argument.token,
                        "binding",
                        "real matrix does not match the formal row width",
                    )
                child.classical_matrices[name] = matrix
            elif parameter.kind == "real[]":
                values = self.real_array(argument)
                if not values or (
                    parameter.width is not None and len(values) != parameter.width
                ):
                    reject(
                        argument.token,
                        "binding",
                        "helper real array length disagrees",
                    )
                child.classical_arrays[name] = values
            else:
                child.classical_values[name] = self.angle(argument, token)
            continue
        raw_field = parameter.kind == "qbit[]" and parameter.role in {"const", "inout"}
        field_port = (
            self.record_field(argument, allow_fractional=raw_field)
            if argument.token.text == "."
            else None
        )
        actual_port = (
            field_port[0]
            if field_port is not None
            else (
                self.parameters.get(argument.token.text)
                if not argument.children
                else None
            )
        )
        signed_actual = (
            actual_port is not None
            and actual_port.kind == "qnum_signed"
            or not argument.children
            and argument.token.text in self.signed
        )
        if parameter.kind == "qnum_signed" and not signed_actual:
            reject(
                argument.token,
                "type",
                "signed helper requires a signed integer actual",
            )
        if (
            actual_port is not None
            and actual_port.kind == "qphase_signed"
            and (parameter.kind != "qbit[]" or parameter.role not in {"const", "inout"})
        ):
            reject(
                argument.token,
                "precision",
                "signed phase storage requires a borrowed bit view",
            )
        borrowed_counter = (
            name in modular_phase_ports
            and actual_port is not None
            and actual_port.kind == "qphase"
            and actual_port.role != "const"
            and parameter.role == "inout"
            and parameter.kind == "qnum"
        )
        if borrowed_counter or (
            actual_port is not None
            and actual_port.kind == "qphase"
            and parameter.kind == "qnum"
            and parameter.width is None
            and parameter.role in {"const", "inout"}
        ):
            parameter = replace(parameter, kind="qphase")
            child.parameters[name] = parameter
        if (
            actual_port is not None
            and actual_port.kind == "qphase"
            and (
                parameter.kind not in {"qphase", "qbit[]"}
                or (actual_port.role == "const" and parameter.role != "const")
            )
        ):
            reject(
                argument.token,
                "precision",
                "fixed-point helper ports require a checked bit view or matching type",
            )
        if parameter.kind == "qphase" and (
            actual_port is None or actual_port.kind != "qphase"
        ):
            reject(
                argument.token,
                "precision",
                "fixed-point helper ports require matching unsigned fractional storage",
            )
        if parameter.role == "output" and (
            actual_port is None or actual_port.role != "output"
        ):
            reject(
                argument.token,
                "ownership",
                "bound helper Output requires a whole caller Output",
            )
        if parameter.role == "output" and self.widths.get(argument.token.text) is None:
            if any(
                previous.text == argument.token.text for previous in deferred.values()
            ):
                reject(
                    argument.token,
                    "alias",
                    "writable helper arguments must be disjoint",
                )
            deferred[name] = argument.token
            continue
        view = (
            record_view(self, parameter, argument)
            if parameter.record is not None
            else (
                self.array_view(argument, allow_phase=True, allow_signed=True)
                if parameter.kind in {"qbit[]", "qnum", "qnum_signed", "qphase"}
                else (self.gate_target(argument, allow_numeric=True),)
            )
        )
        if parameter.length is not None and (
            actual_port is None or actual_port.length != parameter.length
        ):
            reject(argument.token, "width", "helper numeric array shape disagrees")
        if parameter.width is not None and len(view) != parameter.width:
            reject(argument.token, "width", "helper quantum array length disagrees")
        if len(set(view)) != len(view) or actuals.intersection(view):
            reject(
                argument.token,
                "alias",
                "writable helper arguments must be disjoint",
            )
        actuals.update(view)
        child.widths[name] = len(view)
        if (
            actual_port is not None
            and actual_port.kind == "qphase_signed"
            and parameter.role == "inout"
        ):
            signed_bit_exports[argument.token.text] = len(view)
        if parameter.kind in {"qnum", "qnum_signed"} and signed_actual:
            child.signed.add(name)
        value = view[0][0]
        if view == tuple((value, index) for index in range(register_width(value))):
            child.registers[name] = value
        else:
            child.views[name] = view
        if borrowed_counter:
            last = next(
                (op for op in reversed(self.operations) if value in op.operands[1:]),
                None,
            )
            if name not in child.registers or not (
                last is not None
                and last.name == "qreg.allocate"
                or last is None
                and value in self.borrowed_zero_phases
            ):
                reject(
                    argument.token,
                    "qpe",
                    "phase output must be initialized by allocate and unused",
                )
            child.borrowed_zero_phases.add(value)
            phase_exports[name] = argument.token.text
        if parameter.role == "output":
            if name not in child.registers:
                reject(
                    argument.token,
                    "ownership",
                    "bound helper Output cannot be a partial view",
                )
            outputs.add(value)
        if parameter.kind == "qphase" and parameter.role == "const":
            phase_borrows.add(value)
        elif parameter.role == "const":
            readonly.update(view)
    borrowed_owners = {value for value, _ in actuals} - outputs
    child.zero_entry = lambda value: value in borrowed_owners and known_zero(
        self, value
    )
    if not lower_narrow_pauli_sequence(child, syntax.statements):
        snapshot = snapshot_start(syntax.statements)
        for index, statement in enumerate(syntax.statements):
            if index + 3 == len(syntax.statements) and (
                lower_computed_rotation(child, syntax.statements[index:])
                or lower_fixed_payoff(child, syntax.statements[index:])
                or lower_copied_selector(child, syntax.statements[index:], str(index))
            ):
                break
            if index == snapshot:
                lower_snapshot(child, syntax.statements, index)
                break
            child.statement(statement, str(index))
    if any(port.name.text in child.suspended for port in syntax.parameters):
        reject(
            token,
            "binding",
            "bound helper must restore every formal port before return",
        )
    materialized: dict[Value, Value] = {}
    for name, actual in deferred.items():
        child_output = child.registers.get(name)
        if child_output is None:
            reject(actual, "ownership", "bound helper must initialize each Output")
        parameter = child.parameters[name]
        if parameter.kind == "qphase":
            if self.parameters[actual.text].kind != "qnum":
                reject(
                    actual,
                    "precision",
                    "inferred fractional Output requires a numeric caller port",
                )
            self.parameters[actual.text] = replace(
                self.parameters[actual.text], kind="qphase"
            )
        output = self.register(
            actual,
            register_width(child_output),
            allow_qubit_array=True,
            allow_fractional_output=parameter.kind == "qphase",
        )
        materialized[child_output] = output
        outputs.add(child_output)
    readonly_owners = {owner for owner, _ in readonly}
    initialized_outputs = set()
    for op in child.operations:
        operands = op.operands[1:]
        if op.name in {"qreg.allocate", "qreg.prepare_amplitudes"}:
            initialized_outputs.add(operands[0])
        if readonly_owners.intersection(operands) and not preserves_basis_labels(
            op, readonly
        ):
            reject(token, "ownership", "bound helper cannot write a Const quantum port")
        if phase_borrows.intersection(operands):
            if op.name == "qreg.control_phase" or (
                op.name == "qreg.pair_gate"
                and op.attributes.get("gate") == "cx"
                and operands[0] in phase_borrows
                and operands[1] not in phase_borrows
            ):
                continue
            reject(
                token,
                "ownership",
                "bound helper cannot write a Const fixed-point port",
            )
    if not outputs <= initialized_outputs:
        reject(token, "ownership", "bound helper must initialize each Output")
    if any(
        op.name == "qreg.hidden" and op.results[0] not in child.certified_workspace
        for op in child.operations
    ):
        reject(token, "free", "bound helper must free each allocated local qbit")
    for name, actual_name in phase_exports.items():
        if name in child.fractional_bits:
            self.fractional_bits[actual_name] = child.fractional_bits[name]
    self.fractional_bits.update(signed_bit_exports)
    self.helper_expansions = child.helper_expansions
    self.loop_iterations = child.loop_iterations
    for op in child.operations:
        if op.name == "qreg.hidden":
            materialized[op.results[0]] = declare_workspace(self, token)
            continue
        operands = tuple(materialized.get(value, value) for value in op.operands[1:])
        if op.name == "qreg.local":
            reject(
                token,
                "scope",
                "bound helper needs a checked lowering for computed local storage",
            )
        self.effect_op(op.name, token, operands, **op.attributes)


def call(
    self: _Adapt,
    token: Token,
    arguments: tuple[Expression, ...],
    *,
    const_only: bool = False,
    control_bit: Value | None = None,
    condition_inputs: tuple[Value, ...] = (),
) -> None:
    callback = self.callback_bindings.get(token.text)
    if callback is not None:
        if const_only or control_bit is not None:
            reject(token, "scope", "callbacks require ordinary static calls")
        start = len(self.operations)
        self.call_bound_helper(callback, token, arguments)
        if self.parameters[token.text].kind in BORROWED_EFFECT_SIGNATURES and any(
            op.name not in PURE_EFFECTS for op in self.operations[start:]
        ):
            reject(
                token,
                "callback.escape",
                "borrowed callback requires pure unitary effects",
            )
        return
    if token.text in self.parameters:
        reject(token, "call", "an output variable shadows this helper call")
    definition = self.definitions.get(token.text)
    if definition is not None and token.text != "main" and _bound_helper(definition):
        if const_only or control_bit is not None:
            reject(
                token,
                "scope",
                "classical and writable-array helpers require ordinary calls",
            )
        self.call_bound_helper(definition, token, arguments)
        return
    syntax, program = self.resolve(token)
    if control_bit is not None and any(p.role != "const" for p in syntax.parameters):
        reject(token, "control", "controlled helpers require Const-only ports")
    if const_only and any(p.role != "const" for p in syntax.parameters):
        reject(token, "scope", "apply helper calls require Const-only parameters")
    if len(arguments) != len(syntax.parameters):
        reject(token, "signature", "helper argument count disagrees")
    const = iter(program.body.blocks[0].arguments[1:])
    outputs = iter(
        op.results[0]
        for op in program.body.blocks[0].operations
        if op.name == "qreg.declare"
    )
    reads: list[Value] = []
    writes: list[Value] = []
    for argument, parameter in zip(arguments, syntax.parameters, strict=True):
        borrowed = parameter.role == "const"
        formal = next(const if borrowed else outputs)
        actual_name = self.atom(argument)
        if (
            parameter.record is not None
            and self.parameters.get(actual_name.text) is not None
            and self.parameters[actual_name.text].record != parameter.record
        ):
            reject(
                actual_name,
                "signature",
                "record helper requires the matching record type",
            )
        actual = self.register(
            actual_name,
            None if borrowed else formal.type.parameters[0],
            allow_qubit_array=parameter.record is not None
            or parameter.kind == "qbit[]",
            allocate_local=not borrowed and self.scope == "main",
        )
        actual_kind = self.parameters[actual_name.text].kind
        raw_array_output = (
            not borrowed
            and parameter.kind == "qbit[]"
            and actual_kind == "qnum"
            and parameter.record is None
            and self.parameters[actual_name.text].role == "local"
        )
        if (
            actual_name.text in self.signed
            or actual_kind != parameter.kind
            and not raw_array_output
            or actual.type != formal.type
        ):
            reject(
                argument.token,
                "signature",
                "helper argument kind/width must match exactly",
            )
        (reads if borrowed else writes).append(actual)
    if control_bit is None:
        self.effect_op("qreg.call", token, (*reads, *writes), callee=program.name)
    else:
        if len(set(reads)) != len(reads) or set(reads).intersection(condition_inputs):
            reject(token, "control", "controlled helper arguments must be disjoint")
        self.effect_op(
            "qreg.control_call", token, (control_bit, *reads), callee=program.name
        )
