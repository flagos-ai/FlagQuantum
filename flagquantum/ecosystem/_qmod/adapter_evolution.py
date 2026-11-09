"""Qmod state preparation and physical evolution."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from ...compiler import _pauli_schedule
from ...compiler._construction.register_profile import (
    MAX_PREPARED_AMPLITUDES,
    register_width,
)
from ...compiler._hybrid.model import (
    Value,
)
from .adapter_model import (
    _INTRINSICS,
    _STATIC_COMPARISONS,
    MAX_QDRIFT_SAMPLES,
    QDRIFT_SEED,
    _literal_angle,
    _pauli_support,
    _preparation_value,
    _sparse_width,
)
from .indexed_control import (
    lower_indexed_control,
)
from .pauli_schedule import ceiling_repetitions, compact_commuting
from .preparation import exponential_angles
from .qpe import (
    is_modular_power,
    lower_modular_qpe,
    specialize_power_body,
)
from .syntax import (
    Expression,
    Statement,
    Token,
    reject,
)

if TYPE_CHECKING:
    from .context import _Adapt


def evolution_angle(self: _Adapt, node: Expression, token: Token) -> float:
    if node.token.text == "-":
        return -self.angle(node.children[0], token)
    return self.angle(node, token)


def linear_pauli_rotations(
    self: _Adapt, token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 5:
        reject(
            token,
            "signature",
            "linear_pauli_rotations requires bases, slopes, offsets, control and target",
        )
    bases, slopes, offsets, control_arg, target_arg = args

    def bit_view(node: Expression) -> tuple[tuple[Value, int], ...]:
        if node.token.text == "[" and not node.children[0].children:
            return (self.gate_target(node),)
        return self.array_view(node)

    controls = bit_view(control_arg)
    targets = bit_view(target_arg)
    if len(set((*controls, *targets))) != len(controls) + len(targets):
        reject(token, "alias", "linear_pauli_rotations requires disjoint qubit arrays")
    width = len(targets)
    if any(len(node.children) != width for node in (bases, slopes, offsets)):
        reject(token, "width", "bases, slopes and offsets must match the target width")
    for (target, wire), basis, slope, offset in zip(
        targets, bases.children, slopes.children, offsets.children, strict=True
    ):
        if basis.token.text not in {"Y", "2"}:
            reject(basis.token, "pauli", "linear_pauli_rotations supports only Y")
        self.effect_op(
            "qreg.gate",
            token,
            (target,),
            gate="ry",
            index=wire,
            angle=self.classical_number(offset),
        )
        coefficient = self.classical_number(slope)
        for index, (control, control_wire) in enumerate(controls):
            self.effect_op(
                "qreg.pair_gate",
                token,
                (control, target),
                gate="cry",
                indices=(control_wire, wire),
                angle=math.ldexp(coefficient, index),
            )


def single_pauli_exponent(
    self: _Adapt, token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 3:
        reject(
            token,
            "signature",
            "single_pauli_exponent requires a Pauli word, coefficient and target",
        )
    axes, coefficient, target_arg = args
    target = self.atom(target_arg)
    register = self.register(target, allow_qubit_array=True)
    if self.parameters[target.text].kind != "qbit[]":
        reject(target, "type", "single_pauli_exponent requires a qubit array")
    support = _pauli_support(axes, register_width(register), sparse=False)
    angle = self.evolution_angle(coefficient, token)
    if support and not math.isfinite(2 * angle):
        reject(coefficient.token, "angle", "Pauli rotation angle exceeds finite range")
    self.pauli_exponential(token, register, support, angle)


def suzuki_trotter(
    self: _Adapt,
    token: Token,
    args: tuple[Expression, ...],
    *,
    emit: (
        Callable[[Token, Expression, float, Expression, int, int], None] | None
    ) = None,
) -> None:
    if len(args) != 5:
        reject(
            token,
            "signature",
            f"{token.text} requires terms, time, order, repetitions and target",
        )
    terms, time_arg, order_arg, repetitions_arg, target_arg = args
    if token.text == "multi_suzuki_trotter":
        if terms.token.text != "[" or not 1 <= len(terms.children) <= 16:
            reject(token, "evolution", "multi_suzuki_trotter requires 1..16 operators")
        times = self.real_array(time_arg)
        if len(times) != len(terms.children):
            reject(
                time_arg.token,
                "evolution",
                "operator and time arrays must have equal lengths",
            )
        if len(times) > 1:
            if any(
                arg.children or arg.token.text != "1"
                for arg in (order_arg, repetitions_arg)
            ):
                reject(
                    token,
                    "evolution",
                    "multiple operators require order 1 and one repetition",
                )
            for operator, operator_time in zip(terms.children, times, strict=True):
                self.pauli_evolution(token, operator, operator_time, target_arg, 1, 1)
            return
        terms, time = terms.children[0], times[0]
    else:
        time = self.evolution_angle(time_arg, token)
    computed = repetitions_arg.token.text == "$ceiling_repetitions"
    if (
        order_arg.children
        or not order_arg.token.text.isdecimal()
        or not computed
        and (repetitions_arg.children or not repetitions_arg.token.text.isdecimal())
    ):
        reject(
            token,
            "evolution",
            "order and repetitions must be integer literals or a static ceiling",
        )
    order = int(order_arg.token.text)
    repetitions = (
        ceiling_repetitions(self, repetitions_arg.children[0])
        if computed
        else int(repetitions_arg.token.text)
    )
    if order not in {1, 2} or not 1 <= repetitions <= 256:
        reject(token, "evolution", "use order 1 or 2 and 1..256 repetitions")
    if emit is not None:
        emit(token, terms, time, target_arg, order, repetitions)
    else:
        self.pauli_evolution(
            token, terms, time, target_arg, order, repetitions, compact=computed
        )


def exponentiate(self: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
    if len(args) != 3:
        reject(token, "signature", "exponentiate requires an operator, time and target")
    operator, time_arg, target_arg = args
    # Classiq's Hamiltonian Simulation guide specifies a first-order single step.
    self.pauli_evolution(
        token, operator, self.evolution_angle(time_arg, token), target_arg, 1, 1
    )


def pauli_evolution(
    self: _Adapt,
    token: Token,
    operator: Expression,
    time: float,
    target_arg: Expression,
    order: int,
    repetitions: int,
    *,
    compact: bool = False,
) -> None:
    register, terms = self.pauli_terms(token, operator, target_arg)
    if compact:
        angles = [
            (terms[index][0], step_time * terms[index][1])
            for index, step_time in _pauli_schedule.ordered_pauli_steps(
                len(terms), time / repetitions, order, 1
            )
        ]
        coalesced = compact_commuting(token, angles, repetitions)
        if coalesced is not None:
            for support, angle in coalesced:
                self.pauli_exponential(token, register, support, angle)
            return
    for index, step_time in _pauli_schedule.ordered_pauli_steps(
        len(terms), time, order, repetitions
    ):
        support, weight = terms[index]
        self.pauli_exponential(token, register, support, step_time * weight)


def qdrift(self: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
    if len(args) != 4:
        reject(
            token,
            "signature",
            "qdrift requires an operator, time, samples and target",
        )
    operator, time_arg, count_arg, target_arg = args
    if (
        count_arg.children
        or not count_arg.token.text.isdecimal()
        or not 1 <= int(count_arg.token.text) <= MAX_QDRIFT_SAMPLES
    ):
        reject(
            count_arg.token,
            "evolution",
            f"qdrift requires a literal sample count in 1..{MAX_QDRIFT_SAMPLES}",
        )
    count = int(count_arg.token.text)
    register, terms = self.pauli_terms(token, operator, target_arg)
    time = self.evolution_angle(time_arg, token)
    weights = [abs(coefficient) for _, coefficient in terms]
    total = math.fsum(weights)
    if not total or not time:
        self.effect_op("qreg.gate", token, (register,), gate="rz", index=0, angle=0.0)
        return
    try:
        for index, angle in _pauli_schedule.sampled_pauli_steps(
            [coefficient for _, coefficient in terms], time, count, QDRIFT_SEED
        ):
            self.pauli_exponential(token, register, terms[index][0], angle)
    except ValueError as error:
        reject(token, "angle", str(error))


def pauli_terms(
    self: _Adapt, token: Token, operator: Expression, target_arg: Expression
) -> tuple[Value, list[tuple[tuple[tuple[int, str], ...], float]]]:
    view = self.array_view(target_arg)
    owners = tuple(dict.fromkeys(value for value, _ in view))
    if len(owners) != 1 or len(set(view)) != len(view):
        reject(
            target_arg.token,
            "alias",
            "Pauli view requires distinct locations within one storage owner",
        )
    register = owners[0]
    width = len(view)
    if not operator.children and operator.token.text != "[":
        bound = self.pauli_bindings.get(operator.token.text)
        if bound is not None:
            operator = bound
        elif operator.token.text not in self.constants:
            family = "PauliTerm[]" if token.text == "qaoa_penalty" else "SparsePauliOp"
            reject(operator.token, "pauli", f"unknown {family} constant")
        else:
            operator = self.constants[operator.token.text]
        if operator.token.text not in {"SparsePauliOp", "PauliTerm[]"}:
            reject(target_arg.token, "pauli", "operator constant must be a Pauli sum")
    sparse = operator.token.text == "SparsePauliOp"
    terms = operator
    if sparse:
        terms, width_arg = operator.children
        if _sparse_width(width_arg) != width:
            reject(
                width_arg.token,
                "pauli",
                "num_qubits must match the target array width",
            )
    kind = (
        self.parameters[target_arg.token.text].kind
        if not target_arg.children
        else "qbit[]"
    )
    if not terms.children or (kind != "qbit[]" and not (sparse and kind == "qnum")):
        reject(
            token,
            "evolution",
            "evolution requires a nonempty Pauli sum and a qubit array or sparse qnum target",
        )
    prepared = []
    for term in terms.children:
        axes, coefficient = term.children
        support = _pauli_support(axes, width, sparse=sparse)
        mapped_support = tuple((view[index][1], axis) for index, axis in support)
        prepared.append((mapped_support, self.evolution_angle(coefficient, token)))
    return register, prepared


def qpe_flexible(self: _Adapt, statement: Statement) -> None:
    token = statement.token
    assert statement.power_callback is not None
    power, body = statement.power_callback
    if (
        power.text in self.parameters
        or power.text in self.constants
        or power.text in _INTRINSICS
        or power.text in self.loop_indices
    ):
        reject(power, "qpe", "power parameter must be unshadowed")
    if is_modular_power(statement):
        lower_modular_qpe(self, statement)
        return
    body = specialize_power_body(self, body, power)
    if (
        len(body) != 1
        or body[0].token.text != "suzuki_trotter"
        or len(body[0].arguments) != 5
    ):
        reject(token, "qpe", "power callback requires one suzuki_trotter call")
    phase_name = self.atom(statement.arguments[0])
    parameter = self.parameters.get(phase_name.text)
    if (
        parameter is None
        or parameter.kind not in {"qphase", "qphase_signed"}
        or parameter.width is None
    ):
        reject(phase_name, "qpe", "expected an all-fractional phase output")
    width = parameter.width
    if width > 8:
        reject(
            phase_name,
            "qpe",
            "phase estimation supports at most eight phase qubits",
        )
    phase = self.register(phase_name, allow_fractional_output=True)
    last_use = next(
        (op for op in reversed(self.operations) if phase in op.operands[1:]), None
    )
    if last_use is None or last_use.name != "qreg.allocate":
        reject(
            phase_name,
            "qpe",
            "phase output must be initialized by allocate and unused",
        )
    operator, time, order, repetitions, target = body[0].arguments
    if any(arg.children or arg.token.text != "1" for arg in (order, repetitions)):
        reject(token, "qpe", "power callback requires first order and one repetition")
    state, terms = self.pauli_terms(body[0].token, operator, target)

    def coefficient(node: Expression, value: int) -> float:
        text = node.token.text
        if text == "*" and len(node.children) == 2:
            left, right = node.children
            return coefficient(left, value) * coefficient(right, value)
        if text == "-" and len(node.children) == 1:
            return -coefficient(node.children[0], value)
        if not node.children:
            if text == power.text:
                return float(value)
            if text == "pi" and "pi" not in self.parameters:
                return math.pi
        return _literal_angle(node, token)

    for index in range(width):
        self.effect_op("qreg.gate", token, (phase,), gate="h", index=index)
    for index in range(width):
        evolution = coefficient(time, 1 << index)
        for support, weight in terms:
            angle = evolution * weight
            if not math.isfinite(2 * angle):
                reject(time.token, "qpe", "controlled evolution angle must be finite")
            self.pauli_exponential(token, state, support, angle, control=(phase, index))
    self.effect_op("qreg.inverse_qft", token, (phase,))
    self.fractional_bits[phase_name.text] = width


def pauli_exponential(
    self: _Adapt,
    token: Token,
    register: Value,
    support: tuple[tuple[int, str], ...],
    angle: float,
    *,
    control: tuple[Value, int] | None = None,
) -> None:
    if not math.isfinite(angle) or not math.isfinite(2 * angle):
        reject(token, "angle", "Pauli rotation angle and doubled angle must be finite")
    values = (register,) if control is None else (register, control[0])
    attributes: dict[str, object] = {"support": support, "angle": angle}
    if control is not None:
        attributes["control_index"] = control[1]
    self.effect_op("qreg.pauli_rotation", token, values, **attributes)


def prepare_exponential_state(
    self: _Adapt, token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 2:
        reject(
            token,
            "signature",
            "prepare_exponential_state requires a rate and qubit array",
        )
    rate = self.evolution_angle(args[0], token)
    target = self.atom(args[1])
    register = self.register(target, allow_qubit_array=True)
    if self.parameters[target.text].kind != "qbit[]":
        reject(target, "type", "prepare_exponential_state requires a qubit array")
    for index, angle in enumerate(exponential_angles(rate, register_width(register))):
        self.effect_op(
            "qreg.gate", token, (register,), gate="ry", index=index, angle=angle
        )


def prepare_state_vector(
    self: _Adapt, token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 3:
        reject(
            token,
            "signature",
            f"{token.text} requires a vector, bound and output",
        )
    vector, bound_arg, output = args
    values = (
        self.classical_arrays.get(vector.token.text) if not vector.children else None
    )
    if values is None and vector.token.text != "[":
        constant = self.constants.get(vector.token.text)
        if constant is None or constant.token.text != "[":
            reject(vector.token, "amplitudes", "expected a declared real[] vector")
        vector = constant
    if not bound_arg.children and bound_arg.token.text in self.constants:
        constant = self.constants[bound_arg.token.text]
        if constant.token.text in {"[", "SparsePauliOp"}:
            reject(bound_arg.token, "amplitudes", "bound requires a scalar real")
        bound_arg = constant
    count = len(values) if values is not None else len(vector.children)
    if not 2 <= count <= MAX_PREPARED_AMPLITUDES or count & (count - 1):
        reject(
            vector.token,
            "amplitudes",
            "amplitude count must be a power of two in 2..256",
        )

    amplitudes = (
        values
        if values is not None
        else tuple(_preparation_value(item) for item in vector.children)
    )
    bound = _preparation_value(bound_arg)
    inplace = token.text == "inplace_prepare_amplitudes"
    if inplace and (
        not math.isclose(math.hypot(*amplitudes), 1.0, abs_tol=1e-12, rel_tol=0)
        or not math.isfinite(bound)
        or bound < 0
    ):
        reject(
            vector.token,
            "amplitudes",
            "inplace preparation requires a normalized finite real vector and a nonnegative finite bound",
        )
    if token.text == "prepare_state":
        if any(value < 0 for value in amplitudes) or not math.isclose(
            math.fsum(amplitudes), 1.0, rel_tol=0, abs_tol=1e-12
        ):
            reject(
                vector.token,
                "probabilities",
                "probabilities must be nonnegative and sum to one",
            )
        if bound < 0:
            reject(
                bound_arg.token,
                "probabilities",
                "probability bound must be nonnegative",
            )
        amplitudes = tuple(math.sqrt(value) for value in amplitudes)
        # The shared preparation is exact, independent of the probability tolerance.
        bound = 0.0
    if inplace and (output.children or output.token.text in self.views):
        locations = tuple(reversed(self.array_view(output, allow_signed=True)))
        if len(locations) != count.bit_length() - 1:
            reject(
                output.token,
                "width",
                "amplitude vector must match the output width",
            )
        self.effect_op(
            "qreg.inplace_prepare_amplitudes",
            token,
            tuple(value for value, _ in locations),
            indices=tuple(index for _, index in locations),
            amplitudes=amplitudes,
            bound=bound,
        )
        return
    target = self.atom(output)
    width = count.bit_length() - 1
    destination = self.register(
        target, width, allow_qubit_array=True, allocate_local=not inplace
    )
    if register_width(destination) != width:
        reject(target, "width", "amplitude vector must match the output width")
    self.effect_op(
        "qreg.inplace_prepare_amplitudes" if inplace else "qreg.prepare_amplitudes",
        token,
        (destination,),
        amplitudes=amplitudes,
        bound=bound,
    )


def xor_update(
    self: _Adapt, token: Token, destination: Expression, source: Expression
) -> None:
    if source.token.text == "==":
        tested, literal = source.children
        if (
            not tested.children
            and not literal.children
            and literal.token.text.isdecimal()
            and 6 <= (self.widths.get(tested.token.text) or 0) <= 8
            and self.parameters[tested.token.text].role != "const"
            and lower_indexed_control(
                self,
                source,
                Statement(Token("X", token.location), (destination,)),
            )
        ):
            return
    if source.token.text in _STATIC_COMPARISONS or source.token.text == "and":
        register, bit = self.gate_target(destination, allow_numeric=True)
        pi = Expression(Token("pi", token.location))
        self.real_phase(
            token,
            destination,
            pi,
            condition=source,
            disjoint_condition_target=(register, bit),
            conjugate_xor=True,
        )
        return
    field = self.record_field(destination)
    if field is not None:
        parameter, view = field
        if parameter.kind == "qbit" and source.token.text == ".":
            control = self.gate_target(source)
            self.emit_indexed_x(token, (control, view[0]))
            return
        if parameter.kind not in {"qbit", "qnum"}:
            reject(destination.token, "type", "XOR requires an unsigned scalar field")
        mask = self.static_integer(source)
        if not 0 <= mask < 1 << len(view):
            reject(source.token, "width", "XOR mask must fit the target width")
        for i, (owner, bit) in enumerate(view):
            if mask & (1 << i):
                self.effect_op("qreg.gate", token, (owner,), gate="x", index=bit)
        if not mask:
            owner, bit = view[0]
            self.effect_op("qreg.gate", token, (owner,), gate="i", index=bit)
        return
    target = self.atom(destination)
    if (
        target.text in self.views
        and self.parameters[target.text].kind == "qbit"
        and self.parameters[target.text].role == "inout"
        and not source.children
        and source.token.text in {"0", "1"}
    ):
        value, index = self.gate_target(destination)
        self.effect_op(
            "qreg.gate",
            token,
            (value,),
            gate="x" if source.token.text == "1" else "i",
            index=index,
        )
        return
    register = self.register(target)
    if self.local_scope is not None or self.parameters[target.text].kind != "qnum":
        reject(
            token,
            "statement",
            "in-place XOR requires an integer output at function scope",
        )
    width = register_width(register)
    magnitude = source.children[0] if source.token.text == "-" else source
    if not magnitude.children and magnitude.token.text.isdecimal():
        mask = int(magnitude.token.text)
        if source.token.text == "-":
            if target.text not in self.signed or mask > 1 << (width - 1):
                reject(
                    source.token,
                    "width",
                    "negative XOR mask must fit the signed target",
                )
            mask = (-mask) & ((1 << width) - 1)
        if mask >= 1 << width:
            reject(source.token, "width", "XOR mask must fit the target width")
        for bit in range(width):
            if mask & (1 << bit):
                self.effect_op("qreg.gate", token, (register,), gate="x", index=bit)
        if not mask:
            # The verifier must still check that the target is writable and initialized.
            self.effect_op(
                "qreg.gate", token, (register,), gate="rz", index=0, angle=0.0
            )
        return
    other = self.atom(source)
    operand = self.register(other)
    operand_width = register_width(operand)
    if (
        operand == register
        or operand_width > width
        or self.parameters[other.text].kind != "qnum"
    ):
        reject(
            source.token,
            "signature",
            "XOR source must be a distinct integer fitting the target",
        )
    count = width if other.text in self.signed else operand_width
    for bit in range(count):
        self.effect_op(
            "qreg.pair_gate",
            token,
            (operand, register),
            gate="cx",
            indices=(min(bit, operand_width - 1), bit),
        )
