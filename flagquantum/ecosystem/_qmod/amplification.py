"""Checked product preparations and exact Grover rotation with a retained ancilla."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_type
from ...compiler._hybrid.model import Operation, Value, ValueId
from .phase import expand_phase
from .predicates import expand_predicate_oracles
from .quantum_callbacks import Lambda
from .syntax import Expression, Function, Parameter, Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt

MAX_AMPLIFICATION_BITS = 7
MAX_AMPLIFICATION_ROUNDS = 32
AMPLITUDE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Amplification:
    width: int
    angles: tuple[float, ...]
    oracle: tuple[Statement, ...]
    marked: tuple[int, ...]
    amplitude: float
    actual_amplitude: float
    rounds: int
    ancilla_angle: float


def number(
    node: Expression, constants: Mapping[str, Expression], depth: int = 0
) -> float:
    if depth >= 16:
        reject(
            node.token,
            "amplification.budget",
            "classical constant depth exceeds sixteen",
        )
    if not node.children and node.token.text in constants:
        return number(constants[node.token.text], constants, depth + 1)
    if node.token.text == "-" and len(node.children) == 1:
        return -number(node.children[0], constants, depth + 1)
    if node.children or (
        node.token.text != "pi" and not node.token.text.replace(".", "", 1).isdecimal()
    ):
        reject(
            node.token,
            "amplification.capture",
            "preparation requires closed real constants",
        )
    value = math.pi if node.token.text == "pi" else float(node.token.text)
    if not math.isfinite(value) or abs(value) > 2**53:
        reject(
            node.token,
            "amplification.parameter",
            "constant must be finite within ±2**53",
        )
    return value


def product_angles(
    vector: Expression, bound: Expression, constants: Mapping[str, Expression]
) -> tuple[float, ...]:
    vector = constants.get(vector.token.text, vector) if not vector.children else vector
    if vector.token.text not in {"[", "[]"} or not vector.children:
        reject(
            vector.token,
            "amplification.preparation",
            "expected a closed probability vector",
        )
    size = len(vector.children)
    if size < 2 or size > 1 << MAX_AMPLIFICATION_BITS or size & (size - 1):
        reject(
            vector.token,
            "amplification.budget",
            "product preparation requires one through seven bits",
        )
    values = tuple(number(v, constants) for v in vector.children)
    total = math.fsum(values)
    if (
        any(v < 0 for v in values)
        or not math.isclose(total, 1, abs_tol=1e-12, rel_tol=0)
        or number(bound, constants) < 0
    ):
        reject(
            vector.token,
            "amplification.preparation",
            "probabilities must be normalized and nonnegative with nonnegative bound",
        )
    values = tuple(v / total for v in values)
    width = size.bit_length() - 1
    marginals = tuple(
        math.fsum(v for word, v in enumerate(values) if word & (1 << bit))
        for bit in range(width)
    )
    reconstructed = tuple(
        math.prod(p if word & (1 << bit) else 1 - p for bit, p in enumerate(marginals))
        for word in range(size)
    )
    defect = math.sqrt(
        math.fsum(
            (math.sqrt(expected) - math.sqrt(value)) ** 2
            for expected, value in zip(reconstructed, values, strict=True)
        )
    )
    if defect > 1e-12:
        reject(
            vector.token,
            "amplification.preparation",
            "this preparation requires a product distribution with amplitude L2 defect at most 1e-12",
        )
    return tuple(2 * math.asin(math.sqrt(min(1, max(0, p)))) for p in marginals)


def substitute(
    body: tuple[Statement, ...], formal: str, target: Expression
) -> tuple[Statement, ...]:
    def expr(node: Expression) -> Expression:
        if not node.children and node.token.text == formal:
            return target
        return replace(node, children=tuple(expr(c) for c in node.children))

    def optional(value: tuple[Statement, ...] | None) -> tuple[Statement, ...] | None:
        return None if value is None else substitute(value, formal, target)

    result = []
    for s in body:
        if (
            s.declaration is not None
            or s.repeat_index is not None
            or s.within is not None
            or s.inverse_body is not None
            or s.predicate_callback is not None
            or s.oracle_callback is not None
        ):
            reject(
                s.token,
                "amplification.escape",
                "callback cannot declare, capture, allocate or change ownership",
            )
        result.append(
            replace(
                s,
                arguments=tuple(expr(a) for a in s.arguments),
                control_body=optional(s.control_body),
                control_else=optional(s.control_else),
            )
        )
    return tuple(result)


def callback(
    reference: Expression,
    target: Expression,
    width: int,
    definitions: Mapping[str, Function],
) -> tuple[Statement, ...]:
    if isinstance(reference, Lambda):
        if len(reference.parameters) != 1 or not reference.body:
            reject(
                reference.token,
                "amplification.signature",
                "callback requires one distinct borrowed target",
            )
        formal = reference.parameters[0]
        body = reference.body
    else:
        function = definitions.get(reference.token.text)
        if function is None or len(function.parameters) != 1:
            reject(
                reference.token,
                "amplification.signature",
                "callback requires one borrowed quantum parameter",
            )
        port = function.parameters[0]
        if (
            port.role not in {"const", "inout"}
            or port.record is not None
            or port.kind not in {"qbit", "qnum", "qbit[]"}
            or (port.width is not None and port.width != width)
        ):
            reject(
                port.name,
                "amplification.escape",
                "callback cannot consume or export quantum values and width must match",
            )
        formal, body = port.name, function.statements
    return substitute(body, formal.text, target)


def preparation(
    reference: Expression,
    target: Expression,
    width: int,
    definitions: Mapping[str, Function],
    constants: Mapping[str, Expression],
) -> tuple[float, ...]:
    if reference.token.text == "hadamard_transform":
        return (math.pi / 2,) * width
    body = callback(reference, target, width, definitions)
    if len(body) != 1:
        reject(
            reference.token,
            "amplification.preparation",
            "preparation requires one checked product-state call",
        )
    s = body[0]
    args = s.arguments
    if s.control_body is not None or s.control_else is not None:
        reject(
            s.token,
            "amplification.preparation",
            "conditional preparations are outside this product profile",
        )
    if (
        s.token.text == "inplace_prepare_state"
        and len(args) == 3
        and args[-1] == target
    ):
        angles = product_angles(args[0], args[1], constants)
        if len(angles) != width:
            reject(
                s.token, "amplification.signature", "probability vector width disagrees"
            )
        return angles
    if (
        s.token.text == "prepare_exponential_state"
        and len(args) == 2
        and args[-1] == target
    ):
        rate = number(args[0], constants)
        result = []
        for index in range(width):
            magnitude = abs(rate) * math.ldexp(1.0, index - 1)
            angle = 2 * math.atan(math.exp(-magnitude))
            result.append(math.pi - angle if rate < 0 else angle)
        return tuple(result)
    reject(
        s.token,
        "amplification.preparation",
        "expected a closed product probability or exponential preparation",
    )


def oracle(
    reference: Expression,
    target: Expression,
    actual: Parameter,
    definitions: Mapping[str, Function],
    constants: Mapping[str, Expression],
) -> tuple[Statement, ...]:
    if reference.token.text == "Z":
        if actual.width != 1:
            reject(
                reference.token,
                "amplification.signature",
                "Z oracle requires a one-bit target",
            )
        phase_target = (
            Expression(
                Token("[", target.token.location),
                (target, Expression(Token("0", target.token.location))),
            )
            if actual.kind == "qbit[]"
            else target
        )
        return (
            Statement(
                Token("phase", reference.token.location),
                (phase_target, Expression(Token("pi", reference.token.location))),
            ),
        )
    assert actual.width is not None
    body = callback(reference, target, actual.width, definitions)
    mini = {name: f for name, f in definitions.items() if f.permutation}
    mini["main"] = Function(Token("main", reference.token.location), (actual,), body)
    normalized = expand_predicate_oracles(mini, constants, {})["main"].statements

    def check(statements: tuple[Statement, ...]) -> None:
        for s in statements:
            if (
                s.token.text == "phase"
                and len(s.arguments) in {1, 2}
                and s.control_body is None
            ):
                continue
            if (
                s.token.text == "control"
                and s.control_body is not None
                and s.control_else is None
            ):
                check(s.control_body)
                continue
            reject(
                s.token,
                "amplification.oracle",
                "oracle must have closed diagonal sign-phase effects only",
            )

    check(normalized)
    return normalized


def oracle_phases(
    statements: tuple[Statement, ...], condition: Expression | None = None
) -> tuple[tuple[Expression, Expression, Expression | None], ...]:
    result: list[tuple[Expression, Expression, Expression | None]] = []
    for s in statements:
        if s.control_body is not None:
            guard = (
                s.arguments[0]
                if condition is None
                else Expression(
                    Token("&", s.token.location), (condition, s.arguments[0])
                )
            )
            result.extend(oracle_phases(s.control_body, guard))
        else:
            expr, angle = (
                (Expression(Token("1", s.token.location)), s.arguments[0])
                if len(s.arguments) == 1
                else s.arguments
            )
            result.append((expr, angle, condition))
    return tuple(result)


def marked_words(
    statements: tuple[Statement, ...],
    target: Expression,
    width: int,
    constants: Mapping[str, Expression],
) -> tuple[int, ...]:
    value = Value(ValueId("amplification.proof", 0), register_type(width))

    def quantum(node: Expression) -> tuple[tuple[Value, int], ...] | None:
        if not node.children and node.token.text == target.token.text:
            return tuple((value, i) for i in range(width))
        if (
            node.token.text == "["
            and node.children[0] == target
            and not node.children[1].children
            and node.children[1].token.text.isdecimal()
        ):
            bit = int(node.children[1].token.text)
            if not 0 <= bit < width:
                reject(
                    node.token, "amplification.bounds", "oracle index is outside target"
                )
            return ((value, bit),)
        return None

    phases = [0.0] * (1 << width)
    for expr, angle, condition in oracle_phases(statements):
        polynomial, locations = expand_phase(
            expr, quantum, lambda n: number(n, constants), condition=condition
        )
        coefficient = number(angle, constants)
        for word in range(1 << width):
            basis = sum(
                ((word >> bit) & 1) << i for i, (_, bit) in enumerate(locations)
            )
            phases[word] += coefficient * math.fsum(
                weight for mask, weight in polynomial.items() if mask & basis == mask
            )
    marked = []
    for word, phase in enumerate(phases):
        multiple = phase / math.pi
        if not math.isclose(multiple, round(multiple), abs_tol=1e-12, rel_tol=0):
            reject(
                target.token,
                "amplification.oracle",
                "oracle must apply signs +1 or -1 on every word",
            )
        if round(multiple) % 2:
            marked.append(word)
    return tuple(marked)


def expand_amplification(
    definitions: Mapping[str, Function], constants: Mapping[str, Expression]
) -> dict[str, Function]:
    result = dict(definitions)
    main = result["main"]
    ports = {p.name.text: p for p in main.parameters}
    statements = []
    for s in main.statements:
        if s.token.text == "allocate" and len(s.arguments) == 2:
            size, target = s.arguments
            if (
                not size.children
                and size.token.text.isdecimal()
                and target.token.text in ports
            ):
                p = ports[target.token.text]
                if p.width is None:
                    ports[p.name.text] = replace(p, width=int(size.token.text))
        if s.token.text != "exact_amplitude_amplification":
            statements.append(s)
            continue
        if len(s.arguments) != 4:
            reject(
                s.token,
                "amplification.signature",
                "exact amplification requires amplitude, oracle, preparation and target",
            )
        amplitude_arg, oracle_ref, prep_ref, target = s.arguments
        actual = ports.get(target.token.text)
        if (
            target.children
            or actual is None
            or actual.role != "output"
            or actual.kind not in {"qbit", "qnum", "qbit[]"}
            or actual.record is not None
            or actual.width is None
            or not 1 <= actual.width <= MAX_AMPLIFICATION_BITS
        ):
            reject(
                target.token,
                "amplification.signature",
                "exact amplification requires one declared unsigned output of one through seven bits",
            )
        width = actual.width
        angles = preparation(prep_ref, target, width, result, constants)
        phases = oracle(oracle_ref, target, actual, result, constants)
        marked = marked_words(phases, target, width, constants)
        probabilities = [
            math.prod(
                (
                    math.sin(angle / 2) ** 2
                    if word & (1 << bit)
                    else math.cos(angle / 2) ** 2
                )
                for bit, angle in enumerate(angles)
            )
            for word in range(1 << width)
        ]
        actual_amplitude = math.sqrt(math.fsum(probabilities[w] for w in marked))
        amplitude = number(amplitude_arg, constants)
        if not 0 < amplitude <= 1:
            reject(
                amplitude_arg.token,
                "amplification.amplitude",
                "success amplitude must lie in (0,1]",
            )
        if abs(amplitude - actual_amplitude) > AMPLITUDE_TOLERANCE:
            reject(
                amplitude_arg.token,
                "amplification.amplitude",
                f"supplied amplitude {amplitude:.17g} disagrees with proven amplitude {actual_amplitude:.17g}; original parameters are not corrected",
            )
        theta = math.asin(amplitude)
        if theta < math.pi / (4 * MAX_AMPLIFICATION_ROUNDS + 2) - 1e-15:
            reject(
                amplitude_arg.token,
                "amplification.budget",
                "exact amplification exceeds 32 rounds",
            )
        rounds = max(0, math.ceil(math.pi / (4 * theta) - 0.5 - 1e-14))
        if rounds > MAX_AMPLIFICATION_ROUNDS:
            reject(
                amplitude_arg.token,
                "amplification.budget",
                "exact amplification exceeds 32 rounds",
            )
        ratio = min(1.0, math.sin(math.pi / (4 * rounds + 2)) / amplitude)
        rotated = min(1.0, actual_amplitude * ratio)
        residual = abs(math.cos((2 * rounds + 1) * math.asin(rotated)))
        if residual > 1e-12:
            reject(
                amplitude_arg.token,
                "amplification.amplitude",
                "supplied amplitude leaves a nonzero Grover remainder above 1e-12; exact cleanup is not certified",
            )
        recipe = Amplification(
            width,
            angles,
            phases,
            marked,
            amplitude,
            actual_amplitude,
            rounds,
            2 * math.acos(ratio),
        )
        statements.append(replace(s, arguments=(target,), amplification=recipe))
        # Width specialization is justified only for predicates participating here.
        source_oracle = (
            ()
            if oracle_ref.token.text == "Z"
            else callback(oracle_ref, target, width, result)
        )
        used_predicates = {
            node.arguments[0].token.text
            for node in source_oracle
            if node.token.text == "phase_oracle" and node.arguments
        }
        for name, f in tuple(result.items()):
            if f.permutation and name in used_predicates:
                result[name] = replace(
                    f,
                    parameters=tuple(
                        (
                            replace(p, width=width)
                            if p.role == "const"
                            and p.kind == "qnum"
                            and p.width is None
                            else p
                        )
                        for p in f.parameters
                    ),
                )
    if any(s.amplification is not None for s in statements):
        result["main"] = replace(
            main,
            parameters=tuple(ports[p.name.text] for p in main.parameters),
            statements=tuple(statements),
        )
    return result


def lower_amplification(
    owner: _Adapt, token: Token, target: Expression, recipe: Amplification
) -> None:
    if owner.scope != "main" or owner.helper_stack:
        reject(
            token,
            "amplification.scope",
            "exact amplification is admitted only at the root",
        )
    register = owner.register(owner.atom(target), allow_qubit_array=True)
    history = [op for op in owner.operations if register in op.operands]
    if (
        len(history) != 1
        or history[0].name not in {"qreg.allocate", "qreg.initialize"}
        or history[0].attributes.get("value", 0) != 0
    ):
        reject(
            token,
            "amplification.zero",
            "exact amplification requires a proven initialized zero input",
        )

    def rotate(angles: tuple[float, ...], inverse: bool = False) -> None:
        order = reversed(tuple(enumerate(angles))) if inverse else enumerate(angles)
        for index, angle in order:
            owner.effect_op(
                "qreg.gate",
                token,
                (register,),
                gate="ry",
                index=index,
                angle=-angle if inverse else angle,
            )

    rotate(recipe.angles)
    if recipe.rounds == 0:
        return
    name = f"$amplification_aux{owner.counter}"
    aux_token = Token(name, token.location)
    owner.parameters[name] = Parameter(aux_token, "qbit[]", 1, "local")
    owner.widths[name] = 1
    aux = Value(ValueId(owner.scope, owner.counter), register_type(1))
    owner.counter += 1
    owner.registers[name] = aux
    owner.append_op(
        token,
        Operation(
            "qreg.hidden",
            results=(aux,),
            attributes={"name": name, "workspace": True},
            location=token.location,
        ),
    )
    owner.effect_op("qreg.allocate", token, (aux,))

    def expression(text: str, *children: Expression) -> Expression:
        return Expression(Token(text, token.location), tuple(children))

    aux_zero = expression(
        "==", expression("[", expression(name), expression("0")), expression("0")
    )
    target_zero = expression("==", target, expression("0"))
    if owner.parameters[target.token.text].kind == "qbit[]":
        target_zero = expression("1")
        for bit in range(recipe.width):
            target_zero = expression(
                "&",
                target_zero,
                expression(
                    "==", expression("[", target, expression(str(bit))), expression("0")
                ),
            )
    zero = expression("&", target_zero, aux_zero)
    pi = expression("pi")

    def aux_rotate(sign: int) -> None:
        owner.effect_op(
            "qreg.gate",
            token,
            (aux,),
            gate="ry",
            index=0,
            angle=sign * recipe.ancilla_angle,
        )

    aux_rotate(1)
    for _ in range(recipe.rounds):
        for expr, angle, condition in oracle_phases(recipe.oracle):
            guard = (
                aux_zero if condition is None else expression("&", condition, aux_zero)
            )
            owner.real_phase(token, expr, angle, condition=guard)
        aux_rotate(-1)
        rotate(recipe.angles, True)
        owner.real_phase(token, expression("1"), pi, condition=zero)
        owner.effect_op("qreg.phase", token, (), angle=math.pi)
        rotate(recipe.angles)
        aux_rotate(1)
    owner.effect_op("qreg.drop", token, (aux,))
