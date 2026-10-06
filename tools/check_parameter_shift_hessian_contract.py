"""Verify the parameter-shift Hessian against the opcode declaration.

``fq.gradients.parameter_shift_hessian`` answers a second derivative with the
first-order rule read twice: the mixed second derivative of a gate parameter is
the product rule of that parameter's *declared* ``(coefficient, shift)`` terms,
evaluated at the sums of the two shifts. The coefficients therefore live in
``flagquantum/core/operator_schema.py`` and nowhere else, and this gate proves
that by re-deriving the composition from ``OPERATOR_SCHEMAS`` and comparing both
the values and the evaluation counts with the implementation.

Three routes are computed and only two of them count as evidence. The composed
rule and a second application of the declared rule share the same analytic
statement, so their agreement would show self-consistency and not correctness;
that pair is compared at ``1e-12`` to catch a divergence between the gate's
derivation and the implementation, and the correctness claim is the distance
from Richardson central differences of ``fq.run`` alone, which shares no code
with either. A third defect mode -- a second-order coefficient table appearing in
the implementation -- shows up as a value difference against the from-declaration
derivation, which is why the gate derives rather than imports.

The gate fails closed: an opcode row that disagrees with the declaration, a cost
row the implementation does not charge, a refused case that stops refusing, a
result that carries an autograd graph, or a ``hessian`` method value accepted by
``fq.gradient`` are each a failure rather than a silent drift.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS  # noqa: E402

CONTRACT = ROOT / "contracts" / "parameter-shift-hessian-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum" / "gradients.py"
ENTRY_POINT = "parameter_shift_hessian"

# The exception classes a refusal row may name. A name outside this map is a
# contract defect rather than an unmeasured case.
EXCEPTIONS: dict[str, type[BaseException]] = {}

# Two analytic routes that share the declared rule must agree far more tightly
# than either agrees with a difference scheme; this is the self-consistency
# bound, not the correctness bound.
DERIVATION_TOLERANCE = 1e-12
# Cells agree with the composed rule through floating-point summation order.
SYMMETRY_TOLERANCE = 1e-12


def _exceptions() -> dict[str, type[BaseException]]:
    from flagquantum.errors import CapabilityError, ValidationError

    return {
        "TypeError": TypeError,
        "ValueError": ValueError,
        "ValidationError": ValidationError,
        "CapabilityError": CapabilityError,
    }


def _load_contract(path: Path = CONTRACT) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _gradients_module() -> ModuleType:
    import flagquantum.gradients as gradients

    return gradients


# The two symbols this route is made of, so the census below can be scoped to
# them. The batch profile further down the module legitimately names opcodes in
# code -- it states a protocol scope that is a frozen list by design -- and
# counting its literals here would confuse the two contracts.
ROUTE_SYMBOLS = ("parameter_shift_hessian", "_composed_shift_terms")


def _opcodes_named_in_route(path: Path = IMPLEMENTATION) -> list[str]:
    """Opcodes the second-order route names as string literals in its own code.

    A name in the route's code would let it special-case a gate instead of
    reading the declaration. Its docstring is prose and does not decide
    anything, so only string expressions and not docstrings are counted.
    """

    import ast

    tree = ast.parse(path.read_text())
    literals: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in ROUTE_SYMBOLS:
            continue
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            body = body[1:]
        for statement in body:
            for inner in ast.walk(statement):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    literals.add(inner.value)
    return sorted(literals & set(OPERATOR_SCHEMAS))


# --------------------------------------------------------------------------------------
# The declaration, read from `OPERATOR_SCHEMAS` and never restated
# --------------------------------------------------------------------------------------


def _rules(
    opcodes: Sequence[str], parameters: torch.Tensor
) -> list[list[tuple[float, float]]]:
    """Each input parameter's declared first-order rule, in declaration order."""

    collected: list[list[tuple[float, float]]] = []
    cursor = 0
    for opcode in opcodes:
        schema = OPERATOR_SCHEMAS[opcode]
        for name in schema.parameters:
            if len(parameters.reshape(-1)) <= cursor:
                raise AssertionError("witness program wants more parameters than given")
            collected.append([tuple(term) for term in schema.shift_rule(name)])
            cursor += 1
    return collected


def _witness(opcodes: Sequence[str]) -> Callable[[torch.Tensor], Any]:
    """A program whose gate parameters are driven one-to-one by the inputs."""

    fixed = (0.9, 1.4, 0.6, 1.1)

    def build(values: torch.Tensor):
        flat = values.reshape(-1)
        circuit = fq.Circuit(3, dtype=torch.complex128)
        for index in range(3):
            circuit = circuit.ry(index, theta=fixed[index])
        circuit = circuit.cx(0, 1).cx(1, 2)
        cursor = 0
        for position, opcode in enumerate(opcodes):
            schema = OPERATOR_SCHEMAS[opcode]
            qubits = tuple(range(position, position + max(1, schema.arity)))
            driven = {
                name: flat[cursor + offset]
                for offset, name in enumerate(schema.parameters)
            }
            cursor += len(schema.parameters)
            circuit = circuit.gate(opcode, qubits, **driven)
        # Turn the measurement basis so no cell of the reference is degenerate.
        return circuit.cx(2, 0).ry(0, theta=fixed[3])

    return build


def _width(opcodes: Sequence[str]) -> int:
    return sum(len(OPERATOR_SCHEMAS[opcode].parameters) for opcode in opcodes)


def _point(width: int) -> torch.Tensor:
    return torch.tensor((0.31, 0.72, -0.41, 0.55)[:width], dtype=torch.float64)


# --------------------------------------------------------------------------------------
# The three routes
# --------------------------------------------------------------------------------------


def _scalar(circuit: Any, mode: str) -> torch.Tensor:
    result = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode=mode),
        outputs=fq.expectation(fq.Z(0)),
    )
    return result.expectations[0].reshape(()).to(torch.float64)


def _evaluate(
    opcodes: Sequence[str], mode: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The loss of the witness program, as a function of the parameters."""

    build = _witness(opcodes)
    return lambda values: _scalar(build(values), mode)


def _loss(opcodes: Sequence[str], mode: str) -> Callable[[Any], torch.Tensor]:
    """The same loss as the implementation receives: circuit in, scalar out."""

    return lambda circuit: _scalar(circuit, mode)


def _composed_terms(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
    position: tuple[int, int],
    width: int,
) -> list[tuple[float, tuple[float, ...]]]:
    """Every product-rule term pair, with no merging and no rounding.

    Merging equal displacements changes how many circuits are evaluated and not
    what the sum is, so the value comparison runs on the unmerged pairs and the
    merge rule is checked separately by :func:`_distinct_displacements`.
    """

    terms: list[tuple[float, tuple[float, ...]]] = []
    for left_coefficient, left_shift in left:
        for right_coefficient, right_shift in right:
            displacement = [0.0] * width
            displacement[position[0]] += left_shift
            displacement[position[1]] += right_shift
            terms.append((left_coefficient * right_coefficient, tuple(displacement)))
    return terms


def _distinct_displacements(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
    position: tuple[int, int],
    width: int,
) -> int:
    """How many circuits the contract says a merged cell charges for."""

    collected: dict[tuple[float, ...], float] = {}
    for coefficient, displacement in _composed_terms(left, right, position, width):
        key = tuple(round(value, 12) for value in displacement)
        collected[key] = collected.get(key, 0.0) + coefficient
    return sum(1 for coefficient in collected.values() if coefficient != 0.0)


def _cell_matrix(
    rules: Sequence[Sequence[tuple[float, float]]],
    parameters: torch.Tensor,
    evaluate: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    width = len(rules)
    rows = []
    for left in range(width):
        cells = []
        for right in range(width):
            total = torch.zeros((), dtype=torch.float64)
            for coefficient, displacement in _composed_terms(
                rules[left], rules[right], (left, right), width
            ):
                shifted = parameters + torch.tensor(displacement, dtype=torch.float64)
                total = total + coefficient * evaluate(shifted)
            cells.append(total)
        rows.append(torch.stack(cells))
    return torch.stack(rows)


@functools.cache
def _derived(opcodes: tuple[str, ...], mode: str) -> tuple[torch.Tensor, int]:
    """The gate's own composition of the declared rule, plus its cell counts."""

    width = _width(opcodes)
    rules = _rules(opcodes, _point(width))
    count = sum(
        _distinct_displacements(rules[left], rules[right], (left, right), width)
        for left in range(width)
        for right in range(width)
    )
    matrix = _cell_matrix(rules, _point(width), _evaluate(opcodes, mode))
    return matrix, count


@functools.cache
def _measured(opcodes: tuple[str, ...], mode: str) -> tuple[torch.Tensor, int]:
    """What the implementation returns, and how many circuits it charged."""

    width = _width(opcodes)
    counter: list[int] = []
    build = _witness(opcodes)

    def loss(circuit: Any) -> torch.Tensor:
        counter.append(1)
        return _scalar(circuit, mode)

    matrix = _gradients_module().parameter_shift_hessian(build, _point(width), loss)
    return matrix, len(counter)


@functools.cache
def _difference(opcodes: tuple[str, ...], mode: str) -> torch.Tensor:
    """Richardson central differences of ``fq.run`` alone."""

    width = _width(opcodes)
    parameters = _point(width)
    evaluate = _evaluate(opcodes, mode)
    steps = (0.01, 0.005, 0.0025)
    matrix = torch.zeros((width, width), dtype=torch.float64)
    for left in range(width):
        for right in range(width):

            def mixed(displacement: float, left=left, right=right) -> torch.Tensor:
                def at(one: float, two: float) -> torch.Tensor:
                    shifted = parameters.clone()
                    shifted[left] += one
                    shifted[right] += two
                    return evaluate(shifted)

                return (
                    at(displacement, displacement)
                    - at(displacement, -displacement)
                    - at(-displacement, displacement)
                    + at(-displacement, -displacement)
                ) / (4 * displacement * displacement)

            samples = [mixed(step) for step in steps]
            for level in range(2):
                factor = 4 ** (level + 1)
                samples = [
                    (factor * samples[index + 1] - samples[index]) / (factor - 1)
                    for index in range(len(samples) - 1)
                ]
            matrix[left, right] = samples[0]
    return matrix


def _first_order_cost(opcodes: tuple[str, ...], mode: str) -> int:
    """Evaluations the first-order entry point charges for the same program."""

    width = _width(opcodes)
    counter: list[int] = []
    build = _witness(opcodes)

    def loss(circuit: Any) -> torch.Tensor:
        counter.append(1)
        return _scalar(circuit, mode)

    _gradients_module().parameter_shift_gradient(build, _point(width), loss)
    return len(counter)


# --------------------------------------------------------------------------------------
# The refusal surface
# --------------------------------------------------------------------------------------


def _refusal_programs() -> dict[str, tuple[Any, torch.Tensor]]:
    """One program per refusal case, keyed by the case name in the contract."""

    fixed = (0.9, 1.4, 0.6, 1.1)

    def prepare(circuit, qubits: int):
        for index in range(qubits):
            circuit = circuit.ry(index, theta=fixed[index])
        for index in range(qubits - 1):
            circuit = circuit.cx(index, index + 1)
        return circuit

    def closing(circuit):
        return circuit.ry(0, theta=fixed[3])

    def no_occurrence(values: torch.Tensor):
        circuit = prepare(fq.Circuit(2, dtype=torch.complex128), 2)
        return closing(circuit.rz(0, theta=values[0]))

    def two_occurrences(values: torch.Tensor):
        circuit = prepare(fq.Circuit(2, dtype=torch.complex128), 2)
        return closing(circuit.rz(0, theta=values[0]).rz(1, theta=values[0]))

    def scaled_angle(values: torch.Tensor):
        circuit = prepare(fq.Circuit(2, dtype=torch.complex128), 2)
        return closing(circuit.rz(0, theta=2.0 * values[0]).rz(1, theta=values[1]))

    def structure_changes(values: torch.Tensor):
        circuit = prepare(fq.Circuit(2, dtype=torch.complex128), 2)
        if float(values[0]) > 0:
            circuit = circuit.rz(0, theta=values[0])
        return closing(circuit.rz(1, theta=values[1]))

    def custom_matrix(values: torch.Tensor):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        return circuit.gate("bit_flip", (0,), probability=values[0])

    return {
        "empty_parameters": (
            _witness(("rz", "rz")),
            torch.tensor([], dtype=torch.float64),
        ),
        "complex_parameters": (
            _witness(("rz", "rz")),
            torch.tensor([0.31 + 0.0j, 0.72 + 0.0j], dtype=torch.complex128),
        ),
        "non_finite_parameters": (
            _witness(("rz", "rz")),
            torch.tensor([0.31, float("nan")], dtype=torch.float64),
        ),
        "not_a_tensor": (_witness(("rz", "rz")), [0.31, 0.72]),
        "one_input_controls_no_occurrence": (
            no_occurrence,
            torch.tensor([0.31, 0.72], dtype=torch.float64),
        ),
        "one_input_controls_two_occurrences": (
            two_occurrences,
            torch.tensor([0.31], dtype=torch.float64),
        ),
        "scaled_angle": (
            scaled_angle,
            torch.tensor([0.31, 0.72], dtype=torch.float64),
        ),
        "structure_changes_with_the_parameter": (
            structure_changes,
            torch.tensor([0.31, 0.72], dtype=torch.float64),
        ),
        "custom_matrix": (custom_matrix, torch.tensor([0.31], dtype=torch.float64)),
        "loss_is_not_scalar": (
            _witness(("rz", "rz")),
            torch.tensor([0.31, 0.72], dtype=torch.float64),
        ),
        "stabilizer_mode": (
            _witness(("rz", "rz")),
            torch.tensor([0.31, 0.72], dtype=torch.float64),
        ),
    }


@functools.cache
def _refusal_measurement(case: str) -> tuple[str, str]:
    """The exception class name and message the case actually produces."""

    build, parameters = _refusal_programs()[case]
    if case == "loss_is_not_scalar":

        def loss(circuit: Any) -> torch.Tensor:
            value = _scalar(circuit, "statevector")
            return torch.stack([value, value])

    elif case == "stabilizer_mode":

        def loss(circuit: Any) -> torch.Tensor:
            return _scalar(circuit, "stabilizer")

    else:

        def loss(circuit: Any) -> torch.Tensor:
            return _scalar(circuit, "statevector")

    try:
        _gradients_module().parameter_shift_hessian(build, parameters, loss)
    except BaseException as error:
        return type(error).__name__, str(error)
    return "", ""


# --------------------------------------------------------------------------------------
# Contract checks
# --------------------------------------------------------------------------------------


def _declaration_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = {row["name"]: row for row in contract.get("opcode", ())}
    if len(rows) != len(contract.get("opcode", ())):
        errors.append("the opcode rows repeat a name")
    missing = sorted(set(OPERATOR_SCHEMAS) - set(rows))
    extra = sorted(set(rows) - set(OPERATOR_SCHEMAS))
    if missing:
        errors.append(f"the opcode rows omit registered opcodes {missing}")
    if extra:
        errors.append(f"the opcode rows name unregistered opcodes {extra}")
    for name in sorted(set(OPERATOR_SCHEMAS) & set(rows)):
        schema = OPERATOR_SCHEMAS[name]
        row = rows[name]
        if list(row["declared_parameters"]) != list(schema.parameters):
            errors.append(
                f"{name}: declared_parameters is {list(row['declared_parameters'])} "
                f"but the declaration says {list(schema.parameters)}"
            )
            continue
        frequencies = [list(values) for values in schema.parameter_frequencies]
        if [list(values) for values in row["declared_frequencies"]] != frequencies:
            errors.append(
                f"{name}: declared_frequencies is "
                f"{[list(values) for values in row['declared_frequencies']]} but the "
                f"declaration says {frequencies}"
            )
        terms = (
            [len(schema.shift_rule(parameter)) for parameter in schema.parameters]
            if schema.differentiable
            else []
        )
        if list(row["shift_terms"]) != terms:
            errors.append(
                f"{name}: shift_terms is {list(row['shift_terms'])} but the "
                f"declaration produces {terms}"
            )
        if bool(row["admitted"]) is not bool(schema.differentiable):
            errors.append(
                f"{name}: admitted is {row['admitted']} but the declaration's "
                f"differentiable flag is {schema.differentiable}"
            )
        if not row["admitted"] and not row.get("refusal_reason"):
            errors.append(f"{name}: a refused opcode row states no refusal_reason")
        if row["admitted"] and row.get("refusal_reason"):
            errors.append(f"{name}: an admitted opcode row states a refusal_reason")
    return errors


def _admitted(contract: Mapping[str, Any]) -> list[str]:
    return sorted(row["name"] for row in contract.get("opcode", ()) if row["admitted"])


def _cost_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = {row["opcode"]: row for row in contract.get("cost", ())}
    admitted = _admitted(contract)
    if sorted(rows) != admitted:
        errors.append(
            f"the cost rows name {sorted(rows)} but the admitted opcodes are "
            f"{admitted}"
        )
    for name in sorted(set(rows) & set(admitted)):
        row = rows[name]
        opcodes = (name,)
        _, derived = _derived(opcodes, "statevector")
        _, measured = _measured(opcodes, "statevector")
        if list(row["declared_terms"]) != [
            len(OPERATOR_SCHEMAS[name].shift_rule(parameter))
            for parameter in OPERATOR_SCHEMAS[name].parameters
        ]:
            errors.append(
                f"{name}: declared_terms is {list(row['declared_terms'])} but the "
                "declaration produces a different term count per parameter"
            )
        width = _width(opcodes)
        rules = _rules(opcodes, _point(width))
        diagonal = sum(
            _distinct_displacements(rules[index], rules[index], (index, index), width)
            for index in range(width)
        )
        if int(row["diagonal_evaluations"]) != diagonal:
            errors.append(
                f"{name}: diagonal_evaluations is {row['diagonal_evaluations']} but "
                f"the composed diagonal costs {diagonal}"
            )
        if int(row["total_evaluations"]) != derived:
            errors.append(
                f"{name}: total_evaluations is {row['total_evaluations']} but the "
                f"composed rule charges {derived}"
            )
        if measured != derived:
            errors.append(
                f"{name}: the implementation charged {measured} evaluations and the "
                f"composed rule charges {derived}"
            )
    return errors


def _cell_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for row in contract.get("cell", ()):
        opcodes = tuple(row["witness"])
        width = _width(opcodes)
        rules = _rules(opcodes, _point(width))
        total = 0
        actual = None
        for left in range(width):
            for right in range(width):
                count = _distinct_displacements(
                    rules[left], rules[right], (left, right), width
                )
                total += count
                pair = sorted(
                    [
                        len(rules[left]),
                        len(rules[right]),
                    ]
                )
                on_diagonal = left == right
                if on_diagonal == (row["position"] == "diagonal") and pair == sorted(
                    row["declared_terms"]
                ):
                    actual = count
        if actual is None:
            errors.append(
                f"the {row['position']} cell row for declared_terms "
                f"{list(row['declared_terms'])} matches no cell of the witness "
                f"{list(opcodes)}"
            )
            continue
        if actual != int(row["evaluations"]):
            errors.append(
                f"the {row['position']} cell row for declared_terms "
                f"{list(row['declared_terms'])} records {row['evaluations']} "
                f"evaluations but the composed rule charges {actual}"
            )
        _, measured = _measured(opcodes, "statevector")
        if measured != total:
            errors.append(
                f"the witness {list(opcodes)} charged {measured} evaluations "
                f"overall but the composed rule charges {total}"
            )
    return errors


def _exactness_errors(contract: Mapping[str, Any]) -> list[str]:
    exactness = contract["exactness"]
    bound = float(exactness["agreement_bound"])
    floor = float(exactness["minimum_cell_magnitude"])
    modes = [str(mode) for mode in contract["protocol"]["serving_modes"]]
    errors: list[str] = []
    worst = 0.0
    smallest = math.inf
    for name in _admitted(contract):
        opcodes = (name,)
        for mode in modes:
            derived, derived_count = _derived(opcodes, mode)
            measured, measured_count = _measured(opcodes, mode)
            if measured_count != derived_count:
                errors.append(
                    f"{name} in {mode}: the implementation charged {measured_count} "
                    f"evaluations and the composed rule charges {derived_count}"
                )
            if measured.shape != derived.shape:
                errors.append(
                    f"{name} in {mode}: the implementation returned shape "
                    f"{tuple(measured.shape)} and the composed rule returns "
                    f"{tuple(derived.shape)}"
                )
                continue
            divergence = (measured - derived).abs().max().item()
            if divergence > DERIVATION_TOLERANCE:
                errors.append(
                    f"{name} in {mode}: the implementation differs from the "
                    f"from-declaration composition by {divergence}, so it reads a "
                    "coefficient the declaration does not own"
                )
            deviation = (measured - _difference(opcodes, mode)).abs().max().item()
            worst = max(worst, deviation)
            smallest = min(smallest, measured.abs().min().item())
            asymmetry = (measured - measured.transpose(0, 1)).abs().max().item()
            if asymmetry > SYMMETRY_TOLERANCE:
                errors.append(
                    f"{name} in {mode}: the matrix is asymmetric by {asymmetry}"
                )
            if deviation > bound:
                errors.append(
                    f"{name} in {mode}: the Hessian differs from the difference "
                    f"route by {deviation}, above the recorded bound {bound}"
                )
    if errors:
        return errors
    if smallest < floor:
        return [
            f"the smallest reference cell is {smallest}, below the recorded floor "
            f"{floor}, so the comparison above is vacuous"
        ]
    return []


def _refusal_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    exceptions = _exceptions()
    programs = _refusal_programs()
    rows = contract.get("refusal", ())
    if sorted(row["case"] for row in rows) != sorted(programs):
        errors.append(
            f"the refusal rows name {sorted(row['case'] for row in rows)} but the "
            f"gate drives {sorted(programs)}"
        )
    for row in rows:
        case = row["case"]
        if case not in programs:
            continue
        name = str(row["exception"])
        if name not in exceptions:
            errors.append(
                f"the refusal row {case} names the exception {name}, which is not "
                f"one of {sorted(exceptions)}"
            )
            continue
        measured_name, message = _refusal_measurement(case)
        if measured_name != name:
            errors.append(
                f"the refusal row {case} records the exception {name} but the "
                f"measurement produced {measured_name!r}"
            )
            continue
        if row["message_phrase"] not in message:
            errors.append(
                f"the refusal row {case} records the message phrase "
                f"{row['message_phrase']!r}, which is not a substring of the "
                f"measured message {message!r}"
            )
    return errors


def _rule_errors(contract: Mapping[str, Any]) -> list[str]:
    rules = contract["rules"]
    errors: list[str] = []
    gradients = _gradients_module()

    reachable = ENTRY_POINT in gradients.__all__
    if bool(rules["implementation_is_reachable_from_the_module_surface"]) != reachable:
        errors.append(
            f"implementation_is_reachable_from_the_module_surface is "
            f"{rules['implementation_is_reachable_from_the_module_surface']} but "
            f"the entry point is {reachable} in {ENTRY_POINT!r} membership"
        )

    absent = ENTRY_POINT not in fq.__all__ and not hasattr(fq, ENTRY_POINT)
    if bool(rules["implementation_is_absent_from_the_root_surface"]) != absent:
        errors.append(
            "implementation_is_absent_from_the_root_surface is "
            f"{rules['implementation_is_absent_from_the_root_surface']} but the "
            f"root surface exposes it: {absent}"
        )

    # A second coefficient table would show as a value difference against the
    # from-declaration composition, which `_exactness_errors` measures. What is
    # checked here is the weaker structural statement the rule also makes: the
    # route names no opcode in its own code, so it cannot special-case one.
    admitted = set(_admitted(contract))
    named = _opcodes_named_in_route()
    if bool(rules["coefficients_come_only_from_the_declared_rule"]) != (not named):
        errors.append(
            "coefficients_come_only_from_the_declared_rule is "
            f"{rules['coefficients_come_only_from_the_declared_rule']} but the "
            f"second-order route names the opcodes {named} in code"
        )

    width = 3
    opcodes = ("rz", "rz", "rz")
    costs = [_measured(opcodes, mode)[1] for mode in ("statevector",)]
    every = all(
        _measured((name,), "statevector")[1] == _derived((name,), "statevector")[1]
        for name in admitted
    )
    if bool(rules["parameter_count_drives_every_evaluation"]) != every:
        errors.append(
            "parameter_count_drives_every_evaluation is "
            f"{rules['parameter_count_drives_every_evaluation']} but the "
            f"implementation agrees with the composed cost for every admitted "
            f"opcode: {every}"
        )

    detached = not _measured(opcodes, "statevector")[0].requires_grad
    if bool(rules["result_carries_no_autograd_graph"]) != detached:
        errors.append(
            f"result_carries_no_autograd_graph is "
            f"{rules['result_carries_no_autograd_graph']} but the result carries a "
            f"graph: {not detached}"
        )

    pointed = _point(width)
    shaped = _measured(opcodes, "statevector")[0].shape == (width, width)
    if bool(rules["result_shape_is_the_parameter_shape_twice"]) != shaped:
        errors.append(
            f"result_shape_is_the_parameter_shape_twice is "
            f"{rules['result_shape_is_the_parameter_shape_twice']} but the result "
            f"shape is {tuple(_measured(opcodes, 'statevector')[0].shape)}"
        )

    followed = True
    for dtype in (torch.float32, torch.float64):
        values = pointed.to(dtype)
        build = _witness(opcodes)
        outcome = gradients.parameter_shift_hessian(
            build, values, _loss(opcodes, "statevector")
        )
        followed = followed and outcome.dtype == dtype
        followed = followed and outcome.device == values.device
    if bool(rules["result_dtype_and_device_follow_the_parameters"]) != followed:
        errors.append(
            "result_dtype_and_device_follow_the_parameters is "
            f"{rules['result_dtype_and_device_follow_the_parameters']} but the "
            f"result follows the parameters: {followed}"
        )

    refused = "hessian" not in gradients._GRADIENT_METHODS
    try:
        gradients.gradient(
            _witness(opcodes),
            pointed,
            _loss(opcodes, "statevector"),
            method="hessian",
        )
    except (ValueError, TypeError):
        pass
    else:
        refused = False
    if bool(rules["hessian_is_not_an_accepted_gradient_method"]) != refused:
        errors.append(
            f"hessian_is_not_an_accepted_gradient_method is "
            f"{rules['hessian_is_not_an_accepted_gradient_method']} but the method "
            f"is refused as a method value: {refused}"
        )

    first = _first_order_cost(opcodes, "statevector")
    cheaper = first == 2 * width and costs[0] > first
    if bool(rules["first_order_entry_point_keeps_its_own_cost"]) != cheaper:
        errors.append(
            f"first_order_entry_point_keeps_its_own_cost is "
            f"{rules['first_order_entry_point_keeps_its_own_cost']} but the "
            f"first-order route charged {first} against {costs[0]} for the second"
        )

    rejected = _exactness_errors(contract) != []
    if bool(rules["no_silent_fallback"]) == rejected:
        errors.append(
            f"no_silent_fallback is {rules['no_silent_fallback']} but a cell "
            f"outside the recorded bound is refused: {rejected}"
        )

    if errors:
        return errors
    # The reference floors above only mean something if the returned values are
    # the ones the two analytic routes agree on, which the previous checks
    # already require; this final statement records that the whole set was read.
    if not admitted:
        return ["no opcode row is admitted, so every rule above is vacuous"]
    return []


def _verification_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    verification = contract["verification"]
    contract_test = ROOT / str(verification["contract"])
    if not contract_test.is_file():
        errors.append(f"the contract test {contract_test} does not exist")
    for relative in verification["expansion_tests"]:
        if not (ROOT / str(relative)).is_file():
            errors.append(f"the expansion test {relative} does not exist")
    symbol = str(verification["implementation_symbol"])
    module_name, _, attribute = symbol.rpartition(".")
    if module_name != "flagquantum.gradients" or not hasattr(
        _gradients_module(), attribute
    ):
        errors.append(f"the implementation symbol {symbol} is not importable")
    source = IMPLEMENTATION.read_text()
    for phrase in verification["census"]:
        if str(phrase) not in source:
            errors.append(
                f"flagquantum/gradients.py no longer contains the census string "
                f"{phrase!r}"
            )
    return errors


def _vocabulary_errors(contract: Mapping[str, Any]) -> list[str]:
    """The contract names the declaration it read, and that declaration exists."""

    errors: list[str] = []
    vocabulary = contract.get("declaration_vocabulary", ())
    if not vocabulary:
        errors.append(
            "declaration_vocabulary is empty, so the contract names no declaration"
        )
    declared = contract.get("declaration_source", "")
    if declared != "flagquantum/core/operator_schema.py":
        errors.append(
            f"declaration_source is {declared!r}, which is not the opcode declaration "
            "this gate reads"
        )
    from flagquantum.core.operator_schema import OperatorSchema

    for symbol in vocabulary:
        head, _, tail = str(symbol).partition(".")
        if head == "OPERATOR_SCHEMAS" and not tail:
            continue
        if head != "OperatorSchema" or not tail:
            errors.append(
                f"declaration_vocabulary names {symbol!r}, which the gate never reads"
            )
            continue
        fields = {field.name for field in dataclasses.fields(OperatorSchema)}
        if tail not in fields and not hasattr(OperatorSchema, tail):
            errors.append(
                f"declaration_vocabulary names {symbol!r}, which the declaration "
                "does not expose"
            )
    return errors


def _scope_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    scope = contract["scope"]
    for name in scope["provided"]:
        if name != f"fq.gradients.{ENTRY_POINT}":
            errors.append(f"the scope provides {name!r}, which this gate cannot check")
        elif not hasattr(_gradients_module(), ENTRY_POINT):
            errors.append(f"the scope provides {name!r} but the entry point is absent")
    for name in scope["not_provided"]:
        if not isinstance(name, str) or not name:
            errors.append("a not_provided entry is empty")
    return errors


def contract_errors(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Every disagreement between the contract and the measured implementation."""

    errors: list[str] = []
    errors.extend(_declaration_errors(contract))
    errors.extend(_cost_errors(contract))
    errors.extend(_cell_errors(contract))
    errors.extend(_refusal_errors(contract))
    errors.extend(_vocabulary_errors(contract))
    errors.extend(_scope_errors(contract))
    errors.extend(_verification_errors(contract))
    errors.extend(_exactness_errors(contract))
    errors.extend(_rule_errors(contract))
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    arguments = parser.parse_args(argv)
    contract = _load_contract(arguments.contract)
    errors = contract_errors(contract)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    admitted = _admitted(contract)
    modes = len(contract["protocol"]["serving_modes"])
    cells = len(admitted) * modes
    print(
        "Parameter-shift Hessian contract passed: "
        f"{len(admitted)} admitted opcodes x {modes} serving modes = {cells} "
        "cells re-measured against central differences of fq.run, "
        f"{len(contract.get('cell', ()))} cell-cost rows and "
        f"{len(contract.get('refusal', ()))} refusals driven "
        f"(contract {contract['schema']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
