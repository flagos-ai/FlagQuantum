"""Verify the Fubini-Study metric tensor against two routes that share no method.

``fq.gradients.metric_tensor`` differentiates the state a program delivers and
reads no derivative rule at all. That is the claim this gate exists to defend,
because the tempting implementation is to reuse ``OperatorSchema.shift_rule``
the way ``fq.gradient`` does -- and that table describes an *expectation value*,
not a state. Measured on a single ``ry`` the declared rule applied to the state
is wrong by ``sqrt(2)`` per component and the metric reads twice its true value;
measured on ``phase`` the same rule is right to ``5.9e-11``. A rule that is right
for one opcode and wrong by ``sqrt(2)`` for another cannot be repaired by a scale
factor, and ``OperatorSchema`` records no generator spectrum that would let a
state rule be derived from it. So the gate requires the implementation's own code
to name no registered opcode as a string literal and none of the declaration's
symbols as an identifier: a later change that reaches back for the declaration
turns the gate red instead of quietly moving the numbers.

Three routes are computed and two of them count as evidence. The implementation
is a central difference of the delivered state. The independent route is
reverse-mode autodiff through the same ``fq.run``, which differentiates the same
state analytically and shares no arithmetic with a difference scheme. The
definitional route builds the witness programs from hand-written operators with
no import from ``flagquantum`` at all and computes the quantum Fisher information
the way it is defined -- ``Re Tr[rho L_i L_j]`` with ``L`` the symmetric
logarithmic derivative -- which is what pins the convention: the definition
contains no factor of four, so ``metric_tensor`` agreeing with ``QFIM / 4`` is a
measurement and not a restatement of the docstring. The matrix cells, the
evaluation counts, the recorded falsification readings and every refusal message
are then measured against the contract.

The gate fails closed: a witness that drifts from its recorded value, an opcode
literal appearing in the route, a declaration symbol appearing in the route, a
cost the implementation does not charge, a refusal that stops refusing, a mode
the caller names being silently replaced, or a result carrying an autograd graph
are each a failure rather than a silent drift.
"""

from __future__ import annotations

import ast
import dataclasses
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

CONTRACT = ROOT / "contracts" / "metric-tensor-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum" / "gradients.py"
ENTRY_POINT = "metric_tensor"

# The two symbols the route is made of, so the census below can be scoped to
# them. The batch profile further down the module names opcodes in code by
# design -- it states a frozen protocol scope -- and counting its literals here
# would confuse the two contracts.
ROUTE_SYMBOLS = ("metric_tensor", "_state_of")

# The exception classes a refusal row may name. A name outside this map is a
# contract defect rather than an unmeasured case.
EXCEPTIONS: dict[str, type[BaseException]] = {}

# The opcode kinds the gate can build both a FlagQuantum circuit and a
# hand-written operator from. A witness naming anything else is a contract
# defect and not a skipped row.
ORACLE_OPCODES = ("ry", "rx", "rz", "cx")

# How closely a recorded deviation has to reproduce. The bound above is the
# assertion; a recorded reading is documentation, and two correct references
# differ at the twelfth digit through summation order alone, so comparing a
# reading to its own measurement is bounded below by this floor rather than by a
# relative tolerance that would treat a `1e-12` reading as a claim to six digits.
_READING_FLOOR = 1e-11

# The scale at which a deviation that *is* the difference scheme's own round-off
# floor is compared, which is a different question from `_READING_FLOOR`.
# `_declared_rule_state_deviation` reports `max|declared - central difference|`,
# so where the declared rule coincides with the state derivative the residual is
# not the rule's error but the difference's: `eps * |psi| / (2 * step)`, about
# `1.1e-10` at the step this gate uses. Two builds read `5.860885e-11`/`5.810119e-11`
# and `5.052384e-11` for the same program -- they differ by `7.6e-12` -- so a
# recorded floor reading cannot be held to three significant digits. Comparing it
# at the size of the floor is still a check: `1e-10` is nine orders below the
# `1.7e-01` to `2.0e-01` a rule that stopped coinciding would report, and the
# component-ratio column is what fails first in that case. The rotation rows are
# not floor readings and keep a relative tolerance, because their `2e-01` is the
# declared rule's own discrepancy.
_FLOOR_READING_TOLERANCE = 1e-10


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


# --------------------------------------------------------------------------------------
# What the route's own code is allowed to name
# --------------------------------------------------------------------------------------


def _route_bodies(path: Path = IMPLEMENTATION) -> list[ast.stmt]:
    """The statements of the route's own functions, docstrings removed.

    A docstring is prose and does not decide anything, so counting it would
    report a route as reading a declaration it only mentions.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    bodies: list[ast.stmt] = []
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
        bodies.extend(body)
    return bodies


def _opcodes_named_in_route(path: Path = IMPLEMENTATION) -> list[str]:
    """Registered opcodes the route names as string literals."""

    literals: set[str] = set()
    for statement in _route_bodies(path):
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
    return sorted(literals & set(OPERATOR_SCHEMAS))


def _identifiers_named_in_route(path: Path = IMPLEMENTATION) -> set[str]:
    """Every identifier the route's own code reads."""

    found: set[str] = set()
    for statement in _route_bodies(path):
        for node in ast.walk(statement):
            if isinstance(node, ast.Name):
                found.add(node.id)
            elif isinstance(node, ast.Attribute):
                found.add(node.attr)
    return found


def _declaration_symbols_named_in_route(
    contract: Mapping[str, Any], path: Path = IMPLEMENTATION
) -> list[str]:
    found = _identifiers_named_in_route(path)
    return sorted(found & set(contract.get("unread", {}).get("symbols", ())))


# --------------------------------------------------------------------------------------
# The witness programs, built from the contract so the two cannot drift
# --------------------------------------------------------------------------------------


def _witness_param_tensor(witness: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [float(value) for value in witness["parameters"]], dtype=torch.float64
    )


def _witness_circuit(
    witness: Mapping[str, Any], parameters: torch.Tensor, dtype: Any = None
):
    from flagquantum.errors import ValidationError

    n_qubits = int(witness["qubits"])
    circuit = (
        fq.Circuit(n_qubits, dtype=dtype) if dtype is not None else fq.Circuit(n_qubits)
    )
    for gate in witness["gates"]:
        opcode = str(gate["opcode"])
        if opcode in ("ry", "rx", "rz"):
            getattr(circuit, opcode)(
                int(gate["target"]), theta=parameters[int(gate["parameter"])]
            )
        elif opcode == "cx":
            circuit.cx(int(gate["control"]), int(gate["target"]))
        else:
            raise ValidationError(
                f"the witness {witness['name']!r} names the opcode {opcode!r}, which "
                "this gate cannot build an independent operator from, so the row "
                f"cannot be measured; the gate builds {list(ORACLE_OPCODES)}"
            )
    return circuit


def _witness_builder(
    witness: Mapping[str, Any], dtype: Any = None
) -> Callable[[torch.Tensor], Any]:
    def build(parameters: torch.Tensor):
        return _witness_circuit(witness, parameters, dtype)

    return build


def _witness_point(witness: Mapping[str, Any]) -> torch.Tensor:
    return _witness_param_tensor(witness)


def _recorded_metric(witness: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(witness["recorded_metric"], dtype=torch.float64)


# --------------------------------------------------------------------------------------
# Route one: the implementation
# --------------------------------------------------------------------------------------


def _implementation_metric(
    witness: Mapping[str, Any], dtype: Any = None
) -> torch.Tensor:
    gradients = _gradients_module()
    return gradients.metric_tensor(
        _witness_builder(witness, dtype), _witness_point(witness)
    )


# --------------------------------------------------------------------------------------
# Route two: reverse-mode autodiff through the same execution
# --------------------------------------------------------------------------------------


def _delivered_state(
    witness: Mapping[str, Any], parameters: torch.Tensor, dtype: Any = None
) -> torch.Tensor:
    """The state one execution delivers, at the precision the witness pins.

    The precision belongs to the program and not to the parameters, so this
    route carries the same `dtype` the implementation's witness does. Leaving it
    unset would differentiate a `complex64` state and read `float32`, which is a
    statement about the executor default rather than about this route.
    """

    result = fq.run(
        _witness_circuit(witness, parameters, dtype),
        options=fq.ExecutionOptions(mode="statevector"),
    )
    return result.to_statevector()


def _autodiff_metric(witness: Mapping[str, Any], dtype: Any = None) -> torch.Tensor:
    """The state derivative computed analytically, so no step size enters.

    The formula this route evaluates is the same bilinear form the
    implementation evaluates; what it does not share is the derivative. One
    route takes a central difference of the delivered state and the other asks
    reverse-mode autodiff for the exact Jacobian of that state, so the two agree
    only if the value is right.
    """

    point = _witness_point(witness).requires_grad_(True)
    width = point.numel()

    def real_vector(values: torch.Tensor) -> torch.Tensor:
        state = _delivered_state(witness, values, dtype)
        return torch.cat([state.real.reshape(-1), state.imag.reshape(-1)])

    jacobian = torch.autograd.functional.jacobian(real_vector, point)
    half = jacobian.shape[0] // 2
    derivatives = (jacobian[:half] + 1j * jacobian[half:]).to(torch.complex128)
    base = (
        _delivered_state(witness, point.detach(), dtype)
        .reshape(-1)
        .to(torch.complex128)
    )
    overlap = derivatives.conj().T @ derivatives
    projection = derivatives.conj().T @ base
    metric = (overlap - torch.outer(projection, projection.conj())).real
    return metric.reshape(width, width)


# --------------------------------------------------------------------------------------
# Route three: the definition, with no import from the package under test
# --------------------------------------------------------------------------------------

_IDENTITY = torch.eye(2, dtype=torch.complex128)
_PAULI = {
    "x": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}
# A pure state's density matrix has zero eigenvalues, so the logarithmic
# derivative is taken on the support and zero outside it.
_SUPPORT = 1e-12


def _kron(operators: Sequence[torch.Tensor]) -> torch.Tensor:
    out = torch.ones(1, 1, dtype=torch.complex128)
    for operator in operators:
        out = torch.kron(out, operator)
    return out


def _vacuum(n_qubits: int) -> torch.Tensor:
    out = torch.ones(1, dtype=torch.complex128)
    for _ in range(n_qubits):
        out = torch.kron(out, torch.tensor([1, 0], dtype=torch.complex128))
    return out.reshape(-1)


def _controlled_not(n_qubits: int, control: int, target: int) -> torch.Tensor:
    dimension = 2**n_qubits
    out = torch.zeros(dimension, dimension, dtype=torch.complex128)
    for basis in range(dimension):
        bits = [(basis >> (n_qubits - 1 - qubit)) & 1 for qubit in range(n_qubits)]
        if bits[control]:
            bits[target] ^= 1
        out[
            sum(bit << (n_qubits - 1 - qubit) for qubit, bit in enumerate(bits)), basis
        ] = 1
    return out


def _oracle_state(witness: Mapping[str, Any], parameters: torch.Tensor) -> torch.Tensor:
    """The witness ket, built from this file's own operators."""

    from flagquantum.errors import ValidationError

    n_qubits = int(witness["qubits"])
    vector = _vacuum(n_qubits)
    for gate in witness["gates"]:
        opcode = str(gate["opcode"])
        if opcode in ("ry", "rx", "rz"):
            generator = _kron(
                [
                    _PAULI[opcode[1]] if qubit == int(gate["target"]) else _IDENTITY
                    for qubit in range(n_qubits)
                ]
            )
            angle = parameters[int(gate["parameter"])]
            vector = torch.matrix_exp(-0.5j * angle * generator) @ vector
        elif opcode == "cx":
            vector = (
                _controlled_not(n_qubits, int(gate["control"]), int(gate["target"]))
                @ vector
            )
        else:  # pragma: no cover - _witness_circuit refuses this first
            raise ValidationError(f"the oracle cannot build {opcode!r}")
    return vector


def _fisher_information(witness: Mapping[str, Any]) -> torch.Tensor:
    """``Re Tr[rho L_i L_j]``, which is the definition and carries no factor of four."""

    point = _witness_point(witness)
    n_qubits = int(witness["qubits"])
    dimension = 2**n_qubits
    width = point.numel()

    def real_vector(values: torch.Tensor) -> torch.Tensor:
        vector = _oracle_state(witness, values.to(torch.complex128))
        density = torch.outer(vector, vector.conj())
        return torch.cat([density.real.reshape(-1), density.imag.reshape(-1)])

    jacobian = torch.autograd.functional.jacobian(real_vector, point)
    vector = _oracle_state(witness, point.to(torch.complex128))
    density = torch.outer(vector, vector.conj())
    derivatives = [
        (
            jacobian[: dimension * dimension, index]
            + 1j * jacobian[dimension * dimension :, index]
        )
        .to(torch.complex128)
        .reshape(dimension, dimension)
        for index in range(width)
    ]
    values, vectors = torch.linalg.eigh(density)
    logarithmic: list[torch.Tensor] = []
    for index in range(width):
        rotated = vectors.conj().T @ derivatives[index] @ vectors
        total = values.unsqueeze(0) + values.unsqueeze(1)
        safe = torch.where(total > _SUPPORT, total, torch.ones_like(total))
        scaled = torch.where(
            total > _SUPPORT, 2 * rotated / safe, torch.zeros_like(rotated)
        )
        logarithmic.append(vectors @ scaled @ vectors.conj().T)
    out = torch.zeros(width, width, dtype=torch.float64)
    for left in range(width):
        for right in range(width):
            out[left, right] = torch.trace(
                density @ logarithmic[left] @ logarithmic[right]
            ).real
    return out


# --------------------------------------------------------------------------------------
# The declaration read as a state derivative, which is why the route reads none
# --------------------------------------------------------------------------------------


def _declared_rule_state_deviation(
    opcode: str, theta: float, superposition: bool
) -> tuple[list[float], float]:
    """The declared rule applied to a state, against a central difference of it."""

    point = torch.tensor([theta], dtype=torch.float64)
    schema = OPERATOR_SCHEMAS[opcode]
    rule = schema.shift_rule(schema.parameters[0])

    def state(values: torch.Tensor) -> torch.Tensor:
        circuit = fq.Circuit(1, dtype=torch.complex128)
        if superposition:
            circuit.h(0)
        getattr(circuit, opcode)(0, theta=values[0])
        return (
            fq.run(circuit, options=fq.ExecutionOptions(mode="statevector"))
            .to_statevector()
            .reshape(-1)
        )

    step = 1e-6
    displacement = torch.tensor([step], dtype=torch.float64)
    exact = (state(point + displacement) - state(point - displacement)) / (2 * step)
    applied = state(point) * 0
    for coefficient, shift in rule:
        applied = applied + coefficient * state(
            point + torch.tensor([shift], dtype=torch.float64)
        )
    ratios = [
        float(abs(complex(left)) / abs(complex(right)))
        for left, right in zip(applied, exact, strict=True)
        if abs(complex(right)) > 1e-9
    ]
    return ratios, float((applied - exact).abs().max())


def _declared_rule_metric(witness: Mapping[str, Any]) -> torch.Tensor:
    """The metric a route would assemble from the declared rule instead of a state derivative."""

    point = _witness_point(witness)
    width = point.numel()
    base = (
        fq.run(
            _witness_circuit(witness, point),
            options=fq.ExecutionOptions(mode="statevector"),
        )
        .to_statevector()
        .reshape(-1)
    )
    derivatives = []
    for index in range(width):
        declared = OPERATOR_SCHEMAS["ry"].shift_rule("theta")
        total = base * 0
        for coefficient, shift in declared:
            displaced = point.clone()
            displaced[index] += shift
            total = total + coefficient * (
                fq.run(
                    _witness_circuit(witness, displaced),
                    options=fq.ExecutionOptions(mode="statevector"),
                )
                .to_statevector()
                .reshape(-1)
            )
        derivatives.append(total)
    stacked = torch.stack(derivatives)
    projection = stacked @ base.conj()
    return (
        stacked @ stacked.conj().T - torch.outer(projection, projection.conj())
    ).real


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def _close(measured: torch.Tensor, recorded: torch.Tensor, bound: float) -> float:
    if measured.shape != recorded.shape:
        raise ValueError(
            f"a comparison between {tuple(measured.shape)} and {tuple(recorded.shape)}"
        )
    return float((measured.double() - recorded.double()).abs().max())


def _witness_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    exactness = contract["exactness"]
    bound = float(exactness["agreement_bound"])
    definitional_bound = float(exactness["definitional_bound"])
    floor = float(exactness["minimum_reference_magnitude"])
    pinned = (
        torch.complex128
        if str(exactness["program_dtype"]) == "complex128"
        else torch.complex64
    )
    for witness in contract.get("witness", ()):
        name = str(witness["name"])
        referenced = sorted(
            {int(gate["parameter"]) for gate in witness["gates"] if "parameter" in gate}
        )
        if referenced != list(range(len(witness["parameters"]))):
            errors.append(
                f"the witness {name!r} declares {len(witness['parameters'])} parameters "
                f"but its gates read positions {referenced}"
            )
            continue
        recorded = _recorded_metric(witness)
        measured = _implementation_metric(witness, pinned)
        if tuple(measured.shape) != tuple(recorded.shape):
            errors.append(
                f"the witness {name!r} records a {tuple(recorded.shape)} metric and the "
                f"implementation returned {tuple(measured.shape)}"
            )
            continue
        if float(recorded.abs().max()) < floor:
            errors.append(
                f"the witness {name!r} records a metric whose largest cell is "
                f"{float(recorded.abs().max()):.3e}, below the {floor:.3e} floor, so "
                "the comparison would be vacuous"
            )
        deviation = _close(measured, recorded, bound)
        if deviation > bound:
            errors.append(
                f"the witness {name!r}: the implementation is {deviation:.6e} from the "
                f"recorded metric, above the {bound:.3e} bound"
            )
        note = witness.get("measured_deviation")
        if note is not None and abs(float(note) - deviation) > max(
            1e-03 * abs(deviation), _READING_FLOOR
        ):
            errors.append(
                f"the witness {name!r} records a deviation of {float(note):.6e} and the "
                f"gate measured {deviation:.6e}"
            )
        independent = _autodiff_metric(witness, pinned)
        independent_deviation = _close(independent, recorded, bound)
        if independent_deviation > bound:
            errors.append(
                f"the witness {name!r}: the autodiff route is {independent_deviation:.6e} "
                f"from the recorded metric, above the {bound:.3e} bound, so the recorded "
                "value is not the value this program has"
            )
        fisher = _fisher_information(witness)
        factor = float(contract["convention"]["fisher_factor"])
        definitional = _close(factor * measured.double(), fisher, definitional_bound)
        if definitional > definitional_bound:
            errors.append(
                f"the witness {name!r}: {factor:g} times the implementation is "
                f"{definitional:.6e} from the Fisher information built from the "
                f"symmetric logarithmic derivative, above the {definitional_bound:.3e} "
                "bound, so the convention this contract records is not the one the "
                "implementation follows"
            )
        note = witness.get("definitional_deviation")
        if note is not None and abs(float(note) - definitional) > max(
            1e-03 * float(note), _READING_FLOOR
        ):
            errors.append(
                f"the witness {name!r} records a definitional deviation of {float(note):.3e} "
                f"and the gate measured {definitional:.3e}"
            )
        if not bool(torch.allclose(measured, measured.T, atol=bound, rtol=0)):
            errors.append(f"the witness {name!r} produced an asymmetric metric")
    return errors


def _witness_coverage_errors(contract: Mapping[str, Any]) -> list[str]:
    """The witness set must exercise every structure the claims rest on.

    A witness is not decoration: the folding claim needs a program where one
    parameter drives two gates, the width claim needs a matrix wider than two,
    and the diagonality of an untouched block needs a program that leaves it
    alone. Dropping a witness would otherwise leave the contract passing while
    the property it carried is no longer measured anywhere.
    """

    errors: list[str] = []
    witnesses = list(contract.get("witness", ()))
    if not witnesses:
        return ["the contract records no witness program, so nothing is measured"]
    shared = [
        witness
        for witness in witnesses
        if len(
            {int(gate["parameter"]) for gate in witness["gates"] if "parameter" in gate}
        )
        < len([gate for gate in witness["gates"] if "parameter" in gate])
    ]
    if not shared:
        errors.append(
            "the contract records no witness whose parameter drives more than one gate, "
            "so the claim that the matrix is indexed by parameter rather than by gate "
            "occurrence is not measured anywhere"
        )
    widest = max(len(witness["parameters"]) for witness in witnesses)
    if widest < 3:
        errors.append(
            f"the widest witness records {widest} parameters, so a matrix wider than two "
            "is never exercised"
        )
    diagonal = [
        witness
        for witness in witnesses
        if not any(
            _recorded_metric(witness)[row, column] != 0.0
            for row in range(len(witness["parameters"]))
            for column in range(len(witness["parameters"]))
            if row != column
        )
    ]
    if not diagonal:
        errors.append(
            "the contract records no witness with a purely diagonal metric, so the "
            "off-diagonal cells are never measured against a known zero"
        )
    return errors


def _convention_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    convention = contract["convention"]
    factor = float(convention["fisher_factor"])
    if not math.isfinite(factor) or factor <= 0:
        errors.append(
            f"the recorded fisher_factor is {factor!r}, which is not a positive number"
        )
    recorded = [
        str(convention[key]) for key in ("quantity", "cell", "fisher_definition")
    ]
    if any(not value.strip() for value in recorded):
        errors.append("a convention entry is empty")
    if "fubini" not in str(convention["quantity"]).lower():
        errors.append(
            f"the convention names {convention['quantity']!r} as the quantity, which is "
            "not the Fubini-Study metric this route returns"
        )
    return errors


def _falsification_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    row = contract["falsification"]
    if str(row["route"]) != "declared_shift_rule_applied_to_the_state":
        errors.append(
            f"the falsification route is {row['route']!r}, which this gate does not measure"
        )
        return errors
    opcode = str(row["rotation_opcode"])
    schema = OPERATOR_SCHEMAS[opcode]
    frequencies = [list(value) for value in schema.parameter_frequencies]
    recorded_frequencies = [
        [float(item) for item in value] for value in row["rotation_frequencies"]
    ]
    if frequencies != recorded_frequencies:
        errors.append(
            f"the falsification row records the frequencies {recorded_frequencies} for "
            f"{opcode!r} and the declaration now says {frequencies}"
        )
    for index, theta in enumerate(row["rotation_theta"]):
        ratios, deviation = _declared_rule_state_deviation(opcode, float(theta), False)
        if len(ratios) != len(row["rotation_component_ratio"]):
            errors.append(
                f"the falsification row records {len(row['rotation_component_ratio'])} "
                f"component ratios at theta={theta} and the gate measured {len(ratios)}"
            )
        else:
            for measured, note in zip(
                ratios, row["rotation_component_ratio"], strict=True
            ):
                if abs(measured - float(note)) > 1e-06:
                    errors.append(
                        f"the falsification row records a component ratio of {float(note):.9f} "
                        f"at theta={theta} and the gate measured {measured:.9f}"
                    )
        note = float(row["rotation_deviation"][index])
        if abs(deviation - note) > max(1e-03 * abs(note), 1e-12):
            errors.append(
                f"the falsification row records a deviation of {note:.6e} at theta={theta} "
                f"and the gate measured {deviation:.6e}"
            )
    for index, opcode in enumerate(row["coincident_opcodes"]):
        theta = float(row["coincident_theta"][index])
        ratios, deviation = _declared_rule_state_deviation(str(opcode), theta, True)
        if not ratios:
            errors.append(
                f"the coincident reading for {opcode!r} at theta={theta} has no component "
                "to compare, so the row cannot be measured"
            )
            continue
        note = float(row["coincident_ratio"][index])
        if any(abs(measured - note) > 1e-06 for measured in ratios):
            errors.append(
                f"the falsification row records a component ratio of {note:.9f} for "
                f"{opcode!r} at theta={theta} and the gate measured "
                f"{[round(value, 9) for value in ratios]}"
            )
        recorded_deviation = float(row["coincident_deviation"][index])
        if abs(deviation - recorded_deviation) > max(
            1e-03 * abs(recorded_deviation), _FLOOR_READING_TOLERANCE
        ):
            errors.append(
                f"the falsification row records a deviation of {recorded_deviation:.6e} for "
                f"{opcode!r} at theta={theta} and the gate measured {deviation:.6e}"
            )
    name = str(row["reference_witness"])
    witnesses = {str(item["name"]): item for item in contract.get("witness", ())}
    if name not in witnesses:
        errors.append(
            f"the falsification row names the witness {name!r}, which is not one of "
            f"{sorted(witnesses)}"
        )
        return errors
    measured = _declared_rule_metric(witnesses[name])
    recorded_diagonal = float(row["declared_rule_metric_diagonal"])
    if abs(float(measured[0, 0]) - recorded_diagonal) > 1e-06:
        errors.append(
            f"the falsification row records a declared-rule diagonal of "
            f"{recorded_diagonal:.9f} on {name!r} and the gate measured "
            f"{float(measured[0, 0]):.9f}"
        )
    truth = float(row["true_metric_diagonal"])
    if abs(float(_recorded_metric(witnesses[name])[0, 0]) - truth) > 1e-09:
        errors.append(
            f"the falsification row records a true diagonal of {truth:.9f} on {name!r} "
            f"and the contract records {float(_recorded_metric(witnesses[name])[0, 0]):.9f}"
        )
    deviation = abs(float(measured[0, 0]) - truth)
    if abs(deviation - float(row["declared_rule_deviation"])) > 1e-06:
        errors.append(
            f"the falsification row records a declared-rule deviation of "
            f"{float(row['declared_rule_deviation']):.6e} and the gate measured {deviation:.6e}"
        )
    return errors


def _cost_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    protocol = contract["protocol"]
    base = int(protocol["base_evaluations"])
    per_parameter = int(protocol["evaluations_per_parameter"])
    if str(protocol["cost_growth"]) != "linear_in_the_parameter_count":
        errors.append(
            f"the contract records the cost growth as {protocol['cost_growth']!r}, which "
            "this gate does not measure"
        )
    rows = {
        int(row["parameters"]): int(row["evaluations"])
        for row in contract.get("evaluation", ())
    }
    if not rows:
        errors.append(
            "the contract records no evaluation rows, so the cost is unmeasured"
        )
    for count, recorded in sorted(rows.items()):
        calls = {"count": 0}

        def builder(
            parameters: torch.Tensor, calls: dict[str, int] = calls, width: int = count
        ) -> Any:
            calls["count"] += 1
            circuit = fq.Circuit(width, dtype=torch.complex128)
            for index in range(width):
                circuit.ry(index, theta=parameters[index])
            return circuit

        point = torch.linspace(0.3, 1.1, count, dtype=torch.float64)
        _gradients_module().metric_tensor(builder, point)
        measured = calls["count"]
        expected = base + per_parameter * count
        if measured != expected:
            errors.append(
                f"a {count}-parameter call made {measured} state evaluations; the contract "
                f"records {base} plus {per_parameter} per parameter, which is {expected}"
            )
        if recorded != measured:
            errors.append(
                f"the contract records {recorded} evaluations for {count} parameters and "
                f"the implementation made {measured}"
            )
    if per_parameter != 2:
        errors.append(
            f"the contract records {per_parameter} evaluations per parameter, which is "
            "not the two a central difference costs"
        )
    return errors


def _refusal_programs() -> dict[str, Callable[[], None]]:
    def good(parameters: torch.Tensor):
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.ry(1, theta=parameters[1])
        circuit.cx(0, 1)
        return circuit

    def noisy(parameters: torch.Tensor):
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.cx(0, 1)
        circuit.depolarizing(1, 0.1)
        circuit.ry(1, theta=parameters[1])
        return circuit

    point = torch.tensor([0.4, -0.9], dtype=torch.float64)
    gradients = _gradients_module()

    def run(builder: Any, parameters: Any, **kwargs: Any) -> None:
        gradients.metric_tensor(builder, parameters, **kwargs)

    return {
        "not_callable": lambda: run(None, point),
        "not_a_tensor": lambda: run(good, [0.4, -0.9]),
        "empty_parameters": lambda: run(good, torch.zeros(0, dtype=torch.float64)),
        "complex_parameters": lambda: run(good, torch.tensor([0.4 + 0j, -0.9 + 0j])),
        "integral_parameters": lambda: run(good, torch.tensor([1, 2])),
        "non_finite_parameters": lambda: run(
            good, torch.tensor([0.4, float("inf")], dtype=torch.float64)
        ),
        "zero_step": lambda: run(good, point, step=0.0),
        "negative_step": lambda: run(good, point, step=-1e-3),
        "infinite_step": lambda: run(good, point, step=float("inf")),
        "nan_step": lambda: run(good, point, step=float("nan")),
        "density_matrix_state": lambda: run(noisy, point),
        "batched_state": lambda: run(
            lambda parameters: fq.Circuit(2, bsz=2, dtype=torch.complex128).ry(
                0, theta=parameters[0]
            ),
            torch.tensor([0.4], dtype=torch.float64),
        ),
        "non_static_program": lambda: run(
            lambda parameters: fq.Circuit(
                2 if float(parameters[0]) > 0.4 else 1, dtype=torch.complex128
            ).ry(0, theta=parameters[0]),
            torch.tensor([0.4], dtype=torch.float64),
        ),
    }


def _measure(call: Callable[[], None]) -> tuple[str, str]:
    try:
        call()
    except BaseException as error:
        return type(error).__name__, str(error)
    return "", ""


def _refusal_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    exceptions = _exceptions()
    programs = _refusal_programs()
    rows = contract.get("refusal", ())
    if sorted(str(row["case"]) for row in rows) != sorted(programs):
        errors.append(
            f"the refusal rows name {sorted(str(row['case']) for row in rows)} but the "
            f"gate drives {sorted(programs)}"
        )
    for row in rows:
        case = str(row["case"])
        if case not in programs:
            continue
        name = str(row["exception"])
        if name not in exceptions:
            errors.append(
                f"the refusal row {case} names the exception {name}, which is not one of "
                f"{sorted(exceptions)}"
            )
            continue
        measured_name, message = _measure(programs[case])
        if measured_name != name:
            errors.append(
                f"the refusal row {case} records the exception {name} but the measurement "
                f"produced {measured_name or 'no exception'!r}"
            )
            continue
        if str(row["message_phrase"]) not in message:
            errors.append(
                f"the refusal row {case} records the message phrase "
                f"{row['message_phrase']!r}, which is not a substring of the measured "
                f"message {message!r}"
            )
    return errors


def _mode_refusals() -> dict[str, tuple[str, Callable[[], None]]]:
    """Each excluded mode with who refuses it, and the program that refusal needs."""

    def plain(parameters: torch.Tensor):
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.ry(1, theta=parameters[1])
        circuit.cx(0, 1)
        return circuit

    def noisy(parameters: torch.Tensor):
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.depolarizing(1, 0.1)
        return circuit

    point = torch.tensor([0.4, -0.9], dtype=torch.float64)
    gradients = _gradients_module()
    out: dict[str, tuple[str, Callable[[], None]]] = {}
    # The density-matrix mode serves the density operator on the state accessor,
    # so this route is what refuses it. Every other exclusion belongs to the
    # execution, and is driven through the call that produces it: `stabilizer`
    # never carries a program to a state at all, while a state-producing mode
    # asked for on a program carrying a channel is refused by the execution
    # before this route sees anything.
    out["density_matrix"] = (
        "metric_tensor",
        lambda: gradients.metric_tensor(
            noisy, point, options=fq.ExecutionOptions(mode="density_matrix")
        ),
    )
    out["stabilizer"] = (
        "fq.run",
        lambda: fq.run(plain(point), options=fq.ExecutionOptions(mode="stabilizer")),
    )
    for mode in ("statevector", "mps", "tensor_network"):
        out[mode] = (
            "fq.run",
            lambda mode=mode: fq.run(
                noisy(point), options=fq.ExecutionOptions(mode=mode)
            ),
        )
    return out


def _excluded_mode_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    exceptions = _exceptions()
    cases = _mode_refusals()
    rows = list(contract.get("excluded_mode", ())) + list(
        contract.get("noise_refusal", ())
    )
    if not rows:
        errors.append("the contract records no excluded mode")
        return errors
    for row in rows:
        mode = str(row["mode"])
        if mode not in cases:
            errors.append(
                f"the contract excludes {mode!r}, which the gate does not drive"
            )
            continue
        owner, call = cases[mode]
        if str(row["refused_by"]) != owner:
            errors.append(
                f"the contract says {mode!r} is refused by {row['refused_by']!r} and the "
                f"gate drives it through {owner!r}"
            )
            continue
        name = str(row["exception"])
        if name not in exceptions:
            errors.append(
                f"the excluded mode {mode!r} names the exception {name}, which is not one "
                f"of {sorted(exceptions)}"
            )
            continue
        measured_name, message = _measure(call)
        if measured_name != name:
            errors.append(
                f"the excluded mode {mode!r} records the exception {name} but the "
                f"measurement produced {measured_name or 'no exception'!r}"
            )
            continue
        if str(row["message_phrase"]) not in message:
            errors.append(
                f"the excluded mode {mode!r} records the message phrase "
                f"{row['message_phrase']!r}, which is not a substring of the measured "
                f"message {message!r}"
            )
    # `excluded_mode` is the partition of the execution modes with `serving_modes`;
    # a noisy program's mode refusal is an additional reading against the same
    # mode names and not a second exclusion.
    modes = {str(row["mode"]) for row in contract.get("excluded_mode", ())}
    serving = {str(mode) for mode in contract["protocol"]["serving_modes"]}
    if modes & serving:
        errors.append(
            f"the contract both serves and excludes {sorted(modes & serving)}, so the two "
            "lists are not a partition of the execution modes"
        )
    registered = {
        "auto",
        "statevector",
        "mps",
        "tensor_network",
        "density_matrix",
        "stabilizer",
    }
    if modes | serving != registered:
        errors.append(
            f"the contract accounts for {sorted(modes | serving)} of the execution modes, "
            f"which is not all of {sorted(registered)}"
        )
    return errors


def _rule_errors(contract: Mapping[str, Any]) -> list[str]:
    rules = contract["rules"]
    errors: list[str] = []
    gradients = _gradients_module()

    reachable = ENTRY_POINT in gradients.__all__
    if bool(rules["implementation_is_reachable_from_the_module_surface"]) != reachable:
        errors.append(
            "implementation_is_reachable_from_the_module_surface is "
            f"{rules['implementation_is_reachable_from_the_module_surface']} but the "
            f"entry point is {'present' if reachable else 'absent'} in the module __all__"
        )
    absent = ENTRY_POINT not in fq.__all__ and not hasattr(fq, ENTRY_POINT)
    if bool(rules["implementation_is_absent_from_the_root_surface"]) != absent:
        errors.append(
            f"implementation_is_absent_from_the_root_surface is "
            f"{rules['implementation_is_absent_from_the_root_surface']} but the root "
            f"surface is {'clear' if absent else 'carrying the name'}"
        )
    named_opcodes = _opcodes_named_in_route()
    if bool(rules["no_opcode_drives_the_route"]) != (not named_opcodes):
        errors.append(
            f"no_opcode_drives_the_route is {rules['no_opcode_drives_the_route']} but the "
            f"route names {named_opcodes}"
        )
    named_symbols = _declaration_symbols_named_in_route(contract)
    if bool(rules["no_declared_frequency_is_read"]) != (not named_symbols):
        errors.append(
            f"no_declared_frequency_is_read is {rules['no_declared_frequency_is_read']} but "
            f"the route names {named_symbols}"
        )

    witness = contract["witness"][0]
    point = _witness_point(witness).requires_grad_(True)
    measured = gradients.metric_tensor(_witness_builder(witness), point)
    if bool(rules["result_carries_no_autograd_graph"]) != (not measured.requires_grad):
        errors.append(
            "result_carries_no_autograd_graph is "
            f"{rules['result_carries_no_autograd_graph']} but the result carries "
            f"requires_grad={measured.requires_grad}"
        )
    twice = tuple(point.shape) * 2
    if bool(rules["result_shape_is_the_parameter_shape_twice"]) != (
        tuple(measured.shape) == twice
    ):
        errors.append(
            f"result_shape_is_the_parameter_shape_twice is "
            f"{rules['result_shape_is_the_parameter_shape_twice']} but the result has shape "
            f"{tuple(measured.shape)} for a parameter tensor of shape {tuple(point.shape)}"
        )
    follows = True
    for dtype, expected in (
        (torch.complex128, torch.float64),
        (torch.complex64, torch.float32),
    ):
        plain = _witness_point(witness)
        got = gradients.metric_tensor(_witness_builder(witness, dtype), plain)
        follows = follows and got.dtype == expected
    if bool(rules["result_dtype_follows_the_delivered_state"]) != follows:
        errors.append(
            f"result_dtype_follows_the_delivered_state is "
            f"{rules['result_dtype_follows_the_delivered_state']} but a `complex64` state "
            "does not yield a `float32` metric beside a `complex128` state yielding a "
            "`float64` one"
        )
    if bool(rules["result_device_follows_the_parameters"]) != (
        measured.device == point.device
    ):
        errors.append(
            f"result_device_follows_the_parameters is "
            f"{rules['result_device_follows_the_parameters']} but the result is on "
            f"{measured.device} and the parameters on {point.device}"
        )

    # A program carrying an explicit matrix has a state and no declared rule, so
    # this is the one capability the choice of route buys.
    matrix = torch.eye(2, dtype=torch.complex128) / math.sqrt(2)

    def custom(parameters: torch.Tensor):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        circuit.gate("h", (0,), matrix=matrix)
        circuit.ry(0, theta=parameters[0])
        return circuit

    try:
        served = gradients.metric_tensor(
            custom, torch.tensor([0.4], dtype=torch.float64)
        )
        custom_served = served.shape == (1, 1) and math.isfinite(float(served[0, 0]))
    except BaseException as error:
        custom_served = False
        errors.append(
            f"a program carrying an explicit matrix was refused with "
            f"{type(error).__name__}: {error}, and this route's whole claim is that a "
            "state derivative does not need a declared rule"
        )
    if bool(rules["custom_matrix_programs_are_served"]) != custom_served:
        errors.append(
            f"custom_matrix_programs_are_served is "
            f"{rules['custom_matrix_programs_are_served']} but the measurement says "
            f"{custom_served}"
        )

    def noisy(parameters: torch.Tensor):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.depolarizing(0, 0.1)
        return circuit

    name, _ = _measure(
        lambda: gradients.metric_tensor(noisy, torch.tensor([0.4], dtype=torch.float64))
    )
    mixed_refused = name == "CapabilityError"
    if (
        bool(rules["a_mixed_state_is_refused_rather_than_reinterpreted"])
        != mixed_refused
    ):
        errors.append(
            "a_mixed_state_is_refused_rather_than_reinterpreted is "
            f"{rules['a_mixed_state_is_refused_rather_than_reinterpreted']} but the "
            f"measurement produced {name or 'no exception'!r}"
        )
    fallback = []
    for mode in ("statevector", "mps", "tensor_network"):
        fallback.append(
            _measure(
                lambda mode=mode: gradients.metric_tensor(
                    noisy,
                    torch.tensor([0.4], dtype=torch.float64),
                    options=fq.ExecutionOptions(mode=mode),
                )
            )[0]
        )
    no_fallback = all(name == "ValidationError" for name in fallback)
    if bool(rules["no_silent_fallback"]) != no_fallback:
        errors.append(
            f"no_silent_fallback is {rules['no_silent_fallback']} but an explicitly "
            f"requested mode on a noisy program produced {fallback}"
        )
    return errors


def _vocabulary_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    unread = contract.get("unread", {})
    if not unread:
        errors.append(
            "the contract names no declaration, so the route's independence is unchecked"
        )
        return errors
    declared = str(contract.get("declaration_source", ""))
    if declared != "flagquantum/core/operator_schema.py":
        errors.append(
            f"declaration_source is {declared!r}, which is not the opcode declaration this "
            "gate reads"
        )
    if bool(contract.get("declaration_is_read")):
        errors.append(
            "declaration_is_read is set, so the contract claims a route that reads the "
            "declaration; a state derivative must read none"
        )
    from flagquantum.core import operator_schema
    from flagquantum.core.operator_schema import OperatorSchema

    fields = {field.name for field in dataclasses.fields(OperatorSchema)}
    for symbol in unread.get("symbols", ()):
        name = str(symbol)
        exposed = (
            name in fields
            or hasattr(OperatorSchema, name)
            or hasattr(operator_schema, name)
        )
        if not exposed:
            errors.append(
                f"the contract forbids the symbol {name!r}, which the declaration does not "
                "expose, so the prohibition is about nothing"
            )
    if bool(unread.get("opcodes_are_derived_from_the_declaration")) != bool(
        OPERATOR_SCHEMAS
    ):
        errors.append(
            "opcodes_are_derived_from_the_declaration is "
            f"{unread.get('opcodes_are_derived_from_the_declaration')} but the declaration "
            f"registers {len(OPERATOR_SCHEMAS)} opcodes"
        )
    found = _opcodes_named_in_route()
    recorded = int(unread.get("expected_opcode_literals", -1))
    if recorded != len(found):
        errors.append(
            f"the contract records {recorded} opcode literals in the route and the gate "
            f"found {len(found)}: {found}"
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
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    for phrase in verification["census"]:
        if str(phrase) not in source:
            errors.append(
                f"flagquantum/gradients.py no longer contains the census string {phrase!r}"
            )
    return errors


def _precision_errors(contract: Mapping[str, Any]) -> list[str]:
    """The executor-default reading, recorded so it cannot be quoted as the bound."""

    errors: list[str] = []
    exactness = contract["exactness"]
    witnesses = {str(item["name"]): item for item in contract.get("witness", ())}
    name = str(exactness["default_program_dtype_witness"])
    if name not in witnesses:
        errors.append(
            f"the executor-default witness {name!r} is not one of {sorted(witnesses)}"
        )
        return errors
    measured = _implementation_metric(witnesses[name], None)
    expected = (
        torch.float32
        if str(exactness["default_program_dtype"]) == "complex64"
        else torch.float64
    )
    if measured.dtype != expected:
        errors.append(
            f"the executor default produced a {measured.dtype} metric and the contract "
            f"records {expected}"
        )
    deviation = _close(measured, _recorded_metric(witnesses[name]), 1.0)
    recorded = float(exactness["default_program_dtype_deviation"])
    if abs(deviation - recorded) > max(1e-02 * abs(recorded), 1e-09):
        errors.append(
            f"the contract records an executor-default deviation of {recorded:.6e} on "
            f"{name!r} and the gate measured {deviation:.6e}"
        )
    if deviation <= float(exactness["agreement_bound"]):
        errors.append(
            f"the executor-default deviation {deviation:.6e} is inside the "
            f"{float(exactness['agreement_bound']):.3e} bound, so the contract's claim that "
            "the bound is not true at the executor default is wrong"
        )
    return errors


def contract_errors(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Every disagreement between the contract and the measured implementation."""

    errors: list[str] = []
    errors.extend(_scope_errors(contract))
    errors.extend(_vocabulary_errors(contract))
    errors.extend(_witness_errors(contract))
    errors.extend(_witness_coverage_errors(contract))
    errors.extend(_convention_errors(contract))
    errors.extend(_falsification_errors(contract))
    errors.extend(_cost_errors(contract))
    errors.extend(_refusal_errors(contract))
    errors.extend(_excluded_mode_errors(contract))
    errors.extend(_precision_errors(contract))
    errors.extend(_verification_errors(contract))
    errors.extend(_rule_errors(contract))
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    contract = _load_contract()
    errors = contract_errors(contract)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    exactness = contract["exactness"]
    factor = float(contract["convention"]["fisher_factor"])
    print(
        "Metric-tensor contract passed: "
        f"{len(contract.get('witness', ()))} witness programs re-measured against "
        f"reverse-mode autodiff through fq.run at {exactness['program_dtype']} and against "
        f"the symmetric logarithmic derivative, with QFIM = {factor:g} x metric_tensor "
        f"pinned on each; {len(contract.get('evaluation', ()))} evaluation-cost rows and "
        f"{len(contract.get('refusal', ()))} refusals driven, "
        f"{len(contract.get('excluded_mode', ()))} excluded and "
        f"{len(contract.get('noise_refusal', ()))} noisy-program modes re-refused, and the "
        f"{exactness['default_program_dtype_witness']} witness re-read at the executor "
        f"default (contract {contract['schema']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
