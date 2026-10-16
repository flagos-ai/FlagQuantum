"""Verify the batch parameter-shift profile against the opcode declaration.

``batched_parameter_shift_gradient`` sends one positively shifted and one
negatively shifted circuit per input parameter, so it can differentiate exactly
those gate parameters whose declared frequencies produce a two-term rule. Which
opcodes those are is a property of ``flagquantum/core/operator_schema.py``, not
of the protocol, so the protocol has to read the declaration instead of
restating it as a gate list.

This gate re-derives the set from ``OPERATOR_SCHEMAS``, censuses the
implementation's own source for opcode names that are not part of the declared
protocol scope, drives the implementation for every registered opcode, and
compares all of it with ``contracts/parameter-shift-coverage-contract.toml``. It
fails closed: an opcode the derivation admits that the profile refuses, a
multi-pair opcode the profile answers with a truncated rule, and a hardcoded gate
list reappearing in the implementation are each a failure rather than a silent
drift.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.core.operator_schema import OPERATOR_SCHEMAS  # noqa: E402

CONTRACT = ROOT / "contracts" / "parameter-shift-coverage-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum" / "gradients.py"
DECLARATION = ROOT / "flagquantum" / "core" / "operator_schema.py"


def _zero_batch(circuits: tuple[Any, ...]) -> torch.Tensor:
    """The profile's batch-loss stand-in, so the gate never submits anything."""

    return torch.zeros(len(circuits), dtype=torch.float64)


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _gradients_module() -> ModuleType:
    import flagquantum.gradients as gradients

    return gradients


def opcode_literals(source: str, names: set[str]) -> list[tuple[int, str]]:
    """Opcode names written as string literals in code rather than in prose.

    A name inside a docstring documents the profile; a name in code decides it.
    Only the second kind is returned, so the prose in ``gradients.py`` may name
    every opcode it wants to explain while the code may not.
    """

    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        if node.value in names:
            found.append((node.lineno, node.value))
    return sorted(found)


def measure_declaration() -> dict[str, dict[str, Any]]:
    """What each registered opcode declares about its own parameters."""

    measured: dict[str, dict[str, Any]] = {}
    for opcode, schema in OPERATOR_SCHEMAS.items():
        pairs = 0
        if schema.differentiable:
            pairs = max(len(schema.shift_rule(name)) for name in schema.parameters) // 2
        measured[opcode] = {
            "declared_parameters": list(schema.parameters),
            "declared_frequencies": [
                list(item) for item in schema.parameter_frequencies
            ],
            "evaluation_pairs": pairs,
            "differentiable": bool(schema.differentiable),
            "channel": bool(schema.channel),
        }
    return measured


def _scalar(value: torch.Tensor) -> torch.Tensor:
    return value if value.dim() == 0 else value[0]


def _one_parameter_program(opcode: str, name: str, exactness: Mapping[str, Any]):
    """A program whose one input parameter is ``opcode``'s ``name``.

    The preparation and tail are what make the reference non-degenerate for every
    admitted parameter: a bare differentiated gate followed by a Z measurement
    sends several of them to zero, which would make the exactness comparison
    vacuous rather than passing.
    """

    import flagquantum as fq

    schema = OPERATOR_SCHEMAS[opcode]
    arity = max(1, schema.arity)
    fixed = float(exactness["fixed_parameter_value"])

    def build(values: torch.Tensor) -> Any:
        params = {
            parameter: (_scalar(values) if parameter == name else fixed)
            for parameter in schema.parameters
        }
        circuit = fq.Circuit(arity, dtype=torch.complex128)
        for qubit in range(arity):
            circuit = circuit.h(qubit).rz(qubit, theta=0.3 + 0.1 * qubit)
        circuit = circuit.gate(opcode, tuple(range(arity)), params=params)
        return circuit.ry(0, theta=0.4).rz(0, theta=0.13)

    return build


_PROGRAMS: dict[str, Callable[[torch.Tensor], Any]] = {}


def _programs() -> dict[str, Callable[[torch.Tensor], Any]]:
    """The programs the refusal rows and the admission measurement are built on."""

    if _PROGRAMS:
        return _PROGRAMS
    import flagquantum as fq

    def structure_follows_parameter(values: torch.Tensor) -> Any:
        circuit = fq.Circuit(1, dtype=torch.complex128).rx(0, theta=values[0])
        if float(values[0]) > 0:
            circuit = circuit.h(0)
        return circuit

    def custom_matrix(values: torch.Tensor) -> Any:
        return fq.Circuit(1, dtype=torch.complex128).gate(
            "rx",
            (0,),
            matrix=torch.eye(2, dtype=torch.complex128),
            theta=values[0],
        )

    _PROGRAMS.update(
        {
            "rx_theta": lambda v: fq.Circuit(1, dtype=torch.complex128)
            .h(0)
            .rx(0, theta=v[0])
            .ry(0, theta=0.4),
            "crz_theta": lambda v: fq.Circuit(2, dtype=torch.complex128)
            .h(0)
            .crz(0, 1, theta=v[0]),
            "rx_twice": lambda v: fq.Circuit(1, dtype=torch.complex128)
            .rx(0, theta=v[0])
            .ry(0, theta=v[0]),
            "rx_scaled": lambda v: fq.Circuit(1, dtype=torch.complex128).rx(
                0, theta=2 * v[0]
            ),
            "bit_flip": lambda v: fq.Circuit(1).bit_flip(0, probability=v[0]),
            "sdg_then_rx": lambda v: fq.Circuit(1, dtype=torch.complex128)
            .sdg(0)
            .rx(0, theta=v[0]),
            "custom_matrix_gate": custom_matrix,
            "structure_follows_parameter": structure_follows_parameter,
        }
    )
    return _PROGRAMS


def measure_admission(exactness: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Drive the profile for every parameterized opcode and for every constant gate."""

    gradients = _gradients_module()
    import flagquantum as fq

    measured: dict[str, dict[str, Any]] = {}
    parameters = torch.tensor([float(exactness["value"])], dtype=torch.float64)
    for opcode, schema in OPERATOR_SCHEMAS.items():
        if schema.parameters:
            if not schema.differentiable:
                measured[opcode] = {
                    "admitted": False,
                    "message": "no declared derivative rule",
                }
                continue
            for name in schema.parameters:
                build = _one_parameter_program(opcode, name, exactness)
                try:
                    gradients.batched_parameter_shift_gradient(
                        build, parameters, _zero_batch
                    )
                except ValueError as error:
                    measured[opcode] = {"admitted": False, "message": str(error)}
                    break
            else:
                measured[opcode] = {"admitted": True, "message": ""}
            continue
        # A gate with no declared parameter is admitted by scope, not by rule.
        arity = max(1, schema.arity)

        def build(values: torch.Tensor, a: int = arity, name: str = opcode) -> Any:
            return (
                fq.Circuit(a, dtype=torch.complex128)
                .gate(name, tuple(range(a)))
                .rx(0, theta=values[0])
            )

        try:
            gradients.batched_parameter_shift_gradient(build, parameters, _zero_batch)
        except ValueError as error:
            measured[opcode] = {"admitted": False, "message": str(error)}
        else:
            measured[opcode] = {"admitted": True, "message": ""}
    return measured


def measure_exactness(exactness: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Compare every admitted parameter with PyTorch autograd on the same circuit.

    Only opcodes the profile admits are measured, because a wider rule is refused
    rather than approximated and so has no profile value to compare. A reference
    at zero would make the relative error meaningless, so the magnitude is
    returned beside it and checked against the contract's declared floor.
    """

    gradients = _gradients_module()
    point = float(exactness["value"])
    measured: dict[str, dict[str, float]] = {}
    for opcode, schema in sorted(OPERATOR_SCHEMAS.items()):
        if not schema.differentiable:
            continue
        if not all(len(schema.shift_rule(name)) == 2 for name in schema.parameters):
            continue
        for name in schema.parameters:
            build = _one_parameter_program(opcode, name, exactness)
            reference_values = torch.tensor(
                [point], dtype=torch.float64, requires_grad=True
            )
            _loss(build(reference_values)).backward()
            if reference_values.grad is None:  # pragma: no cover - defensive
                raise AssertionError(f"{opcode}.{name} produced no autograd gradient")
            reference = float(reference_values.grad[0])

            def batch(circuits: tuple[Any, ...]) -> torch.Tensor:
                return torch.tensor(
                    [float(_loss(circuit)) for circuit in circuits],
                    dtype=torch.float64,
                )

            value = float(
                gradients.batched_parameter_shift_gradient(
                    build, torch.tensor([point], dtype=torch.float64), batch
                )[0]
            )
            measured[f"{opcode}.{name}"] = {
                "reference": reference,
                "relative_error": abs(value - reference) / max(abs(reference), 1e-30),
            }
    return measured


def _loss(circuit: Any) -> torch.Tensor:
    return circuit.expectation_z(tuple(range(circuit.n_qubits))).sum()


def _refusal_errors(contract: Mapping[str, Any]) -> list[str]:
    gradients = _gradients_module()
    errors = []
    programs = _programs()
    for row in contract.get("refusal", []):
        case = row["case"]
        build = programs.get(row["program"])
        if build is None:
            errors.append(f"refusal {case}: unknown program {row['program']!r}")
            continue
        shift = math.pi / 2
        if row.get("shift_source") == "caller_supplied_half":
            shift = 0.5
        try:
            gradients.batched_parameter_shift_gradient(
                build,
                torch.tensor([0.23], dtype=torch.float64),
                _zero_batch,
                shift=shift,
            )
        except ValueError as error:
            if row["message"] not in str(error):
                errors.append(
                    f"refusal {case}: message {str(error)!r} does not contain "
                    f"{row['message']!r}"
                )
        else:
            errors.append(
                f"refusal {case}: the profile accepted it instead of refusing"
            )
    return errors


def contract_errors(contract: Mapping[str, Any], root: Path) -> list[str]:
    errors: list[str] = []

    if contract.get("schema") != "flagquantum_parameter_shift_coverage_contract_v1":
        errors.append(f"unexpected contract schema {contract.get('schema')!r}")

    for key in ("implementation", "declaration_source"):
        relative = contract.get(key)
        if not isinstance(relative, str):
            errors.append(f"{key} must be a path string")
            continue
        if not (root / relative).is_file():
            errors.append(f"{key} {relative!r} does not exist")

    protocol = contract.get("protocol", {})
    evaluations = protocol.get("evaluations_per_parameter")
    if evaluations != 2:
        errors.append(
            f"protocol.evaluations_per_parameter is {evaluations!r}; one pair per "
            "parameter is what a two-term rule needs"
        )
    if protocol.get("max_evaluation_pairs") != 1:
        errors.append(
            "protocol.max_evaluation_pairs must be 1: the profile sends one "
            "evaluation pair per parameter and must refuse a wider rule instead of "
            "truncating it"
        )
    constant_gates = set(protocol.get("constant_gates", []))
    if not constant_gates:
        errors.append("protocol.constant_gates must name the opcodes admitted by scope")

    # The declaration is derived, and the profile depends on that being true.
    declaration = measure_declaration()
    for opcode, entry in declaration.items():
        if entry["differentiable"] != bool(entry["declared_frequencies"]):
            errors.append(
                f"{opcode}: differentiable is {entry['differentiable']} but the "
                f"declared frequencies are {entry['declared_frequencies']}"
            )
        if entry["differentiable"]:
            expected = 2 * entry["evaluation_pairs"]
            for name in entry["declared_parameters"]:
                rule = OPERATOR_SCHEMAS[opcode].shift_rule(name)
                if len(rule) != expected:
                    errors.append(
                        f"{opcode}.{name}: rule has {len(rule)} terms, expected {expected}"
                    )
    for opcode in sorted(constant_gates):
        entry = declaration.get(opcode)
        if entry is None:
            errors.append(f"protocol.constant_gates names unknown opcode {opcode!r}")
        elif entry["declared_parameters"] or entry["channel"]:
            errors.append(
                f"protocol.constant_gates names {opcode!r}, which is not a gate with "
                "no declared parameter"
            )

    # The implementation must not restate the rule as a gate list.
    measured_literals = {
        name
        for _, name in opcode_literals(
            (root / contract["implementation"]).read_text(), set(declaration)
        )
    }
    if measured_literals != constant_gates:
        errors.append(
            "the implementation names opcodes in code outside the declared scope: "
            f"measured {sorted(measured_literals)}, declared {sorted(constant_gates)}. "
            "A gate list that decides admission belongs in the declaration, not here"
        )

    admission = measure_admission(contract["exactness"])
    rows = contract.get("opcode", [])
    if len(rows) != len(declaration):
        errors.append(
            f"contract records {len(rows)} opcodes; the declaration registers "
            f"{len(declaration)}"
        )
    recorded = {row["name"] for row in rows}
    for opcode in sorted(set(declaration) - recorded):
        errors.append(f"contract does not record registered opcode {opcode!r}")
    for opcode in sorted(recorded - set(declaration)):
        errors.append(f"contract records unregistered opcode {opcode!r}")

    for row in rows:
        opcode = row["name"]
        entry = declaration.get(opcode)
        if entry is None:
            continue
        for key in ("declared_parameters", "declared_frequencies", "evaluation_pairs"):
            if row.get(key) != entry[key]:
                errors.append(
                    f"{opcode}: contract {key} is {row.get(key)!r}, measured "
                    f"{entry[key]!r}"
                )
        measured = admission[opcode]
        if bool(row.get("admitted")) != measured["admitted"]:
            errors.append(
                f"{opcode}: contract admitted={row.get('admitted')!r}, measured "
                f"{measured['admitted']}; {measured['message']}"
            )
        # A one-pair rule is admitted, a wider rule is refused, and the refusal
        # has to name the opcode and the number of pairs it needs.
        if entry["differentiable"] and entry["evaluation_pairs"] > 1:
            if measured["admitted"]:
                errors.append(
                    f"{opcode}: needs {entry['evaluation_pairs']} evaluation pairs "
                    "but the profile answered it from one"
                )
            else:
                for token in (opcode, f"needs {entry['evaluation_pairs']} evaluation"):
                    if token not in measured["message"]:
                        errors.append(
                            f"{opcode}: refusal {measured['message']!r} does not name "
                            f"{token!r}"
                        )
        expected_admitted = opcode in constant_gates or (
            entry["differentiable"] and entry["evaluation_pairs"] == 1
        )
        if measured["admitted"] != expected_admitted:
            errors.append(
                f"{opcode}: the derivation admits {expected_admitted}, the profile "
                f"{measured['admitted']}; {measured['message']}"
            )
        if not measured["admitted"] and not measured["message"]:
            errors.append(f"{opcode}: refused without a message")

    errors.extend(_refusal_errors(contract))

    exactness = contract.get("exactness", {})
    for key in (
        "tolerance",
        "reference_minimum_magnitude",
        "value",
        "fixed_parameter_value",
    ):
        if not isinstance(exactness.get(key), float):
            errors.append(f"exactness.{key} must be a real number")
    if errors:
        return errors
    floor = exactness["reference_minimum_magnitude"]
    for key, entry in sorted(measure_exactness(exactness).items()):
        if abs(entry["reference"]) < floor:
            errors.append(
                f"{key}: the autograd reference is {entry['reference']:.3e}, below the "
                f"declared floor {floor:.3e}, so the comparison would be vacuous"
            )
        if entry["relative_error"] > exactness["tolerance"]:
            errors.append(
                f"{key}: the profile disagrees with autograd by "
                f"{entry['relative_error']:.3e} relative, above the declared "
                f"{exactness['tolerance']:.3e} rule-arithmetic bound"
            )

    return errors


def census_self_test() -> list[str]:
    """Negative-test the scanner so a zero finding is evidence rather than silence."""

    errors = []
    names = {"rx", "ry", "rz", "cz"}
    in_code = 'shift_gates = frozenset({"rx", "ry", "rz"})\n'
    found = {name for _, name in opcode_literals(in_code, names)}
    if found != {"rx", "ry", "rz"}:
        errors.append(
            f"the scanner missed a hardcoded list in code: found {sorted(found)}"
        )
    in_prose = (
        'def f():\n    """Differentiates RX, RY, and RZ only."""\n    return None\n'
    )
    found = {name for _, name in opcode_literals(in_prose, names)}
    if found:
        errors.append(
            f"the scanner read opcodes out of a docstring: found {sorted(found)}"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    errors = census_self_test()
    errors.extend(contract_errors(_load_contract(CONTRACT), ROOT))
    if errors:
        print("\n".join(errors))
        return 1
    declaration = measure_declaration()
    pairs = sum(
        1
        for entry in declaration.values()
        if entry["differentiable"] and entry["evaluation_pairs"] == 1
    )
    print(
        "Parameter-shift coverage contract passed: "
        f"{pairs} opcodes differentiate from one evaluation pair, "
        f"{len(declaration) - pairs} do not"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
