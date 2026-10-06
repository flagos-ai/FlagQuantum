#!/usr/bin/env python3
"""Validate the opcode exactness contract by re-measuring every cell of it.

`contracts/opcode-gradient-exactness-contract.toml` states, for every opcode whose
derivative rule is declared and every execution mode that can serve an expectation
value, that the analytic `parameter_shift` rule is the derivative the implementation
actually applies. That is a claim about an executor, not about a formula: `statevector`,
`mps`, `tensor_network` and `density_matrix` are separate executors, so a rule can be
right in the default mode and wrong in another.

This gate does not read the claim back as text. It rebuilds the contract's reference
program for every opcode, runs all seventy cells, and compares three gradients per cell:
`autograd`, `parameter_shift`, and a Richardson-extrapolated central difference built
from `fq.run` alone. The third route is the reason this gate is more than a tautology --
the first two can agree while both being wrong, because they can share the same analytic
rule, so an agreement measured between them is not evidence that the rule is right. The
difference scheme shares no code with either route and is what makes the comparison a
test rather than a consistency check.

Both vocabularies are read from the implementation. The opcode set is the set of
`OperatorSchema`s whose derived `differentiable` property is true, so an opcode that
gains a declared rule fails this gate until it is measured here; the mode set is
`flagquantum/runtime/options.py::_MODES`, so a new execution mode fails it too. The
excluded modes are checked by driving them, not by trusting the exclusion: `stabilizer`
must still refuse every differentiable opcode with the message the contract quotes, and
the phrase is located in the file it names so a reworded message is a contract change.
The relationship to the remote-safe batch profile is re-derived from that contract's own
file rather than restated, so the two cannot drift apart.

The contract's four aggregate numbers -- the observability floor and the three maximum
deviations -- are re-measured rather than reasoned about, because a maximum over cells
is exactly the kind of figure that survives an edit it should not.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import torch

import flagquantum as fq
from flagquantum import gradients as gradient_module
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.errors import CapabilityError
from flagquantum.runtime import options as runtime_options

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "opcode-gradient-exactness-contract.toml"
BATCH_CONTRACT = ROOT / "contracts" / "parameter-shift-coverage-contract.toml"

#: The exception class each contracted `exception` name denotes. A name absent here is a
#: contracted class this gate cannot observe, which is a failure rather than a pass.
EXCEPTIONS: dict[str, type[BaseException]] = {
    "CapabilityError": CapabilityError,
}

#: Every `[rules]` flag, named so that a flag nobody reads cannot be added unnoticed.
RULE_FLAGS = (
    "one_row_per_differentiable_opcode",
    "no_row_for_an_opcode_without_a_declared_rule",
    "every_measured_mode_is_a_cell",
    "every_cell_re_measured_by_the_gate",
    "three_routes_per_cell_with_the_third_independent",
    "agreement_between_two_routes_is_not_evidence_alone",
    "observability_floor_is_measured_not_assumed",
    "excluded_modes_carry_their_reason",
    "a_new_opcode_rule_fails_the_gate_until_measured",
    "a_new_execution_mode_fails_the_gate_until_measured",
    "cross_mode_spread_is_measured_per_opcode",
    "no_silent_fallback",
)

#: The number of significant digits the contract records, used to compare a recorded
#: figure with a re-measured one. A comparison at the wrong precision reported a false
#: mismatch once already, when a checker formatted twelve decimals against a figure
#: stored to three significant digits; the recorded value is the reference here, so the
#: comparison has to be at least as loose as the recording.
RECORDED_RELATIVE_TOLERANCE = 1e-6
RECORDED_ABSOLUTE_TOLERANCE = 1e-18

#: The modes whose derivative is impossible for a reason that belongs to the measurement
#: kind rather than to the opcode.
EXCLUDED_MODE_NAMES = {"stabilizer"}


def _load_contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _differentiable() -> dict[str, Any]:
    """The opcodes whose derivative rule is declared, as the declaration derives it.

    The set is not stored anywhere: `OperatorSchema.differentiable` reads the declared
    frequency sets, so "the executor can differentiate this" and "the derivative rule is
    known" cannot drift apart.
    """
    return {
        name: schema
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.differentiable
    }


def _evaluation_point(contract: dict[str, Any], opcode: str) -> torch.Tensor:
    points = contract["reference"]["evaluation_points"]
    width = len(OPERATOR_SCHEMAS[opcode].parameters)
    return torch.tensor(points[:width], dtype=torch.float64)


def _build(opcode: str, parameters: torch.Tensor, reference: dict[str, Any]) -> Any:
    """The contract's reference program around one differentiated opcode.

    The fixed rotations and the closing rotation are the point of this builder. A
    differentiated phase-type opcode acting on a bare basis state leaves a state that
    differs from ``|0...0>`` by a scalar phase only, so its derivative measures as zero --
    which reads as agreement between the routes and is not. Every qubit is therefore
    prepared off-axis and the measurement basis is turned back onto the differentiated
    parameter.
    """
    schema = OPERATOR_SCHEMAS[opcode]
    qubits = tuple(range(schema.arity))
    rotations = reference["fixed_rotations"]
    circuit = fq.Circuit(reference["qubits"], dtype=torch.complex128)
    for index in range(reference["qubits"]):
        circuit = circuit.ry(index, theta=rotations[index])
    circuit = circuit.cx(0, 1).cx(1, 2)
    values = {name: parameters[index] for index, name in enumerate(schema.parameters)}
    circuit = circuit.gate(opcode, qubits, **values)
    return circuit.cx(2, 0).ry(0, theta=rotations[3])


def _scalar(circuit: Any, mode: str) -> torch.Tensor:
    result = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode=mode),
        outputs=fq.expectation(fq.Z(0)),
    )
    value = result.expectations[0]
    if value.numel() != 1:
        raise AssertionError(f"one expectation output measured {value.numel()} values")
    return value.reshape(()).to(torch.float64)


def _loss(mode: str):
    def loss(circuit: Any) -> torch.Tensor:
        return _scalar(circuit, mode)

    return loss


def _richardson(
    opcode: str,
    parameters: torch.Tensor,
    mode: str,
    scheme: dict[str, Any],
    reference: dict[str, Any],
) -> torch.Tensor:
    """The third route: Richardson-extrapolated central differences of `fq.run` alone.

    Halving the displacement `levels` times gives `levels` central differences whose
    truncation errors fall by four each time; `order` Richardson levels then combine them
    to cancel the leading terms. The result shares no code with either gradient route,
    which is what lets it decide between them.
    """
    levels = scheme["levels"]
    order = scheme["order"]
    base = scheme["base_step"]
    values = []
    for index in range(parameters.numel()):

        def difference(step: float, index: int = index) -> torch.Tensor:
            up = parameters.clone()
            up[index] = parameters[index] + step
            down = parameters.clone()
            down[index] = parameters[index] - step
            return (
                _scalar(_build(opcode, up, reference), mode)
                - _scalar(_build(opcode, down, reference), mode)
            ) / (2 * step)

        samples = [difference(base / 2**level) for level in range(levels)]
        for level in range(order):
            factor = 4 ** (level + 1)
            samples = [
                (factor * samples[position + 1] - samples[position]) / (factor - 1)
                for position in range(len(samples) - 1)
            ]
        values.append(samples[0])
    return torch.stack(values)


def _measure_cell(
    contract: dict[str, Any], opcode: str, mode: str, scheme: dict[str, Any]
) -> dict[str, float]:
    parameters = _evaluation_point(contract, opcode)
    reference = contract["reference"]
    autograd = fq.gradient(
        lambda values: _loss(mode)(_build(opcode, values, reference)),
        parameters,
        method="autograd",
    ).gradient
    shifted = fq.gradient(
        lambda values: _build(opcode, values, reference),
        parameters,
        _loss(mode),
        method="parameter_shift",
    ).gradient
    extrapolated = _richardson(opcode, parameters, mode, scheme, reference)
    for name, value in (
        ("autograd", autograd),
        ("parameter_shift", shifted),
        ("richardson", extrapolated),
    ):
        if value.shape != parameters.shape:
            raise AssertionError(
                f"{opcode}/{mode}: {name} produced shape {tuple(value.shape)}, "
                f"expected {tuple(parameters.shape)}"
            )
    return {
        "shift_norm": float(torch.linalg.vector_norm(shifted)),
        "parameter_shift_vs_autograd": float(
            torch.linalg.vector_norm(shifted - autograd)
        ),
        "autograd_vs_richardson": float(
            torch.linalg.vector_norm(autograd - extrapolated)
        ),
        "parameter_shift_vs_richardson": float(
            torch.linalg.vector_norm(shifted - extrapolated)
        ),
    }


def _recorded_matches(recorded: Any, measured: float) -> bool:
    return isinstance(recorded, (int, float)) and math.isclose(
        float(recorded),
        measured,
        rel_tol=RECORDED_RELATIVE_TOLERANCE,
        abs_tol=RECORDED_ABSOLUTE_TOLERANCE,
    )


def _header_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_opcode_gradient_exactness_contract_v1":
        errors.append(f"opcode exactness schema is {contract.get('schema')!r}")
    if contract.get("maturity") != "development_evidence":
        errors.append(f"opcode exactness maturity is {contract.get('maturity')!r}")
    for name in (
        "implementation",
        "declaration_source",
        "mode_source",
        "batch_profile",
    ):
        raw = contract.get(name)
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(f"opcode exactness {name} path {raw!r} does not exist")
    authorization = contract.get("authorization")
    if not isinstance(authorization, list) or not authorization:
        errors.append("opcode exactness authorization is not a non-empty list")
    else:
        for raw in authorization:
            if not isinstance(raw, str) or not (ROOT / raw).is_file():
                errors.append(
                    f"opcode exactness authorization record {raw!r} does not exist"
                )
    return errors


def _vocabulary_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    modes = sorted(runtime_options._MODES)
    if list(contract.get("declared_modes", ())) != modes:
        errors.append(
            f"opcode exactness declared_modes is {contract.get('declared_modes')!r} "
            f"but the runtime declares {modes!r}"
        )
    methods = sorted(gradient_module._GRADIENT_METHODS)
    if list(contract.get("declared_methods", ())) != methods:
        errors.append(
            f"opcode exactness declared_methods is {contract.get('declared_methods')!r} "
            f"but the implementation declares {methods!r}"
        )
    vocabulary = contract.get("declaration_vocabulary")
    if not isinstance(vocabulary, list) or not vocabulary:
        errors.append("opcode exactness declaration_vocabulary is not a non-empty list")
    else:
        sample = next(iter(OPERATOR_SCHEMAS.values()))
        for name in vocabulary:
            head, _, tail = str(name).partition(".")
            if head == "OPERATOR_SCHEMAS":
                if not isinstance(OPERATOR_SCHEMAS, Mapping) or not OPERATOR_SCHEMAS:
                    errors.append(
                        f"opcode exactness declaration_vocabulary names {name!r}, "
                        "which the declaration does not provide"
                    )
            elif head == "OperatorSchema":
                if not hasattr(sample, tail):
                    errors.append(
                        f"opcode exactness declaration_vocabulary names {name!r}, "
                        "which OperatorSchema does not provide"
                    )
            else:
                errors.append(
                    f"opcode exactness declaration_vocabulary names {name!r}, which this "
                    "gate cannot resolve to the declaration"
                )
    excluded = contract.get("excluded_modes")
    if not isinstance(excluded, list) or not excluded:
        errors.append("opcode exactness excluded_modes is not a non-empty list")
    else:
        names = {entry.get("mode") for entry in excluded if isinstance(entry, dict)}
        if names != EXCLUDED_MODE_NAMES:
            errors.append(
                f"opcode exactness excluded_modes names {sorted(names)!r}, "
                f"expected {sorted(EXCLUDED_MODE_NAMES)!r}"
            )
        for entry in excluded:
            if not isinstance(entry, dict) or not entry.get("reason"):
                errors.append("opcode exactness an excluded mode carries no reason")
    measured = list(contract.get("measured_modes", ()))
    expected = [mode for mode in modes if mode not in EXCLUDED_MODE_NAMES]
    if sorted(measured) != expected:
        errors.append(
            f"opcode exactness measured_modes is {sorted(measured)!r} but the serving "
            f"modes are {expected!r}; a mode that starts serving an expectation value "
            "has to be measured here"
        )
    return errors


def _rule_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rules = contract.get("rules", {})
    for flag in RULE_FLAGS:
        if rules.get(flag) is not True:
            errors.append(f"opcode exactness rule {flag!r} is not contracted as true")
    for flag in sorted(set(rules) - set(RULE_FLAGS)):
        errors.append(f"opcode exactness rule {flag!r} is not read by this gate")
    return errors


def _reference_errors(contract: dict[str, Any], scheme: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reference = contract.get("reference", {})
    if reference.get("dtype") != "complex128":
        errors.append(
            f"opcode exactness reference dtype is {reference.get('dtype')!r}; the "
            "comparison is contracted in complex128"
        )
    rotations = reference.get("fixed_rotations")
    if (
        not isinstance(rotations, list)
        or len(rotations) != reference.get("qubits", 0) + 1
    ):
        errors.append(
            "opcode exactness fixed_rotations does not turn the measurement basis back "
            "onto the differentiated parameter"
        )
    if not isinstance(reference.get("evaluation_points"), list) or not reference.get(
        "evaluation_points"
    ):
        errors.append("opcode exactness evaluation_points is empty")
    for name in (
        "cell_count",
        "observability_floor",
        "max_parameter_shift_vs_autograd",
        "max_autograd_vs_richardson",
        "max_parameter_shift_vs_richardson",
        "max_cross_mode_spread",
    ):
        if name not in reference:
            errors.append(f"opcode exactness reference {name} is missing")
    for name in ("kind", "levels", "order", "base_step"):
        if name not in scheme:
            errors.append(f"opcode exactness difference_scheme {name} is missing")
    if scheme.get("kind") != "richardson_extrapolated_central_difference":
        errors.append(
            f"opcode exactness difference_scheme kind is {scheme.get('kind')!r}"
        )
    for flag in (
        "base_step_is_a_probe_choice",
        "shares_no_code_with_the_compared_routes",
    ):
        if scheme.get(flag) is not True:
            errors.append(
                f"opcode exactness difference_scheme {flag} is not contracted as true"
            )
    levels = scheme.get("levels")
    order = scheme.get("order")
    if isinstance(levels, int) and isinstance(order, int) and not 0 < order < levels:
        errors.append(
            f"opcode exactness difference_scheme removes {order} level(s) from "
            f"{levels} sample(s), which cannot extrapolate"
        )
    return errors


def _opcode_row_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = contract.get("opcode")
    if not isinstance(rows, list):
        return ["opcode exactness [[opcode]] is not a list of tables"]
    declared = _differentiable()
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            errors.append("opcode exactness an [[opcode]] row carries no name")
            continue
        name = row["name"]
        if name in seen:
            errors.append(
                f"opcode exactness {name!r} is described by more than one row"
            )
        seen[name] = row
        schema = declared.get(name)
        if schema is None:
            errors.append(
                f"opcode exactness describes {name!r}, which declares no derivative rule; "
                "a row for a rule-less opcode claims a derivative it does not have"
            )
            continue
        if row.get("arity") != schema.arity:
            errors.append(
                f"opcode exactness {name}.arity is {row.get('arity')!r}, measured {schema.arity}"
            )
        if list(row.get("declared_parameters", ())) != list(schema.parameters):
            errors.append(
                f"opcode exactness {name}.declared_parameters is "
                f"{row.get('declared_parameters')!r}, measured {list(schema.parameters)!r}"
            )
        frequencies = [list(freqs) for freqs in schema.parameter_frequencies]
        if [
            list(freqs) for freqs in row.get("declared_frequencies", ())
        ] != frequencies:
            errors.append(
                f"opcode exactness {name}.declared_frequencies is "
                f"{row.get('declared_frequencies')!r}, measured {frequencies!r}"
            )
        terms = [len(schema.shift_rule(parameter)) for parameter in schema.parameters]
        if list(row.get("terms_per_parameter", ())) != terms:
            errors.append(
                f"opcode exactness {name}.terms_per_parameter is "
                f"{row.get('terms_per_parameter')!r}, measured {terms!r}"
            )
        for parameter, declared_frequencies, rule_length in zip(
            schema.parameters, schema.parameter_frequencies, terms, strict=True
        ):
            if rule_length != 2 * len(declared_frequencies):
                errors.append(
                    f"opcode exactness {name}.{parameter} declares {len(declared_frequencies)} "
                    f"frequenc(ies) and its rule has {rule_length} term(s), which is not "
                    "two terms per frequency"
                )
        needed = max(len(freqs) for freqs in schema.parameter_frequencies)
        if row.get("pair_count_needed") != needed:
            errors.append(
                f"opcode exactness {name}.pair_count_needed is "
                f"{row.get('pair_count_needed')!r}, measured {needed}"
            )
    for name in sorted(set(declared) - set(seen)):
        errors.append(
            f"opcode exactness has no [[opcode]] row for {name!r}, whose derivative rule "
            "is declared; an opcode that gains a rule has to be measured here"
        )
    return errors


def _cell_shape_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cells = contract.get("cell")
    if not isinstance(cells, list):
        return ["opcode exactness [[cell]] is not a list of tables"]
    expected = {
        (opcode, mode)
        for opcode in _differentiable()
        for mode in contract.get("measured_modes", ())
    }
    seen = set()
    for cell in cells:
        if not isinstance(cell, dict):
            errors.append("opcode exactness a [[cell]] row is not a table")
            continue
        key = (cell.get("opcode"), cell.get("mode"))
        if key in seen:
            errors.append(f"opcode exactness cell {key!r} is recorded more than once")
        seen.add(key)
        if key not in expected:
            errors.append(
                f"opcode exactness records cell {key!r}, which is not a differentiable "
                "opcode in a measured mode"
            )
    for key in sorted(expected - seen, key=lambda item: (str(item[0]), str(item[1]))):
        errors.append(f"opcode exactness has no cell for {key!r}")
    if contract.get("reference", {}).get("cell_count") != len(expected):
        errors.append(
            f"opcode exactness reference.cell_count is "
            f"{contract.get('reference', {}).get('cell_count')!r}, measured {len(expected)}"
        )
    return errors


def _cell_measurement_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    scheme = contract.get("difference_scheme", {})
    for cell in contract.get("cell", ()):
        opcode = cell["opcode"]
        mode = cell["mode"]
        try:
            measured = _measure_cell(contract, opcode, mode, scheme)
        except Exception as error:
            # A cell that cannot be measured is a failure with a reason, not a traceback:
            # an uncontracted mode or a difference scheme that cannot extrapolate has to
            # be reported like any other disagreement.
            errors.append(
                f"opcode exactness {opcode}/{mode} could not be re-measured: "
                f"{type(error).__name__}: {error}"
            )
            continue
        for key, value in measured.items():
            if not _recorded_matches(cell.get(key), value):
                errors.append(
                    f"opcode exactness {opcode}/{mode} records {key}="
                    f"{cell.get(key)!r} but re-measured {value:.12e}"
                )
    return errors


def _aggregate_errors(contract: dict[str, Any]) -> list[str]:
    """Re-measure the four figures that summarize the whole matrix.

    A maximum over cells is the kind of figure that survives an edit it should not, so
    it is recomputed from the cells rather than checked for plausibility. The
    observability floor is recomputed for the same reason: a comparison against an
    all-zero reference would be vacuous, so the floor is asserted as measured rather
    than assumed.
    """
    errors: list[str] = []
    reference = contract.get("reference", {})
    cells = list(contract.get("cell", ()))
    if not cells:
        return ["opcode exactness has no cells to summarize"]
    floor = min(cell["shift_norm"] for cell in cells)
    max_ap = max(cell["parameter_shift_vs_autograd"] for cell in cells)
    max_ar = max(cell["autograd_vs_richardson"] for cell in cells)
    max_pr = max(cell["parameter_shift_vs_richardson"] for cell in cells)
    for name, measured in (
        ("observability_floor", floor),
        ("max_parameter_shift_vs_autograd", max_ap),
        ("max_autograd_vs_richardson", max_ar),
        ("max_parameter_shift_vs_richardson", max_pr),
    ):
        if not _recorded_matches(reference.get(name), measured):
            errors.append(
                f"opcode exactness reference.{name} is {reference.get(name)!r} but the "
                f"cells re-measure {measured:.12e}"
            )
    tolerance = reference.get("tolerance")
    if not isinstance(tolerance, (int, float)):
        errors.append("opcode exactness reference.tolerance is not a number")
        return errors
    for name, measured in (
        ("max_parameter_shift_vs_autograd", max_ap),
        ("max_autograd_vs_richardson", max_ar),
        ("max_parameter_shift_vs_richardson", max_pr),
    ):
        if measured > tolerance:
            errors.append(
                f"opcode exactness {name} is {measured:.12e}, above the contracted "
                f"tolerance {tolerance!r}"
            )
    if floor < tolerance:
        errors.append(
            f"opcode exactness observability_floor is {floor:.12e}, which is not above "
            f"the tolerance {tolerance!r}; a comparison against a reference that small "
            "is vacuous"
        )
    return errors


def _cross_mode_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    recorded = {
        row["name"]: row.get("cross_mode_spread")
        for row in contract.get("opcode", ())
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    worst = 0.0
    for name in sorted(_differentiable()):
        parameters = _evaluation_point(contract, name)
        gradients = {}
        for mode in contract.get("measured_modes", ()):
            try:
                gradients[mode] = fq.gradient(
                    lambda values, opcode=name, current=mode: _build(
                        opcode, values, contract["reference"]
                    ),
                    parameters,
                    _loss(mode),
                    method="parameter_shift",
                ).gradient
            except Exception as error:
                errors.append(
                    f"opcode exactness {name} could not be measured in mode {mode!r}: "
                    f"{type(error).__name__}: {error}"
                )
        if len(gradients) < 2:
            continue
        distinct = sorted(mode for mode in gradients if mode != "auto") or sorted(
            gradients
        )
        spread = max(
            float(torch.linalg.vector_norm(gradients[first] - gradients[second]))
            for index, first in enumerate(distinct)
            for second in distinct[index + 1 :]
        )
        worst = max(worst, spread)
        if not _recorded_matches(recorded.get(name), spread):
            errors.append(
                f"opcode exactness {name}.cross_mode_spread is {recorded.get(name)!r} "
                f"but the modes re-measure {spread:.12e}"
            )
    reference = contract.get("reference", {})
    if not _recorded_matches(reference.get("max_cross_mode_spread"), worst):
        errors.append(
            f"opcode exactness reference.max_cross_mode_spread is "
            f"{reference.get('max_cross_mode_spread')!r} but the opcodes re-measure "
            f"{worst:.12e}"
        )
    tolerance = reference.get("tolerance")
    if isinstance(tolerance, (int, float)) and worst > tolerance:
        errors.append(
            f"opcode exactness max_cross_mode_spread is {worst:.12e}, above the "
            f"contracted tolerance {tolerance!r}; the modes disagree about the "
            "derivative"
        )
    return errors


def _excluded_mode_errors(contract: dict[str, Any]) -> list[str]:
    """Drive every excluded mode, because an exclusion is a claim like any other."""
    errors: list[str] = []
    for entry in contract.get("excluded_modes", ()):
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode")
        if not isinstance(mode, str):
            errors.append("opcode exactness an excluded mode carries no name")
            continue
        served = []
        for name in sorted(_differentiable()):
            parameters = _evaluation_point(contract, name)
            try:
                fq.gradient(
                    lambda values, opcode=name: _build(
                        opcode, values, contract["reference"]
                    ),
                    parameters,
                    _loss(mode),
                    method="parameter_shift",
                )
            except CapabilityError:
                continue
            except Exception as error:
                errors.append(
                    f"opcode exactness mode {mode!r} refused {name!r} with "
                    f"{type(error).__name__} rather than CapabilityError"
                )
                continue
            served.append(name)
        if served:
            errors.append(
                f"opcode exactness mode {mode!r} is excluded as "
                f"{entry.get('reason')!r} but it served {len(served)} differentiable "
                f"opcode(s): {', '.join(served)}"
            )
    return errors


def _refusal_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    refusals = contract.get("refusal")
    if not isinstance(refusals, list) or not refusals:
        return ["opcode exactness [[refusal]] is not a non-empty list of tables"]
    for row in refusals:
        if not isinstance(row, dict):
            errors.append("opcode exactness a [[refusal]] row is not a table")
            continue
        exception = EXCEPTIONS.get(row.get("exception"))
        if exception is None:
            errors.append(
                f"opcode exactness refusal names the exception "
                f"{row.get('exception')!r}, which this gate cannot observe"
            )
        phrase = row.get("message_phrase")
        if not isinstance(phrase, str) or not phrase:
            errors.append("opcode exactness a refusal carries no message_phrase")
            continue
        for field in ("message_source", "second_message_source"):
            raw = row.get(field)
            if raw is None:
                continue
            if not isinstance(raw, str) or not (ROOT / raw).is_file():
                errors.append(
                    f"opcode exactness refusal {field} path {raw!r} does not exist"
                )
                continue
            if phrase not in (ROOT / raw).read_text(encoding="utf-8"):
                errors.append(
                    f"opcode exactness refusal phrase {phrase!r} does not occur in {raw}"
                )
        if not isinstance(row.get("trigger"), str) or not row.get("trigger"):
            errors.append("opcode exactness a refusal carries no trigger")
        if not isinstance(row.get("reachable_from"), str) or not row.get(
            "reachable_from"
        ):
            errors.append("opcode exactness a refusal carries no reachable_from")
    return errors


def _batch_profile_errors(contract: dict[str, Any]) -> list[str]:
    """Re-derive the relationship to the remote-safe batch profile from that file."""
    errors: list[str] = []
    relation = contract.get("batch_profile_relation")
    if not isinstance(relation, dict):
        return ["opcode exactness [batch_profile_relation] is missing"]
    batch = tomllib.loads(BATCH_CONTRACT.read_text(encoding="utf-8"))
    entries = {row["name"]: row for row in batch.get("opcode", ())}
    declared = _differentiable()
    refused = sorted(
        name for name in declared if not entries.get(name, {}).get("admitted", False)
    )
    admitted = len(declared) - len(refused)
    max_pairs = batch.get("protocol", {}).get("max_evaluation_pairs")
    recorded_refused = list(relation.get("refused_differentiable_opcodes", ()))
    if recorded_refused != refused:
        errors.append(
            f"opcode exactness batch_profile_relation.refused_differentiable_opcodes is "
            f"{recorded_refused!r} but the batch profile refuses {refused!r}"
        )
    if relation.get("max_evaluation_pairs") != max_pairs:
        errors.append(
            f"opcode exactness batch_profile_relation.max_evaluation_pairs is "
            f"{relation.get('max_evaluation_pairs')!r}, but the batch profile declares "
            f"{max_pairs!r}"
        )
    if relation.get("differentiable_opcodes") != len(declared):
        errors.append(
            f"opcode exactness batch_profile_relation.differentiable_opcodes is "
            f"{relation.get('differentiable_opcodes')!r}, measured {len(declared)}"
        )
    if relation.get("admitted_differentiable_opcodes") != admitted:
        errors.append(
            f"opcode exactness batch_profile_relation.admitted_differentiable_opcodes is "
            f"{relation.get('admitted_differentiable_opcodes')!r}, measured {admitted}"
        )
    for row in contract.get("opcode", ()):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        entry = entries.get(name)
        if entry is None:
            errors.append(f"opcode exactness {name!r} has no row in the batch profile")
            continue
        if row.get("batch_profile_admitted") is not bool(entry.get("admitted")):
            errors.append(
                f"opcode exactness {name}.batch_profile_admitted is "
                f"{row.get('batch_profile_admitted')!r} but the batch profile admits "
                f"{entry.get('admitted')!r}"
            )
        if row.get("batch_profile_evaluation_pairs") != entry.get("evaluation_pairs"):
            errors.append(
                f"opcode exactness {name}.batch_profile_evaluation_pairs is "
                f"{row.get('batch_profile_evaluation_pairs')!r} but the batch profile "
                f"records {entry.get('evaluation_pairs')!r}"
            )
    # The batch profile records no reason for a multi-pair refusal: it records the pair
    # count the opcode needs, and the reason follows from that count exceeding the
    # profile's limit. That derivation is what is checked, rather than a reason string
    # that would be a second statement of the same fact.
    if relation.get("split_reason") != "pair_limit":
        errors.append(
            f"opcode exactness batch_profile_relation.split_reason is "
            f"{relation.get('split_reason')!r}, which this gate cannot derive"
        )
    for name in refused:
        entry = entries[name]
        recorded_reason = entry.get("refusal_reason")
        if recorded_reason is not None:
            errors.append(
                f"opcode exactness {name!r} is refused by the batch profile as "
                f"{recorded_reason!r} rather than by the pair limit, so the recorded "
                "split reason does not hold for every refused opcode"
            )
        pairs = entry.get("evaluation_pairs")
        if (
            not isinstance(pairs, int)
            or not isinstance(max_pairs, int)
            or pairs <= max_pairs
        ):
            errors.append(
                f"opcode exactness {name!r} needs {pairs!r} evaluation pair(s) and the "
                f"batch profile limits them to {max_pairs!r}, so its refusal is not the "
                "pair limit"
            )
        row = next(
            (
                item
                for item in contract.get("opcode", ())
                if isinstance(item, dict) and item.get("name") == name
            ),
            None,
        )
        if row is None or row.get("pair_count_needed") != pairs:
            errors.append(
                f"opcode exactness {name!r} records pair_count_needed "
                f"{None if row is None else row.get('pair_count_needed')!r} but the batch "
                f"profile refuses it at {pairs!r} evaluation pair(s)"
            )
    return errors


def _verification_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    verification = contract.get("verification", {})
    for name in (
        "every_cell_is_remeasured_by_the_gate",
        "every_opcode_row_is_derived_from_the_declaration",
        "refusal_phrase_occurs_in_its_message_source",
        "vocabularies_are_compared_with_the_implementation",
        "batch_profile_relation_is_remeasured",
    ):
        if verification.get(name) is not True:
            errors.append(
                f"opcode exactness verification {name} is not contracted as true"
            )
    for name in ("contract", "gate"):
        raw = verification.get(name)
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(
                f"opcode exactness verification {name} path {raw!r} does not exist"
            )
    return errors


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    scheme = contract.get("difference_scheme", {})
    errors = _header_errors(contract)
    errors.extend(_vocabulary_errors(contract))
    errors.extend(_rule_errors(contract))
    errors.extend(_reference_errors(contract, scheme))
    errors.extend(_opcode_row_errors(contract))
    errors.extend(_cell_shape_errors(contract))
    errors.extend(_cell_measurement_errors(contract))
    errors.extend(_aggregate_errors(contract))
    errors.extend(_cross_mode_errors(contract))
    errors.extend(_excluded_mode_errors(contract))
    errors.extend(_refusal_errors(contract))
    errors.extend(_batch_profile_errors(contract))
    errors.extend(_verification_errors(contract))
    return tuple(errors)


def main() -> int:
    contract = _load_contract()
    errors = contract_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Opcode gradient exactness contract passed: "
        f"{len(contract.get('opcode', ()))} differentiable opcodes x "
        f"{len(contract.get('measured_modes', ()))} modes = "
        f"{len(contract.get('cell', ()))} cells re-measured against an independent "
        f"difference scheme, {len(contract.get('refusal', ()))} excluded mode, "
        f"{len(contract.get('batch_profile_relation', {}).get('refused_differentiable_opcodes', ()))} "
        "opcodes split off by the batch profile"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
