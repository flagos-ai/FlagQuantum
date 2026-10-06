#!/usr/bin/env python3
"""Validate the gradient-methods contract by re-measuring both of its axes.

`contracts/gradient-methods-contract.toml` states, for every gradient method this
repository accepts and every execution mode `ExecutionOptions` accepts, whether
`fq.gradient` produces a derivative for that combination and what the result reports.
It states the same thing a second time for a program carrying a noise model, because
noise is a `fq.run` argument rather than a mode, and it narrows the reachable set.

This gate does not read either statement back as text. It rebuilds the contract's
reference program, runs all sixty combinations, and compares each measured verdict with
the recorded one, so a cell that stopped being true fails here rather than in a user's
notebook. The two vocabularies are read from the implementation for the same reason: a
new gradient method or a new execution mode must fail this gate until the matrix is
revisited, rather than silently leaving a combination no cell describes. The refusal
rows are checked against their `message_source` files, so a reworded message is a
contract change instead of a drift nobody notices. The five unreachable mode names are
checked by constructing `ExecutionOptions` with them, because the reason they have no
row is that the option layer refuses them before any gradient code runs.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import torch

import flagquantum as fq
from flagquantum import gradients as gradient_module
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.noise import NoiseModel, depolarizing_channel
from flagquantum.runtime import options as runtime_options

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "gradient-methods-contract.toml"

#: The exception class each contracted `exception` name denotes. A name absent here is a
#: contracted class this gate cannot observe, which is a failure rather than a pass.
EXCEPTIONS: dict[str, type[BaseException]] = {
    "ValidationError": ValidationError,
    "CapabilityError": CapabilityError,
}

#: Every `[rules]` flag, named so that a flag nobody reads cannot be added unnoticed.
RULE_FLAGS = (
    "no_silent_fallback",
    "refusal_names_the_reason",
    "refusal_is_fail_closed",
    "approximate_methods_report_their_step",
    "statistical_method_reports_a_range_not_a_value",
    "step_is_ignored_by_the_exact_methods",
    "distributed_modes_are_absent_rather_than_fallback",
    "noise_axis_is_a_second_measured_axis",
    "vocabulary_is_read_from_the_implementation",
)

#: The contract's reference program, written once. `program_steps` in the contract is
#: this sequence; the gate asserts its own reference gradient matches the recorded one,
#: which is what proves both sides are measuring the same program.
REFERENCE_PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)
REFERENCE_QUBITS = 2
REFERENCE_TOLERANCE = 1e-12

#: The contract's reference noise, written once, and asserted against `[noise]`.
NOISE_PROBABILITY = 0.05
NOISE_GATE = "cx"


def _build_reference(parameters: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(REFERENCE_QUBITS, dtype=torch.complex128)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def _noise_model() -> NoiseModel:
    return NoiseModel().add(NOISE_GATE, depolarizing_channel(NOISE_PROBABILITY))


class _Counter:
    """The contract's loss, optionally noisy, counting its own evaluations."""

    def __init__(self, mode: str, noise: NoiseModel | None = None) -> None:
        self.mode = mode
        self.noise = noise
        self.calls = 0

    def __call__(self, circuit: fq.Circuit) -> torch.Tensor:
        self.calls += 1
        result = fq.run(
            circuit,
            options=fq.ExecutionOptions(mode=self.mode),
            outputs=fq.expectation(fq.Z(0)),
            noise_model=self.noise,
        )
        return result.expectations[0]


def _load_contract(path: Path = CONTRACT) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _expected_pairs(contract: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (method, mode)
        for method in contract.get("declared_methods", ())
        for mode in contract.get("serving_modes", ())
    }


def _header_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_gradient_methods_contract_v1":
        errors.append(
            "gradient methods schema must be flagquantum_gradient_methods_contract_v1"
        )
    if contract.get("maturity") != "development_evidence":
        errors.append("gradient methods maturity must be development_evidence")
    for name in ("implementation", "mode_source"):
        raw = contract.get(name)
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(f"gradient methods {name} path {raw!r} does not exist")
    for raw in contract.get("authorization", ()):
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(f"gradient methods authorization {raw!r} does not exist")
    return errors


def _vocabulary_errors(contract: dict[str, Any]) -> list[str]:
    """The two vocabularies are the implementation's, not the contract's."""

    errors: list[str] = []
    declared = list(contract.get("declared_methods", ()))
    implemented = list(gradient_module._GRADIENT_METHODS)
    if declared != implemented:
        errors.append(
            f"gradient methods declared_methods drifted: contract={declared}, "
            f"implementation={implemented}"
        )
    if list(contract.get("refused_methods", ())) != ["adjoint"]:
        errors.append("gradient methods refused_methods must name exactly adjoint")
    if "adjoint" in declared:
        errors.append("gradient methods must not declare adjoint as a method")

    modes = list(contract.get("serving_modes", ()))
    implemented_modes = sorted(runtime_options._MODES)
    if modes != implemented_modes:
        errors.append(
            f"gradient methods serving_modes drifted: contract={modes}, "
            f"implementation={implemented_modes}"
        )

    taxonomy = contract.get("taxonomy", {})
    for name in (
        "mode_is_not_a_gradient_argument",
        "mode_enters_through_the_loss",
        "noise_enters_through_fq_run",
        "noise_is_not_a_mode",
        "reported_method_is_the_method_that_ran",
        "exact_flag_is_the_only_accuracy_signal",
    ):
        if taxonomy.get(name) is not True:
            errors.append(f"gradient methods {name} is not contracted as true")
    if (
        taxonomy.get("method_vocabulary")
        != "flagquantum/gradients.py::_GRADIENT_METHODS"
    ):
        errors.append(
            "gradient methods method_vocabulary does not name the implementation"
        )
    if taxonomy.get("mode_vocabulary") != "flagquantum/runtime/options.py::_MODES":
        errors.append(
            "gradient methods mode_vocabulary does not name the implementation"
        )

    fields = runtime_options.ExecutionOptions.__dataclass_fields__
    if "mode" not in fields:
        errors.append(
            "gradient methods: ExecutionOptions no longer carries a mode field"
        )
    if "noise" in fields or "noise_model" in fields:
        errors.append(
            "gradient methods: noise appears to have become an execution mode field"
        )
    if "noise_model" not in inspect.signature(fq.run).parameters:
        errors.append("gradient methods: fq.run no longer accepts noise_model")

    # "mode is not a gradient argument" is a claim about a signature, so it is read off
    # the signature rather than trusted.
    parameters = inspect.signature(gradient_module.gradient).parameters
    if "mode" in parameters:
        errors.append("gradient methods: fq.gradient accepts a mode argument")
    if "noise_model" in parameters:
        errors.append("gradient methods: fq.gradient accepts a noise_model argument")
    for name in ("step", "directions", "generator"):
        if name not in parameters:
            errors.append(f"gradient methods: fq.gradient lost its {name!r} argument")

    lost = [
        mode
        for mode in contract.get("unreachable_modes", ())
        if mode in implemented_modes
    ]
    if lost:
        errors.append(f"gradient methods unreachable_modes are now reachable: {lost}")
    for mode in contract.get("unreachable_modes", ()):
        try:
            fq.ExecutionOptions(mode=mode)
        except ValidationError:
            continue
        except BaseException as error:
            errors.append(
                f"gradient methods unreachable mode {mode!r} refuses as "
                f"{type(error).__name__} instead of ValidationError"
            )
        else:
            errors.append(f"gradient methods unreachable mode {mode!r} is accepted")
    return errors


def _reference_errors(
    contract: dict[str, Any],
) -> tuple[list[str], torch.Tensor | None]:
    """The gate's own program must reproduce the recorded reference gradient."""

    errors: list[str] = []
    reference = contract.get("reference", {})
    parameters = reference.get("parameters")
    if (
        not isinstance(parameters, list)
        or len(parameters) != REFERENCE_PARAMETERS.numel()
    ):
        errors.append(
            "gradient methods reference.parameters does not match the gate program"
        )
    elif any(
        abs(float(recorded) - float(measured)) > 1e-12
        for recorded, measured in zip(
            parameters, REFERENCE_PARAMETERS.tolist(), strict=False
        )
    ):
        errors.append(
            "gradient methods reference.parameters differ from the gate program"
        )
    if reference.get("qubit_count") != REFERENCE_QUBITS:
        errors.append(
            "gradient methods reference.qubit_count differs from the gate program"
        )
    if reference.get("observable") != "Z(0)":
        errors.append("gradient methods reference.observable must be Z(0)")
    if reference.get("parameter_dtype") != "float64":
        errors.append("gradient methods reference.parameter_dtype must be float64")
    if reference.get("reference_method") not in contract.get("declared_methods", ()):
        errors.append(
            "gradient methods reference.reference_method is not a declared method"
        )
    if reference.get("reference_mode") not in contract.get("serving_modes", ()):
        errors.append("gradient methods reference.reference_mode is not a serving mode")

    measured = (
        fq.gradient(
            _build_reference,
            REFERENCE_PARAMETERS,
            _Counter(str(reference.get("reference_mode"))),
            method=str(reference.get("reference_method")),
        )
        .gradient.detach()
        .reshape(-1)
    )
    recorded = reference.get("reference_gradient")
    if not isinstance(recorded, list) or len(recorded) != measured.numel():
        errors.append(
            "gradient methods reference_gradient length differs from the program"
        )
        return errors, measured
    for index, (left, right) in enumerate(
        zip(recorded, measured.tolist(), strict=False)
    ):
        if abs(float(left) - float(right)) > REFERENCE_TOLERANCE:
            errors.append(
                f"gradient methods reference_gradient[{index}] is {float(left)!r} but the "
                f"program produces {float(right)!r}"
            )
    return errors, measured


def _noise_reference_errors(contract: dict[str, Any]) -> list[str]:
    """The `[noise]` block is one measurement, so it is measured once."""

    errors: list[str] = []
    noise = contract.get("noise", {})
    if noise.get("channel") != "depolarizing":
        errors.append("gradient methods noise.channel must be depolarizing")
    if noise.get("attached_to_gate") != NOISE_GATE:
        errors.append(
            "gradient methods noise.attached_to_gate differs from the gate's model"
        )
    probability = noise.get("channel_probability")
    if (
        not isinstance(probability, (int, float))
        or float(probability) != NOISE_PROBABILITY
    ):
        errors.append(
            "gradient methods noise.channel_probability differs from the gate's model"
        )
    for flag in (
        "noise_narrows_the_reachable_mode_set",
        "noise_does_not_change_the_reported_method",
    ):
        if noise.get(flag) is not True:
            errors.append(f"gradient methods noise {flag} is not contracted as true")

    method = str(noise.get("reference_method"))
    mode = str(noise.get("reference_mode"))
    if method not in contract.get("declared_methods", ()):
        errors.append(
            "gradient methods noise.reference_method is not a declared method"
        )
    measured = (
        fq.gradient(
            _build_reference,
            REFERENCE_PARAMETERS,
            _Counter(mode, _noise_model()),
            method=method,
        )
        .gradient.detach()
        .reshape(-1)
    )
    recorded = noise.get("noisy_reference_gradient")
    if not isinstance(recorded, list) or len(recorded) != measured.numel():
        errors.append(
            "gradient methods noisy_reference_gradient length differs from the program"
        )
        return errors
    for index, (left, right) in enumerate(
        zip(recorded, measured.tolist(), strict=False)
    ):
        if abs(float(left) - float(right)) > 1e-12:
            errors.append(
                f"gradient methods noisy_reference_gradient[{index}] is {float(left)!r} but "
                f"the noisy program produces {float(right)!r}"
            )
    spread = noise.get("noisy_cross_method_max_spread")
    # A spread is a claim about agreement, so it is re-measured rather than range-checked:
    # a number nobody recomputes can drift into a disagreement nobody notices.
    if not isinstance(spread, (int, float)):
        errors.append("gradient methods noise records no cross-method spread")
    else:
        measured_spread = _measured_noisy_spread()
        if abs(float(spread) - measured_spread) > 1e-14:
            errors.append(
                f"gradient methods noise.noisy_cross_method_max_spread is {float(spread)!r} "
                f"but the noisy methods differ by {measured_spread!r}"
            )
    delta = noise.get("noisy_minus_unnoisy_gradient")
    if not isinstance(delta, list) or len(delta) != measured.numel():
        errors.append(
            "gradient methods noise records no noisy-minus-unnoisy difference"
        )
    return errors


def _measured_noisy_spread() -> float:
    """The widest disagreement among the noisy deterministic methods, measured."""

    reference = (
        fq.gradient(
            _build_reference,
            REFERENCE_PARAMETERS,
            _Counter("density_matrix", _noise_model()),
            method="parameter_shift",
        )
        .gradient.detach()
        .reshape(-1)
    )
    results = []
    for method in ("autograd", "finite_difference"):
        candidate = (
            fq.gradient(
                _build_reference,
                REFERENCE_PARAMETERS,
                _Counter("density_matrix", _noise_model()),
                method=method,
            )
            .gradient.detach()
            .reshape(-1)
        )
        results.append(candidate)
    return max(
        [float((reference - candidate).abs().max()) for candidate in results]
        + [float((results[0] - results[1]).abs().max())]
    )


def _rule_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rules = contract.get("rules", {})
    for flag in RULE_FLAGS:
        if rules.get(flag) is not True:
            errors.append(f"gradient methods rule {flag!r} is not contracted as true")
    for flag in rules:
        if flag not in RULE_FLAGS:
            errors.append(f"gradient methods rule {flag!r} is not read by this gate")

    taxonomy = contract.get("taxonomy", {})
    if taxonomy.get("cell_count") != len(_expected_pairs(contract)):
        errors.append("gradient methods taxonomy.cell_count is not the pair count")
    cells = list(contract.get("cells", ()))
    if taxonomy.get("serving_cell_count") != sum(
        1 for cell in cells if cell.get("verdict") == "serves"
    ):
        errors.append("gradient methods taxonomy.serving_cell_count is not measured")
    if taxonomy.get("refusing_cell_count") != sum(
        1 for cell in cells if cell.get("verdict") == "refuses"
    ):
        errors.append("gradient methods taxonomy.refusing_cell_count is not measured")
    noise_cells = list(contract.get("noise_cells", ()))
    if taxonomy.get("noise_cell_count") != len(noise_cells):
        errors.append(
            "gradient methods taxonomy.noise_cell_count is not the pair count"
        )
    if taxonomy.get("noise_serving_cell_count") != sum(
        1 for cell in noise_cells if cell.get("verdict") == "serves"
    ):
        errors.append(
            "gradient methods taxonomy.noise_serving_cell_count is not measured"
        )
    if taxonomy.get("noise_refusing_cell_count") != sum(
        1 for cell in noise_cells if cell.get("verdict") == "refuses"
    ):
        errors.append(
            "gradient methods taxonomy.noise_refusing_cell_count is not measured"
        )
    return errors


def _cell_shape_errors(contract: dict[str, Any], key: str) -> list[str]:
    """Every pair exactly once, and every reference from a cell resolvable."""

    errors: list[str] = []
    cells = list(contract.get(key, ()))
    expected = _expected_pairs(contract)
    seen: list[tuple[str, str]] = []
    codes = {row.get("code") for row in contract.get("refusals", ())}
    methods = set(contract.get("declared_methods", ()))
    modes = set(contract.get("serving_modes", ()))

    for cell in cells:
        method = cell.get("method")
        mode = cell.get("mode")
        if method not in methods or mode not in modes:
            errors.append(
                f"gradient methods {key} ({method!r}, {mode!r}) is outside the vocabulary"
            )
            continue
        seen.append((method, mode))
        verdict = cell.get("verdict")
        if verdict == "serves":
            if cell.get("reported_method") not in methods:
                errors.append(
                    f"gradient methods {key} ({method}, {mode}) reports an unknown method"
                )
            if not isinstance(cell.get("exact"), bool):
                errors.append(
                    f"gradient methods {key} ({method}, {mode}) has no exact flag"
                )
            if not isinstance(cell.get("loss_evaluations"), int):
                errors.append(
                    f"gradient methods {key} ({method}, {mode}) has no loss_evaluations"
                )
        elif verdict == "refuses":
            if cell.get("refusal") not in codes:
                errors.append(
                    f"gradient methods {key} ({method}, {mode}) names an unlisted refusal"
                )
        else:
            errors.append(f"gradient methods {key} ({method}, {mode}) has no verdict")

    duplicated = sorted({pair for pair in seen if seen.count(pair) > 1})
    if duplicated:
        errors.append(f"gradient methods {key} repeat: {duplicated}")
    missing = sorted(expected - set(seen))
    if missing:
        errors.append(f"gradient methods {key} are missing: {missing}")
    extra = sorted(set(seen) - expected)
    if extra:
        errors.append(f"gradient methods {key} are extra: {extra}")
    return errors


def _measure(
    method: str, mode: str, noise: NoiseModel | None
) -> tuple[str, str | None, bool | None]:
    """Run one cell and report its verdict, reported method, and exactness."""

    loss = _Counter(mode, noise)
    if method == "spsa":
        # One direction, fixed seed: enough to observe the verdict and the reported
        # method. Accuracy is a property of the seed and is not asserted per cell.
        kwargs: dict[str, Any] = {
            "directions": 1,
            "generator": torch.Generator().manual_seed(0),
        }
    else:
        kwargs = {}
    try:
        result = fq.gradient(
            _build_reference, REFERENCE_PARAMETERS, loss, method=method, **kwargs
        )
    except (ValidationError, CapabilityError) as error:
        return "refuses", type(error).__name__, None
    return "serves", str(result.method), bool(result.exact)


def _cell_measurement_errors(
    contract: dict[str, Any], key: str, noisy: bool
) -> list[str]:
    errors: list[str] = []
    noise = _noise_model() if noisy else None
    for cell in contract.get(key, ()):
        verdict, reported, exact = _measure(
            str(cell["method"]), str(cell["mode"]), noise
        )
        label = f"({cell['method']}, {cell['mode']})"
        if verdict != cell.get("verdict"):
            errors.append(
                f"gradient methods {key} {label} is contracted as {cell.get('verdict')!r} "
                f"but measures {verdict!r}"
            )
            continue
        if verdict == "serves":
            if reported != cell.get("reported_method"):
                errors.append(
                    f"gradient methods {key} {label} reports {reported!r}, not "
                    f"{cell.get('reported_method')!r}"
                )
            if exact is not cell.get("exact"):
                errors.append(
                    f"gradient methods {key} {label} reports exact={exact}, not "
                    f"{cell.get('exact')}"
                )
    return errors


def _refusal_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for row in contract.get("refusals", ()):
        code = row.get("code")
        name = row.get("exception")
        if name not in EXCEPTIONS:
            errors.append(
                f"gradient methods refusal {code!r} names unobservable class {name!r}"
            )
        phrase = row.get("message_phrase")
        if not isinstance(phrase, str) or not phrase:
            errors.append(f"gradient methods refusal {code!r} has no message phrase")
            continue
        source = row.get("message_source")
        if not isinstance(source, str) or not (ROOT / source).is_file():
            errors.append(f"gradient methods refusal {code!r} names a missing source")
            continue
        if phrase not in (ROOT / source).read_text(encoding="utf-8"):
            errors.append(
                f"gradient methods refusal {code!r} phrase {phrase!r} is not in {source}"
            )
        trigger = row.get("trigger")
        if not isinstance(trigger, str) or not trigger:
            errors.append(f"gradient methods refusal {code!r} has no trigger")
        reachable = row.get("reachable_from")
        if not isinstance(reachable, list) or not reachable:
            errors.append(f"gradient methods refusal {code!r} has no reachable_from")

    errors.extend(_reframing_errors(contract))
    return errors


def _reframing_errors(contract: dict[str, Any]) -> list[str]:
    """The recorded asymmetry between a ValidationError and a CapabilityError.

    A `ValidationError` is a `ValueError` and a `CapabilityError` is not, so a
    `ValidationError` raised inside a `parameter_shift` loss is reframed by the
    shift-rule wrapper while a stabilizer refusal is not. That is a claim about the
    code, so both halves are measured instead of quoted.
    """

    errors: list[str] = []
    rows = list(contract.get("reframings", ()))
    if len(rows) != 1:
        errors.append("gradient methods must record exactly one reframing")
        return errors
    row = rows[0]
    if row.get("verdict") != "recorded_not_corrected":
        errors.append(
            "gradient methods reframing must be recorded, not declared corrected"
        )
    for name in ("outer_exception", "inner_exception"):
        if row.get(name) not in EXCEPTIONS:
            errors.append(
                f"gradient methods reframing names unobservable class {row.get(name)!r}"
            )
    inner = EXCEPTIONS.get(row.get("inner_exception"), Exception)
    outer = EXCEPTIONS.get(row.get("outer_exception"), Exception)
    if not issubclass(inner, ValueError):
        errors.append("gradient methods reframing inner class is not a ValueError")
    if issubclass(outer, ValueError):
        errors.append("gradient methods reframing outer class is a ValueError")

    phrase = row.get("outer_phrase")
    if not isinstance(phrase, str) or phrase not in (
        ROOT / "flagquantum/gradients.py"
    ).read_text(encoding="utf-8"):
        errors.append(
            "gradient methods reframing outer phrase is not in the implementation"
        )
        return errors

    instances = list(row.get("instances", ()))
    if not instances:
        errors.append("gradient methods reframing records no measured instance")
        return errors

    def reframed(method: str, inner_phrase: str, mode: str) -> str:
        noise = _noise_model() if inner_phrase.startswith("stable noisy") else None

        def loss(circuit: fq.Circuit) -> torch.Tensor:
            return fq.run(
                circuit,
                options=fq.ExecutionOptions(mode=mode),
                outputs=fq.expectation(fq.Z(0)),
                noise_model=noise,
            ).expectations[0]

        try:
            fq.gradient(_build_reference, REFERENCE_PARAMETERS, loss, method=method)
        except BaseException as error:
            return f"{type(error).__name__}: {error}"
        return "no refusal"

    for instance in instances:
        inner_phrase = str(instance.get("inner_phrase"))
        modes = list(instance.get("measured_refusing_modes", ()))
        if not modes:
            errors.append("gradient methods reframing instance names no refusing mode")
            continue
        for mode in modes:
            measured = reframed("parameter_shift", inner_phrase, str(mode))
            if not measured.startswith("CapabilityError") or phrase not in measured:
                errors.append(
                    f"gradient methods reframing does not hold for mode={mode!r}: {measured[:140]}"
                )
            elif inner_phrase not in measured:
                errors.append(
                    f"gradient methods reframing instance for mode={mode!r} lost its inner "
                    f"phrase {inner_phrase!r}"
                )
    for method in row.get("not_applied_to", ()):
        measured = reframed(str(method), "must be one of", "mps")
        if phrase in measured:
            errors.append(
                f"gradient methods reframing wrongly applies to method={method!r}"
            )
    return errors


def _accuracy_errors(
    contract: dict[str, Any], measured: torch.Tensor | None
) -> list[str]:
    errors: list[str] = []
    accuracy = contract.get("accuracy", {})
    tolerance = accuracy.get("relative_tolerance")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        errors.append(
            "gradient methods accuracy.relative_tolerance is not a positive bound"
        )
        return errors
    if measured is None:
        return errors

    scale = float(measured.abs().max())
    if scale <= 0.0:
        errors.append("gradient methods reference gradient is degenerate")
        return errors

    for method, key in (
        ("autograd", "autograd_max_deviation"),
        ("parameter_shift", "parameter_shift_max_deviation"),
    ):
        result = fq.gradient(
            _build_reference,
            REFERENCE_PARAMETERS,
            _Counter("statevector"),
            method=method,
        )
        deviation = (
            float((result.gradient.detach().reshape(-1) - measured).abs().max()) / scale
        )
        recorded = accuracy.get(key)
        if not isinstance(recorded, (int, float)):
            errors.append(
                f"gradient methods accuracy.{key} is not a measured deviation"
            )
            continue
        if deviation > tolerance:
            errors.append(
                f"gradient methods {method} deviates {deviation:.3e} from the reference, above "
                f"the contracted {tolerance:.3e}"
            )
        if deviation > 1e-09:
            errors.append(f"gradient methods {method} is no longer exact at 1e-09")

    result = fq.gradient(
        _build_reference,
        REFERENCE_PARAMETERS,
        _Counter("statevector"),
        method="finite_difference",
    )
    deviation = (
        float((result.gradient.detach().reshape(-1) - measured).abs().max()) / scale
    )
    recorded = accuracy.get("finite_difference_max_deviation")
    if not isinstance(recorded, (int, float)):
        errors.append(
            "gradient methods accuracy.finite_difference_max_deviation is missing"
        )
    elif deviation > max(float(recorded) * 100.0, 1e-09):
        errors.append(
            f"gradient methods finite_difference deviates {deviation:.3e}, far above the "
            f"recorded {float(recorded):.3e}"
        )
    if result.step != 6.055454452393343e-06:
        errors.append(
            f"gradient methods finite_difference step is {result.step!r} for float64, not the "
            "cube root of the loss dtype's epsilon"
        )

    spsa_rows = list(accuracy.get("spsa", ()))
    if not spsa_rows:
        errors.append("gradient methods accuracy records no spsa range")
    for row in spsa_rows:
        directions = row.get("directions")
        for key in ("min_deviation", "median_deviation", "max_deviation"):
            if not isinstance(row.get(key), (int, float)):
                errors.append(
                    f"gradient methods spsa directions={directions} has no {key}"
                )
        if not isinstance(row.get("loss_evaluations"), int):
            errors.append(f"gradient methods spsa directions={directions} has no cost")
        elif directions and row["loss_evaluations"] != 2 * int(directions) + 1:
            errors.append(
                f"gradient methods spsa directions={directions} cost is not two per direction"
            )
        low, high = row.get("min_deviation"), row.get("max_deviation")
        if all(isinstance(value, (int, float)) for value in (low, high)) and float(
            low
        ) > float(high):
            errors.append(
                f"gradient methods spsa directions={directions} range is inverted"
            )
    ordered = [row.get("directions") for row in spsa_rows]
    if ordered != sorted(ordered):
        errors.append("gradient methods spsa rows are not ordered by direction count")
    return errors


class _Counting(_Counter):
    """A loss that records how many times it ran."""

    def __init__(self, mode: str, counts: dict[str, int]) -> None:
        super().__init__(mode)
        self._counts = counts

    def __call__(self, circuit: fq.Circuit) -> torch.Tensor:
        self._counts["loss"] = self._counts.get("loss", 0) + 1
        return super().__call__(circuit)


def _cost_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = {
        row.get("method"): row for row in contract.get("cost", {}).get("methods", ())
    }
    if (
        contract.get("cost", {}).get("one_direction_of_spsa_costs_two_loss_evaluations")
        is not True
    ):
        errors.append("gradient methods cost rule is not contracted as true")

    counts: dict[str, int] = {}
    for method in ("autograd", "parameter_shift", "finite_difference"):
        counts["loss"] = 0
        fq.gradient(
            _build_reference,
            REFERENCE_PARAMETERS,
            _Counting("statevector", counts),
            method=method,
        )
        recorded = rows.get(method, {}).get("loss_evaluations")
        if recorded != counts["loss"]:
            errors.append(
                f"gradient methods cost for {method} records {recorded} loss evaluations but "
                f"measures {counts['loss']}"
            )
    counts["loss"] = 0
    fq.gradient(
        _build_reference,
        REFERENCE_PARAMETERS,
        _Counting("statevector", counts),
        method="spsa",
        directions=4,
        generator=torch.Generator().manual_seed(0),
    )
    recorded = rows.get("spsa", {}).get("loss_evaluations")
    if recorded != counts["loss"]:
        errors.append(
            f"gradient methods cost for spsa records {recorded} loss evaluations but measures "
            f"{counts['loss']}"
        )
    return errors


def _resolution_errors(contract: dict[str, Any]) -> list[str]:
    """The three `auto` measurements, each reproduced."""

    errors: list[str] = []
    resolution = contract.get("resolution", {})
    if resolution.get("resolution_is_measured_not_declared") is not True:
        errors.append("gradient methods resolution is not contracted as measured")

    graph = fq.gradient(
        _build_reference, REFERENCE_PARAMETERS, _Counter("statevector"), method="auto"
    )
    if resolution.get("graph_present") != graph.method:
        errors.append(
            f"gradient methods resolution.graph_present is {resolution.get('graph_present')!r} "
            f"but a graph-carrying loss resolves to {graph.method!r}"
        )

    def detached_loss(circuit: fq.Circuit) -> torch.Tensor:
        return _Counter("statevector")(circuit).detach()

    scored = fq.gradient(
        _build_reference, REFERENCE_PARAMETERS, detached_loss, method="auto"
    )
    if resolution.get("no_graph_with_loss") != scored.method:
        errors.append(
            f"gradient methods resolution.no_graph_with_loss is "
            f"{resolution.get('no_graph_with_loss')!r} but measured {scored.method!r}"
        )

    def detached_program(parameters: torch.Tensor) -> torch.Tensor:
        return _Counter("statevector")(_build_reference(parameters)).detach()

    bare = fq.gradient(detached_program, REFERENCE_PARAMETERS, method="auto")
    if resolution.get("no_graph_without_loss") != bare.method:
        errors.append(
            f"gradient methods resolution.no_graph_without_loss is "
            f"{resolution.get('no_graph_without_loss')!r} but measured {bare.method!r}"
        )
    return errors


def _verification_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    verification = contract.get("verification", {})
    for name in (
        "cell_matrix_is_remeasured_by_the_gate",
        "noise_axis_is_remeasured_by_the_gate",
        "every_refusal_phrase_occurs_in_its_message_source",
        "vocabularies_are_compared_with_the_implementation",
    ):
        if verification.get(name) is not True:
            errors.append(
                f"gradient methods verification {name} is not contracted as true"
            )
    for name in ("contract", "gate"):
        raw = verification.get(name)
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(
                f"gradient methods verification {name} path {raw!r} does not exist"
            )
    return errors


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors = _header_errors(contract)
    errors.extend(_vocabulary_errors(contract))
    errors.extend(_rule_errors(contract))
    errors.extend(_cell_shape_errors(contract, "cells"))
    errors.extend(_cell_shape_errors(contract, "noise_cells"))
    reference_errors, measured = _reference_errors(contract)
    errors.extend(reference_errors)
    errors.extend(_noise_reference_errors(contract))
    errors.extend(_accuracy_errors(contract, measured))
    errors.extend(_cost_errors(contract))
    errors.extend(_resolution_errors(contract))
    errors.extend(_cell_measurement_errors(contract, "cells", noisy=False))
    errors.extend(_cell_measurement_errors(contract, "noise_cells", noisy=True))
    errors.extend(_refusal_errors(contract))
    errors.extend(_verification_errors(contract))
    return tuple(errors)


def main() -> int:
    contract = _load_contract()
    errors = contract_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Gradient methods contract passed: "
        f"{len(contract.get('cells', ()))} cells and "
        f"{len(contract.get('noise_cells', ()))} noisy cells re-measured, "
        f"{len(contract.get('refusals', ()))} refusals, "
        f"{len(contract.get('reframings', ()))} reframing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
