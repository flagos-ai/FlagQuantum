"""Conformance of the metric tensor with two routes that share no method.

``tests/test_gradient_api.py``, ``tests/integration/test_gradient_modes.py`` and
``tests/test_native_circuit.py`` own the behaviour of ``fq.gradient`` and the
batch profile. This file owns ``contracts/metric-tensor-contract.toml`` and
``tools/check_metric_tensor_contract.py``, and asks one question of them -- is
the matrix the Fubini-Study metric of the state the program delivers, and is it
computed without reading the derivative rule that describes an expectation value
instead?

That question is not ``is the number big or small''. The tempting implementation
of a metric tensor is to reuse ``OperatorSchema.shift_rule`` the way
``fq.gradient`` does, and that reuse is wrong in a way no shape assertion sees:
measured on a single ``ry``, the declared rule applied to the state is wrong by
``sqrt(2)`` per component and the matrix reads exactly twice its true value,
while on ``phase`` the same rule is right to ``5.9e-11``. So the tests here move
the declaration, count the evaluations the route actually charges, and pin the
factor of four that separates the Fubini-Study metric from the quantum Fisher
information against the definition -- ``Re Tr[rho L_i L_j]`` with ``L`` the
symmetric logarithmic derivative -- built from hand-written operators with no
import from the package under test.

Two routes count as evidence: the implementation's central difference of the
delivered state, and reverse-mode autodiff through the same ``fq.run``, which
differentiates that state analytically and shares no arithmetic with a
difference scheme. A third route -- central differences of the oracle's own
state -- is deliberately not counted, because it is the implementation's method
with the oracle's constants and would show consistency rather than correctness.

The final section asks whether the gate is wired into CI and
``tools/pre_push.py`` at all.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum import gradients as gradient_module
from flagquantum.core import OPERATOR_SCHEMAS
from flagquantum.errors import CapabilityError, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "metric-tensor-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum" / "gradients.py"

_GATE_PATH = ROOT / "tools" / "check_metric_tensor_contract.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_metric_tensor_contract", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)

ENTRY_POINT = gradient_module.metric_tensor

WITNESSES = ("product", "repeated", "three")
REFUSAL_CASES = (
    "batched_state",
    "complex_parameters",
    "density_matrix_state",
    "empty_parameters",
    "infinite_step",
    "integral_parameters",
    "nan_step",
    "negative_step",
    "non_finite_parameters",
    "non_static_program",
    "not_a_tensor",
    "not_callable",
    "zero_step",
)


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _witness(name: str) -> dict[str, Any]:
    return next(row for row in _contract()["witness"] if row["name"] == name)


def _point(name: str) -> torch.Tensor:
    return _GATE._witness_point(_witness(name))


def _declared(opcode: str) -> Any:
    return OPERATOR_SCHEMAS[opcode]


# --------------------------------------------------------------------------------------
# The convention, which is the part a shape assertion cannot see
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WITNESSES)
def test_every_witness_is_the_fubini_study_metric_and_not_the_fisher_information(
    name: str,
) -> None:
    """``QFIM == 4 * metric_tensor`` against the definition, on every witness.

    The definition is ``Re Tr[rho L_i L_j]`` and contains no factor of four, so
    this comparison is a measurement of the convention rather than a restatement
    of it. An implementation that returned the Fisher information instead would
    be wrong by four in every cell and would pass every other test in this file.
    """

    witness = _witness(name)
    measured = _GATE._implementation_metric(witness, torch.complex128)
    fisher = _GATE._fisher_information(witness)
    factor = float(_contract()["convention"]["fisher_factor"])
    assert factor == 4.0
    assert measured.shape == fisher.shape
    deviation = float((factor * measured.double() - fisher).abs().max())
    assert deviation <= float(_contract()["exactness"]["definitional_bound"])
    assert float(fisher.abs().max()) > float(
        _contract()["exactness"]["minimum_reference_magnitude"]
    )


@pytest.mark.parametrize("name", WITNESSES)
def test_every_witness_agrees_with_reverse_mode_autodiff(name: str) -> None:
    """The independent route differentiates the state analytically."""

    witness = _witness(name)
    measured = _GATE._implementation_metric(witness, torch.complex128)
    independent = _GATE._autodiff_metric(witness, torch.complex128)
    bound = float(_contract()["exactness"]["agreement_bound"])
    assert measured.shape == independent.shape
    assert float((measured.double() - independent.double()).abs().max()) <= bound


@pytest.mark.parametrize("name", WITNESSES)
def test_the_recorded_metric_is_the_value_the_program_has(name: str) -> None:
    """A recorded cell that belongs to another program would pass nothing else."""

    witness = _witness(name)
    recorded = _GATE._recorded_metric(witness)
    assert recorded.shape == (len(witness["parameters"]), len(witness["parameters"]))
    for route in (
        _GATE._implementation_metric(witness, torch.complex128),
        _GATE._autodiff_metric(witness, torch.complex128),
    ):
        assert float((route.double() - recorded).abs().max()) <= float(
            _contract()["exactness"]["agreement_bound"]
        )


def test_the_repeated_witness_folds_two_occurrences_into_one_coordinate() -> None:
    """The matrix is indexed by parameter, not by gate occurrence.

    PennyLane's ``qml.metric_tensor`` reports a coordinate per gate occurrence
    for this program; folding the two occurrences of the shared parameter into
    one coordinate is what makes the two conventions comparable, and the
    off-diagonal cell is the part of the value that only exists after folding.
    """

    recorded = _GATE._recorded_metric(_witness("repeated"))
    assert recorded[0, 1] != 0.0
    assert abs(float(recorded[0, 1]) - float(recorded[1, 0])) <= 1e-12
    assert float(recorded[0, 0]) > float(recorded[1, 1])
    assert float(recorded[0, 0]) == pytest.approx(0.5, abs=1e-12)


def test_the_matrix_is_symmetric_to_summation_order() -> None:
    for name in WITNESSES:
        measured = _GATE._implementation_metric(_witness(name), torch.complex128)
        assert float((measured - measured.T).abs().max()) <= float(
            _contract()["exactness"]["agreement_bound"]
        )


def test_a_metric_that_vanishes_would_make_the_comparison_vacuous() -> None:
    """Every witness has a cell above the floor the contract records."""

    floor = float(_contract()["exactness"]["minimum_reference_magnitude"])
    for name in WITNESSES:
        recorded = _GATE._recorded_metric(_witness(name))
        assert float(recorded.abs().max()) >= floor


# --------------------------------------------------------------------------------------
# The route reads no rule, and that is enforced rather than promised
# --------------------------------------------------------------------------------------


def test_the_route_names_no_registered_opcode_in_its_own_code() -> None:
    """A special case for one gate would be a second rule by another name."""

    assert _GATE._opcodes_named_in_route() == []
    assert _GATE.ROUTE_SYMBOLS == ("metric_tensor", "_state_of")
    assert len(OPERATOR_SCHEMAS) > 0


def test_the_route_names_no_symbol_of_the_declaration() -> None:
    """The declaration is the second source of truth this route must not grow."""

    named = _GATE._declaration_symbols_named_in_route(_contract())
    assert named == []
    assert "shift_rule" in _contract()["unread"]["symbols"]


def test_every_forbidden_symbol_is_something_the_declaration_exposes() -> None:
    """A prohibition about a name that does not exist is not a prohibition."""

    from flagquantum.core import operator_schema
    from flagquantum.core.operator_schema import OperatorSchema

    for symbol in _contract()["unread"]["symbols"]:
        assert (
            hasattr(operator_schema, symbol)
            or hasattr(OperatorSchema, symbol)
            or symbol in {field for field in OperatorSchema.__dataclass_fields__}
        ), symbol


def test_the_route_serves_a_program_the_declared_rule_cannot_describe() -> None:
    """A program carrying an explicit matrix has a state and no declared rule.

    This is the capability the choice of route buys and the reason the choice is
    not merely a matter of taste: ``fq.gradient`` refuses the same program,
    because a shift rule for a matrix no opcode declares does not exist.
    """

    matrix = torch.eye(2, dtype=torch.complex128) / math.sqrt(2)

    def custom(parameters: torch.Tensor):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        circuit.gate("h", (0,), matrix=matrix)
        circuit.ry(0, theta=parameters[0])
        return circuit

    point = torch.tensor([0.4], dtype=torch.float64)
    measured = gradient_module.metric_tensor(custom, point)
    assert measured.shape == (1, 1)
    assert math.isfinite(float(measured[0, 0]))

    def loss_fn(circuit):
        return circuit.expectation_z((0,)).sum()

    with pytest.raises(
        ValueError, match="parameter shift does not support custom matrices"
    ):
        gradient_module.parameter_shift_gradient(custom, point, loss_fn)


# --------------------------------------------------------------------------------------
# Why the rule is not reused: measured on the declaration itself
# --------------------------------------------------------------------------------------


def test_the_declared_rule_read_as_a_state_derivative_is_wrong_on_a_rotation() -> None:
    """``ry`` declares frequency one; its state derivative is off by ``sqrt(2)``."""

    row = _contract()["falsification"]
    assert row["rotation_opcode"] == "ry"
    ratios, _ = _GATE._declared_rule_state_deviation("ry", 0.4, False)
    assert ratios, "no component of the state moved, so the reading is vacuous"
    for value in ratios:
        assert value == pytest.approx(math.sqrt(2.0), rel=1e-06)


def test_the_same_declared_rule_is_right_on_a_phase_gate() -> None:
    """A rule right for one opcode and wrong by ``sqrt(2)`` for another has no scale factor.

    This is the reading that rules out repairing the reuse with a constant: the
    declared table is not uniformly wrong, so no single multiplier fixes it.
    """

    ratios, deviation = _GATE._declared_rule_state_deviation("phase", 0.4, True)
    assert ratios, "the phase gate moved no component, so the reading is vacuous"
    for value in ratios:
        assert value == pytest.approx(1.0, rel=1e-06)
    assert deviation < 1e-09


def test_the_declared_rule_frequencies_are_recorded_from_the_declaration() -> None:
    """Moving the declaration moves this row rather than leaving it stale."""

    row = _contract()["falsification"]
    assert [
        [float(item) for item in value]
        for value in _declared(row["rotation_opcode"]).parameter_frequencies
    ] == [[float(item) for item in value] for value in row["rotation_frequencies"]]


def test_the_reference_witness_would_read_twice_the_fisher_information() -> None:
    """The clearest single reading: ``2.0`` where the truth is ``1.0``."""

    row = _contract()["falsification"]
    witness = _witness(row["reference_witness"])
    measured = _GATE._declared_rule_metric(witness)
    truth = float(_GATE._recorded_metric(witness)[0, 0])
    assert float(measured[0, 0]) == pytest.approx(2 * truth, rel=1e-06)
    assert float(measured[0, 0]) == pytest.approx(
        float(row["declared_rule_metric_diagonal"]), abs=1e-06
    )
    assert float(4 * measured[0, 0]) == pytest.approx(2.0, abs=1e-05)


# --------------------------------------------------------------------------------------
# Cost, shape, precision and graph
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("width", "evaluations"), ((1, 3), (2, 5), (3, 7)))
def test_the_route_charges_a_base_state_and_a_central_pair_per_parameter(
    width: int, evaluations: int
) -> None:
    """A route that re-evaluated once per matrix cell would cost the square."""

    calls = {"count": 0}

    def builder(parameters: torch.Tensor) -> Any:
        calls["count"] += 1
        circuit = fq.Circuit(width, dtype=torch.complex128)
        for index in range(width):
            circuit.ry(index, theta=parameters[index])
        return circuit

    gradient_module.metric_tensor(
        builder, torch.linspace(0.3, 1.1, width, dtype=torch.float64)
    )
    assert calls["count"] == evaluations
    assert (
        evaluations
        == int(_contract()["protocol"]["base_evaluations"])
        + int(_contract()["protocol"]["evaluations_per_parameter"]) * width
    )


@pytest.mark.parametrize("shape", ((1,), (2,), (3,), (4,)))
def test_the_result_shape_is_the_parameter_shape_twice(shape: tuple[int, ...]) -> None:
    def builder(parameters: torch.Tensor):
        circuit = fq.Circuit(parameters.numel(), dtype=torch.complex128)
        for index in range(parameters.numel()):
            circuit.ry(index, theta=parameters[index])
        return circuit

    point = torch.linspace(0.2, 0.9, shape[0], dtype=torch.float64)
    measured = gradient_module.metric_tensor(builder, point)
    assert measured.shape == shape * 2


@pytest.mark.parametrize(
    ("dtype", "expected"),
    ((torch.complex128, torch.float64), (torch.complex64, torch.float32)),
)
def test_the_result_follows_the_delivered_state_in_dtype(
    dtype: Any, expected: Any
) -> None:
    """The precision belongs to the program, so the parameters do not set it."""

    witness = _witness("product")
    point = _GATE._witness_point(witness)
    assert point.dtype == torch.float64
    assert _GATE._implementation_metric(witness, dtype).dtype == expected


def test_the_result_carries_no_autograd_graph() -> None:
    """A second derivative is out of scope, and a graph would imply one."""

    witness = _witness("product")
    point = _GATE._witness_point(witness).requires_grad_(True)
    measured = gradient_module.metric_tensor(_GATE._witness_builder(witness), point)
    assert not measured.requires_grad
    assert measured.device == point.device


def test_the_recorded_bound_is_a_claim_about_a_declared_precision() -> None:
    """The executor default is roughly five thousand times the bound.

    A single absolute number would be a false claim at one of the two
    precisions, so the contract records both and the gate requires the default
    reading to stay outside the bound rather than relying on it.
    """

    exactness = _contract()["exactness"]
    assert exactness["program_dtype"] == "complex128"
    assert exactness["default_program_dtype"] == "complex64"
    pinned = _GATE._implementation_metric(
        _witness(exactness["default_program_dtype_witness"]), torch.complex128
    )
    default = _GATE._implementation_metric(
        _witness(exactness["default_program_dtype_witness"]), None
    )
    assert default.dtype == torch.float32
    recorded = _GATE._recorded_metric(
        _witness(exactness["default_program_dtype_witness"])
    )
    default_deviation = float((default.double() - recorded).abs().max())
    pinned_deviation = float((pinned.double() - recorded).abs().max())
    assert default_deviation > float(exactness["agreement_bound"])
    assert pinned_deviation <= float(exactness["agreement_bound"])
    assert default_deviation == pytest.approx(
        float(exactness["default_program_dtype_deviation"]), rel=1e-02
    )


# --------------------------------------------------------------------------------------
# The surface, the refusals, and the modes that are out of reach
# --------------------------------------------------------------------------------------


def test_the_route_is_reachable_from_the_module_and_not_from_the_root() -> None:
    """The placement is deliberate and needs no Stable Core authorization."""

    assert ENTRY_POINT is gradient_module.metric_tensor
    assert "metric_tensor" in gradient_module.__all__
    assert "metric_tensor" not in fq.__all__
    assert not hasattr(fq, "metric_tensor")


def test_the_root_surface_did_not_change() -> None:
    """The frozen export list is read from the manifest, never restated."""

    import json

    manifest = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )
    stable = set(manifest["stable_exports"])
    assert set(fq.__all__) == stable
    assert "metric_tensor" not in stable


@pytest.mark.parametrize("case", REFUSAL_CASES)
def test_every_recorded_refusal_still_fires(case: str) -> None:
    row = next(item for item in _contract()["refusal"] if item["case"] == case)
    measured, message = _GATE._measure(_GATE._refusal_programs()[case])
    assert measured == row["exception"]
    assert row["message_phrase"] in message


def test_the_refusals_cover_every_program_the_gate_can_drive() -> None:
    recorded = sorted(item["case"] for item in _contract()["refusal"])
    assert recorded == sorted(_GATE._refusal_programs())


def test_a_mixed_state_is_refused_rather_than_reinterpreted() -> None:
    """``2 Tr[d_i rho d_j rho]`` is not the Fisher information and is not offered.

    On a single ``depolarizing(0.4)`` the two differ by ``7.2e-01`` under a
    matrix whose cells are around one, so a route that quietly returned the
    cheaper expression would be wrong by most of its own value.
    """

    def noisy(parameters: torch.Tensor):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.depolarizing(0, 0.4)
        return circuit

    message = ""
    with pytest.raises(CapabilityError) as refusal:
        gradient_module.metric_tensor(noisy, torch.tensor([0.4], dtype=torch.float64))
    message = str(refusal.value)
    assert "symmetric logarithmic derivative" in message
    assert "density matrix" in message


def test_the_two_modes_out_of_reach_are_recorded_as_exclusions() -> None:
    contract = _contract()
    excluded = {row["mode"] for row in contract["excluded_mode"]}
    serving = set(contract["protocol"]["serving_modes"])
    assert excluded == {"density_matrix", "stabilizer"}
    assert excluded & serving == set()
    assert excluded | serving == {
        "auto",
        "statevector",
        "mps",
        "tensor_network",
        "density_matrix",
        "stabilizer",
    }


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_a_state_producing_mode_asked_for_on_a_noisy_program_is_refused(
    mode: str,
) -> None:
    """A caller's mode is honoured and refused, never quietly replaced."""

    def noisy(parameters: torch.Tensor):
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit.ry(0, theta=parameters[0])
        circuit.depolarizing(1, 0.1)
        return circuit

    point = torch.tensor([0.4, -0.9], dtype=torch.float64)
    with pytest.raises(ValidationError, match="stable noisy execution supports"):
        gradient_module.metric_tensor(
            noisy, point, options=fq.ExecutionOptions(mode=mode)
        )
    with pytest.raises(ValidationError, match="stable noisy execution supports"):
        fq.run(noisy(point), options=fq.ExecutionOptions(mode=mode))


def test_the_stabilizer_mode_serves_no_state_and_is_not_approximated() -> None:
    """It refuses without a sampling measurement, and again once one is asked for.

    The second refusal has two reachable sentences because the sampling backend
    is an optional dependency: without it the execution names the dependency it
    would need, and with it the execution names the non-Clifford gate it will not
    approximate. Both are `CapabilityError`, which is the point -- neither is a
    state, and the route never returns a number instead.
    """

    witness = _witness("product")
    point = _GATE._witness_point(witness)
    with pytest.raises(ValidationError, match="mode='stabilizer' requires shots"):
        gradient_module.metric_tensor(
            _GATE._witness_builder(witness),
            point,
            options=fq.ExecutionOptions(mode="stabilizer"),
        )
    clifford = fq.Circuit(1, dtype=torch.complex128)
    clifford.h(0)
    clifford.ry(0, theta=0.4)
    message = ""
    with pytest.raises(CapabilityError) as refusal:
        fq.run(clifford, options=fq.ExecutionOptions(mode="stabilizer", shots=100))
    message = str(refusal.value)
    assert (
        "optional dependency" in message or "is not a Clifford gate" in message
    ), message


def test_a_batched_program_is_refused_rather_than_averaged() -> None:
    """One state per parameter set is required, so a batch has no single matrix."""

    def batched(parameters: torch.Tensor):
        return fq.Circuit(2, bsz=3, dtype=torch.complex128).ry(0, theta=parameters[0])

    with pytest.raises(
        CapabilityError, match="a batch must be differentiated one member"
    ):
        gradient_module.metric_tensor(batched, torch.tensor([0.4], dtype=torch.float64))


def test_a_program_whose_state_changes_shape_is_refused() -> None:
    """The same builder must structure the same state at every displacement."""

    def shifting(parameters: torch.Tensor):
        return fq.Circuit(
            2 if float(parameters[0]) > 0.4 else 1, dtype=torch.complex128
        ).ry(0, theta=parameters[0])

    with pytest.raises(ValidationError, match="the program is not static"):
        gradient_module.metric_tensor(
            shifting, torch.tensor([0.4], dtype=torch.float64), step=0.5
        )


@pytest.mark.parametrize(
    ("builder", "parameters", "exception", "phrase"),
    (
        (None, torch.tensor([0.4]), TypeError, "circuit_builder must be callable"),
        (
            lambda p: fq.Circuit(1),
            [0.4],
            TypeError,
            "parameters must be a torch.Tensor",
        ),
        (
            lambda p: fq.Circuit(1),
            torch.zeros(0),
            ValidationError,
            "parameters must not be empty",
        ),
        (
            lambda p: fq.Circuit(1),
            torch.zeros(2, dtype=torch.complex128),
            ValidationError,
            "real floating-point",
        ),
        (
            lambda p: fq.Circuit(1),
            torch.zeros(2, dtype=torch.int64),
            ValidationError,
            "real floating-point",
        ),
        (
            lambda p: fq.Circuit(1),
            torch.tensor([float("nan")]),
            ValidationError,
            "must be finite",
        ),
    ),
)
def test_the_argument_refusals_are_reachable_directly(
    builder: Any, parameters: Any, exception: Any, phrase: str
) -> None:
    with pytest.raises(exception, match=phrase):
        gradient_module.metric_tensor(builder, parameters)


@pytest.mark.parametrize("step", (0.0, -1e-3, float("inf"), float("nan")))
def test_a_step_that_is_not_a_positive_finite_displacement_is_refused(
    step: float,
) -> None:
    witness = _witness("product")
    with pytest.raises(ValueError, match="positive finite displacement"):
        gradient_module.metric_tensor(
            _GATE._witness_builder(witness), _GATE._witness_point(witness), step=step
        )


def test_the_default_step_is_derived_from_the_delivered_precision() -> None:
    """An explicit step changes the reading; the default is not a constant.

    The central-difference floor follows the state's precision, so a caller who
    passes one number for both a `complex64` and a `complex128` program has
    chosen a step that is too coarse for one of them.
    """

    witness = _witness("product")
    recorded = _GATE._recorded_metric(witness)
    coarse = gradient_module.metric_tensor(
        _GATE._witness_builder(witness, torch.complex128),
        _GATE._witness_point(witness),
        step=1e-1,
    )
    assert float((coarse.double() - recorded).abs().max()) > float(
        _contract()["exactness"]["agreement_bound"]
    )


# --------------------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------------------


def test_the_gate_accepts_the_checked_in_contract() -> None:
    assert _GATE.contract_errors(_contract()) == ()


def test_the_gate_runs_in_ci_and_before_push() -> None:
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_metric_tensor_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_metric_tensor_contract.py"' in pre_push


def test_the_gate_reports_what_it_measured(capsys: pytest.CaptureFixture[str]) -> None:
    """The success line is the measurement summary, not a bare ``ok``."""

    assert _GATE.main([]) == 0
    printed = capsys.readouterr().out
    assert "Metric-tensor contract passed" in printed
    assert "symmetric logarithmic derivative" in printed
    assert "excluded" in printed


def test_the_gate_refuses_a_contract_whose_unit_no_longer_holds() -> None:
    """A tampered convention must be refused by the gate and must not crash it."""

    contract = _contract()
    contract["convention"]["fisher_factor"] = 1.0
    errors = _GATE.contract_errors(contract)
    assert errors
    assert all(isinstance(error, str) and error for error in errors)


def _miscount_the_cost(contract: dict[str, Any]) -> None:
    contract["evaluation"][0]["evaluations"] = 99


def _loosen_the_bound(contract: dict[str, Any]) -> None:
    contract["exactness"]["agreement_bound"] = 1e-30


def _make_the_floor_vacuous(contract: dict[str, Any]) -> None:
    contract["exactness"]["minimum_reference_magnitude"] = 1.0


def _misstate_a_witness_value(contract: dict[str, Any]) -> None:
    contract["witness"][0]["recorded_metric"][0][0] = 9.0


def _misstate_a_witness_deviation(contract: dict[str, Any]) -> None:
    contract["witness"][0]["measured_deviation"] = 1e-03


def _misstate_the_default_precision_reading(contract: dict[str, Any]) -> None:
    contract["exactness"]["default_program_dtype_deviation"] = 1e-12


def _drop_a_witness(contract: dict[str, Any]) -> None:
    contract["witness"].pop()


def _drop_a_witness_gate(contract: dict[str, Any]) -> None:
    contract["witness"][0]["gates"] = [
        gate for gate in contract["witness"][0]["gates"] if "parameter" not in gate
    ]


def _rename_a_refusal_exception(contract: dict[str, Any]) -> None:
    contract["refusal"][0]["exception"] = "PlanningError"


def _misquote_a_refusal(contract: dict[str, Any]) -> None:
    contract["refusal"][0]["message_phrase"] = "this message does not exist"


def _drop_a_refusal(contract: dict[str, Any]) -> None:
    contract["refusal"].pop()


def _serve_an_excluded_mode(contract: dict[str, Any]) -> None:
    contract["protocol"]["serving_modes"].append("stabilizer")


def _misattribute_an_exclusion(contract: dict[str, Any]) -> None:
    row = next(
        item for item in contract["excluded_mode"] if item["mode"] == "stabilizer"
    )
    row["refused_by"] = "metric_tensor"


def _forge_a_stabilizer_refusal(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["noise_refusal"] if item["mode"] == "mps")
    row["mode"] = "sampling"
    row["message_phrase"] = "stable noisy execution"


def _claim_a_root_export(contract: dict[str, Any]) -> None:
    contract["rules"]["implementation_is_absent_from_the_root_surface"] = False


def _claim_an_opcode_drives_the_route(contract: dict[str, Any]) -> None:
    contract["rules"]["no_opcode_drives_the_route"] = False


def _claim_the_declaration_is_read(contract: dict[str, Any]) -> None:
    contract["declaration_is_read"] = True


def _count_a_literal_that_is_not_there(contract: dict[str, Any]) -> None:
    contract["unread"]["expected_opcode_literals"] = 3


def _forbid_a_symbol_that_does_not_exist(contract: dict[str, Any]) -> None:
    contract["unread"]["symbols"].append("shift_rule_for_the_state")


def _claim_a_graph_is_kept(contract: dict[str, Any]) -> None:
    contract["rules"]["result_carries_no_autograd_graph"] = False


def _claim_a_mixed_state_is_served(contract: dict[str, Any]) -> None:
    contract["rules"]["a_mixed_state_is_refused_rather_than_reinterpreted"] = False


def _claim_a_custom_matrix_is_refused(contract: dict[str, Any]) -> None:
    contract["rules"]["custom_matrix_programs_are_served"] = False


def _move_the_convention_off_the_fubini_study_metric(contract: dict[str, Any]) -> None:
    contract["convention"]["quantity"] = "quantum_fisher_information"


def _blank_a_convention_entry(contract: dict[str, Any]) -> None:
    contract["convention"]["cell"] = ""


def _change_the_falsification_route(contract: dict[str, Any]) -> None:
    contract["falsification"]["route"] = "some_other_reading"


def _misstate_the_declared_frequencies(contract: dict[str, Any]) -> None:
    contract["falsification"]["rotation_frequencies"] = [[2.0]]


def _misstate_a_declared_component_ratio(contract: dict[str, Any]) -> None:
    contract["falsification"]["rotation_component_ratio"] = [1.0, 1.0]


def _misstate_a_coincident_reading(contract: dict[str, Any]) -> None:
    contract["falsification"]["coincident_ratio"] = [2.0, 2.0]


def _name_a_witness_that_does_not_exist(contract: dict[str, Any]) -> None:
    contract["falsification"]["reference_witness"] = "absent"


def _stop_deriving_the_opcodes_from_the_declaration(contract: dict[str, Any]) -> None:
    contract["unread"]["opcodes_are_derived_from_the_declaration"] = False


def _claim_the_wrong_scope(contract: dict[str, Any]) -> None:
    contract["scope"]["provided"] = ["fq.metric_tensor"]


def _vanish_the_contract_test(contract: dict[str, Any]) -> None:
    contract["verification"]["contract"] = "tests/unit/absent.py"


def _vanish_an_expansion_test(contract: dict[str, Any]) -> None:
    contract["verification"]["expansion_tests"].append("tests/unit/absent.py")


def _drop_a_census_string(contract: dict[str, Any]) -> None:
    contract["verification"]["census"] = ["def metric_tensor_that_is_not_there("]


def _misname_the_implementation_symbol(contract: dict[str, Any]) -> None:
    contract["verification"]["implementation_symbol"] = "flagquantum.metric_tensor"


def _record_a_step_that_is_not_two_per_parameter(contract: dict[str, Any]) -> None:
    contract["protocol"]["evaluations_per_parameter"] = 1


def _claim_the_cost_grows_quadratically(contract: dict[str, Any]) -> None:
    contract["protocol"]["cost_growth"] = "quadratic_in_the_parameter_count"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (_miscount_the_cost, "made"),
        (_loosen_the_bound, "above the"),
        (_make_the_floor_vacuous, "vacuous"),
        (_misstate_a_witness_value, "from the recorded metric"),
        (_misstate_a_witness_deviation, "records a deviation"),
        (_misstate_the_default_precision_reading, "executor-default deviation"),
        (_drop_a_witness, "never exercised"),
        (_drop_a_witness_gate, "its gates read positions"),
        (_rename_a_refusal_exception, "not one of"),
        (_misquote_a_refusal, "not a substring of"),
        (_drop_a_refusal, "the refusal rows name"),
        (_serve_an_excluded_mode, "both serves and excludes"),
        (_misattribute_an_exclusion, "refused by"),
        (_forge_a_stabilizer_refusal, "which the gate does not drive"),
        (_claim_a_root_export, "implementation_is_absent_from_the_root_surface"),
        (_claim_an_opcode_drives_the_route, "no_opcode_drives_the_route"),
        (_claim_the_declaration_is_read, "declaration_is_read"),
        (_count_a_literal_that_is_not_there, "opcode literals"),
        (_forbid_a_symbol_that_does_not_exist, "does not expose"),
        (_claim_a_graph_is_kept, "result_carries_no_autograd_graph"),
        (_claim_a_mixed_state_is_served, "a_mixed_state_is_refused"),
        (_claim_a_custom_matrix_is_refused, "custom_matrix_programs_are_served"),
        (
            _move_the_convention_off_the_fubini_study_metric,
            "not the Fubini-Study metric",
        ),
        (_blank_a_convention_entry, "convention entry is empty"),
        (_change_the_falsification_route, "which this gate does not measure"),
        (_misstate_the_declared_frequencies, "records the frequencies"),
        (_misstate_a_declared_component_ratio, "component ratio"),
        (_misstate_a_coincident_reading, "component ratio"),
        (_name_a_witness_that_does_not_exist, "not one of"),
        (_stop_deriving_the_opcodes_from_the_declaration, "opcodes_are_derived"),
        (_claim_the_wrong_scope, "which this gate cannot check"),
        (_vanish_the_contract_test, "does not exist"),
        (_vanish_an_expansion_test, "does not exist"),
        (_drop_a_census_string, "census string"),
        (_misname_the_implementation_symbol, "is not importable"),
        (_record_a_step_that_is_not_two_per_parameter, "evaluations per parameter"),
        (_claim_the_cost_grows_quadratically, "cost growth"),
    ),
)
def test_the_gate_refuses_a_contract_that_stopped_measuring(
    mutate: Any, expected: str
) -> None:
    """Each mutation breaks one distinct claim, and the gate names which."""

    contract = _contract()
    mutate(contract)
    errors = _GATE.contract_errors(contract)
    assert errors, "the gate accepted a mutated contract"
    assert any(expected in error for error in errors), errors


def test_the_gate_refuses_a_contract_that_dropped_its_declaration_vocabulary() -> None:
    """The vocabulary is what the gate read; an empty one is not a pass."""

    contract = _contract()
    contract["unread"] = {}
    errors = _GATE.contract_errors(contract)
    assert any("names no declaration" in error for error in errors), errors
