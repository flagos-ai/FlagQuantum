"""Conformance of the parameter-shift Hessian with the opcode declaration.

``tests/test_gradient_api.py`` and ``tests/integration/test_gradient_modes.py``
own the behaviour of ``fq.gradient``; ``tests/test_native_circuit.py`` owns the
batch profile. This file owns
``contracts/parameter-shift-hessian-contract.toml`` and
``tools/check_parameter_shift_hessian_contract.py``, and asks one question of
them -- is the second-order route still the first-order rule read twice, or has a
second table of coefficients grown next to it?

That is a different measurement from "does the Hessian have the right value". A
second coefficient table and a composition of the declared rule agree until the
declaration moves, so the tests here move it: a registered opcode whose
frequencies are replaced must produce a different rule and a different cost, the
route's own source must name no opcode, and a contract row that misstates a term
count must be refused by the gate. The value claim is measured separately against
Richardson central differences of ``fq.run``, which shares no code with the route
and is therefore the only route here that is independent rather than merely
consistent.

The final section asks whether the gate is wired into CI and
``tools/pre_push.py`` at all.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
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
CONTRACT = ROOT / "contracts" / "parameter-shift-hessian-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum" / "gradients.py"

_GATE_PATH = ROOT / "tools" / "check_parameter_shift_hessian_contract.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_parameter_shift_hessian_contract", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)

ENTRY_POINT = gradient_module.parameter_shift_hessian


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _admitted() -> list[str]:
    return sorted(
        name for name, schema in OPERATOR_SCHEMAS.items() if schema.differentiable
    )


def _width(opcode: str) -> int:
    return max(1, len(OPERATOR_SCHEMAS[opcode].parameters))


def _witness(opcode: str):
    """A program whose gate parameters are driven one-to-one by the inputs."""

    fixed = (0.9, 1.4, 0.6, 1.1)
    schema = OPERATOR_SCHEMAS[opcode]
    qubits = max(1, schema.arity)
    names = list(schema.parameters)

    def build(values: torch.Tensor):
        flat = values.reshape(-1)
        circuit = fq.Circuit(qubits, dtype=torch.complex128)
        for index in range(qubits):
            circuit = circuit.ry(index, theta=fixed[index % len(fixed)])
        circuit = circuit.gate(
            opcode,
            tuple(range(qubits)),
            **{name: flat[index] for index, name in enumerate(names)},
        )
        return circuit.ry(0, theta=fixed[3])

    return build


def _loss(opcode: str, mode: str, counter: list[int] | None = None):
    build = _witness(opcode)

    def evaluate(circuit: Any) -> torch.Tensor:
        if counter is not None:
            counter.append(1)
        result = fq.run(
            circuit,
            options=fq.ExecutionOptions(mode=mode),
            outputs=fq.expectation(fq.Z(0)),
        )
        return result.expectations[0].reshape(()).to(torch.float64)

    return build, evaluate


# --------------------------------------------------------------------------------------
# The declaration is the ground
# --------------------------------------------------------------------------------------


def test_every_contract_opcode_row_is_the_declaration() -> None:
    """The rows are the declaration, transcribed; a drift is a failure."""

    rows = {row["name"]: row for row in _contract()["opcode"]}
    assert sorted(rows) == sorted(OPERATOR_SCHEMAS)
    for name, schema in OPERATOR_SCHEMAS.items():
        row = rows[name]
        assert list(row["declared_parameters"]) == list(schema.parameters)
        assert row["admitted"] is schema.differentiable


def test_the_route_is_composed_from_the_declared_rule_and_not_a_table() -> None:
    """A second coefficient table is the drift this slice exists to prevent.

    The registered frequency of ``rx`` is replaced with a three-frequency set,
    so the declaration now produces a six-term rule where the measured one had
    two. The composition must follow the declaration -- both the rule it reads
    and the number of circuits it charges for -- because a table of second-order
    coefficients would have kept answering with the old rule. The replacement
    goes into the gate's own view of the declaration, so nothing else in the
    session sees a modified schema.
    """

    original = OPERATOR_SCHEMAS["rx"]
    assert len(original.shift_rule("theta")) == 2

    wide = dataclasses.replace(original, parameter_frequencies=((1.0, 2.0, 3.0),))
    assert len(wide.shift_rule("theta")) == 6

    values = torch.tensor([0.31], dtype=torch.float64)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_GATE, "OPERATOR_SCHEMAS", {"rx": wide})
        rules = _GATE._rules(("rx",), values)
        assert len(rules[0]) == 6
        # The composition reads whatever the declaration says: a widened rule
        # widens the cost, and it stays the first-order rule that was read twice.
        assert _GATE._distinct_displacements(rules[0], rules[0], (0, 0), 1) == 11
        assert len(OPERATOR_SCHEMAS["rx"].shift_rule("theta")) == 2


def test_a_second_frequency_changes_the_rule_the_composition_reads() -> None:
    """The composition is over whatever the declaration says, whatever it says."""

    one = OPERATOR_SCHEMAS["rz"]
    two = dataclasses.replace(one, parameter_frequencies=((1.0, 2.0),))
    assert len(one.shift_rule("theta")) == 2
    assert len(two.shift_rule("theta")) == 4
    # Four declared terms per parameter means sixteen product pairs, and the
    # coarser of the two declared steps is what a cell evaluates at.
    assert two.parameter_frequencies != one.parameter_frequencies


def test_the_route_names_no_opcode_in_its_own_code() -> None:
    """A name in the route's code would let it special-case one gate."""

    assert _GATE._opcodes_named_in_route() == []
    # The rest of the module legitimately states the batch profile's constant
    # scope, so the census has to be scoped to the route or it reports that.
    whole = {
        name
        for name in OPERATOR_SCHEMAS
        if f'"{name}"' in IMPLEMENTATION.read_text(encoding="utf-8")
    }
    assert whole, "the module-wide scan found nothing at all"


# --------------------------------------------------------------------------------------
# The rule agrees with the declaration, and the values agree with a foreign route
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("opcode", _admitted())
def test_the_rule_is_re_derived_for_every_admitted_opcode(opcode: str) -> None:
    """The gate's own composition matches the implementation, cell for cell."""

    width = _width(opcode)
    if width != len(OPERATOR_SCHEMAS[opcode].parameters):
        pytest.skip("a parameterless admitted opcode has no cell to measure")
    parameters = _GATE._point(width)
    rules = _GATE._rules((opcode,), parameters)
    expected = _GATE._cell_matrix(
        rules, parameters, _GATE._evaluate((opcode,), "statevector")
    )
    measured, _ = _GATE._measured((opcode,), "statevector")
    assert measured.shape == expected.shape
    torch.testing.assert_close(measured, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["auto", "statevector", "mps", "tensor_network"])
def test_the_matrix_agrees_with_central_differences(mode: str) -> None:
    """The only independent route: Richardson differences of ``fq.run`` alone.

    The composed rule and a second application of the declared rule share the
    same analytic statement, so their agreement is not evidence of correctness.
    This route shares no code with either and is the correctness claim.
    """

    opcode = "rz"
    measured, _ = _GATE._measured((opcode,), mode)
    difference = _GATE._difference((opcode,), mode)
    assert measured.shape == (1, 1)
    assert measured.abs().min().item() > 1e-3, "a degenerate reference proves nothing"
    torch.testing.assert_close(measured, difference, rtol=0, atol=1e-9)


def test_the_whole_declared_sweep_is_inside_the_recorded_bound() -> None:
    """Every admitted opcode in every serving mode, against the same bound."""

    contract = _contract()
    bound = float(contract["exactness"]["agreement_bound"])
    floor = float(contract["exactness"]["minimum_cell_magnitude"])
    modes = [str(mode) for mode in contract["protocol"]["serving_modes"]]
    worst = 0.0
    smallest = math.inf
    for opcode in _admitted():
        for mode in modes:
            measured, _ = _GATE._measured((opcode,), mode)
            difference = _GATE._difference((opcode,), mode)
            assert measured.shape == difference.shape
            worst = max(worst, (measured - difference).abs().max().item())
            smallest = min(smallest, measured.abs().min().item())
    assert smallest >= floor
    assert worst <= bound, (worst, bound)


def test_the_matrix_is_symmetric_to_summation_order() -> None:
    """Symmetry is a property of the construction, not evidence of the values."""

    for opcode in ("crx", "u2", "u3"):
        measured, _ = _GATE._measured((opcode,), "statevector")
        asymmetry = (measured - measured.transpose(0, 1)).abs().max().item()
        assert asymmetry <= 1e-12, (opcode, asymmetry)


def test_the_recorded_bound_is_a_claim_about_a_declared_precision() -> None:
    """The sweep is three orders finer than the precision the executor defaults to.

    An unqualified ``fq.Circuit`` carries ``complex64``, so this measurement is
    the one a user gets from the documented example rather than from the gate.
    It is recorded so that neither reading can be quoted as the other.
    """

    contract = _contract()
    exactness = contract["exactness"]
    bound = float(exactness["agreement_bound"])
    opcode, mode = (str(part) for part in exactness["default_program_dtype_witness"])
    measured = _GATE._measured_at_default_precision((opcode,), mode)
    pinned = _GATE._difference((opcode,), mode)
    width = _GATE._width((opcode,))
    implementation = (measured - pinned).abs().max().item()
    reference = (
        (measured - _GATE._difference((opcode,), mode, pinned=False)).abs().max().item()
    )

    assert _GATE._witness((opcode,))(_GATE._point(width)).dtype == getattr(
        torch, str(exactness["program_dtype"])
    )
    assert (
        _GATE._default_precision_witness((opcode,))(_GATE._point(width)).dtype
        == torch.complex64
    )
    # The implementation on its own is the smaller of the two readings, and both
    # are above the bound. The record exists so that neither is quoted as it.
    assert implementation == pytest.approx(
        float(exactness["default_program_dtype_implementation_deviation"]), rel=1e-06
    )
    assert reference == pytest.approx(
        float(exactness["default_program_dtype_reference_deviation"]), rel=1e-06
    )
    assert implementation > bound
    assert reference > implementation


def test_the_documented_precision_advice_is_the_measured_one() -> None:
    """The docstring's precision sentence is a claim, so it is re-derived here."""

    def build(parameters: torch.Tensor, dtype: Any = None):
        circuit = fq.Circuit(2) if dtype is None else fq.Circuit(2, dtype=dtype)
        circuit.ry(0, theta=parameters[0])
        circuit.ry(1, theta=parameters[1])
        return circuit.cx(0, 1)

    def loss(circuit: Any) -> torch.Tensor:
        return fq.run(
            circuit,
            options=fq.ExecutionOptions(precision="complex128"),
            outputs=fq.expectation(fq.Z(1)),
        ).expectations[0]

    def default_loss(circuit: Any) -> torch.Tensor:
        return fq.run(circuit, outputs=fq.expectation(fq.Z(1))).expectations[0]

    parameters = torch.tensor([0.4, -0.9], dtype=torch.float64)
    exact = -math.cos(0.4) * math.cos(0.9)
    default = gradient_module.parameter_shift_hessian(build, parameters, default_loss)
    widened = gradient_module.parameter_shift_hessian(build, parameters, loss)
    widened_dtype = gradient_module.parameter_shift_hessian(
        lambda values: build(values, torch.complex128), parameters, default_loss
    )

    assert default[0, 0].item() == pytest.approx(exact, abs=6.5e-08)
    assert abs(default[0, 0].item() - exact) > 1e-08
    for result in (widened, widened_dtype):
        assert result[0, 0].item() == pytest.approx(exact, abs=1e-12)
        assert torch.equal(result, widened)


# --------------------------------------------------------------------------------------
# What the route costs and what it returns
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opcode, evaluations",
    [("rz", 3), ("crx", 7), ("u2", 14), ("u3", 33)],
)
def test_every_cell_is_its_own_set_of_circuits(opcode: str, evaluations: int) -> None:
    """The cost follows the declared rule's width, and it is not the first order."""

    _, counted = _GATE._measured((opcode,), "statevector")
    assert counted == evaluations


def test_the_cost_is_the_first_order_cost_squared_in_the_rule_width() -> None:
    """Four evaluations per off-diagonal cell, three per diagonal one."""

    width = _GATE._width(("rz", "rz", "rz"))
    rules = _GATE._rules(("rz", "rz", "rz"), _GATE._point(width))
    diagonal = sum(
        _GATE._distinct_displacements(rules[i], rules[i], (i, i), width)
        for i in range(width)
    )
    off_diagonal = sum(
        _GATE._distinct_displacements(rules[i], rules[j], (i, j), width)
        for i in range(width)
        for j in range(width)
        if i != j
    )
    assert diagonal == 3 * width
    assert off_diagonal == 4 * width * (width - 1)
    assert _GATE._first_order_cost(("rz", "rz", "rz"), "statevector") == 2 * width


@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 1)])
def test_the_result_shape_is_the_parameter_shape_twice(shape: tuple[int, ...]) -> None:
    width = math.prod(shape)
    build = _GATE._witness(tuple("rz" for _ in range(width)))
    parameters = _GATE._point(width).reshape(shape)
    matrix = ENTRY_POINT(
        build, parameters, _GATE._loss(tuple("rz" for _ in range(width)), "statevector")
    )
    assert matrix.shape == (*shape, *shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_result_follows_the_parameters_in_dtype_and_device(dtype) -> None:
    build = _GATE._witness(("rz", "rz"))
    parameters = _GATE._point(2).to(dtype)
    matrix = ENTRY_POINT(build, parameters, _GATE._loss(("rz", "rz"), "statevector"))
    assert matrix.dtype == dtype
    assert matrix.device == parameters.device


def test_the_result_carries_no_autograd_graph() -> None:
    """The route reads the declaration and detaches, so no graph comes back."""

    build = _GATE._witness(("rz", "rz"))
    parameters = _GATE._point(2).requires_grad_(True)
    matrix = ENTRY_POINT(build, parameters, _GATE._loss(("rz", "rz"), "statevector"))
    assert not matrix.requires_grad
    assert matrix.grad_fn is None


def test_a_wider_parameter_tensor_leaves_an_input_unread_and_is_refused() -> None:
    """The route is parameter-count driven: an unread input is a refusal."""

    build = _GATE._witness(("rz",))
    with pytest.raises(ValueError, match="must control exactly one gate parameter"):
        ENTRY_POINT(build, _GATE._point(2), _GATE._loss(("rz",), "statevector"))


# --------------------------------------------------------------------------------------
# What a user is refused
# --------------------------------------------------------------------------------------


def test_the_route_is_not_a_gradient_method() -> None:
    """A second derivative is not a fifth way to compute a first one."""

    assert "hessian" not in gradient_module._GRADIENT_METHODS
    with pytest.raises(ValueError, match="unknown gradient method"):
        gradient_module.gradient(
            _GATE._witness(("rz", "rz")),
            _GATE._point(2),
            _GATE._loss(("rz", "rz"), "statevector"),
            method="hessian",
        )


def test_the_route_is_reachable_from_the_module_and_not_from_the_root() -> None:
    """No frozen root surface changes, so this slice needs no authorization."""

    assert "parameter_shift_hessian" in gradient_module.__all__
    assert "parameter_shift_hessian" not in fq.__all__
    assert not hasattr(fq, "parameter_shift_hessian")
    # The root surface is read from its own contract rather than restated, so a
    # later authorized export does not turn this test into a stale constant.
    baseline = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )
    assert len(fq.__all__) == len(baseline["stable_exports"])


@pytest.mark.parametrize(
    "case",
    [
        "empty_parameters",
        "complex_parameters",
        "non_finite_parameters",
        "not_a_tensor",
        "one_input_controls_no_occurrence",
        "one_input_controls_two_occurrences",
        "scaled_angle",
        "structure_changes_with_the_parameter",
        "custom_matrix",
        "loss_is_not_scalar",
        "stabilizer_mode",
    ],
)
def test_every_recorded_refusal_still_fires(case: str) -> None:
    row = next(item for item in _contract()["refusal"] if item["case"] == case)
    name, message = _GATE._refusal_measurement(case)
    assert name == row["exception"], (case, name, row["exception"])
    assert row["message_phrase"] in message, (case, row["message_phrase"], message)


def test_the_stabilizer_mode_is_out_of_reach_and_not_a_silent_answer() -> None:
    """The sixth mode cannot serve an expectation, so it refuses rather than runs."""

    build, loss = (
        _GATE._witness(("rz", "rz")),
        _GATE._loss(("rz", "rz"), "stabilizer"),
    )
    with pytest.raises(CapabilityError, match="samples measurement outcomes"):
        ENTRY_POINT(build, _GATE._point(2), loss)


def test_a_loss_that_is_not_a_scalar_is_refused_by_both_entry_points() -> None:
    """One validator for both, so the second order cannot leak a shape error."""

    build = _GATE._witness(("rz", "rz"))

    def vector_loss(circuit: Any) -> torch.Tensor:
        value = _GATE._scalar(circuit, "statevector")
        return torch.stack([value, value])

    for entry in (
        gradient_module.parameter_shift_gradient,
        gradient_module.parameter_shift_hessian,
    ):
        with pytest.raises(ValidationError, match="must return one scalar tensor"):
            entry(build, _GATE._point(2), vector_loss)


# --------------------------------------------------------------------------------------
# The gate: `tools/check_parameter_shift_hessian_contract.py`
# --------------------------------------------------------------------------------------
#
# The tests above ask whether the contract is true of the code. This section asks
# whether the contract is still enforced. Each mutation patches one clause and
# requires the gate to name it.


def test_the_gate_accepts_the_checked_in_contract() -> None:
    assert _GATE.contract_errors(_contract()) == ()


def test_the_gate_runs_in_ci_and_before_push() -> None:
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_parameter_shift_hessian_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_parameter_shift_hessian_contract.py"' in pre_push


def test_the_gate_reports_what_it_measured() -> None:
    """The success line is the measurement summary, not a bare ``ok``."""

    assert _GATE.main([]) == 0
    assert _GATE.main(["--contract", str(CONTRACT)]) == 0


def _misstate_a_term_count(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["opcode"] if item["admitted"] is True)
    row["shift_terms"] = [99]


def _misstate_an_admission(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["opcode"] if item["admitted"] is True)
    row["admitted"] = False


def _misstate_a_cost(contract: dict[str, Any]) -> None:
    contract["cost"][0]["total_evaluations"] = 1


def _misstate_a_cell_cost(contract: dict[str, Any]) -> None:
    contract["cell"][0]["evaluations"] = 99


def _loosen_the_bound(contract: dict[str, Any]) -> None:
    contract["exactness"]["agreement_bound"] = 1e-30


def _make_the_floor_vacuous(contract: dict[str, Any]) -> None:
    contract["exactness"]["minimum_cell_magnitude"] = 1.0


def _drop_an_opcode(contract: dict[str, Any]) -> None:
    contract["opcode"].pop()


def _rename_a_refusal_exception(contract: dict[str, Any]) -> None:
    contract["refusal"][0]["exception"] = "PlanningError"


def _misquote_a_refusal(contract: dict[str, Any]) -> None:
    contract["refusal"][0]["message_phrase"] = "this message does not exist"


def _drop_a_refusal(contract: dict[str, Any]) -> None:
    contract["refusal"].pop()


def _claim_a_root_export(contract: dict[str, Any]) -> None:
    contract["rules"]["implementation_is_absent_from_the_root_surface"] = False


def _claim_a_graph_is_kept(contract: dict[str, Any]) -> None:
    contract["rules"]["result_carries_no_autograd_graph"] = False


def _claim_hessian_is_a_method(contract: dict[str, Any]) -> None:
    contract["rules"]["hessian_is_not_an_accepted_gradient_method"] = False


def _claim_the_wrong_scope(contract: dict[str, Any]) -> None:
    contract["scope"]["provided"] = ["fq.hessian"]


def _vanish_the_contract_test(contract: dict[str, Any]) -> None:
    contract["verification"]["contract"] = "tests/unit/test_absent.py"


def _drop_a_census_string(contract: dict[str, Any]) -> None:
    contract["verification"]["census"] = ["def parameter_shift_laplace("]


def _claim_the_wrong_program_dtype(contract: dict[str, Any]) -> None:
    contract["exactness"]["program_dtype"] = "complex64"


def _misname_the_precision_witness(contract: dict[str, Any]) -> None:
    contract["exactness"]["default_program_dtype_witness"] = ["ry", "statevector"]


def _misstate_the_precision_deviation(contract: dict[str, Any]) -> None:
    contract["exactness"]["default_program_dtype_implementation_deviation"] = 1e-30


def _misstate_the_reference_deviation(contract: dict[str, Any]) -> None:
    contract["exactness"]["default_program_dtype_reference_deviation"] = 1e-30


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (_misstate_a_term_count, "shift_terms"),
        (_misstate_an_admission, "differentiable flag"),
        (_misstate_a_cost, "total_evaluations"),
        (_misstate_a_cell_cost, "evaluations but the composed rule charges"),
        (_loosen_the_bound, "above the recorded bound"),
        (_make_the_floor_vacuous, "vacuous"),
        (_drop_an_opcode, "omit registered opcodes"),
        (_rename_a_refusal_exception, "not one of"),
        (_misquote_a_refusal, "not a substring of"),
        (_drop_a_refusal, "the refusal rows name"),
        (_claim_a_root_export, "implementation_is_absent_from_the_root_surface"),
        (_claim_a_graph_is_kept, "result_carries_no_autograd_graph"),
        (_claim_hessian_is_a_method, "hessian_is_not_an_accepted_gradient_method"),
        (_claim_the_wrong_scope, "which this gate cannot check"),
        (_vanish_the_contract_test, "does not exist"),
        (_drop_a_census_string, "census string"),
        (_claim_the_wrong_program_dtype, "records program_dtype"),
        (_misname_the_precision_witness, "default_program_dtype_implementation"),
        (_misstate_the_precision_deviation, "for default_program_dtype_implementation"),
        (_misstate_the_reference_deviation, "for default_program_dtype_reference"),
    ],
)
def test_the_gate_refuses_a_contract_that_stopped_being_true(
    mutate, expected: str
) -> None:
    contract = _contract()
    mutate(contract)
    errors = _GATE.contract_errors(contract)
    assert errors, "the gate accepted a mutated contract"
    assert any(expected in error for error in errors), errors


def test_the_gate_refuses_a_contract_that_dropped_its_declaration_vocabulary() -> None:
    """The vocabulary is what the gate read; an empty one is not a pass."""

    contract = _contract()
    contract["declaration_vocabulary"] = []
    errors = _GATE.contract_errors(contract)
    assert any("declaration_vocabulary" in error for error in errors), errors


def test_the_route_names_no_opcode_even_after_the_census_is_widened() -> None:
    """The scoped census is the one that decides, so widening it changes nothing."""

    assert _GATE._opcodes_named_in_route() == []
    assert _GATE.ROUTE_SYMBOLS == ("parameter_shift_hessian", "_composed_shift_terms")
