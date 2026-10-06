"""Conformance of the batch parameter-shift profile with the opcode declaration.

``tests/test_native_circuit.py`` owns the semantics of
``batched_parameter_shift_gradient``: which circuit it accepts, what value it
returns, and what it refuses. This file owns
``contracts/parameter-shift-coverage-contract.toml`` and
``tools/check_parameter_shift_coverage_contract.py``, and asks one question of
them -- is the profile still derived from the opcode declaration rather than from
a gate list that happens to agree with it today?

That is a different measurement. A gate list and a derivation agree until the
declaration moves, so the tests here move it: a synthetic three-frequency opcode
must leave the derived set, a hardcoded list must be found by the source scan,
and a contract row that misstates an opcode's pair count must be refused by the
gate. The final section asks whether the gate is wired into CI and
``tools/pre_push.py`` at all.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core import OPERATOR_SCHEMAS
from flagquantum.core.operator_schema import OperatorSchema
from flagquantum.gradients import (
    _BATCH_PROFILE_CONSTANT_GATES,
    _BATCH_PROFILE_GATES,
    _single_pair_shift_opcodes,
    batched_parameter_shift_gradient,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "parameter-shift-coverage-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum" / "gradients.py"

_GATE_PATH = ROOT / "tools" / "check_parameter_shift_coverage_contract.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_parameter_shift_coverage_contract", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# The declaration is the ground
# --------------------------------------------------------------------------------------


def test_the_profile_is_the_constant_scope_plus_the_derivation() -> None:
    """The set the profile admits is derived, not restated."""

    derived = _single_pair_shift_opcodes(OPERATOR_SCHEMAS)
    assert derived, "the derivation collapsed to nothing"
    assert _BATCH_PROFILE_GATES == _BATCH_PROFILE_CONSTANT_GATES | derived
    assert not derived & _BATCH_PROFILE_CONSTANT_GATES


def test_a_two_frequency_opcode_leaves_the_derived_set_by_itself() -> None:
    """A wider declared rule is refused because of the declaration, not a list.

    The wide entry is derived from a registered one rather than written out, so
    the test moves the declaration and nothing else -- and it never installs the
    entry, so no other test in the session sees it.
    """

    one_frequency = OPERATOR_SCHEMAS["rx"]
    wide = dataclasses.replace(one_frequency, parameter_frequencies=((0.5, 1.0),))
    schemas: Mapping[str, OperatorSchema] = {"rx": one_frequency, "wide": wide}

    derived = _single_pair_shift_opcodes(schemas)
    assert derived == frozenset({"rx"})
    assert len(one_frequency.shift_rule("theta")) == 2
    assert len(wide.shift_rule("theta")) > 2


def test_every_differentiated_opcode_declares_equally_long_rules() -> None:
    """``shift_rule`` is per parameter, and the profile measures the widest one."""

    for opcode, schema in OPERATOR_SCHEMAS.items():
        if not schema.differentiable:
            continue
        lengths = {len(schema.shift_rule(name)) for name in schema.parameters}
        assert len(lengths) == 1, (opcode, schema.parameters, sorted(lengths))


def test_the_implementation_declares_no_opcode_outside_the_constant_scope() -> None:
    """A hardcoded gate list in code is the drift this whole slice removed."""

    names = set(OPERATOR_SCHEMAS)
    found = {
        name
        for _, name in _GATE.opcode_literals(
            IMPLEMENTATION.read_text(encoding="utf-8"), names
        )
    }
    assert found == set(_BATCH_PROFILE_CONSTANT_GATES), (
        "gradients.py names opcodes in code outside the declared protocol scope; "
        "admission belongs in the opcode declaration"
    )


def test_the_source_scan_ignores_prose_and_finds_code() -> None:
    """A scan that reported nothing would otherwise be indistinguishable from silence."""

    assert _GATE.census_self_test() == []
    assert _GATE.opcode_literals('x = "rx"\n', {"rx"}) == [(1, "rx")]
    assert _GATE.opcode_literals('"""Differentiates RX only."""\n', {"RX"}) == []


# --------------------------------------------------------------------------------------
# What a user gets
# --------------------------------------------------------------------------------------


def _remote_shaped_gradient(
    build, parameters: torch.Tensor, batch_loss_fn
) -> tuple[torch.Tensor, list[int]]:
    sizes: list[int] = []

    def counting(circuits):
        sizes.append(len(circuits))
        return batch_loss_fn(circuits)

    return batched_parameter_shift_gradient(build, parameters, counting), sizes


def test_a_two_qubit_entangling_circuit_matches_autograd_in_one_request() -> None:
    """The workflow this profile exists for: one remote batch, two calls per parameter."""

    def build(values: torch.Tensor):
        return (
            fq.Circuit(2, dtype=torch.complex128)
            .h(0)
            .rz(0, theta=0.3)
            .rzz(0, 1, theta=values[0])
            .phase(1, theta=values[1])
            .ry(0, theta=0.4)
            .rz(0, theta=0.13)
        )

    def loss(circuit):
        return circuit.expectation_z((0, 1)).sum()

    parameters = torch.tensor([0.37, -0.21], dtype=torch.float64)
    reference = parameters.detach().clone().requires_grad_(True)
    loss(build(reference)).backward()

    gradient, sizes = _remote_shaped_gradient(
        build,
        parameters,
        lambda circuits: torch.tensor(
            [float(loss(circuit)) for circuit in circuits], dtype=torch.float64
        ),
    )

    assert sizes == [4], "one request of two evaluations per parameter"
    assert reference.grad is not None
    torch.testing.assert_close(gradient, reference.grad, rtol=0, atol=1e-12)
    assert not gradient.requires_grad


def test_a_multidimensional_parameter_tensor_keeps_its_shape_and_dtype() -> None:
    """One parameter per gate occurrence, and the result comes back in shape."""

    def build(values: torch.Tensor):
        return (
            fq.Circuit(2, dtype=torch.complex128)
            .h(0)
            .h(1)
            .rx(0, theta=values[0, 0])
            .ry(0, theta=values[0, 1])
            .rx(1, theta=values[1, 0])
            .rzz(0, 1, theta=values[1, 1])
            .rz(0, theta=0.13)
        )

    def loss(circuit):
        return circuit.expectation_z((0, 1)).sum()

    parameters = torch.tensor([[0.1, 0.7], [-0.3, 0.2]], dtype=torch.float64)
    gradient, sizes = _remote_shaped_gradient(
        build,
        parameters,
        lambda circuits: torch.tensor(
            [float(loss(circuit)) for circuit in circuits], dtype=torch.float64
        ),
    )

    assert sizes == [8]
    assert gradient.shape == parameters.shape
    assert gradient.dtype == parameters.dtype
    assert gradient.device == parameters.device

    reference = parameters.detach().clone().requires_grad_(True)
    loss(build(reference)).backward()
    assert reference.grad is not None
    torch.testing.assert_close(gradient, reference.grad, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "opcode, name",
    sorted(
        (opcode, name)
        for opcode, schema in OPERATOR_SCHEMAS.items()
        if schema.differentiable
        and all(len(schema.shift_rule(parameter)) == 2 for parameter in schema.parameters)
        for name in schema.parameters
    ),
)
def test_every_admitted_parameter_agrees_with_autograd(opcode: str, name: str) -> None:
    """One row per admitted parameter, on a loss that is not degenerate for it."""

    schema = OPERATOR_SCHEMAS[opcode]
    arity = max(1, schema.arity)

    def build(values: torch.Tensor):
        circuit = fq.Circuit(arity, dtype=torch.complex128)
        for qubit in range(arity):
            circuit = circuit.h(qubit).rz(qubit, theta=0.3 + 0.1 * qubit)
        parameters = {
            parameter: (values[0] if parameter == name else 0.23)
            for parameter in schema.parameters
        }
        return (
            circuit.gate(opcode, tuple(range(arity)), params=parameters)
            .ry(0, theta=0.4)
            .rz(0, theta=0.13)
        )

    def loss(circuit):
        return circuit.expectation_z(tuple(range(circuit.n_qubits))).sum()

    point = torch.tensor([0.37], dtype=torch.float64)
    reference = point.detach().clone().requires_grad_(True)
    loss(build(reference)).backward()
    assert reference.grad is not None

    gradient, sizes = _remote_shaped_gradient(
        build,
        point,
        lambda circuits: torch.tensor(
            [float(loss(circuit)) for circuit in circuits], dtype=torch.float64
        ),
    )

    assert sizes == [2]
    torch.testing.assert_close(gradient, reference.grad, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------
# What a user is refused
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opcode",
    sorted(
        opcode
        for opcode, schema in OPERATOR_SCHEMAS.items()
        if schema.differentiable and not schema.channel
        if any(len(schema.shift_rule(name)) > 2 for name in schema.parameters)
    ),
)
def test_a_wider_declared_rule_is_refused_by_name_and_pair_count(opcode: str) -> None:
    """A truncated rule is not an approximation, so it is a refusal instead."""

    schema = OPERATOR_SCHEMAS[opcode]
    arity = max(1, schema.arity)
    name = schema.parameters[0]
    pairs = len(schema.shift_rule(name)) // 2

    def build(values: torch.Tensor):
        parameters = {parameter: 0.23 for parameter in schema.parameters}
        parameters[name] = values[0]
        return fq.Circuit(arity, dtype=torch.complex128).gate(
            opcode, tuple(range(arity)), params=parameters
        )

    def unexpected_batch(_circuits):
        pytest.fail("a rule wider than the profile must fail before batch execution")

    with pytest.raises(ValueError) as error:
        batched_parameter_shift_gradient(
            build, torch.tensor([0.23], dtype=torch.float64), unexpected_batch
        )
    message = str(error.value)
    assert opcode in message
    assert f"needs {pairs} evaluation pairs" in message


def test_a_gate_with_no_declared_parameter_is_scope_not_a_rule() -> None:
    """The constant gates are the profile's one exemption, and it is named."""

    for opcode in sorted(_BATCH_PROFILE_CONSTANT_GATES):
        schema = OPERATOR_SCHEMAS[opcode]
        assert schema.parameters == ()
        assert not schema.differentiable

    def build(values: torch.Tensor):
        return fq.Circuit(1, dtype=torch.complex128).sdg(0).rx(0, theta=values[0])

    with pytest.raises(ValueError, match="sdg carries no declared parameter"):
        batched_parameter_shift_gradient(
            build, torch.tensor([0.23], dtype=torch.float64), lambda circuits: None
        )


def test_a_channel_is_refused_as_a_channel() -> None:
    def build(values: torch.Tensor):
        return fq.Circuit(1).bit_flip(0, probability=values[0])

    with pytest.raises(ValueError, match="bit_flip is a channel"):
        batched_parameter_shift_gradient(
            build, torch.tensor([0.23], dtype=torch.float64), lambda circuits: None
        )


def test_a_shift_that_is_not_the_declared_one_names_the_declared_one() -> None:
    """The displacement is read from the rule, so a wrong one is diagnosable."""

    def build(values: torch.Tensor):
        return fq.Circuit(1, dtype=torch.complex128).h(0).rx(0, theta=values[0])

    with pytest.raises(ValueError, match="must be shifted by 1.5707963267948966"):
        batched_parameter_shift_gradient(
            build,
            torch.tensor([0.23], dtype=torch.float64),
            lambda circuits: None,
            shift=0.5,
        )


# --------------------------------------------------------------------------------------
# The gate: `tools/check_parameter_shift_coverage_contract.py`
# --------------------------------------------------------------------------------------
#
# The tests above ask whether the contract is true of the code. This section asks
# whether the contract is still enforced. Each mutation patches one clause and
# requires the gate to name it.


def test_the_gate_accepts_the_checked_in_contract() -> None:
    assert _GATE.contract_errors(_contract(), ROOT) == []


def test_the_gate_runs_in_ci_and_before_push() -> None:
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_parameter_shift_coverage_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_parameter_shift_coverage_contract.py"' in pre_push


def _misstate_a_pair_count(contract: dict[str, Any]) -> None:
    row = next(
        item
        for item in contract["opcode"]
        if item["admitted"] is False and item["evaluation_pairs"] > 1
    )
    row["evaluation_pairs"] = 1


def _misstate_an_admission(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["opcode"] if item["admitted"] is True)
    row["admitted"] = False


def _loosen_the_tolerance(contract: dict[str, Any]) -> None:
    contract["exactness"]["tolerance"] = 1e-30


def _make_the_floor_vacuous(contract: dict[str, Any]) -> None:
    contract["exactness"]["reference_minimum_magnitude"] = 1.0


def _drop_an_opcode(contract: dict[str, Any]) -> None:
    contract["opcode"].pop()


def _widen_the_constant_scope(contract: dict[str, Any]) -> None:
    contract["protocol"]["constant_gates"].append("rx")


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (_misstate_a_pair_count, "evaluation_pairs"),
        (_misstate_an_admission, "measured"),
        (_loosen_the_tolerance, "rule-arithmetic bound"),
        (_make_the_floor_vacuous, "vacuous"),
        (_drop_an_opcode, "does not record registered opcode"),
        (_widen_the_constant_scope, "is not a gate with no declared parameter"),
    ],
)
def test_the_gate_refuses_a_contract_that_stopped_being_true(
    mutate, expected: str
) -> None:
    contract = _contract()
    mutate(contract)
    errors = _GATE.contract_errors(contract, ROOT)
    assert errors, "the gate accepted a mutated contract"
    assert any(expected in error for error in errors), errors
