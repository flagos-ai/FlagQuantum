"""The primitives admission rule is measured, and the measurement refuses a hypothetical.

`flagquantum.algorithms.primitives` admits an export on three dimensions: the consumer that
grounds it, the distribution semantics it runs under, and its differentiability. Those were
prose before this contract, and prose cannot be wrong in a way a check notices.

The tests below do two different jobs. The first group reads the shipped artifacts and fails if
the contract and the tree disagree. The second group feeds the gate deliberately broken
contracts, because a gate whose refusal has never been exercised is a gate whose refusal is
unknown: the first measurement of this rule found an export with no consumer anywhere, and the
only way to know that finding is refused rather than grandfathered is to hand the gate the
shape that used to be accepted.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest
import torch

import flagquantum.algorithms.primitives as primitives

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "primitives-admission-contract.toml"
TYPES = ROOT / "flagquantum" / "algorithms" / "primitives" / "types.py"


def _load_gate():
    """Import `tools/check_primitives_admission_contract.py` by path.

    `tools/` is not a package on `sys.path` in the unit-test lane, so the module is loaded from
    its file the same way the repository loads other standalone gate tools.
    """
    path = ROOT / "tools" / "check_primitives_admission_contract.py"
    spec = importlib.util.spec_from_file_location("_primitives_admission_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


@pytest.fixture(scope="module")
def declared() -> dict:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured() -> dict:
    return gate.measure()


def test_the_contract_and_the_tree_agree(declared: dict, measured: dict) -> None:
    assert gate.contract_errors(declared, measured) == ()


def test_the_package_surface_is_what_the_contract_records(
    declared: dict, measured: dict
) -> None:
    recorded = [row["name"] for row in declared["primitive"]]
    assert recorded == sorted(primitives.__all__)
    assert recorded == [row["name"] for row in measured["rows"]]


def test_every_export_has_a_consumer(measured: dict) -> None:
    """The zero this slice was written for: nothing on the surface is `unadmitted`."""
    bases = {row["name"]: row["admission_basis"] for row in measured["rows"]}
    assert gate.UNADMITTED not in set(bases.values())
    assert set(bases.values()) <= set(gate.ADMISSION_BASES)


def test_derive_basis_refuses_a_hypothetical_consumer() -> None:
    """A consumer nobody wrote is not a ground, and the derivation says so."""
    assert gate.derive_basis(
        {"framework": ["a", "b"], "package": [], "tests": [], "examples": []}
    ) == ("shared")
    assert gate.derive_basis(
        {"framework": ["a"], "package": [], "tests": [], "examples": []}
    ) == ("confirmed")
    assert gate.derive_basis(
        {"framework": [], "package": ["a"], "tests": [], "examples": []}
    ) == ("grounded_expectation")
    assert gate.derive_basis(
        {"framework": [], "package": [], "tests": ["a"], "examples": []}
    ) == ("public_unit")
    assert (
        gate.derive_basis({"framework": [], "package": [], "tests": [], "examples": []})
        == gate.UNADMITTED
    )


def test_a_recorded_unadmitted_export_is_refused(
    declared: dict, measured: dict
) -> None:
    """The gate has to fail on the shape that motivated it, not just report it.

    `derive_basis` returning `unadmitted` is a measurement; a contract that records a basis for
    such an export is what the rule forbids. The probe replaces one measured row with an
    export that has no consumer at all, which is exactly what the deleted protocol was.
    """
    hollow = copy.deepcopy(measured)
    hollow["rows"].append(
        {
            "name": "StatePreparationOperator",
            "module": "flagquantum/algorithms/primitives/types.py",
            "admission_basis": gate.UNADMITTED,
            "framework_consumers": [],
            "package_consumers": [],
            "test_consumers": [],
            "example_consumers": [],
            "distribution_semantics": "single_device_fast_path",
            "accepts_tensor": False,
        }
    )
    probe = copy.deepcopy(declared)
    probe["primitive"].append(
        {
            "name": "StatePreparationOperator",
            "module": "flagquantum/algorithms/primitives/types.py",
            "admission_basis": "public_unit",
            "framework_consumers": [],
            "package_consumers": [],
            "distribution_semantics": "single_device_fast_path",
            "accepts_tensor": False,
            "differentiability_shape": "no_tensor_argument",
            "differentiable": False,
            "differentiability_note": "a protocol with no tensor argument",
        }
    )
    errors = gate.contract_errors(probe, hollow)
    assert any(
        "StatePreparationOperator" in error and "no consumer" in error
        for error in errors
    ), errors


def test_the_deleted_protocol_is_gone_from_the_package_and_the_module() -> None:
    assert "StatePreparationOperator" not in primitives.__all__
    assert not hasattr(primitives, "StatePreparationOperator")
    assert "StatePreparationOperator" not in TYPES.read_text(encoding="utf-8")
    recorded = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))["retired_export"]
    assert [row["name"] for row in recorded] == ["StatePreparationOperator"]


def test_a_retired_export_that_came_back_fails(declared: dict, measured: dict) -> None:
    """A retirement is a record, not a permission: reviving the name has to fail.

    Somebody re-adding the protocol would update the export list, and the surface then matches
    again. The retirement row is what catches that, because it still names a live export.
    """
    returned = copy.deepcopy(measured)
    returned["rows"].append(
        {
            "name": "StatePreparationOperator",
            "module": "flagquantum/algorithms/primitives/types.py",
            "admission_basis": "public_unit",
            "framework_consumers": [],
            "package_consumers": [],
            "test_consumers": ["tests/unit/test_primitives_admission_contract.py"],
            "example_consumers": [],
            "distribution_semantics": "single_device_fast_path",
            "accepts_tensor": False,
        }
    )
    probe = copy.deepcopy(declared)
    probe["primitive"].append(
        {
            "name": "StatePreparationOperator",
            "module": "flagquantum/algorithms/primitives/types.py",
            "admission_basis": "public_unit",
            "framework_consumers": [],
            "package_consumers": [],
            "distribution_semantics": "single_device_fast_path",
            "accepts_tensor": False,
            "differentiability_shape": "no_tensor_argument",
            "differentiable": False,
            "differentiability_note": "a protocol with no tensor argument",
        }
    )
    errors = gate.contract_errors(probe, returned)
    assert any("still on the export list" in error for error in errors), errors


def test_each_dimension_has_to_be_stated_where_the_contract_says(
    declared: dict, measured: dict
) -> None:
    """Naming a file that does not state the rule fails, in every dimension.

    `qft.py` is a real module in the package that states none of the three, which is the shape a
    contributor produces by listing a file because it is nearby rather than because it states the
    rule. Every dimension is required in every naming file, so all three are reported.
    """
    dropped = copy.deepcopy(declared)
    dropped["stated_in"] = ["flagquantum/algorithms/primitives/qft.py"]
    errors = gate.contract_errors(dropped, measured)
    for dimension in gate.DIMENSIONS:
        assert any(
            f"does not state the {dimension} dimension" in error for error in errors
        ), (dimension, errors)

    missing = copy.deepcopy(declared)
    missing["stated_in"] = ["docs/guides/NO_SUCH_GUIDE.md"]
    errors = gate.contract_errors(missing, measured)
    assert any("names a missing stating file" in error for error in errors), errors

    unmarked = copy.deepcopy(declared)
    unmarked["dimension"][1]["marker"] = "a phrase no file in this repository contains"
    errors = gate.contract_errors(unmarked, measured)
    assert any(
        "does not state the differentiability dimension" in error for error in errors
    )

    renamed = copy.deepcopy(declared)
    renamed["dimension"][2]["name"] = "callers"
    errors = gate.contract_errors(renamed, measured)
    assert any("dimensions must be the three" in error for error in errors)


def test_a_scalability_claim_from_a_primitive_fails(
    declared: dict, measured: dict
) -> None:
    probe = copy.deepcopy(declared)
    probe["primitive"][0]["distribution_semantics"] = "sharded_across_ranks"
    errors = gate.contract_errors(probe, measured)
    assert any("distribution semantic drifted" in error for error in errors), errors


def test_a_consumer_that_moved_fails(declared: dict, measured: dict) -> None:
    probe = copy.deepcopy(declared)
    probe["primitive"][0]["framework_consumers"] = []
    errors = gate.contract_errors(probe, measured)
    assert any("framework_consumers drifted" in error for error in errors), errors


def test_a_gradient_claim_must_name_an_existing_test(declared: dict) -> None:
    """The third answer, `differentiable_tensor_input`, is reachable and is checked."""
    row = {
        "name": "probe",
        "differentiability_shape": "differentiable_tensor_input",
        "differentiable": True,
        "differentiability_note": "the gradient flows back through the tensor",
        "differentiability_reproduce": "pytest",
    }
    actual = {"accepts_tensor": True}
    declared_shapes = {
        shape["name"] for shape in declared["differentiability"]["shape"]
    }

    missing = gate._differentiability_errors("probe", row, actual, declared_shapes)
    assert any("path::test_name" in error for error in missing), missing

    absent = dict(row, gradient_test="tests/unit/test_nothing_here.py::test_gone")
    errors = gate._differentiability_errors("probe", absent, actual, declared_shapes)
    assert any("missing gradient test file" in error for error in errors), errors

    unnamed = dict(
        row,
        gradient_test="tests/unit/test_primitives_admission_contract.py::no_such_test",
    )
    errors = gate._differentiability_errors("probe", unnamed, actual, declared_shapes)
    assert any("does not define" in error for error in errors), errors

    real = dict(
        row,
        gradient_test=(
            "tests/unit/test_primitives_admission_contract.py"
            "::test_the_contract_and_the_tree_agree"
        ),
    )
    assert gate._differentiability_errors("probe", real, actual, declared_shapes) == []


def test_a_tensor_with_no_gradient_has_to_say_why(declared: dict) -> None:
    declared_shapes = {
        shape["name"] for shape in declared["differentiability"]["shape"]
    }
    row = {
        "name": "probe",
        "differentiability_shape": "no_tensor_argument",
        "differentiable": False,
        "differentiability_note": "a note that claims nothing",
        "differentiability_reproduce": "pytest",
    }
    errors = gate._differentiability_errors(
        "probe", row, {"accepts_tensor": True}, declared_shapes
    )
    assert any("classical_tensor_input" in error for error in errors), errors

    undeclared = dict(row, differentiability_shape="differentiable")
    errors = gate._differentiability_errors(
        "probe", undeclared, {"accepts_tensor": False}, declared_shapes
    )
    assert any(
        "which the contract does not declare" in error for error in errors
    ), errors


def test_the_default_distribution_semantic_is_the_core_vocabulary(
    declared: dict, measured: dict
) -> None:
    from flagquantum.core.contracts import DistributionSemantics

    assert measured["default_semantics"] == "single_device_fast_path"
    assert tuple(measured["vocabulary"]) == DistributionSemantics.__args__
    assert declared["semantics_source"] == "flagquantum/core/contracts.py"
    literal = str(DistributionSemantics)
    for name in measured["vocabulary"]:
        assert f"'{name}'" in literal


def test_state_preparation_reads_its_tensor_as_classical_data() -> None:
    """The `classical_tensor_input` record is a behaviour, and the behaviour is asserted.

    An amplitude vector that requires a gradient used to raise a PyTorch warning about reading a
    `requires_grad` tensor as a scalar: the module detaches it on entry now, and the warning is
    gone along with any suggestion that a gradient could have arrived.
    """
    amplitudes = torch.tensor([0.6, 0.8], dtype=torch.complex128, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit = primitives.arbitrary_state(amplitudes)
    assert [str(item.message) for item in caught] == []
    assert [instruction.name for instruction in circuit._instructions] == ["ry", "rz"]
    assert amplitudes.grad_fn is None
    for instruction in circuit._instructions:
        assert not any(
            isinstance(param, torch.Tensor) for param in (instruction.params or ())
        )


def test_appending_a_state_reads_its_tensor_as_classical_data_too() -> None:
    import flagquantum as fq

    amplitudes = torch.tensor([0.6, 0.8], dtype=torch.complex128, requires_grad=True)
    circuit = fq.Circuit(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        primitives.append_arbitrary_state(circuit, amplitudes, [0])
    assert [str(item.message) for item in caught] == []
    assert [instruction.name for instruction in circuit._instructions] == ["ry", "rz"]
