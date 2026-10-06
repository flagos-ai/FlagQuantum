"""Focused tests for the density-matrix output contract gate.

The gate exists so that three properties of a reduced matrix are re-measured rather than
remembered. A gate whose own measurements cannot fail is a gate that reports success for the
wrong reason, so every test here fails the measurement on purpose and asserts that the gate
notices: a perturbed recorded number, an implementation that answers in ascending order, a
rescaled matrix, and a refusal sentence the package does not say.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - the 3.10 interpreter
    import tomli as tomllib

from flagquantum.simulation import density_matrix as density_matrix_module

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "density-matrix-output-contract.toml"


def _load_gate() -> Any:
    """Import ``tools/check_density_matrix_output_contract.py`` by path.

    The checker is a script rather than an importable package member, so loading it by path is
    what keeps this test measuring the file the CI step runs.
    """

    spec = importlib.util.spec_from_file_location(
        "check_density_matrix_output_contract",
        ROOT / "tools" / "check_density_matrix_output_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_the_live_tree_passes_every_measurement_the_contract_records() -> None:
    gate = _load_gate()

    assert gate.contract_errors(_contract()) == []


def test_a_perturbed_recorded_number_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["baseline"][2]["real"][0][1] += 1e-3

    errors = gate.contract_errors(contract)

    assert errors
    assert any("differs from the recorded matrix" in error for error in errors)


def test_an_ascending_order_answer_is_refused_as_a_wrong_matrix() -> None:
    """A reduction that ignored the caller's order must fail, not pass as a second reading."""

    gate = _load_gate()
    original = density_matrix_module.reduced_density_matrix

    def ascending(rho: torch.Tensor, qubits: Any = None) -> torch.Tensor:
        if qubits is None:
            return original(rho, qubits)
        named = (qubits,) if isinstance(qubits, int) else tuple(qubits)
        return original(rho, tuple(sorted(named)))

    density_matrix_module.reduced_density_matrix = ascending
    try:
        errors = gate.contract_errors(_contract())
    finally:
        density_matrix_module.reduced_density_matrix = original

    assert any("basis permutation" in error for error in errors)


def test_a_rescaled_matrix_fails_both_the_trace_and_the_agreement_measurement() -> None:
    gate = _load_gate()
    original = density_matrix_module.reduced_density_matrix

    density_matrix_module.reduced_density_matrix = (
        lambda rho, qubits=None: 0.5 * original(rho, qubits)
    )
    try:
        errors = gate.contract_errors(_contract())
    finally:
        density_matrix_module.reduced_density_matrix = original

    assert any("has trace" in error for error in errors)
    assert any("the observable path reports" in error for error in errors)


def test_a_refusal_sentence_the_package_does_not_say_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["refusal"][0]["message_fragment"] = "a sentence the package does not say"

    errors = gate.contract_errors(contract)

    assert any("does not say" in error for error in errors)


def test_every_recorded_refusal_has_a_trigger_the_gate_can_run() -> None:
    """A recorded sentence with no trigger would be prose in a contract's clothing."""

    gate = _load_gate()
    contract = _contract()
    contract["refusal"] = [*contract["refusal"], {"name": "a refusal with no trigger"}]

    errors = gate.contract_errors(contract)

    assert any("cannot trigger" in error for error in errors)


def test_the_bound_is_refused_at_the_recorded_width_rather_than_inside_it() -> None:
    """A contract that probed inside the bound would report a bound it never tested."""

    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["bound"]["probed_qubits"] = contract["bound"]["max_statevector_qubits"]

    errors = gate.contract_errors(contract)

    assert any("inside the bound" in error for error in errors)


def test_the_trace_measurement_refuses_a_pure_probe() -> None:
    """A pure probe cannot tell a trace from a marginal, so the gate has to say so."""

    gate = _load_gate()
    contract = _contract()
    original = density_matrix_module.reduced_density_matrix

    def pure_single_qubit_answer(rho: torch.Tensor, qubits: Any = None) -> torch.Tensor:
        # The probe is the gate's two-qubit program reduced to one qubit, so answering that one
        # selection with a pure matrix is exactly the situation the guard exists for.
        named = (qubits,) if isinstance(qubits, int) else tuple(qubits or ())
        if rho.shape[-1] == 4 and named == (0,):
            pure = torch.zeros((rho.shape[0], 2, 2), dtype=rho.dtype)
            pure[:, 0, 0] = 1.0
            return pure
        return original(rho, qubits)

    density_matrix_module.reduced_density_matrix = pure_single_qubit_answer
    try:
        errors = gate.contract_errors(contract)
    finally:
        density_matrix_module.reduced_density_matrix = original

    assert any("cannot tell a trace from a marginal" in error for error in errors)


def test_a_route_that_disagrees_with_the_default_one_is_reported() -> None:
    """The reduction is arithmetic on amplitudes, so a mode must not answer a different matrix.

    The live modes agree, which is the property under test, so the disagreement is injected: a
    route that answered a scaled matrix has to be caught by name rather than by the numerical
    baseline, because the baseline is measured on the default route only.
    """

    gate = _load_gate()
    import flagquantum

    original_run = flagquantum.run

    class _Disagreeing:
        def __init__(self, value: torch.Tensor) -> None:
            self.density_matrix = value

    def run(program: Any, **kwargs: Any) -> Any:
        result = original_run(program, **kwargs)
        options = kwargs.get("options")
        if getattr(options, "mode", None) == "mps":
            return _Disagreeing(0.5 * result.density_matrix)
        return result

    flagquantum.run = run  # type: ignore[assignment]
    try:
        errors = gate.contract_errors(_contract())
    finally:
        flagquantum.run = original_run  # type: ignore[assignment]

    assert any("differs from the default route" in error for error in errors)


def test_a_contract_that_names_no_modes_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["request"]["modes"] = []

    errors = gate.contract_errors(contract)

    assert any("names no execution modes" in error for error in errors)


def test_a_constructor_without_the_selection_keyword_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["request"]["selection_parameter"] = "some_other_name"

    errors = gate.contract_errors(contract)

    assert any("selection parameter" in error for error in errors)


def test_the_measure_switch_reports_the_same_matrices_the_contract_records() -> None:
    """``--measure`` is what a contributor runs to refresh the baseline, so it must agree."""

    gate = _load_gate()
    measured = gate.measure()
    contract = _contract()

    assert [row["name"] for row in measured["baselines"]] == [
        row["name"] for row in contract["baseline"]
    ]
    for recorded, live in zip(contract["baseline"], measured["baselines"], strict=True):
        assert live["qubits"] == recorded["qubits"]
        recorded_matrix = gate._matrix(recorded)
        live_matrix = torch.tensor(
            live["real"], dtype=torch.float64
        ) + 1j * torch.tensor(live["imag"], dtype=torch.float64)
        torch.testing.assert_close(live_matrix, recorded_matrix)
