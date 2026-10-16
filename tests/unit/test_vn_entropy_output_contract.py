"""Focused tests for the von Neumann entropy output contract gate.

The gate exists so that the properties of an entropy are re-measured rather than remembered.
A gate whose own measurements cannot fail is a gate that reports success for the wrong reason,
so the tests here break the measurement on purpose and assert that the gate notices: a
perturbed recorded number, an implementation that answers the whole state's entropy instead of
the selection's, an implementation that ignores the base, a mode whose shortcut disagrees, a
constructor that normalises what the caller wrote, and refusal sentences the package does not
say.
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

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "vn-entropy-output-contract.toml"


def _load_gate() -> Any:
    """Import ``tools/check_vn_entropy_output_contract.py`` by path.

    The checker is a script rather than an importable package member, so loading it by path is
    what keeps this test measuring the file the CI step runs.
    """

    spec = importlib.util.spec_from_file_location(
        "check_vn_entropy_output_contract",
        ROOT / "tools" / "check_vn_entropy_output_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class _FakeResult:
    """The smallest result the gate reads: one entropy per request."""

    def __init__(self, value: Any) -> None:
        self.vn_entropy = value


def test_the_live_tree_passes_every_measurement_the_contract_records() -> None:
    gate = _load_gate()

    assert gate.contract_errors(_contract()) == []


def test_a_perturbed_recorded_value_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["baseline"][3]["value"] += 1e-3

    errors = gate.contract_errors(contract)

    assert errors
    assert any("rather than the recorded" in error for error in errors)


def test_an_answer_that_ignores_the_selection_is_reported() -> None:
    """The whole state's entropy is the failure mode this measurement exists for.

    A number that is always zero would satisfy a "is it small" eyeball and always zero is also
    what the whole chain's entropy is for a pure state, so the gate has to separate them by the
    amplitude-side reference rather than by a bound.
    """

    gate = _load_gate()
    import flagquantum

    original_run = flagquantum.run

    def run(program: Any, **kwargs: Any) -> Any:
        result = original_run(program, **kwargs)
        if getattr(kwargs.get("outputs"), "kind", None) != "vn_entropy":
            return result
        return _FakeResult(result.vn_entropy * 0.0)

    flagquantum.run = run  # type: ignore[assignment]
    try:
        errors = gate.contract_errors(_contract())
    finally:
        flagquantum.run = original_run  # type: ignore[assignment]

    assert any("the amplitude-side reference is" in error for error in errors)


def test_an_answer_that_ignores_the_base_is_reported() -> None:
    """A named base is a division, so an implementation that skipped it must fail by name."""

    gate = _load_gate()
    import flagquantum

    original_run = flagquantum.run
    natural: dict[int, Any] = {}

    def run(program: Any, **kwargs: Any) -> Any:
        outputs = kwargs.get("outputs")
        log_base = getattr(outputs, "log_base", None)
        result = original_run(program, **kwargs)
        if getattr(outputs, "kind", None) != "vn_entropy":
            return result
        if log_base is None:
            natural[id(outputs)] = result.vn_entropy
            return result
        return _FakeResult(result.vn_entropy * 0.0 + 1.0)

    flagquantum.run = run  # type: ignore[assignment]
    try:
        errors = gate.contract_errors(_contract())
    finally:
        flagquantum.run = original_run  # type: ignore[assignment]

    assert any(
        "log base" in error and "rather than the recorded" in error for error in errors
    )


def test_a_route_that_disagrees_with_the_default_one_is_reported() -> None:
    """The reduction and the spectral shortcut are two routes to one number.

    The live modes agree, which is the property under test, so the disagreement is injected: a
    route that answered a scaled entropy has to be caught by name rather than by the numerical
    baseline, because the baseline is measured on the default route only.
    """

    gate = _load_gate()
    import flagquantum

    original_run = flagquantum.run

    def run(program: Any, **kwargs: Any) -> Any:
        result = original_run(program, **kwargs)
        options = kwargs.get("options")
        if getattr(options, "mode", None) == "mps":
            return _FakeResult(result.vn_entropy * 0.5)
        return result

    flagquantum.run = run  # type: ignore[assignment]
    try:
        errors = gate.contract_errors(_contract())
    finally:
        flagquantum.run = original_run  # type: ignore[assignment]

    assert any("answers 'mps' with" in error for error in errors)


def test_a_constructor_that_normalises_the_base_is_reported() -> None:
    """The base the caller wrote has to reach the request unchanged."""

    gate = _load_gate()
    import flagquantum

    original = flagquantum.vn_entropy
    baseline = _contract()["baseline"][0]

    def vn_entropy(qubits: Any, *, log_base: Any = None, name: Any = None) -> Any:
        return original(qubits, log_base=log_base, name=name).__class__(
            "vn_entropy",
            tuple(qubits) if not isinstance(qubits, int) else (qubits,),
            log_base=None,
        )

    flagquantum.vn_entropy = vn_entropy  # type: ignore[assignment]
    try:
        errors = gate.contract_errors(_contract())
    finally:
        flagquantum.vn_entropy = original  # type: ignore[assignment]

    assert baseline["value"] == 0.6931471805599454
    assert any("does not keep the base" in error for error in errors)


def test_a_refusal_sentence_the_package_does_not_say_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["refusal"][0]["message_phrase"] = "a sentence the package does not say"

    errors = gate.contract_errors(contract)

    assert any("does not say" in error for error in errors)


def test_every_recorded_refusal_has_a_trigger_the_gate_can_run() -> None:
    """A recorded sentence with no trigger would be prose in a contract's clothing."""

    gate = _load_gate()
    contract = _contract()
    contract["refusal"] = [*contract["refusal"], {"name": "a refusal with no trigger"}]

    errors = gate.contract_errors(contract)

    assert any("cannot trigger" in error for error in errors)


def test_a_trigger_the_contract_does_not_record_is_reported() -> None:
    """The reverse direction matters too: an unrecorded sentence is an undocumented refusal."""

    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["refusal"] = [
        row for row in contract["refusal"] if row["name"] != "a boolean is not a qubit"
    ]

    errors = gate.contract_errors(contract)

    assert any("does not record" in error for error in errors)


def test_a_contract_that_names_no_modes_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["request"]["modes"] = []

    errors = gate.contract_errors(contract)

    assert any("names no execution modes" in error for error in errors)


def test_a_contract_that_names_no_bases_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["log_base"] = []

    errors = gate.contract_errors(contract)

    assert any("records no bases" in error for error in errors)


def test_a_constructor_without_the_base_keyword_is_reported() -> None:
    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["request"]["base_parameter"] = "some_other_name"

    errors = gate.contract_errors(contract)

    assert any("base_parameter" in error for error in errors)


def test_a_probed_size_the_gate_does_not_measure_is_reported() -> None:
    """A contract that recorded a size outside its own probes would describe a different run."""

    gate = _load_gate()
    contract = copy.deepcopy(_contract())
    contract["bound"]["probed_qubits"] = 7

    errors = gate.contract_errors(contract)

    assert any("the gate measures" in error for error in errors)


def test_a_user_asks_for_an_entropy_of_a_named_subsystem() -> None:
    """The shortest supported path, written the way a user writes it.

    This is the scenario the contract's numbers describe: build a program, ask for the entropy
    of one qubit by name, and read it off the result. Keeping the scenario in the same file as
    the tamper tests is what stops the contract from drifting away from a workflow nobody runs.
    """

    import flagquantum as fq

    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    result = fq.run(circuit, outputs=fq.vn_entropy([0]))
    assert float(result.vn_entropy[0]) == pytest.approx(0.6931471805599454, abs=1e-12)

    # A named request is read back by name, which is what makes it addressable when more
    # than one of them is asked for in the same run.
    named = fq.run(
        circuit,
        outputs=[
            fq.vn_entropy(0, name="nats"),
            fq.vn_entropy([0], log_base=2, name="bits"),
        ],
    )
    assert float(named.measurement("nats").value[0]) == pytest.approx(
        0.6931471805599454, abs=1e-12
    )
    assert float(named.measurement("bits").value[0]) == pytest.approx(1.0, abs=1e-12)


def test_a_user_sees_which_request_the_package_will_not_answer() -> None:
    """Every refusal a user can meet, asserted on the sentence rather than on the class alone.

    A refusal that names the reason is what makes the failure actionable; the contract's rows
    quote these sentences, so a reworded message would otherwise pass unnoticed.
    """

    import flagquantum as fq

    bell = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    with pytest.raises(ValueError, match="requires at least one qubit"):
        fq.vn_entropy([])
    with pytest.raises(ValueError, match="must not be 1"):
        fq.vn_entropy([0], log_base=1)
    with pytest.raises(ValueError, match="must be positive"):
        fq.vn_entropy([0], log_base=-2)
    with pytest.raises(TypeError, match="must be a number"):
        fq.vn_entropy([0], log_base="2")
    with pytest.raises(TypeError, match="does not accept a log_base"):
        fq.OutputRequest("probabilities", (0,), log_base=2)
    with pytest.raises(ValueError, match="shots requires"):
        fq.run(bell, outputs=fq.vn_entropy([0]), options=fq.ExecutionOptions(shots=100))
    with pytest.raises(ValueError, match="outside a 2-qubit circuit"):
        fq.run(bell, outputs=fq.vn_entropy([5]))
