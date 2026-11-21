"""Native instance providers retain basic-gate identity and execution semantics."""

import importlib
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms import basic_gate_ui_export as native
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    MeasurementCall,
    ZeroCostCall,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_ui_export import source_metadata
from tests.unit.test_development_tools import REFERENCE
from tools.development.component_ui_export import export_component_ui

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXAMPLES = {
    example.name: example.make()
    for _, _, examples in native.DOCUMENTS
    for example in examples
}


@pytest.fixture
def originals(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    monkeypatch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
    source = importlib.import_module("qualtran_dev_tools.notebook_specs")
    titles = {title for title, _, _ in native.DOCUMENTS}
    specs = [spec for spec in source.NB_BY_SECTION[0][1] if spec.title in titles]
    assert len(specs) == 9
    assert sum(len(spec.bloq_specs) for spec in specs) == 14
    return {
        example.name: example.make()
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_all_original_instances_parameters_ports_and_models(
    originals: dict[str, Any],
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    metadata = source_metadata()
    assert originals.keys() == EXAMPLES.keys() and len(EXAMPLES) == 16
    for identifier, call in EXAMPLES.items():
        source = originals[identifier]
        assert native.name(call) == type(source).__name__
        assert native.attributes(call) == metadata["bloq_attrs"](source)
        assert native.filename(call) == metadata["bloq_filename"](source)
        ports = tuple(native.signature(call))
        assert [port.name for port in ports] == [port.name for port in source.signature]
        for port, original in zip(ports, source.signature, strict=True):
            assert port.bitsize == original.bitsize
            assert port.shape == original.shape
            assert port.side.value == original.side.value
        assert (
            query_qec_gate_cost(call).asdict()
            == get_cost_value(source, QECGatesCost()).asdict()
        )


@pytest.mark.parametrize(
    "identifier",
    tuple(
        name
        for name, call in EXAMPLES.items()
        if not isinstance(call, MeasurementCall) and name != "identity_symb"
    ),
)
def test_concrete_append_full_complex_operator_and_state(
    originals: dict[str, Any],
    identifier: str,
) -> None:
    call = EXAMPLES[identifier]
    op = native.operation(call)
    actual = get_unitary(Circuit(op.n_qubits, dtype=torch.complex128).append(op))
    expected = torch.as_tensor(
        originals[identifier].tensor_contract(), dtype=actual.dtype
    )
    if isinstance(call, ZeroCostCall) and call.opcode == "global_phase":
        expected = expected * torch.eye(2, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    state = torch.tensor(
        [complex(i + 1, (5 * i) % 11 - 3) for i in range(2**op.n_qubits)],
        dtype=torch.complex128,
    )
    state /= state.norm()
    observed = Circuit(op.n_qubits, inputs=state, dtype=state.dtype).append(op).state()
    torch.testing.assert_close(observed.flatten(), expected @ state, rtol=0, atol=2e-12)


@pytest.mark.parametrize("identifier", ("meas_z", "meas_x"))
def test_all_measurement_branches_and_consumed_wire_zero(
    originals: dict[str, Any],
    identifier: str,
) -> None:
    call = EXAMPLES[identifier]
    source = originals[identifier]
    from qualtran.bloqs.basic_gates import (
        MinusEffect,
        OneEffect,
        PlusEffect,
        ZeroEffect,
    )

    with pytest.raises(ValueError, match="discard information"):
        source.tensor_contract()
    # Reuse the source basis effects to retain each unnormalized branch phase.
    effects = (
        (PlusEffect, MinusEffect) if identifier == "meas_x" else (ZeroEffect, OneEffect)
    )
    instrument = torch.stack(
        tuple(
            torch.as_tensor(effect().tensor_contract(), dtype=torch.complex128)
            for effect in effects
        )
    )
    state = torch.tensor([complex(2, 3), complex(-5, 1)], dtype=torch.complex128)
    state /= state.norm()
    probabilities = []
    for outcome in (0, 1):
        actual = forced_branch(state, native.operation(call).instructions, (outcome,))
        expected = torch.stack((instrument[outcome] @ state, state.new_zeros(())))
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
        assert actual[1] == 0
        probabilities.append(actual.abs().square().sum())
    torch.testing.assert_close(
        sum(probabilities), torch.tensor(1.0, dtype=torch.float64)
    )


def test_native_export_all_fresh_scores_and_distinct_shared_class_docs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    native.export_example(tmp_path)
    report = capsys.readouterr().out
    assert "chadamard measured primitives {'ch': 1}" in report
    assert "chadamard model categories {'clifford': 4, 'rotation': 2}" in report
    assert "identity_n measured primitives {'i': 4}" in report
    navigation = json.loads((tmp_path / "navigation.json").read_text())["Basic Gates"]
    assert len(navigation) == 9
    assert sum(map(len, navigation.values())) == 14
    assert (
        sum(len(doc["examples"]) for docs in navigation.values() for doc in docs) == 16
    )
    assert len(list(tmp_path.rglob("docs.txt"))) == 14
    assert len(list(tmp_path.rglob("*.json"))) == 17
    for call in EXAMPLES.values():
        path = tmp_path / native.name(call) / native.filename(call)
        record = json.loads(path.read_text())
        assert record["msd"] == native.score(call)
        assert record["msd"] is not None and record["callees"] == []
        doc = path.with_name("docs.txt").read_text()
        assert doc.startswith("# " + native.name(call) + "\n")
    # PhaseCall and CliffordCall each own several distinct native operations.
    assert not (tmp_path / "PhaseCall").exists()
    assert not (tmp_path / "CliffordCall").exists()
    assert Counter(native.name(call) for call in EXAMPLES.values())["Identity"] == 3
    first = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    native.export_example(tmp_path)
    assert first == {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }


def test_symbolic_identity_and_invalid_instance_boundaries() -> None:
    with pytest.raises(ValueError, match="concrete execution width"):
        native.operation(EXAMPLES["identity_symb"])
    with pytest.raises(TypeError, match="registered native gate"):
        native.name(object())


def test_explicit_documentation_failure_propagates(tmp_path: Path) -> None:
    def fail(_: object) -> str:
        raise RuntimeError("domain documentation unavailable")

    with pytest.raises(RuntimeError, match="domain documentation unavailable"):
        export_component_ui(
            tmp_path,
            native.sections(),
            {},
            name=native.name,
            attributes=native.attributes,
            filename=native.filename,
            score=native.score,
            documentation_name=native.documentation_name,
            documentation_text=fail,
        )
