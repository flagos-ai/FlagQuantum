from __future__ import annotations

import importlib

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.diagnostics import Diagnostic, DiagnosticCode
from flagquantum._compiler.exporters.circuit_ir import (
    ExportStatus,
    export_circuit_ir,
    seal_circuit_ir_round_trip,
)
from flagquantum._compiler.import_models import ImportStatus
from flagquantum._compiler.ir.verifier import VerificationResult
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode

importer = importlib.import_module("flagquantum._compiler.importers.circuit_ir")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_success_cache():
    with importer._SUCCESS_CACHE_LOCK:
        importer._SUCCESS_CACHE.clear()
    yield
    with importer._SUCCESS_CACHE_LOCK:
        importer._SUCCESS_CACHE.clear()


def _rich_source() -> CircuitIR:
    return CircuitIR(
        2,
        (
            Instruction(
                "rx",
                (0,),
                {"theta": 0.25},
                metadata={"diagonal": False},
            ),
            Instruction(
                "custom",
                (1,),
                matrix=[[1.0, 0.0], [0.0, 1.0]],
                metadata={"source": "unit-test"},
            ),
        ),
        observables=(
            ObservableNode(
                "z",
                (0,),
                coefficient=torch.tensor(1.0),
                metadata={"name": "z0"},
            ),
        ),
        measurements=(
            MeasurementNode(
                "sample",
                (0,),
                shots=16,
                metadata={"seed": 7, "postselect": {"value": 0}},
            ),
        ),
        metadata={
            "source": "unit-test",
            "batch_size": 1,
            "runtime_config": {"mode": "exact"},
        },
    )


def _count_verifications(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = importer.verify_module

    def counted(module, registry):
        calls.append(1)
        return original(module, registry)

    monkeypatch.setattr(importer, "verify_module", counted)
    return calls


def test_repeated_success_reuses_only_the_verified_immutable_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_verifications(monkeypatch)
    source = _rich_source()

    first = importer.import_circuit_ir(source)
    second = importer.import_circuit_ir(source)

    assert first.status is ImportStatus.SUPPORTED_EXACT
    assert first.diagnostics == second.diagnostics == ()
    assert second.imported is first.imported
    assert second.imported.internal_program_identity == (
        first.imported.internal_program_identity
    )
    assert calls == [1]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda source: source.instructions[0].params.__setitem__("theta", 0.5),
            id="instruction-parameter",
        ),
        pytest.param(
            lambda source: source.instructions[0].metadata.__setitem__(
                "diagonal", True
            ),
            id="instruction-metadata",
        ),
        pytest.param(
            lambda source: source.instructions[1].matrix.__setitem__(
                slice(None), [[0.0, 1.0], [1.0, 0.0]]
            ),
            id="custom-matrix",
        ),
        pytest.param(
            lambda source: source.observables[0].coefficient.fill_(0.5),
            id="observable-coefficient-tensor",
        ),
        pytest.param(
            lambda source: source.observables[0].metadata.__setitem__("name", "z"),
            id="observable-metadata",
        ),
        pytest.param(
            lambda source: source.measurements[0]
            .metadata["postselect"]
            .__setitem__("value", 1),
            id="measurement-nested-metadata",
        ),
        pytest.param(
            lambda source: source.metadata["runtime_config"].__setitem__(
                "mode", "strict"
            ),
            id="circuit-nested-metadata",
        ),
    ],
)
def test_every_nested_source_layer_change_reimports_and_reverifies(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    calls = _count_verifications(monkeypatch)
    source = _rich_source()
    first = importer.import_circuit_ir(source)
    first_hash = first.imported.source.circuit_ir_content_hash

    mutate(source)
    second = importer.import_circuit_ir(source)

    assert first.ok and second.ok
    assert second.imported is not first.imported
    assert second.imported.source.circuit_ir_content_hash != first_hash
    assert calls == [1, 1]


def test_equal_trainable_tensor_replacement_misses_and_preserves_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_verifications(monkeypatch)
    first_tensor = torch.tensor(0.25, requires_grad=True)
    second_tensor = torch.tensor(0.25, requires_grad=True)
    source = CircuitIR(
        1,
        (Instruction("rx", (0,), {"theta": first_tensor}),),
    )

    first = importer.import_circuit_ir(source).imported
    source.instructions[0].params["theta"] = second_tensor
    second = importer.import_circuit_ir(source).imported
    slot = first.bindings.references[0].slot

    assert second is not first
    assert first.bindings[slot] is first_tensor
    assert second.bindings[slot] is second_tensor
    assert first.internal_program_identity == second.internal_program_identity
    assert calls == [1, 1]


def test_in_place_trainable_mutation_reimports_without_stealing_autograd_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_verifications(monkeypatch)
    parameter = torch.tensor(0.25, requires_grad=True)
    source = CircuitIR(1, (Instruction("rx", (0,), {"theta": parameter}),))
    first = importer.import_circuit_ir(source).imported

    with torch.no_grad():
        parameter.add_(0.5)
    second = importer.import_circuit_ir(source).imported
    slot = second.bindings.references[0].slot

    assert second is not first
    assert second.bindings[slot] is parameter
    assert second.bindings[slot].requires_grad is True
    assert first.internal_program_identity == second.internal_program_identity
    assert calls == [1, 1]


@pytest.mark.parametrize(
    ("source", "status"),
    [
        pytest.param(object(), ImportStatus.INVALID_INPUT, id="wrong-object-type"),
        pytest.param(
            CircuitIR(1, (Instruction("h", (0,), metadata={"mystery": True}),)),
            ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            id="instruction-diagnostic",
        ),
        pytest.param(
            CircuitIR(1, (Instruction("h", (0,)),), metadata={"mystery": True}),
            ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            id="circuit-diagnostic",
        ),
        pytest.param(
            CircuitIR(
                1,
                (Instruction("rx", (0,), {"theta": torch.tensor([0.2, 0.3])}),),
            ),
            ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            id="unrepresentable-parameter",
        ),
    ],
)
def test_input_and_diagnostic_failures_are_never_cached(source, status) -> None:
    first = importer.import_circuit_ir(source)
    second = importer.import_circuit_ir(source)

    assert first.status is second.status is status
    assert first.imported is second.imported is None
    assert first.diagnostics == second.diagnostics
    with importer._SUCCESS_CACHE_LOCK:
        assert id(source) not in importer._SUCCESS_CACHE


def test_verifier_failure_is_never_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    source = CircuitIR(1, (Instruction("h", (0,)),))
    failure = VerificationResult(
        (Diagnostic(DiagnosticCode.UNKNOWN_OPERATION, "forced verifier failure"),)
    )
    original = importer.verify_module
    monkeypatch.setattr(importer, "verify_module", lambda module, registry: failure)

    first = importer.import_circuit_ir(source)
    second = importer.import_circuit_ir(source)
    monkeypatch.setattr(importer, "verify_module", original)
    repaired = importer.import_circuit_ir(source)

    assert first.status is second.status is ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert first.diagnostics == second.diagnostics == failure.diagnostics
    assert repaired.status is ImportStatus.SUPPORTED_EXACT
    with importer._SUCCESS_CACHE_LOCK:
        assert importer._SUCCESS_CACHE[id(source)][2] is repaired.imported


def test_cached_success_preserves_round_trip_identity_and_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _rich_source()
    first = importer.import_circuit_ir(source).imported
    sealed = seal_circuit_ir_round_trip(source)
    exported = export_circuit_ir(sealed.artifact)

    assert sealed.artifact.imported is first
    assert exported.status is ExportStatus.EXPORTED_EXACT
    assert exported.circuit_ir.to_json() == source.to_json()
    assert exported.circuit_ir.content_hash == source.content_hash

    def forbidden(_: object) -> object:
        raise AssertionError("default path invoked private importer")

    monkeypatch.setattr(importer, "import_circuit_ir", forbidden)
    circuit = fq.Circuit(1).h(0)
    plan = fq.plan(circuit)
    result = fq.run(plan)

    assert result.plan is plan
    assert result.state is not None
