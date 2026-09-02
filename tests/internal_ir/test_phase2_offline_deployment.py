from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.offline_deployment import (
    OfflineCompilationStatus,
    OfflineStaticTarget,
    OfflineTextFormat,
    compile_offline_static,
)
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.pipeline_cache import BoundedPipelineCache, CacheDisposition
from flagquantum._compiler.testing.differential import lower_module_for_differential
from flagquantum.utils.qasm_exporter import export_to_qasm_str
from flagquantum.utils.qcis_exporter import export_to_qcis_str

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/fixtures/internal_ir/phase2_batch_f_offline_corpus.json"


def _corpus() -> dict[str, object]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _target(payload: dict[str, object]) -> OfflineStaticTarget:
    return OfflineStaticTarget(
        DirectedCouplingGraph(
            payload["n_qubits"],
            tuple(tuple(edge) for edge in payload["directed_edges"]),
        ),
        payload["calibration_identity"],
        tuple(payload["initial_layout"]) if "initial_layout" in payload else None,
    )


def _parse_qasm(text: str) -> fq.CircuitIR:
    declaration = re.search(r"(?:qreg q|qubit)\[(\d+)\](?: q)?;", text)
    assert declaration is not None
    instructions = []
    gate_pattern = re.compile(r"^(rx|ry|rz)\(([^)]+)\) q\[(\d+)\];$")
    cx_pattern = re.compile(r"^cx q\[(\d+)\], ?q\[(\d+)\];$")
    for line in text.splitlines():
        gate = gate_pattern.match(line)
        if gate:
            instructions.append(
                fq.Instruction(
                    gate.group(1),
                    (int(gate.group(3)),),
                    {"theta": float(gate.group(2))},
                )
            )
        cx = cx_pattern.match(line)
        if cx:
            instructions.append(
                fq.Instruction("cx", (int(cx.group(1)), int(cx.group(2))))
            )
    return fq.CircuitIR(
        int(declaration.group(1)), tuple(instructions), dtype="complex128"
    )


def _parse_qcis(text: str, n_qubits: int) -> fq.CircuitIR:
    lines = text.splitlines()
    instructions = []
    index = 0
    while index < len(lines):
        tokens = lines[index].split()
        if tokens[0] == "RZ":
            instructions.append(
                fq.Instruction("rz", (int(tokens[1][1:]),), {"theta": float(tokens[2])})
            )
            index += 1
            continue
        first, middle, last = (lines[index + offset].split() for offset in range(3))
        wire = int(first[1][1:])
        if first[0] == "Y2M" and middle[0] == "RZ" and last[0] == "Y2P":
            instructions.append(
                fq.Instruction("rx", (wire,), {"theta": float(middle[2])})
            )
        elif first[0] == "X2P" and middle[0] == "RZ" and last[0] == "X2M":
            instructions.append(
                fq.Instruction("ry", (wire,), {"theta": float(middle[2])})
            )
        else:
            assert first[0] == "Y2M" and middle[0] == "CZ" and last[0] == "Y2P"
            instructions.append(
                fq.Instruction("cx", (int(middle[1][1:]), int(middle[2][1:])))
            )
        index += 3
    return fq.CircuitIR(n_qubits, tuple(instructions), dtype="complex128")


def _assert_state_equal(actual: fq.CircuitIR, expected: fq.CircuitIR) -> None:
    actual_state = fq.run(actual).state.reshape(-1)
    expected_state = fq.run(expected).state.reshape(-1)
    pivot = int(torch.argmax(torch.abs(expected_state)).item())
    phase = actual_state[pivot] / expected_state[pivot]
    torch.testing.assert_close(actual_state, phase * expected_state, atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "fixture", _corpus()["positive_fixtures"], ids=lambda item: item["id"]
)
def test_offline_corpus_is_structural_parser_state_order_and_text_exact(
    fixture,
) -> None:
    source = fq.CircuitIR.from_dict(fixture["payload"])
    target = _target(fixture["target"])
    result = compile_offline_static(source, target)

    assert result.ok, result.diagnostics
    assert result.module is not None and result.execution is not None
    assert result.execution.identity is not None
    assert all(
        operation.name in {"quantum.rx", "quantum.ry", "quantum.rz", "quantum.cx"}
        for operation in result.module.body.blocks[0].operations
    )
    lowered = lower_module_for_differential(result.source_artifact, result.module)
    assert lowered.ok and lowered.circuit_ir is not None
    assert all(
        instruction.name != "cx" or target.coupling_graph.has_edge(*instruction.wires)
        for instruction in lowered.circuit_ir.instructions
    )
    _assert_state_equal(lowered.circuit_ir, source)

    qasm2 = result.emission(OfflineTextFormat.OPENQASM2)
    qasm3 = result.emission(OfflineTextFormat.OPENQASM3)
    qcis = result.emission(OfflineTextFormat.QCIS_V1)
    assert {
        "openqasm2": qasm2.content_hash,
        "openqasm3": qasm3.content_hash,
        "qcis_v1": qcis.content_hash,
    } == fixture["expected_hashes"]
    _assert_state_equal(_parse_qasm(qasm2.text), source)
    _assert_state_equal(_parse_qasm(qasm3.text), source)
    _assert_state_equal(_parse_qcis(qcis.text, source.n_wires), source)

    legacy_qasm = _parse_qasm(export_to_qasm_str(lowered.circuit_ir, version=2.0))
    legacy_qcis = _parse_qcis(export_to_qcis_str(lowered.circuit_ir), source.n_wires)
    _assert_state_equal(legacy_qasm, _parse_qasm(qasm2.text))
    _assert_state_equal(legacy_qcis, _parse_qcis(qcis.text, source.n_wires))


def test_pipeline_identity_hash_and_cache_are_deterministic() -> None:
    fixture = _corpus()["positive_fixtures"][1]
    source = fq.CircuitIR.from_dict(fixture["payload"])
    target = _target(fixture["target"])
    cache = BoundedPipelineCache(max_entries=2)

    first = compile_offline_static(source, target, cache=cache)
    second = compile_offline_static(source, target, cache=cache)

    assert first.ok and second.ok
    assert first.execution.cache_disposition is CacheDisposition.MISS
    assert second.execution.cache_disposition is CacheDisposition.HIT
    assert first.execution.identity == second.execution.identity
    assert first.emissions == second.emissions
    changed_target = replace(target, calibration_identity="d" * 64)
    changed = compile_offline_static(source, changed_target, cache=cache)
    assert changed.execution.identity.digest != first.execution.identity.digest


def test_gradient_and_expectation_survive_private_pipeline_without_text_emission() -> (
    None
):
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("ry", (0,), {"theta": theta}),
            fq.Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
    )
    target = OfflineStaticTarget(DirectedCouplingGraph(2, ((0, 1),)), "e" * 64)
    result = compile_offline_static(source, target, output_formats=())
    lowered = lower_module_for_differential(result.source_artifact, result.module)

    assert result.ok and result.execution.cache_disposition is CacheDisposition.BYPASS
    assert lowered.ok and lowered.circuit_ir is not None
    original_value = (
        fq.Circuit.from_ir(source, dtype=torch.complex128).expectation_z((0, 1)).sum()
    )
    lowered_value = (
        fq.Circuit.from_ir(lowered.circuit_ir, dtype=torch.complex128)
        .expectation_z((0, 1))
        .sum()
    )
    torch.testing.assert_close(lowered_value, original_value, atol=1e-12, rtol=0)
    lowered_value.backward()
    assert theta.grad is not None
    torch.testing.assert_close(
        theta.grad, -2 * torch.sin(theta.detach()), atol=1e-12, rtol=0
    )


def _negative_source(feature: str) -> tuple[fq.CircuitIR, OfflineStaticTarget]:
    target = OfflineStaticTarget(DirectedCouplingGraph(1, ()), "f" * 64)
    base = fq.CircuitIR(1, (fq.Instruction("rx", (0,), {"theta": 0.2}),))
    if feature == "measurement":
        return (
            replace(base, measurements=(fq.MeasurementNode("sample", (0,), 8),)),
            target,
        )
    if feature == "observable":
        return replace(base, observables=(fq.ObservableNode("z", (0,)),)), target
    if feature == "dynamic_circuit":
        return replace(base, metadata={"dynamic_circuit": True}), target
    if feature == "unbound_parameter":
        return (
            fq.CircuitIR(
                1, (fq.Instruction("rx", (0,), {"theta": fq.Parameter("t")}),)
            ),
            target,
        )
    if feature == "noise_channel":
        return (
            fq.CircuitIR(
                1, (fq.Instruction("bit_flip", (0,), metadata={"is_channel": True}),)
            ),
            target,
        )
    if feature == "qubit_count_mismatch":
        return base, OfflineStaticTarget(DirectedCouplingGraph(2, ((0, 1),)), "f" * 64)
    raise AssertionError(feature)


@pytest.mark.parametrize(
    "fixture", _corpus()["negative_fixtures"], ids=lambda item: item["id"]
)
def test_unsupported_deployment_features_fail_closed(fixture) -> None:
    source, target = _negative_source(fixture["feature"])
    result = compile_offline_static(source, target)

    assert result.status is OfflineCompilationStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert not result.ok and result.diagnostics
    assert any(
        fixture["expected_diagnostic_contains"] in diagnostic.message
        for diagnostic in result.diagnostics
    )
    assert all(emission.text is None for _, emission in result.emissions)


def test_corpus_and_pipeline_are_provider_free_private_and_default_neutral() -> None:
    corpus = _corpus()
    serialized = json.dumps(corpus, sort_keys=True).lower()

    assert corpus["contains_provider_credentials_or_backend_ids"] is False
    assert "https://" not in serialized and 'token"' not in serialized
    assert not hasattr(fq, "compile_offline_static")
    source = fq.CircuitIR(1, (fq.Instruction("x", (0,)),))
    torch.testing.assert_close(
        fq.run(source).state.reshape(-1), torch.tensor([0j, 1 + 0j])
    )
