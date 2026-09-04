from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.core.ir import IRSerializationError, IRValidationError
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.errors import CapabilityError
from flagquantum.utils.qasm_exporter import export_to_qasm_str
from flagquantum.utils.qcis_exporter import export_to_qcis_str

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/fixtures/internal_ir/circuit_ir_v1/manifest.json"
METADATA = ROOT / "tests/fixtures/internal_ir/metadata_inventory.json"
EVIDENCE = ROOT / "tests/fixtures/internal_ir/phase0_evidence.json"
PERFORMANCE = ROOT / "tests/fixtures/internal_ir/phase0_performance_baseline.json"
BUDGET = ROOT / "tests/fixtures/internal_ir/phase1_performance_budget_candidate.json"


def _corpus() -> dict[str, object]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _metadata_inventory() -> dict[str, object]:
    return json.loads(METADATA.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_oracle_cases() -> list[tuple[dict[str, object], str]]:
    return [
        (fixture, oracle)
        for fixture in _corpus()["fixtures"]
        for oracle in fixture["oracles"]
        if oracle != "serialization_exact"
    ]


SEMANTIC_ORACLE_CASES = _semantic_oracle_cases()


def _metadata_base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _consumed_metadata_keys() -> set[str]:
    keys: set[str] = set()
    for path in (ROOT / "flagquantum").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "pop", "setdefault"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "metadata" in _metadata_base_name(node.func.value).lower()
            ):
                keys.add(node.args[0].value)
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and "metadata" in _metadata_base_name(node.value).lower()
            ):
                keys.add(node.slice.value)
    return keys


def test_corpus_covers_every_registered_canonical_opcode() -> None:
    manifest = _corpus()
    payloads = [fixture["payload"] for fixture in manifest["fixtures"]]
    covered = {
        instruction["opcode"]
        for payload in payloads
        for instruction in payload["instructions"]
        if instruction["opcode"] in OPERATOR_SCHEMAS
    }

    assert manifest["registry_snapshot"]["canonical_opcode_count"] == len(
        OPERATOR_SCHEMAS
    )
    assert covered == set(OPERATOR_SCHEMAS)


@pytest.mark.parametrize(
    "fixture",
    _corpus()["fixtures"],
    ids=lambda fixture: fixture["id"],
)
def test_phase0_fixture_preserves_legacy_serialization(fixture) -> None:
    payload = fixture["payload"]
    ir = fq.CircuitIR.from_dict(payload)

    assert ir.to_dict() == payload
    assert fq.CircuitIR.from_json(ir.to_json()).to_dict() == payload
    assert fq.CircuitIR.from_json(ir.to_json()).content_hash == ir.content_hash


def test_manifest_only_declares_machine_implemented_oracles() -> None:
    declared = {
        oracle for fixture in _corpus()["fixtures"] for oracle in fixture["oracles"]
    }

    assert declared == {
        "emitter_parse_or_golden",
        "backend_selection",
        "expectation",
        "gradient",
        "measurement_exact_seeded",
        "routing_legality",
        "serialization_exact",
        "statevector",
        "unsupported_diagnostic",
    }


@pytest.mark.parametrize(
    ("fixture", "oracle"),
    SEMANTIC_ORACLE_CASES,
    ids=[f"{fixture['id']}::{oracle}" for fixture, oracle in SEMANTIC_ORACLE_CASES],
)
def test_manifest_semantic_oracle_executes(fixture, oracle) -> None:
    ir = fq.CircuitIR.from_dict(fixture["payload"])
    oracle_data = fixture.get("oracle_data", {})

    if oracle == "statevector":
        first = fq.run(ir).statevector()
        second = fq.run(ir).statevector()
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            torch.linalg.vector_norm(first, dim=-1),
            torch.ones(first.shape[:-1], dtype=first.real.dtype),
        )
        return

    if oracle == "measurement_exact_seeded":
        first = fq.run(ir)
        second = fq.run(ir)
        assert first.samples is not None
        assert second.samples is not None
        assert torch.equal(first.samples, second.samples)
        assert first.measurement(0).wires == ir.measurements[0].wires
        assert first.samples.shape[-1] == len(ir.measurements[0].wires)
        # This asymmetric wire request measures a Bell state. Equal columns
        # prove the declared (1, 0) order was preserved rather than sorted.
        assert torch.equal(first.samples[..., 0], first.samples[..., 1])
        return

    if oracle == "expectation":
        actual = fq.run(ir).expectation()
        expected = torch.tensor(
            oracle_data["expected"],
            dtype=torch.float64 if ir.dtype == "complex128" else torch.float32,
        )
        torch.testing.assert_close(
            actual,
            expected,
            atol=float(oracle_data["atol"]),
            rtol=0.0,
        )
        return

    if oracle == "gradient":
        bindings = oracle_data["bindings"]
        theta = torch.tensor(bindings["theta"], dtype=torch.float64, requires_grad=True)
        phi = torch.tensor(bindings["phi"], dtype=torch.float64, requires_grad=True)
        circuit = fq.Circuit.from_ir(ir, dtype=torch.complex128).bind_parameters(
            {"theta": theta, "phi": phi}
        )
        loss = circuit.expectation_z(tuple(oracle_data["expectation_wires"])).sum()
        loss.backward()
        expected_loss = torch.cos(theta.detach()) * (1.0 + torch.cos(phi.detach()))
        expected_theta = -torch.sin(theta.detach()) * (1.0 + torch.cos(phi.detach()))
        expected_phi = -torch.cos(theta.detach()) * torch.sin(phi.detach())
        torch.testing.assert_close(loss.detach(), expected_loss, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(theta.grad, expected_theta, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(phi.grad, expected_phi, atol=1e-12, rtol=0.0)
        return

    if oracle == "routing_legality":
        assert oracle_data["coupling"] == "line"
        coupling = CouplingMap.line(ir.n_wires)
        routed = route_to_topology(
            ir,
            coupling,
            strategy=str(oracle_data["routing_strategy"]),
        )
        assert all(
            len(instruction.wires) != 2
            or instruction.metadata.get("is_channel")
            or coupling.has_edge(*instruction.wires)
            for instruction in routed.instructions
        )
        assert routed.metadata["routing"]["mapping_restored"] is True
        torch.testing.assert_close(
            fq.run(routed).statevector(), fq.run(ir).statevector(), atol=1e-6, rtol=1e-6
        )
        return

    if oracle == "emitter_parse_or_golden":
        qasm = export_to_qasm_str(ir, version=2.0)
        qcis = export_to_qcis_str(ir)
        assert qasm.startswith("OPENQASM 2.0;")
        assert hashlib.sha256(qasm.encode()).hexdigest() == oracle_data["qasm2_sha256"]
        assert hashlib.sha256(qcis.encode()).hexdigest() == oracle_data["qcis_sha256"]
        return

    if oracle == "backend_selection":
        plan = fq.plan(ir)
        payload = plan.to_dict()
        assert payload["requested_options"]["backend"] is None
        assert payload["resolved_options"]["backend"] == "pytorch"
        assert payload["resolved_options"]["mode"] == "auto"
        assert payload["decision"]["backend"] == "pytorch"
        assert payload["decision"]["mode"] == "statevector"
        assert payload["decision"]["allow_backend_fallback"] is False
        result = fq.run(plan)
        assert result.plan is plan
        assert result.runtime["mode"] == "statevector"
        return

    if oracle == "unsupported_diagnostic":
        with pytest.raises(
            CapabilityError, match="does not execute dynamic trajectories"
        ):
            fq.run(ir)
        return

    raise AssertionError(f"unhandled Phase 0 oracle {oracle!r}")


def test_parameter_fixture_preserves_symbolic_types() -> None:
    fixture = next(
        item
        for item in _corpus()["fixtures"]
        if item["id"] == "static.parameter_and_expression.v1"
    )
    ir = fq.CircuitIR.from_dict(fixture["payload"])

    assert isinstance(ir.instructions[0].params["theta"], fq.Parameter)
    assert isinstance(ir.instructions[1].params["theta"], fq.ParameterExpression)
    assert ir.instructions[0].params["theta"].name == "theta"
    assert fq.Circuit.from_ir(ir).parameter_names == ("phi", "theta")

    entangled = next(
        item
        for item in _corpus()["fixtures"]
        if item["id"] == "static.two_parameter_entangled_gradient.v1"
    )
    entangled_ir = fq.CircuitIR.from_dict(entangled["payload"])
    assert fq.Circuit.from_ir(entangled_ir).parameter_names == ("phi", "theta")


def test_metadata_inventory_fails_on_unclassified_consumed_key_drift() -> None:
    inventory = _metadata_inventory()

    assert inventory["consumed_key_count"] == len(inventory["consumed_keys"])
    assert set(inventory["consumed_keys"]) == _consumed_metadata_keys()
    importer_relevant = {
        key
        for keys in inventory["circuit_ir_importer_relevant"].values()
        for key in keys
    } | set(inventory["produced_or_indirect_keys_requiring_manual_review"])
    assert importer_relevant <= set(inventory["typed_destinations"])


def test_phase0_evidence_pins_corpus_inventory_and_public_contracts() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    corpus = _corpus()

    assert evidence["status"] == "in_progress"
    assert evidence["approval"]["phase1_authorized"] is False
    assert evidence["corpus"]["sha256"] == _sha256(CORPUS)
    assert evidence["corpus"]["positive_fixture_count"] == len(corpus["fixtures"])
    assert evidence["corpus"]["negative_fixture_count"] == len(
        corpus["negative_fixtures"]
    )
    assert evidence["corpus"]["semantic_oracle_dispatch"] == "machine_enforced"
    assert evidence["metadata_inventory"]["sha256"] == _sha256(METADATA)
    assert evidence["performance_baseline"]["sha256"] == _sha256(PERFORMANCE)
    assert evidence["phase1_budget_candidate"]["sha256"] == _sha256(BUDGET)
    for relative_path, expected_hash in evidence["public_api_contract_hashes"].items():
        assert _sha256(ROOT / relative_path) == expected_hash
    assert evidence["known_blockers"] == [
        "IR0-EXIT-003: API/compiler/runtime/training owner review is not recorded",
        "IR0-EXIT-004: Phase 0 exit and Phase 1 implementation authorization are not approved",
    ]
    assert any(
        "complex128 precision is preserved" in finding
        for finding in evidence["closed_findings"]
    )


def test_phase1_budget_is_approved_and_derived_from_baseline() -> None:
    baseline = json.loads(PERFORMANCE.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    baseline_by_gates = {case["gate_count"]: case for case in baseline["cases"]}

    assert budget["status"] == "approved_internal_phase1_gate"
    assert budget["approval"]["approved"] is True
    assert budget["approval"]["approved_by"]
    assert "not a public SLA" in budget["approval"]["approved_scope"]
    for gate in budget["budgets"]:
        measured = baseline_by_gates[gate["gate_count"]]
        derived_latency = max(1.0, 1.75 * measured["p95_ms"]["legacy_plan"])
        derived_memory = max(
            1_048_576, 2 * measured["peak_host_memory_bytes"]["legacy_plan"]
        )
        assert gate["import_verify_p95_ms_max"] >= derived_latency
        assert gate["import_verify_p95_ms_max"] <= derived_latency + 1.5
        assert gate["peak_host_memory_bytes_max"] >= derived_memory
        assert gate["peak_host_memory_bytes_max"] <= derived_memory + 1_048_576


def test_phase0_gradient_oracle_preserves_tensor_identity_and_analytic_gradient() -> (
    None
):
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).rx(0, theta=theta)
    ir = circuit.to_ir()

    assert ir.instructions[0].params["theta"] is theta
    value = circuit.expectation_z(0).sum()
    value.backward()

    assert theta.grad is not None
    assert torch.allclose(value.detach(), torch.cos(theta.detach()), atol=1e-12)
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-12)


def test_phase0_seeded_measurement_oracle_is_exact_and_statistically_sound() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    ir = fq.CircuitIR(
        n_wires=2,
        instructions=circuit.to_ir().instructions,
        measurements=(
            fq.MeasurementNode(
                "sample",
                (0, 1),
                shots=4096,
                metadata={"seed": 41},
            ),
        ),
    )

    first = fq.run(ir).samples
    second = fq.run(ir).samples

    assert first is not None
    assert second is not None
    assert torch.equal(first, second)
    assert torch.equal(first[..., 0], first[..., 1])
    probability_one = first[..., 0].to(torch.float64).mean()
    assert 0.45 <= float(probability_one) <= 0.55


@pytest.mark.parametrize(
    "fixture",
    _corpus()["negative_fixtures"],
    ids=lambda fixture: fixture["id"],
)
def test_phase0_machine_readable_negative_corpus(fixture) -> None:
    error_types = {
        "IRSerializationError": IRSerializationError,
        "IRValidationError": IRValidationError,
        "ValueError": ValueError,
    }
    expected_error = error_types[fixture["expected_error"]]

    with pytest.raises(expected_error):
        ir = fq.CircuitIR.from_dict(fixture["payload"])
        if fixture.get("operation") == "plan_with_external_measurement":
            fq.plan(
                ir,
                measurements=(fq.MeasurementNode("sample", (0,), shots=8),),
            )


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"kind": "not.flagquantum", "version": "1.0"}, IRSerializationError),
        (
            {
                "kind": "flagquantum.circuit_ir",
                "version": "2.0",
                "n_wires": 1,
                "instructions": [],
            },
            IRValidationError,
        ),
        (
            {
                "kind": "flagquantum.circuit_ir",
                "version": "1.0",
                "n_wires": 1,
                "instructions": [],
                "unknown": True,
            },
            IRSerializationError,
        ),
    ],
)
def test_phase0_negative_serialization_contract(payload, error) -> None:
    with pytest.raises(error):
        fq.CircuitIR.from_dict(payload)


@pytest.mark.parametrize(
    "instruction",
    [
        fq.Instruction("x", (1,)),
        fq.Instruction("cx", (0, 1)),
    ],
)
def test_phase0_negative_wire_contract(instruction) -> None:
    with pytest.raises(IRValidationError, match="outside circuit range"):
        fq.CircuitIR(n_wires=1, instructions=(instruction,))
