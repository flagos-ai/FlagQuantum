import pytest

import flagquantum as fq
import flagquantum.compiler as compiler
import flagquantum.runtime.planner as fqxp
from flagquantum.runtime.audit import audit_distributed_scalability
from flagquantum.runtime.backend_registry import backend_execution_options

pytestmark = pytest.mark.unit


def test_ir_and_compiler_pure_logic_baseline():
    circuit = fq.Circuit(1)
    circuit.x(0).x(0).h(0)

    ir = circuit.to_ir()
    compiled = compiler.optimize(ir)

    assert len(ir) == 3
    assert len(compiled) == 1
    assert compiled.instructions[0].name == "h"


def test_runtime_planner_reports_local_fast_path_metadata():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    summary = fqxp.plan_runtime_selection(circuit, prefer_jax=False).summary()
    candidate = summary["recommended_candidate"]

    assert summary["world_size"] == 1
    assert candidate["distribution_semantics"] == "single_device_fast_path"
    assert candidate["scalability_claim_allowed"] is False
    assert candidate["release_gate_allowed"] is False
    assert candidate["available"] is True


def test_audit_single_device_metadata_is_non_scalability():
    payload = {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "world_size": 1,
        "backend": "pytorch",
        "mode": "statevector",
    }

    audit = audit_distributed_scalability(payload)

    assert audit.valid
    assert not audit.scalability_claim_allowed
    assert not audit.release_gate_allowed
    assert audit.errors == ()


def test_backend_policy_normalizes_local_runtime_metadata():
    options = backend_execution_options(
        mode="statevector", device="cpu", dtype="complex64"
    )

    assert options["backend"] == "pytorch"
    assert options["mode"] == "statevector"
    assert options["device"] == "cpu"
    assert options["complex_dtype"] is not None
    assert options["real_dtype"] is not None
