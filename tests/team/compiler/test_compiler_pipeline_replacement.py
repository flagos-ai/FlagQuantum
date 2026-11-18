from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.compiler.pipeline as compiler_pipeline
import flagquantum.runtime.planner as runtime_planner
from flagquantum.core.ir import CircuitIR, ensure_circuit_ir

pytestmark = pytest.mark.unit


def test_pipeline_facade_keeps_registry_replacement_in_both_compile_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[bool] = []
    original_registry = compiler_pipeline.default_pass_registry

    def record_phase(
        program: CircuitIR, analyses: Mapping[str, object] | None = None
    ) -> CircuitIR:
        phases.append("routing" in program.metadata)
        return compiler_pipeline.remove_identity_gates(program, analyses)

    def recording_registry() -> compiler_pipeline.PassRegistry:
        registry = original_registry()
        return compiler_pipeline.PassRegistry(
            dict(registry.passes) | {"remove_identity_gates": record_phase},
            dict(registry.specs)
            | {
                "remove_identity_gates": replace(
                    registry.spec("remove_identity_gates"), function=record_phase
                )
            },
            registry.analyses,
        )

    monkeypatch.setattr(compiler_pipeline, "default_pass_registry", recording_registry)
    circuit = fq.Circuit(3).h(0).cx(0, 2)
    compiler_pipeline.optimize(circuit)
    assert phases and not any(phases)

    phases.clear()
    compiled = compiler_pipeline.compile(
        circuit, coupling_map=compiler_pipeline.CouplingMap.line(3)
    )
    assert False in phases and True in phases
    assert compiled.metadata["routing"]["mapping_restored"]


def test_runtime_accepts_a_replacement_compiler_without_consumer_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def replacement_compile(program: Any, **options: Any) -> CircuitIR:
        calls.append(options)
        return ensure_circuit_ir(program)

    monkeypatch.setattr(runtime_planner, "compile_program", replacement_compile)

    options = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        precision="complex128",
        allow_backend_fallback=False,
    )
    plan = fq.plan(fq.Circuit(2).h(0).cx(0, 1), options=options)
    result = fq.run(plan)

    assert len(calls) == 1
    assert calls[0]["optimize"] is True
    assert calls[0]["coupling_map"] is None
    assert result.state is not None
    torch.testing.assert_close(
        result.state,
        torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
