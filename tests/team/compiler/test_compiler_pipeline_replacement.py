from __future__ import annotations

from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.planner as runtime_planner
from flagquantum.core.ir import CircuitIR, ensure_circuit_ir


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
