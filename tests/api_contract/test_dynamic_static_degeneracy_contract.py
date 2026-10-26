"""Stable remote dispatch for statically degenerate DynamicCircuit programs."""

from __future__ import annotations

import pytest

import flagquantum as fq
import flagquantum.errors as fqe
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit


def test_static_dynamic_circuit_reuses_the_static_quafu_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = DynamicCircuit(2).h(0).cx(0, 1)
    marker = object()
    captured: dict[str, object] = {}

    def reject_dynamic_preflight(*args: object, **kwargs: object) -> None:
        raise AssertionError("a statically degenerate program needs no dynamic facts")

    def execute_static(program: object, **kwargs: object) -> object:
        captured["program"] = program
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(
        "flagquantum.runtime.dynamic.remote_preflight.require_dynamic_remote_preflight",
        reject_dynamic_preflight,
    )
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.execute_quafu",
        execute_static,
    )

    result = fq.run(circuit, target="quafu:Baihua", shots=1024)

    assert result is marker
    assert captured["program"] == circuit.to_ir()
    assert captured["target"] == "quafu:Baihua"
    assert captured["shots"] == 1024
    assert captured["compiler"] is None


@pytest.mark.parametrize(
    "circuit",
    (
        DynamicCircuit(1).x(0).measure(0, classical_bit=0),
        DynamicCircuit(1).x(0).reset(0),
        DynamicCircuit(2)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0),
    ),
)
def test_explicit_dynamic_semantics_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    circuit: DynamicCircuit,
) -> None:
    def reject_static_execution(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "dynamic semantics must not enter the static provider path"
        )

    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.execute_quafu",
        reject_static_execution,
    )

    with pytest.raises(fqe.CapabilityError, match="dynamic_remote_preflight_failed"):
        fq.run(circuit, target="quafu:Baihua", shots=1024)
