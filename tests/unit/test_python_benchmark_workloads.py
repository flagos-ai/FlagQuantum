"""Compare complete stateful Python benchmark construction and execution."""

import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from benchmarks import physical_array_perf, physical_builder_perf
from flagquantum.benchmarking._adaptive_timing import TimingSample
from flagquantum.benchmarking._timing_report import format_suite_timing
from flagquantum.benchmarking._timing_suite import TimingSuite
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.registers.layout.operand_word import bind_operand_array
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession

pytestmark = pytest.mark.unit
_DATA: dict[str, Any] = json.loads(
    (Path(__file__).with_name("data") / "python_benchmark_workloads.json").read_text()
)


def _source_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    assert isinstance(value, OperandIdentity)
    if value.kind is OperandKind.BOOLEAN:
        return str(bool(value))
    return value.diagnostic_text


def _observe(case: dict[str, Any]) -> tuple[TimingSample, dict[str, object]]:
    name = str(case["name"]).split(".")[-1]
    module = (
        physical_array_perf if name == "benchmark_construct" else physical_builder_perf
    )
    observation: dict[str, object] = {}

    def capture(
        group: TimingSuite,
        body: Callable[[], object],
        *,
        rates: Mapping[str, float] | None = None,
    ) -> TimingSample:
        for _ in range(case["repeat"]):
            body()
        cells = inspect.getclosurevars(body).nonlocals
        if "builder" in cells:
            builder = cells["builder"]
            assert isinstance(builder, KmxCircuitBuilder)
            circuit = builder.freeze()
            observation["text"] = circuit.to_kmx()
            observation["allocated"] = sum(
                len(register.members) for register in circuit.registers
            )
        else:
            items = cells["items"]
            assert isinstance(items, list)
            observation["items"] = [_source_text(item) for item in items]
            observation["constructed"] = [
                _source_text(item) for item in bind_operand_array(items)
            ]
        observation["rates"] = rates
        return TimingSample(
            0.5,
            case["repeat"],
            tuple(() if rates is None else rates.items()),
            goal_seconds=None,
        )

    with patch.object(TimingSuite, "go", capture):
        result = getattr(module, name)()
    assert isinstance(result, TimingSample)
    return result, observation


@pytest.mark.parametrize("case", _DATA["workloads"])
def test_complete_repeated_original_workloads(case: dict[str, Any]) -> None:
    result, observation = _observe(case)
    for key, value in observation.items():
        assert value == case[key], key
    assert result.goal_seconds == case["goal"]
    assert format_suite_timing(result) == case["formatted"].replace(
        "km.array.", "OperandArray."
    )
    assert len(physical_array_perf.benchmark.benchmarks) == 1
    assert len(physical_builder_perf.benchmark.benchmarks) == 5


@pytest.mark.parametrize("case", _DATA["executions"])
def test_actual_three_lane_execution_phase_counters_and_zero_work(
    case: dict[str, Any],
) -> None:
    source = next(
        item
        for item in _DATA["workloads"]
        if item["name"] == case["name"] and item["repeat"] == case["repeat"]
    )
    _, observation = _observe(source)
    circuit = read_kmx_circuit(str(observation["text"]))
    simulator = PhysicalBasisSession(
        3, seed=0, debug_print=False, counting_operations=True
    )
    simulator.use_same_registers_as(circuit)
    quantum = len(case["values"])
    for wire in range(quantum):
        simulator.write_across_shots(wire, (wire * 5 + 3) & 7)
    if "mixed" in case["name"]:
        for bit in range(256, 512):
            simulator.write_across_shots(
                OperandIdentity(OperandKind.CLASSICAL, bit), (bit * 3 + 1) & 7
            )
    simulator.do(circuit)
    assert simulator.read_across_shots(list(range(quantum)), out=int) == case["values"]
    assert [str(simulator.read_shot_phase(shot)) for shot in range(3)] == case["phases"]
    assert dict(simulator.operation_statistics()) == case["counts"]
    if "iadd" in case["name"]:
        assert (
            simulator.read_across_shots(list(range(512, 766)), out=int)
            == case["work"]
            == [0] * 254
        )
