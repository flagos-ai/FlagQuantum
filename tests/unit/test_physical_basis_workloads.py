"""Compare complete source benchmark workloads through shared simulation."""

import json
from pathlib import Path
from typing import cast, get_args

import pytest

from flagquantum.benchmarking._adaptive_timing import measure
from flagquantum.benchmarking._physical_basis_workloads import (
    SimulationWorkload,
    simulation_workload,
    transpose_workload,
)
from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.core._binary_matrix import BinaryMatrix
from tests.unit.test_physical_basis_lifecycle import _word

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_basis_lifecycle.json").read_text()
)


@pytest.mark.parametrize("mode", get_args(SimulationWorkload))
def test_complete_original_workload_states_buffers_phases_and_lane_counts(
    mode: SimulationWorkload,
) -> None:
    index = get_args(SimulationWorkload).index(mode)
    circuit = read_kmx_circuit(_DATA["workloads"][index])
    lines: list[str] = []
    for width in (64, 128, 256, 512):
        sim, body = simulation_workload(circuit, width, mode)
        lines.append(f"width:{width}")
        for _ in range(2):
            body()
            values = [
                sim.state.sign,
                *(sim.state.qubits[wire] for wire in range(sim.qubits)),
                *(sim.state.bits[wire] for wire in range(sim.bits)),
            ]
            lines.extend(f"state:{_word(value, width)}" for value in values)
            for lane in range(width):
                lines.append(
                    "phase:" + str(PhaseAngle128._from_units(sim.read_shot_phase(lane)))
                )
                lines.extend(f"out:{value}" for value in sim.register_buffer[lane])
                lines.extend(f"before:{value}" for value in sim.expected_buffer[lane])
            if mode == "counting":
                for record in circuit.records:
                    name = record.opcode + (
                        "_IF" if record.condition_bit is not None else ""
                    )
                    lines.append(
                        "count:"
                        + name
                        + "".join(
                            f":{dict(sim.counters.snapshot(lane)[0])[name]}"
                            for lane in range(width)
                        )
                    )
    assert "\n".join(lines) + "\n" == _DATA["workload_observations"][index]


def test_transpose_workload_reuses_core_and_shared_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[BinaryMatrix] = []
    original = BinaryMatrix.transposed

    def observed(matrix: BinaryMatrix) -> BinaryMatrix:
        calls.append(matrix)
        return original(matrix)

    monkeypatch.setattr(BinaryMatrix, "transposed", observed)
    ticks = iter((0, 1_000_000))
    result = measure(
        "transpose", transpose_workload(), target_seconds=0.001, clock=ticks.__next__
    )
    assert len(calls) == result.total_repetitions == 1
    assert calls[0] == BinaryMatrix((1, *([0] * 63)), 64)
    assert result.total_seconds == 0.001


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        simulation_workload(
            read_kmx_circuit(""), 64, cast(SimulationWorkload, "missing")
        )
