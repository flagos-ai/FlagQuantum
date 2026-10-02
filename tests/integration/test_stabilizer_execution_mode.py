"""End-to-end evidence for the stabilizer execution mode.

These tests run the public `fq.run` path and read what the result reports about
how it was produced. They are integration tests because they execute a circuit;
the planning contract is tested in `tests/unit/test_stabilizer_execution_mode.py`
and the numerical contract in `tests/team/simulation/`.
"""

from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.integration

pytest.importorskip("stim")


def _ghz(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires - 1):
        circuit.cx(wire, wire + 1)
    return circuit


def _run(circuit: fq.Circuit, **overrides: object) -> fq.ExecutionResult:
    values = {"mode": "stabilizer", "shots": 64, "seed": 7}
    values.update(overrides)
    return fq.run(
        circuit,
        options=fq.ExecutionOptions(**values),
        outputs=fq.samples(),
    )


def test_the_result_reports_the_mode_that_ran() -> None:
    result = _run(_ghz(3))

    assert result.runtime["mode"] == "stabilizer"
    assert result.runtime["execution_path"] == "local_stabilizer"
    assert result.runtime["simulation_engine"] == "pauli_stabilizer_tableau"
    assert result.runtime["device"] == "cpu"
    assert result.samples is not None
    assert result.samples.shape == (1, 64, 3)


def test_the_plan_a_run_produces_states_the_tableau() -> None:
    plan = _run(_ghz(4)).plan

    assert plan.state_mode == "stabilizer"
    assert plan.state_bytes == 9
    assert plan.summary()["scalability_claim_allowed"] is False


def test_a_ghz_chain_samples_only_its_two_correlated_outcomes() -> None:
    result = _run(_ghz(4))

    assert result.samples is not None
    rows = {tuple(int(bit) for bit in row) for row in result.samples[0]}
    assert rows <= {(0, 0, 0, 0), (1, 1, 1, 1)}


def test_a_repeated_seed_reproduces_the_samples() -> None:
    first = _run(_ghz(3), seed=11)
    second = _run(_ghz(3), seed=11)

    assert bool((first.samples == second.samples).all())


def test_a_different_seed_samples_a_different_stream() -> None:
    circuit = fq.Circuit(6)
    for wire in range(6):
        circuit.h(wire)
    first = _run(circuit, shots=512, seed=11)
    second = _run(circuit, shots=512, seed=12)

    assert first.samples is not None and second.samples is not None
    assert not bool((first.samples == second.samples).all())


def test_the_requested_wire_order_is_the_output_column_order() -> None:
    circuit = _ghz(4)
    all_wires = _run(circuit)
    subset = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode="stabilizer", shots=64, seed=7),
        outputs=fq.samples(qubits=[2, 0]),
    )

    assert subset.samples is not None
    assert subset.samples.shape == (1, 64, 2)
    assert bool((subset.samples == all_wires.samples[..., [2, 0]]).all())


def test_counts_answer_the_same_request_as_samples() -> None:
    circuit = _ghz(4)
    samples = _run(circuit)
    counts = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode="stabilizer", shots=64, seed=7),
        outputs=fq.counts(),
    )

    assert samples.samples is not None
    histogram = counts.measurements[0].value[0]
    assert sum(histogram.values()) == 64
    assert set(histogram) <= {"0000", "1111"}


def test_a_width_no_amplitude_store_can_hold_is_planned_and_run() -> None:
    """16384 wires is a 134 MB tableau; no statevector here is representable."""

    circuit = fq.Circuit(16384)
    for wire in range(16383):
        circuit.cx(wire, wire + 1)

    result = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode="stabilizer", shots=4, seed=1),
        outputs=fq.samples(),
    )

    assert result.samples is not None
    assert result.samples.shape == (1, 4, 16384)
    assert len(set(map(tuple, result.samples[0].tolist()))) == 1


def test_a_non_clifford_circuit_fails_before_any_route_runs() -> None:
    with pytest.raises(CapabilityError, match="is not a Clifford gate"):
        _run(fq.Circuit(2).h(0).t(1))


def test_postselection_is_refused_rather_than_retried() -> None:
    """Every drawing of one Clifford circuit is the same stream.

    The stable output vocabulary cannot request postselection, so this arrives
    through the executor entry point that takes an IR carrying the request.
    """

    from dataclasses import replace

    from flagquantum.core.ir import MeasurementNode
    from flagquantum.runtime import run_native

    circuit = _ghz(2)
    nodes = (
        MeasurementNode("sample", (0, 1), shots=8, metadata={"postselect": {0: 1}}),
    )

    with pytest.raises(CapabilityError, match="cannot postselect"):
        run_native(
            replace(circuit.to_ir(), measurements=nodes),
            mode="stabilizer",
            shots=8,
        )


def test_the_executor_refuses_a_run_with_no_sampling_request() -> None:
    """The executor states the request it serves rather than guessing one."""

    from flagquantum.runtime import run_native

    with pytest.raises(CapabilityError, match="exactly one sampling request"):
        run_native(_ghz(2).to_ir(), mode="stabilizer", shots=8)


def test_a_memory_limit_below_the_tableau_is_refused() -> None:
    with pytest.raises(CapabilityError, match="above the declared memory_limit_bytes"):
        _run(_ghz(4), memory_limit_bytes=4)


def test_the_mode_does_not_disturb_the_amplitude_paths() -> None:
    circuit = _ghz(2)
    automatic = fq.run(
        circuit,
        options=fq.ExecutionOptions(shots=8, seed=5),
        outputs=fq.samples(),
    )

    assert automatic.runtime["mode"] == "statevector"
    assert automatic.runtime["execution_path"] != "local_stabilizer"
