from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import MeasurementNode
from flagquantum.runtime.execution import run as run_internal

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_measurement_nodes_are_consistent_across_local_backends(mode: str) -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    requests = (
        MeasurementNode("expectation_z", (0, 1)),
        MeasurementNode(
            "expectation_ps",
            (0, 1),
            metadata={"x": (0, 1)},
        ),
        MeasurementNode("counts", (0, 1), shots=8, metadata={"seed": 11}),
    )

    result = run_internal(
        circuit,
        options=fq.ExecutionOptions(mode=mode),
        measurements=requests,
    )

    torch.testing.assert_close(
        result.measurements[0].value,
        torch.zeros(1, 2),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        result.measurements[1].value,
        torch.ones(1),
        atol=1e-6,
        rtol=0,
    )
    counts = result.measurements[2].value[0]
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 8


def test_ir_measurements_execute_without_a_parallel_user_api() -> None:
    circuit = fq.Circuit(2).x(1)
    source = circuit.to_ir()
    ir = fq.CircuitIR(
        n_wires=source.n_wires,
        instructions=source.instructions,
        dtype=source.dtype,
        shape=source.shape,
        measurements=(
            MeasurementNode("expectation_z", (0, 1)),
            MeasurementNode("sample", (0,), shots=3, metadata={"seed": 5}),
        ),
        metadata=source.metadata,
    )

    result = fq.run(ir, options=fq.ExecutionOptions(mode="statevector"))

    torch.testing.assert_close(
        result.measurements[0].value,
        torch.tensor([[1.0, -1.0]]),
    )
    assert result.measurements[1].value.tolist() == [[[0], [0], [0]]]


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_joint_marginal_probabilities_do_not_require_full_state(mode: str) -> None:
    circuit = fq.Circuit(3).x(0).h(1)
    result = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode=mode),
        outputs=fq.probabilities((0, 1)),
    )

    torch.testing.assert_close(
        result.measurements[0].value,
        torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_modern_postselection_returns_conditioned_samples_and_statistics(
    mode: str,
) -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = run_internal(
        circuit,
        options=fq.ExecutionOptions(mode=mode),
        measurements=(
            MeasurementNode(
                "counts",
                (1,),
                shots=16,
                metadata={"seed": 13, "postselect": {0: 1}},
            ),
        ),
    )

    measurement = result.measurements[0]
    assert measurement.value == [{"1": 16}]
    assert measurement.statistics["shots"] == 16
    assert measurement.statistics["postselection"] == {0: 1}
    assert 0 < measurement.statistics["acceptance_rate"][0] <= 1
    assert measurement.statistics["outcome_probability"] == [{"1": 1.0}]
    assert measurement.statistics["outcome_standard_error"] == [{"1": 0.0}]


def test_measurement_preflight_rejects_exponential_or_invalid_postselection() -> None:
    circuit = fq.Circuit(9)

    with pytest.raises(ValueError, match="increase max_marginal_wires"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="mps"),
            measurements=(MeasurementNode("probabilities", tuple(range(9))),),
        )
    with pytest.raises(ValueError, match="only for sample/counts"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            measurements=(
                MeasurementNode(
                    "expectation_z",
                    (0,),
                    metadata={"postselect": {0: 1}},
                ),
            ),
        )
    with pytest.raises(ValueError, match="bits must be 0 or 1"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            measurements=(
                MeasurementNode(
                    "sample",
                    (0,),
                    shots=2,
                    metadata={"postselect": {0: 2}},
                ),
            ),
        )


def test_measurement_requests_fail_closed_on_unsupported_or_ambiguous_inputs() -> None:
    circuit = fq.Circuit(2)

    with pytest.raises(ValueError, match="unsupported measurement kind"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            measurements=(MeasurementNode("povm", (0,)),),
        )
    with pytest.raises(ValueError, match="requires a positive shots"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            measurements=(MeasurementNode("sample", (0,)),),
        )
    with pytest.raises(ValueError, match="must match metadata"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            measurements=(
                MeasurementNode(
                    "expectation_ps",
                    (0,),
                    metadata={"x": (1,)},
                ),
            ),
        )
