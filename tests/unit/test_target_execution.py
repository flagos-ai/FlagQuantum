import pytest
import torch

import flagquantum as fq
import flagquantum.runtime as fqr

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["statevector", "mps", "tensor_network"])
@pytest.mark.parametrize("use_ir", [False, True])
def test_single_amplitude_accepts_zero_and_rejects_missing_bitstring(
    mode: str, use_ir: bool
) -> None:
    circuit = fq.Circuit(2).h(0)
    program = circuit.to_ir() if use_ir else circuit
    with pytest.raises(ValueError, match="single_amplitude requires bitstring"):
        fqr.run_target(program, target="single_amplitude", mode=mode)

    result = fqr.run_target(program, target="single_amplitude", bitstring=0, mode=mode)
    torch.testing.assert_close(result.values, circuit.state()[:, 0])


@pytest.mark.parametrize("value", [None, (torch.zeros(2), {})])
def test_statevector_target_rejects_non_tensor_result(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(
        "flagquantum.runtime.execution.run_native", lambda *args, **kwargs: value
    )
    with pytest.raises(
        TypeError, match="statevector execution must return a torch.Tensor"
    ):
        fqr.run_target(fq.Circuit(2), target="full_state", mode="statevector")


@pytest.mark.parametrize("target", ["", "probability", "FULL_STATE"])
def test_run_target_rejects_unknown_output_names(target: str) -> None:
    with pytest.raises(ValueError, match="unsupported output target"):
        fqr.run_target(fq.Circuit(2), target=target)


def test_run_target_uses_statevector_when_dense_path_fits() -> None:
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2)

    result = fqr.run_target(
        circuit,
        target="local_observables",
        observables=({"z": (0, 5)}, {"x": (2,)}),
        memory_limit_bytes=1 << 30,
    )

    assert result.backend == "statevector"
    assert torch.allclose(
        result.values,
        torch.stack(
            [
                circuit.expectation_ps(z=(0, 5)),
                circuit.expectation_ps(x=(2,)),
            ],
            dim=-1,
        ),
    )


def test_run_target_forced_tn_executes_sparse_kernel_not_full_state() -> None:
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2)
    targets = ("000000", "100001", "010100")
    indices = torch.tensor([int(bits, 2) for bits in targets])

    result = fqr.run_target(
        circuit,
        target="few_amplitudes",
        bitstrings=targets,
        mode="tensor_network",
    )

    assert result.backend == "tensor_network"
    assert torch.allclose(result.values, circuit.state()[:, indices], atol=1e-6)
    assert result.summary()["full_state_materialized"] is False


def test_run_target_distributed_tn_preserves_sparse_execution_evidence() -> None:
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2)

    result = fqr.run_target(
        circuit,
        target="local_observables",
        observables=({"z": (0, 5)}, {"x": (2,)}),
        mode="tensor_network",
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )

    assert result.backend == "tensor_network"
    assert result.execution_summary["full_state_materialized"] is False
    assert result.execution_summary["shared_contraction"] is True


def test_run_target_routes_shallow_chain_to_sparse_mps_execution() -> None:
    circuit = fq.Circuit(20)
    for wire in range(19):
        circuit.cx(wire, wire + 1)
    targets = ("0" * 20, "1" * 20)

    result = fqr.run_target(
        circuit,
        target="few_amplitudes",
        bitstrings=targets,
        memory_limit_bytes=1 << 20,
    )

    assert result.backend == "mps"
    assert result.execution_summary["full_state_materialized"] is False
    assert result.execution_summary["max_bond_observed"] <= 2
    assert torch.allclose(
        result.values,
        circuit.state()[:, torch.tensor([0, 2**20 - 1])],
    )
