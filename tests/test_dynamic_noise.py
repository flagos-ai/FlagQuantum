from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.noise as fqn
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic._noise import apply_readout_error

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_dynamic_bit_flip_channel_changes_state_after_matching_gate(
    strategy: str,
) -> None:
    circuit = DynamicCircuit(1).x(0)
    circuit.measure(0, classical_bit=0)
    model = fqn.NoiseModel().add("x", fqn.bit_flip_channel(1.0), wires=0)

    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=4,
        seed=17,
        strategy=strategy,
        noise_model=model,
    )

    assert torch.count_nonzero(result.classical_bits) == 0
    assert torch.count_nonzero(result.samples) == 0
    assert result.statistics["bit_flip_event_count"] == 4
    assert result.statistics["noise_channel_application_count"] == 4
    assert result.statistics["noise_model_identity"] == model.identity


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_dynamic_readout_error_changes_observed_control_not_collapsed_state(
    strategy: str,
) -> None:
    circuit = DynamicCircuit(2).x(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    model = fqn.NoiseModel().add_readout(
        0,
        fqn.ReadoutError(((0.0, 1.0), (1.0, 0.0))),
    )

    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=5,
        seed=19,
        strategy=strategy,
        noise_model=model,
    )

    assert torch.count_nonzero(result.classical_bits) == 0
    assert torch.count_nonzero(result.samples) == 0
    assert torch.allclose(
        torch.abs(result.final_states[:, 2]), torch.ones(5, dtype=torch.float32)
    )
    assert result.statistics["conditional_applied_count"] == 0
    assert result.statistics["readout_error_count"] == 10


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_dynamic_noise_is_seed_reproducible(strategy: str) -> None:
    circuit = DynamicCircuit(1).x(0)
    circuit.measure(0, classical_bit=0)
    model = fqn.NoiseModel().add("x", fqn.bit_flip_channel(0.37), wires=0)

    first = fq.experimental.dynamic.run_dynamic(
        circuit, shots=64, seed=23, strategy=strategy, noise_model=model
    )
    second = fq.experimental.dynamic.run_dynamic(
        circuit, shots=64, seed=23, strategy=strategy, noise_model=model
    )

    assert torch.equal(first.samples, second.samples)
    assert torch.equal(first.classical_bits, second.classical_bits)
    assert (
        first.statistics["bit_flip_event_count"]
        == second.statistics["bit_flip_event_count"]
    )


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_dynamic_gate_noise_applies_only_to_active_conditional_shots(
    strategy: str,
) -> None:
    circuit = DynamicCircuit(2).h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    model = fqn.NoiseModel().add("x", fqn.bit_flip_channel(1.0), wires=1)

    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=128,
        seed=29,
        strategy=strategy,
        noise_model=model,
    )

    active = int(torch.count_nonzero(result.classical_bits[:, 0]).item())
    assert 0 < active < 128
    assert torch.count_nonzero(result.samples[:, 1]) == 0
    assert result.statistics["noise_channel_application_count"] == active
    assert result.statistics["bit_flip_event_count"] == active


def test_dynamic_noise_rejects_unsupported_channel_and_correlated_readout() -> None:
    circuit = DynamicCircuit(1).x(0)
    circuit.measure(0, classical_bit=0)
    amplitude = fqn.NoiseModel().add("x", fqn.amplitude_damping_channel(0.1))
    correlated = fqn.NoiseModel().add_correlated_readout(
        (0,),
        fqn.CorrelatedReadoutError(((1.0, 0.0), (0.0, 1.0))),
    )
    misplaced = fqn.NoiseModel().add("measure", fqn.bit_flip_channel(0.1))

    with pytest.raises(ValueError, match="bit-flip channels only"):
        fq.experimental.dynamic.run_dynamic(circuit, shots=1, noise_model=amplitude)
    with pytest.raises(ValueError, match="correlated readout"):
        fq.experimental.dynamic.run_dynamic(circuit, shots=1, noise_model=correlated)
    with pytest.raises(ValueError, match="must use independent readout rules"):
        fq.experimental.dynamic.run_dynamic(circuit, shots=1, noise_model=misplaced)


def test_readout_boundary_rejects_correlated_error_without_consuming_rng() -> None:
    bits = torch.tensor([0, 1, 0, 1])
    before_bits = bits.clone()
    generator = torch.Generator().manual_seed(42)
    before_rng = generator.get_state().clone()
    model = fqn.NoiseModel().add_correlated_readout(
        (0,), fqn.CorrelatedReadoutError(((1.0, 0.0), (0.0, 1.0)))
    )

    with pytest.raises(ValueError, match="correlated readout is outside"):
        apply_readout_error(bits, 0, model, generator=generator)

    assert torch.equal(bits, before_bits)
    assert torch.equal(generator.get_state(), before_rng)
