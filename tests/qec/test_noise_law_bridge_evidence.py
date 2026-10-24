"""Exact executed-law checks across the native circuit text boundary."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

import flagquantum.qec as qec
from flagquantum.noise import (
    KrausChannel,
    NoiseModel,
    bit_flip_channel,
    phase_flip_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.qec.adapters.stim import memory_to_stim_text, stim_circuit_to_dem_text

pytestmark = pytest.mark.unit


def _actual_rate(factory: Callable[[float], KrausChannel], probability: float) -> float:
    mixture = factory(probability).unitary_mixture
    assert mixture is not None
    return float(mixture.probabilities[-1])


def _selected_joint_rates(model: qec.DetectorErrorModel) -> tuple[float, ...]:
    # Independent Bernoulli parity identity. An exclusive model is deliberately
    # outside this reference's domain, rather than silently treated as independent.
    assert all(error.error_id is None for error in model.errors)
    selections = (("D", 0), ("D", model.num_detectors - 1), ("L", 0))

    def expectation(selected: tuple[tuple[str, int], ...]) -> float:
        return math.prod(
            1 - 2 * error.probability
            for error in model.errors
            if sum(
                index in (error.detectors if kind == "D" else error.observables)
                for kind, index in selected
            )
            % 2
        )

    return tuple(
        (1 - expectation((a,)) - expectation((b,)) + expectation((a, b))) / 4
        for a, b in (
            (selections[0], selections[1]),
            (selections[0], selections[2]),
            (selections[1], selections[2]),
        )
    )


@pytest.mark.parametrize(
    ("family", "distance", "basis"),
    [
        ("repetition", 3, "z"),
        ("surface", 3, "x"),
        ("surface", 3, "z"),
        ("surface", 5, "x"),
        ("surface", 5, "z"),
    ],
)
def test_exported_detector_logical_and_joint_law_matches_executed_kraus(
    family: str,
    distance: int,
    basis: str,
) -> None:
    pytest.importorskip("stim")
    code = (qec.RepetitionCode if family == "repetition" else qec.RotatedSurfaceCode)(
        distance
    )
    product = qec.Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    memory = qec.build_memory_circuit(code, rounds=3, product=product)
    nominal = qec.PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    # Independent analytic construction receives the executed branch weights,
    # never a second physical draw through factories that would quantize again.
    actual = qec.PhenomenologicalNoise(
        data_flip=_actual_rate(bit_flip_channel, 0.005),
        phase_flip=_actual_rate(phase_flip_channel, 0.007),
        measurement_flip=_actual_rate(bit_flip_channel, 0.009),
    )
    reference = qec.DetectorErrorModel.from_memory_circuit(memory, noise=actual)
    text = memory_to_stim_text(memory, noise=nominal)
    imported = qec.DetectorErrorModel.from_stim_text(stim_circuit_to_dem_text(text))
    assert torch.allclose(
        reference.detector_rates(), imported.detector_rates(), atol=1e-12, rtol=0
    )
    assert torch.allclose(
        reference.observable_rates(), imported.observable_rates(), atol=1e-12, rtol=0
    )
    assert _selected_joint_rates(reference) == pytest.approx(
        _selected_joint_rates(imported), abs=1e-12, rel=0
    )


def test_two_qubit_branch_arguments_round_trip_without_decimal_shortening() -> None:
    stim = pytest.importorskip("stim")
    memory = qec.build_memory_circuit(qec.RepetitionCode(3), rounds=2)
    channel = two_qubit_depolarizing_channel(
        torch.tensor(0.1234567890123456, dtype=torch.float64), dtype=torch.complex128
    )
    mixture = channel.unitary_mixture
    assert mixture is not None
    expected = mixture.probabilities[1:].tolist()
    text = memory_to_stim_text(
        memory, noise=NoiseModel().add("cx", channel, qubits=(0, 3))
    )
    instructions = [
        instruction
        for instruction in stim.Circuit(text)
        if instruction.name == "PAULI_CHANNEL_2"
    ]
    assert len(instructions) == 2
    for instruction in instructions:
        assert instruction.gate_args_copy() == expected
