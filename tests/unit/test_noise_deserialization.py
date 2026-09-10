"""Noise restoration must enforce the same constraints as direct construction."""

import pytest
import torch

from flagquantum.noise import CorrelatedReadoutError, NoiseModel, ReadoutError


def test_restored_correlated_readout_rejects_inconsistent_wire_count() -> None:
    probabilities = tuple(tuple(row) for row in torch.eye(4).tolist())
    model = NoiseModel().add_correlated_readout(
        (0, 1), CorrelatedReadoutError(probabilities)
    )
    payload = model.to_dict()
    payload["readout_rules"][0]["wires"] = [0]

    with pytest.raises(ValueError, match="matrix size must match wires"):
        NoiseModel.from_dict(payload)


@pytest.mark.parametrize(
    "probabilities",
    [[], [[1.0, 0.0]], [[1.0], [0.0]], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]],
)
def test_restored_single_qubit_readout_rejects_wrong_shape(
    probabilities: list[list[float]],
) -> None:
    model = NoiseModel().add_readout(0, ReadoutError(((1.0, 0.0), (0.0, 1.0))))
    payload = model.to_dict()
    payload["readout_rules"][0]["probabilities"] = probabilities

    with pytest.raises(ValueError, match="shape"):
        NoiseModel.from_dict(payload)
