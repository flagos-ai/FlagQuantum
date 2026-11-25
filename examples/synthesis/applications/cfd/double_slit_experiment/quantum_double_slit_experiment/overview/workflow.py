"""Verify full-state wave interference and the independent spectral solution."""

from dataclasses import dataclass

import torch

from flagquantum.compiler import ResourceEstimate, estimate_resources

from .model import Inverse, Model, build_model, fidelity, inverse_fit
from .quantum import WIDTH, qsvt_circuit
from .reference import complete_state


@dataclass(frozen=True)
class Result:
    model: Model
    inverse: Inverse
    wave: torch.Tensor
    full_error: float
    spectral_error: float
    work_probability: float
    selected_probability: float
    fidelity: float
    resources: ResourceEstimate


def run() -> Result:
    model = build_model()
    inverse = inverse_fit(model)
    circuit = qsvt_circuit(model, inverse)
    resource = estimate_resources(circuit)
    if resource.n_qubits != WIDTH or resource.max_operation_width > 7:
        raise ValueError("The double-slit program must respect the source width bound")
    if resource.n_operations > 10_000:
        raise ValueError("The bounded simulator instruction budget was exceeded")
    native: torch.Tensor = circuit.state().reshape(1 << 15, 8)
    expected = complete_state(model, inverse)
    full_error = float((native[:, 0] - expected).abs().max())
    work_probability = float(native[:, 1:].abs().square().sum())
    wave = native[:256, 0]
    # The alternating odd QSP convention contributes (-1)^((191-1)/2).
    spectral_error = float((wave + inverse.expected_solution).abs().max())
    success = float(wave.abs().square().sum())
    measured = fidelity(wave, model.reference)
    if max(full_error, spectral_error) > 1e-10 or work_probability > 1e-24:
        raise ValueError(
            "The full physical state disagrees with its tensor or spectral reference"
        )
    if measured < 0.95:
        raise ValueError("The current circuit did not meet the original wave fidelity")
    return Result(
        model,
        inverse,
        wave,
        full_error,
        spectral_error,
        work_probability,
        success,
        measured,
        resource,
    )
