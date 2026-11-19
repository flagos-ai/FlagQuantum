"""Check the full HHL state and apply the source portfolio reconstruction."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.hhl_reference import matrix_hhl_circuit
from flagquantum import synth

from .model import COMPLEX, PHASE_WORDS, Model, build_model
from .quantum import build_program, logical_state


@dataclass(frozen=True)
class Result:
    model: Model
    program: synth.Program
    absolute_error: float
    postselection_probability: float
    direct: torch.Tensor
    fitted: torch.Tensor
    least_squares: torch.Tensor
    relative_error: float
    weights: torch.Tensor
    values: torch.Tensor


def reference_state(model: Model) -> torch.Tensor:
    """Use the independent dense Fourier and controlled-matrix Circuit."""
    circuit = matrix_hhl_circuit(
        11,
        (1, 2, 3, 4),
        (5, 6, 7, 8, 9, 10),
        0,
        tuple(tuple(map(complex, row)) for row in model.evolution),
        tuple(map(float, model.angles)),
        tuple(map(float, model.encoded_rhs)),
    )
    state: torch.Tensor = fq.run(circuit).statevector()
    return state.reshape(2, 16, PHASE_WORDS)


def run_workflow() -> Result:
    """Read the original postselection and preserve its fitted scale and order."""
    model = build_model()
    program = build_program(model)
    logical = logical_state(program)
    expected = reference_state(model)
    absolute_error = float((logical - expected).abs().max())
    if absolute_error > 1e-11:
        raise AssertionError("The full native HHL state differs from the reference")
    postselected = logical[1, :, 0]
    probability = float(postselected.abs().square().sum())
    exact = torch.linalg.solve(model.hermitian, model.encoded_rhs)[:6]
    least_squares = torch.linalg.lstsq(
        model.hermitian, model.encoded_rhs, rcond=0.0
    ).solution[:6]
    raw = postselected[:6] / model.cutoff
    # Preserve the source's bilinear fit before taking its real portfolio weights.
    fitted = raw * (torch.dot(raw, exact.to(COMPLEX)) / torch.dot(raw, raw))
    relative_error = float((fitted - exact).norm() / exact.norm())
    weights = torch.stack((exact[2:6], fitted.real[2:6]), dim=1)
    values = model.prices @ weights
    values = values / values[0]
    return Result(
        model,
        program,
        absolute_error,
        probability,
        exact,
        fitted,
        least_squares,
        relative_error,
        weights,
        values,
    )
