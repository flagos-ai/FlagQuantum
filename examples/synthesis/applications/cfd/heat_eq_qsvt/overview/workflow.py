"""Run every projected matrix column and all four source heat iterations."""

from dataclasses import dataclass

import torch

from flagquantum.compiler import ResourceEstimate

from .model import DTYPE, OBSERVED_STEPS, POINTS, Fit, Model, build_model, fit
from .quantum import (
    full_state,
    projected,
    synthesize_column,
    synthesize_iteration,
)


@dataclass(frozen=True)
class Result:
    model: Model
    fit: Fit
    measured: torch.Tensor
    matrix_error: float
    inputs: tuple[torch.Tensor, ...]
    fields: torch.Tensor
    projection_norms: tuple[float, ...]
    gauge_indices: tuple[int, ...]
    resources: ResourceEstimate


def run_workflow() -> Result:
    model = build_model()
    angles = fit(model)
    columns = []
    for vector in torch.eye(POINTS, dtype=DTYPE):
        program = synthesize_column(model.rotation, vector)
        columns.append(projected(program, full_state(program)))
    measured = model.scaling * torch.stack(columns, dim=1)
    matrix_error = float(torch.linalg.vector_norm(measured - model.operator))
    if not matrix_error < 1e-12:
        raise ValueError("The reconstructed block does not match the heat operator")

    vector = model.initial
    inputs, fields, norms, indices = [], [], [], []
    product = 1.0
    for _ in range(OBSERVED_STEPS):
        inputs.append(vector)
        fields.append(model.initial_norm * product * vector)
        program = synthesize_iteration(model.rotation, vector, angles.phases)
        state = full_state(program)
        # Remove the phase of the first occupied basis amplitude in the native
        # output ordering. This makes the source's first-row rule reproducible.
        index = int(torch.nonzero(state.abs() > 1e-14)[0, 0])
        gauge = torch.exp(1j * torch.angle(state[index]))
        values = (projected(program, state) / gauge).real
        norm = float(values.norm())
        if not norm > 1e-12:
            raise ValueError(
                "The source's real postselected vector is numerically zero"
            )
        indices.append(index)
        norms.append(norm)
        product *= norm
        vector = values / norm
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Result(
        model,
        angles,
        measured,
        matrix_error,
        tuple(inputs),
        torch.stack(fields),
        tuple(norms),
        tuple(indices),
        resources,
    )
