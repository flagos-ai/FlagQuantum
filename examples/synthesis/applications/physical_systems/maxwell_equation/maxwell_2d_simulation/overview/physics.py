"""Full original 32 by 32 Yee grid with the original rectangular conductor."""

import torch

from flagquantum.simulation.matrix_free_exponential import exponential_action

SIZE, LENGTH, TIME = 5, 32, 5.0
BOTTOM, TOP = 9, 24


def electric_mask() -> torch.Tensor:
    mask = torch.ones((LENGTH, LENGTH), dtype=torch.bool)
    mask[0, 1:] = False
    mask[1:, 0] = False
    mask[BOTTOM:TOP, BOTTOM:TOP] = False
    return mask


def initial_state() -> torch.Tensor:
    coordinate = torch.arange(LENGTH, dtype=torch.float64)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    electric = torch.exp(-((x - 4).square() + (y - 4).square()) / 32)
    electric *= electric_mask()
    electric[0, 0] = 0
    result = torch.zeros((4, LENGTH, LENGTH), dtype=torch.complex128)
    result[0] = electric / electric.norm()
    return result.reshape(-1)


def apply(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape != (4 * LENGTH * LENGTH,):
        raise ValueError("The original Maxwell grid requires 4096 amplitudes")
    fields = vector.reshape(4, LENGTH, LENGTH)
    mask = electric_mask()
    electric = fields[0] * mask
    result = torch.zeros_like(fields)
    result[0] = (
        -fields[2] + fields[2].roll(1, 0) + fields[3] - fields[3].roll(1, 1)
    ) * mask
    result[2] = electric - electric.roll(-1, 0)
    result[3] = electric.roll(-1, 1) - electric
    return result.reshape(-1)


def evolve() -> tuple[torch.Tensor, float]:
    """Evaluate the same exp(160 A) action using bounded Krylov windows."""
    vector = initial_state()
    maximum_defect = 0.0
    for _ in range(64):
        vector, defect = exponential_action(
            apply,
            TIME * LENGTH / 64,
            vector,
            tolerance=1e-13,
            max_iterations=48,
            restart=48,
        )
        maximum_defect = max(maximum_defect, defect)
        if defect > 1e-13:
            raise ValueError(
                "The full-grid classical exponential missed its defect tolerance"
            )
    return vector, maximum_defect
