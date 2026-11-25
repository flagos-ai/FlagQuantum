"""Original rounded matrix/RHS and fresh exact evolution for all precisions."""

import math

import torch

MATRIX = torch.tensor(
    [
        [0.135, -0.092, -0.011, -0.045, -0.026, -0.033, 0.03, 0.034],
        [-0.092, 0.115, 0.02, 0.017, 0.044, -0.009, -0.015, -0.072],
        [-0.011, 0.02, 0.073, -0.0, -0.068, -0.042, 0.043, -0.011],
        [-0.045, 0.017, -0.0, 0.043, 0.028, 0.027, -0.047, -0.005],
        [-0.026, 0.044, -0.068, 0.028, 0.21, 0.079, -0.177, -0.05],
        [-0.033, -0.009, -0.042, 0.027, 0.079, 0.121, -0.123, 0.021],
        [0.03, -0.015, 0.043, -0.047, -0.177, -0.123, 0.224, 0.011],
        [0.034, -0.072, -0.011, -0.005, -0.05, 0.021, 0.011, 0.076],
    ],
    dtype=torch.float64,
)
RAW_B = torch.tensor(
    [
        -0.00885448,
        -0.17725898,
        -0.15441119,
        0.17760157,
        0.41428775,
        0.44735303,
        -0.71137715,
        0.1878808,
    ],
    dtype=torch.float64,
)
RAW_B_NORM = float(RAW_B.norm())
# A rounded near-unit vector must be normalized before any unitary preparation.
B = RAW_B / RAW_B.norm()
AMPLITUDES = tuple(float(v) for v in B)
UNITARY = tuple(
    tuple(complex(v) for v in row)
    for row in torch.matrix_exp(2j * math.pi * MATRIX).tolist()
)
CLASSICAL = torch.linalg.solve(MATRIX, RAW_B)
PRECISIONS = tuple(range(2, 9))
HISTORICAL_FIDELITIES = (
    0.3158037121175521,
    0.43599529278857063,
    0.5586003448231571,
    0.6824252904259536,
    0.806169290650212,
    0.9243747525650154,
    1.0,
)
HISTORICAL_DEPTHS = (1921, 4439, 9451, 19455, 39443, 79399, 159291)
HISTORICAL_WIDTHS = (6, 7, 8, 9, 10, 11, 12)
HISTORICAL_CX = (979, 2263, 4819, 9915, 20087, 40407, 81019)


def reciprocal_angles(precision: int) -> tuple[float, ...]:
    if type(precision) is not int or precision not in PRECISIONS:
        raise ValueError("precision must be an integer from two through eight")
    return tuple(
        0.0 if word == 0 else 2 * math.asin(1 / word) for word in range(1 << precision)
    )
