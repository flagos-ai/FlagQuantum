# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/chem_tutorials.py.

"""Classical scaling fits and figures for chemistry resource tutorials."""

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.optimize import curve_fit


def linear_values(
    x: NDArray[np.float64], slope: float, intercept: float
) -> NDArray[np.float64]:
    """Evaluate a line at the supplied coordinates."""
    return slope * x + intercept


def fit_line(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Fit slope and intercept with the tutorial's least-squares solver."""
    parameters, _ = curve_fit(linear_values, x, y)
    return float(parameters[0]), float(parameters[1])


def plot_power_law(
    axes: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    label: str | None = None,
    color: str = "C0",
) -> None:
    """Fit in logarithmic coordinates and plot samples plus the fitted curve.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(1, 6, dtype=float)
        >>> figure, axes = plt.subplots()
        >>> plot_power_law(axes, x, 17 * x**2)
        >>> plt.close(figure)
    """
    slope, intercept = fit_line(np.log(x), np.log(y))
    coordinates = np.linspace(x[0], x[-1], 200)
    fitted = np.exp(intercept) * coordinates**slope
    if label is None:
        label = ""
    axes.loglog(
        x,
        y,
        marker="o",
        ls="None",
        label=rf"{label} ${{{np.exp(intercept):3.1f}}}N^{{{slope:3.2f}}}$",
        color=color,
    )
    axes.loglog(coordinates, fitted, marker="None", linestyle="--", color=color)
    axes.legend()
