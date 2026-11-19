"""Execute the full-size edge detector and compare it with cyclic finite differences."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
from matplotlib.image import imread

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate, estimate_resources

from .quantum import AXIS_BITS, build_edges, prepare_image

SIDE = 1 << AXIS_BITS
SHOTS = 200_000
THRESHOLD = 0.003
SEED = 2026
IMAGE = Path(__file__).parent / "data" / "marina-bay-sands-128.png"


@dataclass(frozen=True)
class Result:
    photo: torch.Tensor
    preview: torch.Tensor
    exact_edge: torch.Tensor
    sampled_edge: torch.Tensor
    thresholded_edge: torch.Tensor
    edge_probability: float
    observed_edge_shots: int
    preparation_error: float
    state_error: float
    work_probability: float
    preparation_resources: ResourceEstimate
    resources: ResourceEstimate


def load_photo(path: Path = IMAGE) -> torch.Tensor:
    photo = torch.as_tensor(imread(path), dtype=torch.float64)
    if photo.shape != (SIDE, SIDE, 3):
        raise ValueError("The workflow uses the full 128-by-128 RGB image")
    return photo


def threshold_preview(photo: torch.Tensor) -> torch.Tensor:
    """Reproduce the source's diagnostic RGB quantization, including the first row."""
    levels = (
        torch.tensor([0, 30, 60, 90, 120, 150, 180, 210, 255], dtype=torch.float64)
        / 255
    )
    bins = torch.bucketize(photo[1:, :, 0].contiguous(), levels, right=True) - 1
    preview = photo.clone()
    preview[1:] = levels[bins].unsqueeze(2)
    return preview


def pixel_amplitudes(photo: torch.Tensor) -> torch.Tensor:
    """Load magnitudes from the unquantized image, as the source does."""
    pixels = photo.mean(dim=2).abs()
    norm = float(pixels.norm())
    if not math.isfinite(norm):
        raise ValueError("Image intensities must be finite")
    if norm == 0:
        return torch.full_like(pixels, 1 / SIDE)
    return pixels / norm


def reference_state(pixels: torch.Tensor, work_states: int) -> torch.Tensor:
    """Derive the four coherent branches before tracing the two direction bits."""
    dx = pixels.roll(-1, dims=0)
    dy = pixels.roll(-1, dims=1)
    dxy = dx.roll(-1, dims=1)
    state = torch.zeros((SIDE, SIDE, 2, 2, 2, work_states), dtype=torch.complex128)
    for vertical, horizontal in product((0, 1), repeat=2):
        x_sign, y_sign = (-1) ** horizontal, (-1) ** vertical
        branch = (pixels + x_sign * dx + y_sign * dy + x_sign * y_sign * dxy) / 4
        state[:, :, horizontal | vertical, vertical, horizontal, 0] = branch
    return state


def align_state(state: torch.Tensor, wires: Sequence[int]) -> torch.Tensor:
    width = state.numel().bit_length() - 1
    workspace = tuple(wire for wire in range(width) if wire not in wires)
    return (
        state.reshape((2,) * width)
        .permute((*wires, *workspace))
        .reshape(SIDE, SIDE, 2, 2, 2, 1 << len(workspace))
    )


def run() -> Result:
    photo = load_photo()
    pixels = pixel_amplitudes(photo)
    preparation = prepare_image(pixels.flatten())
    prepared: torch.Tensor = preparation.state().flatten()
    preparation_error = float((prepared - pixels.flatten()).abs().max())

    detector = build_edges()
    coordinates = (*detector.registers["x"], *detector.registers["y"])
    wires = (*coordinates, *detector.registers["flags"])
    width = detector.circuit.n_wires
    circuit = fq.Circuit(width, dtype=torch.complex128)
    circuit.compose(preparation, qubits=coordinates)
    circuit.compose(detector.circuit, qubits=tuple(range(width)))
    resources = estimate_resources(circuit.to_ir())
    if resources.n_qubits > 20 or resources.n_operations > 35_000:
        raise ValueError("The complete detector exceeded its declared resource bounds")

    native = align_state(circuit.state(), wires)
    exact = reference_state(pixels, native.shape[-1])
    state_error = float((native - exact).abs().max())
    work_probability = float(native[..., 1:].abs().square().sum())
    if max(preparation_error, state_error) > 1e-12 or work_probability > 1e-24:
        raise ValueError("State preparation, coherent differences or cleanup disagree")
    joint = native.abs().square().sum(dim=(3, 4, 5))
    edge_probability = float(joint[:, :, 1].sum())

    generator = torch.Generator().manual_seed(SEED)
    outcomes = torch.multinomial(
        joint.flatten(), SHOTS, replacement=True, generator=generator
    )
    counts = torch.bincount(outcomes, minlength=joint.numel()).reshape_as(joint)
    observed_edge = counts[:, :, 1]
    edge_shots = int(observed_edge.sum())
    sampled_edge = (observed_edge.to(torch.float64) / SHOTS).sqrt()
    exact_edge = joint[:, :, 1].sqrt()
    thresholded = torch.where(sampled_edge > THRESHOLD, sampled_edge, 0.0)
    return Result(
        photo,
        threshold_preview(photo),
        exact_edge,
        sampled_edge,
        thresholded,
        edge_probability,
        edge_shots,
        preparation_error,
        state_error,
        work_probability,
        estimate_resources(preparation.to_ir()),
        resources,
    )
