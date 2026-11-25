"""Check full-state evolution and the original Fermi–Hubbard analytic limits."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def independent_exchange(
    state: torch.Tensor, first: int, second: int, angle: float
) -> torch.Tensor:
    result = state.clone()
    words = torch.arange(256)
    left = ((words >> first) & 1) != 0
    right = ((words >> second) & 1) != 0
    only_left = words[left & ~right]
    only_right = only_left ^ ((1 << first) | (1 << second))
    c, s = math.cos(angle), -1j * math.sin(angle)
    result[only_left] = c * state[only_left] + s * state[only_right]
    result[only_right] = s * state[only_left] + c * state[only_right]
    return result


def independent_step(
    state: torch.Tensor, hopping: float, interaction: float
) -> torch.Tensor:
    words = torch.arange(256)

    def apply_contacts(sites: tuple[int, int], value: torch.Tensor) -> torch.Tensor:
        hits = torch.zeros_like(words)
        for site in sites:
            hits += ((words >> site) & 1) * ((words >> (4 + site)) & 1)
        return value * torch.exp(1j * interaction * hits.to(torch.float64))

    for spin_offset in (0, 4):
        for site in (0, 2):
            state = independent_exchange(
                state, spin_offset + site, spin_offset + site + 1, -hopping
            )
    state = apply_contacts((0, 2), state)
    for site in (1, 5):
        state = independent_exchange(state, site, site + 1, -math.pi / 2)
    state = apply_contacts((2, 3), state)
    for site in (1, 5):
        state = independent_exchange(state, site, site + 1, -hopping + math.pi / 2)
    return state


@pytest.mark.parametrize(
    "case,points,electrons,tau,hopping,contact",
    [
        ("interacting", 10, 2, 0.1, 1.0, 3.0),
        ("free", 11, 1, 0.1, 1.0, 0.0),
        ("frozen", 11, 4, 0.3, 0.0, 2.0),
    ],
)
def test_all_time_points_match_independent_exchange(
    case: str, points: int, electrons: int, tau: float, hopping: float, contact: float
) -> None:
    if case == "free":
        occupied = workflow.localized()
    else:
        matrix = (
            model.hamiltonian(3.0, 0.5)
            if case == "interacting"
            else model.hamiltonian(10.0, 4 / 6)
        )
        occupied = workflow.diagonalize(matrix, electrons).vectors
    observed = workflow.evolve(occupied, points, tau, hopping, contact)
    expected = model.expected_state(occupied)
    for vector in observed.states:
        torch.testing.assert_close(vector, expected, rtol=0, atol=2e-12)
        expected = independent_step(expected, tau * hopping, tau * contact)
    assert observed.program.report["workspace_qubits"] == 0
    assert len(observed.program.registers["q"]) == 8
    torch.testing.assert_close(
        observed.charge.sum(1),
        torch.full((points,), float(electrons), dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    if case == "frozen":
        torch.testing.assert_close(
            observed.charge,
            observed.charge[0].expand_as(observed.charge),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            observed.spin, observed.spin[0].expand_as(observed.spin), rtol=0, atol=1e-12
        )
    if case == "free":
        _, numeric, _ = workflow.one_particle_curves(observed.times)
        assert float((numeric - observed.modes[:, :4]).abs().max()) < 0.04
        assert float(observed.modes[:, 4:].abs().max()) < 1e-12


def test_demo_reconstruction_and_single_particle() -> None:
    demo = workflow.demo_orbitals()
    assert len(demo.energies) == 4 and demo.vectors.shape == (4, 2)
    assert demo.reconstruction_error < 1e-13
    trajectory, determinant, fidelity = workflow.initial_one_electron()
    assert fidelity == pytest.approx(1.0, abs=1e-13)
    torch.testing.assert_close(trajectory.states[0], determinant, rtol=0, atol=1e-12)


@pytest.mark.parametrize("steps", [-1, 11])
def test_out_of_scope_time_budget(steps: int) -> None:
    with pytest.raises(ValueError, match="zero to ten"):
        quantum.build_program(model.factor(workflow.localized()), steps, 0.1, 0.3)


def test_plot_grid_and_notebook_stages() -> None:
    times = torch.linspace(0, 4, 400, dtype=torch.float64)
    _, numeric, exact = workflow.one_particle_curves(times)
    assert numeric.shape == (400, 4)
    torch.testing.assert_close(numeric, exact, rtol=0, atol=1e-14)
    root = Path(model.__file__).parent
    n = json.loads((root / "walkthrough.ipynb").read_text())
    p = json.loads((root / "provenance.json").read_text())
    cells = [c for c in n["cells"] if c["cell_type"] == "code"]
    assert len(cells) == len(p["reviewed_code_cells"]) == 39
    assert all(c["execution_count"] is None and not c["outputs"] for c in cells)
