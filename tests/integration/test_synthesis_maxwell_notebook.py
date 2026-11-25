"""Original-grid Maxwell predicates, coherent evolution and fresh phase fitting."""

import json
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

if find_spec("numpy") is None:
    pytest.skip(
        "The phase factorization requires optional NumPy", allow_module_level=True
    )

import flagquantum as fq
from examples.synthesis.applications.physical_systems.maxwell_equation.maxwell_2d_simulation.overview import (
    controls,
    phases,
    physics,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def random_state(width: int, seed: int) -> torch.Tensor:
    value = torch.randn(
        1 << width,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(seed),
    )
    value /= value.norm()
    return value


@pytest.mark.parametrize("count", [3, 5, 12])
def test_borrowed_control_preserves_every_complex_amplitude(count: int) -> None:
    width = count + 2
    initial = random_state(width, count + 17)
    circuit = fq.Circuit(width, dtype=torch.complex128, inputs=initial)
    controls.controlled_x(circuit, tuple(range(count)), count, count + 1)
    indices = torch.arange(1 << width)
    selected = indices >> (width - count) == (1 << count) - 1
    permutation = indices.clone()
    permutation[selected] ^= 2
    assert (
        torch.max(torch.abs(circuit.state().reshape(-1) - initial[permutation])) < 1e-12
    )


def test_conductor_xor_axes_rectangle_and_entangled_borrowed_storage() -> None:
    initial = random_state(16, 127)
    circuit = fq.Circuit(16, dtype=torch.complex128, inputs=initial)
    quantum.conductor(circuit)
    indices = torch.arange(1 << 16)
    x, y, field = indices & 31, (indices >> 5) & 31, (indices >> 10) & 3
    selected = (field == 0) & (
        ((x == 0) ^ (y == 0)) ^ ((x >= 9) & (x < 24) & (y >= 9) & (y < 24))
    )
    permutation = indices.clone()
    permutation[selected] ^= 1 << 12
    assert (
        torch.max(torch.abs(circuit.state().reshape(-1) - initial[permutation])) < 1e-12
    )


@pytest.mark.parametrize("seed", [17, 29])
def test_full_grid_block_against_independent_finite_differences(seed: int) -> None:
    data = random_state(12, seed)
    initial = torch.zeros(1 << 16, dtype=torch.complex128)
    initial[:4096] = data
    actual = (
        fq.Circuit(16, dtype=torch.complex128, inputs=initial)
        .compose(quantum.block_encoding())
        .state()
        .reshape(16, 4096)[0]
    )
    assert torch.max(torch.abs(actual - physics.apply(data) / 4)) < 1e-12


def test_controlled_walk_preserves_relative_phase_and_inactive_branch() -> None:
    initial = random_state(17, 71)
    actual = (
        fq.Circuit(17, dtype=torch.complex128, inputs=initial)
        .compose(quantum.controlled_walk())
        .state()
        .reshape(2, -1)
    )
    expected = (
        fq.Circuit(16, dtype=torch.complex128, inputs=initial.reshape(2, -1)[1])
        .compose(quantum.block_encoding())
        .state()
        .reshape(16, 4096)
        * 1j
    )
    expected[1:] *= -1
    assert torch.max(torch.abs(actual[0] - initial.reshape(2, -1)[0])) < 1e-12
    assert torch.max(torch.abs(actual[1] - expected.reshape(-1))) < 1e-12


@pytest.mark.parametrize(
    "guards", [(), ((0, 0), (0, 1)), ((12, 0),), ((16, 1),), ((1, 2),)]
)
def test_invalid_predicate_fails_before_partial_mutation(
    guards: tuple[tuple[int, int], ...],
) -> None:
    circuit = fq.Circuit(16, dtype=torch.complex128)
    before = circuit.to_ir()
    with pytest.raises(ValueError, match="disjoint in-bounds"):
        quantum.toggle_pattern(circuit, guards)
    assert circuit.to_ir() == before


def test_fresh_original_degree_and_original_size_workflow() -> None:
    fitted = phases.fit()
    assert len(fitted.rows) == 1373
    result = workflow.run(fitted)
    assert result.state.shape == (1 << 17,)
    assert result.projected.shape == (4096,)
    assert result.operations_executed == 3_272_222
    assert abs(float(result.state.norm()) - 1) < 1e-10
    assert abs(result.projection_norm - 0.5) < 1e-6
    assert result.absolute_error < 1e-6
    assert result.fidelity > 1 - 1e-4
    assert result.classical_defect <= 1e-13
    assert torch.max(torch.abs(result.initial - physics.initial_state())) == 0
    assert (
        torch.max(torch.abs(result.projected - phases.SCALE * result.classical)) < 1e-6
    )


def test_all_original_stages_have_unexecuted_native_cells() -> None:
    base = Path(workflow.__file__).parent
    proof = json.loads((base / "provenance.json").read_text())
    notebook = json.loads((base / "walkthrough.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 20
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
