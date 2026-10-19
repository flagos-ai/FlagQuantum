"""Contract-first conformance and numerical reuse evidence for M1.

The test-only fake is not a production mixed-dimension API. A Core
implementation must replace its factory without changing these assertions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Protocol

import pytest
import torch

from flagquantum.core.ir.state_space import ProductStateSpace
from flagquantum.observables.boson import annihilate
from flagquantum.simulation.lindblad_generator import Liouvillian
from flagquantum.simulation.lindblad_integrators import (
    INTEGRATOR_NAMES,
    advance_interval,
)

pytestmark = pytest.mark.unit


class ProductSpace(Protocol):
    @property
    def dimensions(self) -> tuple[int, ...]: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_qubit_space(self) -> bool: ...

    def basis_index(self, occupations: Sequence[int]) -> int: ...


@dataclass(frozen=True)
class _ProductSpaceFake:
    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, Sequence) or isinstance(
            self.dimensions, (str, bytes)
        ):
            raise TypeError("dimensions must be an ordered sequence of integers")
        dimensions = tuple(self.dimensions)
        if not dimensions or any(
            type(item) is not int or item < 1 for item in dimensions
        ):
            raise ValueError("dimensions must contain positive integers")
        object.__setattr__(self, "dimensions", dimensions)

    @property
    def dimension(self) -> int:
        return math.prod(self.dimensions)

    @property
    def is_qubit_space(self) -> bool:
        return all(dimension == 2 for dimension in self.dimensions)

    def basis_index(self, occupations: Sequence[int]) -> int:
        if len(occupations) != len(self.dimensions):
            raise ValueError("one occupation is required per subsystem")
        index = 0
        for occupation, dimension in zip(occupations, self.dimensions, strict=True):
            if type(occupation) is not int or not 0 <= occupation < dimension:
                raise ValueError("occupation is outside the subsystem basis")
            index = index * dimension + occupation
        return index


@pytest.fixture(params=(_ProductSpaceFake, ProductStateSpace), ids=("contract-fake", "core"))
def space_factory(request) -> Callable[[tuple[int, ...]], ProductSpace]:
    return request.param


def test_candidate_preserves_existing_owners_and_public_ir():
    from flagquantum.core.ir import IR_VERSION

    root = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (root / "contracts/product-state-space-internal-v1-candidate.json").read_text()
    )
    assert contract["status"] == "implementation_authorized"
    assert contract["version"] == "0.1"
    assert contract["representation"]["basis_order"] == "subsystem_0_most_significant"
    assert contract["existing_authorities"]["public_ir_version"] == IR_VERSION == "1.0"
    assert (
        contract["existing_authorities"]["generator"]
        == "flagquantum.simulation.lindblad_generator.Liouvillian"
    )
    assert contract["proposed_api_delta"]["new_stable_root_exports"] == []
    assert (root / contract["approval_basis"]).is_file()


@pytest.mark.parametrize(
    ("dimensions", "dimension"),
    [((3,), 3), ((2, 3), 6), ((3, 2), 6), ((1, 3), 3), ((1,), 1)],
)
def test_product_dimension(space_factory, dimensions, dimension):
    space = space_factory(dimensions)
    assert space.dimensions == dimensions
    assert space.dimension == dimension


def test_basis_order_matches_existing_boson_matrices(space_factory):
    space = space_factory((2, 3))
    assert [
        space.basis_index((spin, level)) for spin in range(2) for level in range(3)
    ] == list(range(6))
    matrix = annihilate(1).to_matrix({0: 2, 1: 3})
    source = space.basis_index((1, 2))
    destination = space.basis_index((1, 1))
    assert matrix[destination, source] == pytest.approx(math.sqrt(2))
    assert torch.count_nonzero(matrix[:, source]) == 1


def test_total_dimension_does_not_determine_subsystem_structure(space_factory):
    qubits = space_factory((2, 2))
    ququart = space_factory((4,))
    assert qubits.dimension == ququart.dimension == 4
    assert qubits.is_qubit_space
    assert not ququart.is_qubit_space
    assert not space_factory((1, 2)).is_qubit_space
    assert space_factory((2, 3)).dimensions != space_factory((3, 2)).dimensions


def test_space_is_immutable(space_factory):
    space = space_factory((2, 3))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        space.dimensions = (3, 2)


@pytest.mark.parametrize("dimensions", [(), (0,), (-1, 2), (True, 3), (2.5,), ("3",)])
def test_invalid_dimensions_are_refused(space_factory, dimensions):
    with pytest.raises(ValueError, match="positive integers"):
        space_factory(dimensions)


@pytest.mark.parametrize("occupations", [(0,), (0, 3), (2, 0), (-1, 0), (True, 0)])
def test_invalid_basis_occupations_are_refused(space_factory, occupations):
    with pytest.raises(ValueError):
        space_factory((2, 3)).basis_index(occupations)


def test_fixed_charge_fusion_space_is_not_implicitly_a_product(space_factory):
    # Four sigma anyons with total charge 1 have these sequential fusion paths.
    fusion_paths = (("1", "sigma", "1"), ("psi", "sigma", "1"))
    assert len(fusion_paths) == 2
    descriptor = {"basis": fusion_paths, "total_charge": "1"}
    with pytest.raises(TypeError, match="ordered sequence"):
        space_factory(descriptor)


@pytest.mark.parametrize("dimensions", [(3,), (2, 3)])
@pytest.mark.parametrize("method", INTEGRATOR_NAMES)
def test_existing_dynamics_core_matches_two_analytic_models(
    space_factory, dimensions, method
):
    space = space_factory(dimensions)
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    if dimensions == (3,):
        operator = torch.zeros((3, 3), dtype=torch.complex128)
        operator[1, 2] = operator[2, 1] = 1
        initial, target = 1, 2
    else:
        lower = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
        ladder = annihilate(0).to_matrix({0: 3})
        operator = torch.kron(lower.T.contiguous(), ladder) + torch.kron(
            lower, ladder.T.contiguous()
        )
        initial = space.basis_index((1, 0))
        target = space.basis_index((0, 1))
    generator = Liouvillian(coupling * operator, [], hilbert_dimension=space.dimension)
    state = torch.zeros((space.dimension, space.dimension), dtype=torch.complex128)
    state[initial, initial] = 1
    for _ in range(50):
        state, residual = advance_interval(
            generator.derivative,
            state,
            torch.tensor(0.01, dtype=torch.float64),
            method=method,
            solve_tolerance=1e-12,
            solve_max_iterations=256,
        )
        assert residual <= 1e-12
    population = state[target, target].real
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    assert float(population.detach()) == pytest.approx(
        math.sin(0.35) ** 2, abs=tolerance
    )
    assert torch.trace(state).item() == pytest.approx(1.0, abs=1e-12)
    if method == "runge-kutta":
        (gradient,) = torch.autograd.grad(population, coupling)
        assert gradient.item() == pytest.approx(0.5 * math.sin(0.7), abs=1e-9)
