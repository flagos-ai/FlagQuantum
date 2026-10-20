"""Replaceable explicit-basis consumers; only a test fake supplies labels today.

The fake delegates numerical work to existing encoded-product plan/run. It does
not implement labelled plan 1.6, public validation or constrained-space support.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.core.ir.state_space import ExplicitBasisSpace
from flagquantum.simulation.lindblad import EvolutionResult

pytestmark = pytest.mark.unit


class _IntegerAlias(int):
    pass


class _StringAlias(str):
    pass


Label = tuple[int | str, ...]


def _label(value: Sequence[int | str]) -> Label:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("a label must be an ordered sequence")
    result = tuple(value)
    if not result or any(
        type(atom) not in (int, str) or (type(atom) is str and not atom)
        for atom in result
    ):
        raise ValueError("labels require nonempty rows of integer or string atoms")
    return result


class Basis(Protocol):
    @property
    def labels(self) -> tuple[Label, ...]: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_qubit_space(self) -> bool: ...

    def basis_index(self, label: Sequence[int | str]) -> int: ...


@dataclass(frozen=True, slots=True, init=False)
class _ExplicitBasisFake:
    labels: tuple[Label, ...]

    def __init__(self, labels: Sequence[Sequence[int | str]]) -> None:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError("labels must be an ordered sequence")
        rows = tuple(_label(row) for row in labels)
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("labels must be nonempty and have a common arity")
        if len(set(rows)) != len(rows):
            raise ValueError("labels must be unique")
        object.__setattr__(self, "labels", rows)

    @property
    def dimension(self) -> int:
        return len(self.labels)

    @property
    def is_qubit_space(self) -> bool:
        return False

    def basis_index(self, label: Sequence[int | str]) -> int:
        return self.labels.index(_label(label))


BasisFactory = Callable[..., Basis]
EvolutionRunner = Callable[..., EvolutionResult]


@pytest.fixture(
    params=(_ExplicitBasisFake, ExplicitBasisSpace), ids=("contract-fake", "core")
)
def basis_factory(request: pytest.FixtureRequest) -> BasisFactory:
    return request.param


def _encoded_product_run_fake(
    hamiltonian: object,
    initial_state: object,
    times: Sequence[float],
    *,
    basis_labels: Sequence[Sequence[int | str]],
    method: str = "krylov-exponential",
    options: fq.ExecutionOptions | None = None,
    parameter_values: Sequence[Mapping[str | fq.Parameter, object]] | None = None,
) -> EvolutionResult:
    basis = _ExplicitBasisFake(basis_labels)
    return fql.run(
        hamiltonian,
        initial_state,
        times,
        dimensions=(basis.dimension,),
        method=method,
        options=options,
        parameter_values=parameter_values,
        return_density_matrices=True,
    )


@pytest.fixture(
    params=(
        _encoded_product_run_fake,
        partial(fql.run, return_density_matrices=True, method="krylov-exponential"),
    ),
    ids=("encoded-product-fake", "public-labelled-run"),
)
def run_evolution(request: pytest.FixtureRequest) -> EvolutionRunner:
    return request.param


def test_candidate_extends_existing_baseline_without_a_parallel_owner() -> None:
    root = Path(__file__).resolve().parents[2]
    candidate = json.loads(
        (root / "contracts/explicit-basis-internal-v1-candidate.json").read_text()
    )
    original = json.loads((root / candidate["extends_contract"]).read_text())
    assert candidate["status"] == "implementation_authorized"
    assert original["existing_authorities"]["public_ir_version"] == "1.0"
    assert candidate["proposed_api_delta"]["new_stable_root_exports"] == []
    assert (root / candidate["approval_basis"]).is_file()
    assert candidate["proposed_api_delta"]["labelled_plan_version"] == "1.6"


def test_nonproduct_order_and_exact_identifier_values(
    basis_factory: BasisFactory,
) -> None:
    labels = (("psi", "sigma", "1"), ("1", "sigma", "1"))
    space = basis_factory(labels)
    assert space.labels == labels
    assert space.dimension == 2
    assert not space.is_qubit_space
    assert space.basis_index(("psi", "sigma", "1")) == 0
    assert space.basis_index(("1", "sigma", "1")) == 1
    identifiers = basis_factory(((1, "1"), ("1", 1), (-1, " 1 ")))
    assert [identifiers.basis_index(row) for row in identifiers.labels] == [0, 1, 2]
    assert not hasattr(space, "dimensions")


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_label_count_never_infers_qubits(
    basis_factory: BasisFactory, dimension: int
) -> None:
    space = basis_factory(tuple((index,) for index in range(dimension)))
    assert space.dimension == dimension
    assert not space.is_qubit_space
    assert not hasattr(space, "dimensions")


def test_basis_copies_nested_mutable_inputs_and_remains_frozen(
    basis_factory: BasisFactory,
) -> None:
    labels = [[0, 1], [1, 0]]
    space = basis_factory(labels)
    labels[0][0] = 9
    labels.append([2, 0])
    assert space.labels == ((0, 1), (1, 0))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        space.labels = ((9, 1),)
    assert space == basis_factory(((0, 1), (1, 0)))
    assert hash(space) == hash(basis_factory(((0, 1), (1, 0))))
    assert space != basis_factory(((1, 0), (0, 1)))


@pytest.mark.parametrize(
    "labels",
    [
        (),
        ((),),
        ((0,), (0,)),
        ((0,), (1, 0)),
        ((True,),),
        ((1.0,),),
        (("",),),
        ((torch.tensor(1),),),
        ((_IntegerAlias(1),),),
        ((_StringAlias("one"),),),
        (([0],),),
    ],
)
def test_malformed_labels_are_refused(
    basis_factory: BasisFactory, labels: object
) -> None:
    with pytest.raises(ValueError):
        basis_factory(labels)


@pytest.mark.parametrize(
    "labels", ["01", b"01", {"basis": ((0,),)}, iter(((0,),)), ((0,), "1")]
)
def test_unordered_or_scalar_label_inputs_are_refused(
    basis_factory: BasisFactory, labels: object
) -> None:
    with pytest.raises(TypeError):
        basis_factory(labels)


@pytest.mark.parametrize("label", [(2, 0), (True, 1), (0,), (0.0, 1), ("0", 1)])
def test_lookup_never_coerces_or_fabricates_labels(
    basis_factory: BasisFactory, label: object
) -> None:
    with pytest.raises(ValueError):
        basis_factory(((0, 1), (1, 0))).basis_index(label)


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_fixed_number_transfer_and_control_derivative_use_one_engine(
    basis_factory: BasisFactory, run_evolution: EvolutionRunner, method: str
) -> None:
    space = basis_factory(((0, 1), (1, 0)))
    coupling = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    exchange = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    result = run_evolution(
        coupling * exchange,
        [1.0, 0.0],
        [index / 200 for index in range(101)],
        basis_labels=space.labels,
        method=method,
        options=fq.ExecutionOptions(require_gradients=True),
    )
    probability = result.populations[-1, space.basis_index((1, 0))]
    (gradient,) = torch.autograd.grad(probability, coupling)
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    assert probability.item() == pytest.approx(math.sin(0.2) ** 2, abs=tolerance)
    assert gradient.item() == pytest.approx(0.5 * math.sin(0.4), abs=tolerance)
    assert result.populations.shape == (101, 2)


def test_fusion_basis_consumer_preserves_coordinate_order(
    basis_factory: BasisFactory, run_evolution: EvolutionRunner
) -> None:
    space = basis_factory((("1", "sigma", "1"), ("psi", "sigma", "1")))
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    f = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2)
    r = torch.diag(
        torch.exp(torch.tensor([-1j, 3j], dtype=torch.complex128) * math.pi / 8)
    )
    braid = f.mH @ r @ f
    result = run_evolution(
        math.pi / 4 * x - math.pi / 8 * identity,
        [1.0, 0.0],
        [0.0, 1.0],
        basis_labels=space.labels,
    )
    assert result.density_matrices is not None
    state = braid[:, 0]
    torch.testing.assert_close(
        result.density_matrices[-1],
        torch.outer(state, state.conj()),
        atol=1e-11,
        rtol=0,
    )
    assert result.populations[
        -1, space.basis_index(("psi", "sigma", "1"))
    ].item() == pytest.approx(0.5, abs=1e-11)
    permutation = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    reordered_space = basis_factory(tuple(reversed(space.labels)))
    reordered = run_evolution(
        permutation @ (math.pi / 4 * x - math.pi / 8 * identity) @ permutation.mH,
        [0.0, 1.0],
        [0.0, 1.0],
        basis_labels=reordered_space.labels,
    )
    assert reordered.density_matrices is not None
    torch.testing.assert_close(
        permutation @ result.density_matrices[-1] @ permutation.mH,
        reordered.density_matrices[-1],
        atol=1e-11,
        rtol=0,
    )


def test_scheduled_batch_gradient_and_optimizer_keep_existing_torch_path(
    run_evolution: EvolutionRunner,
) -> None:
    coupling = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
    exchange = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    terms = [{"operator": exchange, "coefficient": fq.Parameter("g")}]
    optimizer = torch.optim.SGD([coupling], lr=0.2)

    def transfer() -> torch.Tensor:
        result = run_evolution(
            terms,
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
            [0.0, 0.1, 0.3],
            basis_labels=((0, 1), (1, 0)),
            parameter_values=[{"g": coupling}, {"g": coupling}],
            options=fq.ExecutionOptions(require_gradients=True, batch_size=2),
        )
        assert result.populations.shape == (3, 2, 2)
        return result.populations[-1, 0, 1] + result.populations[-1, 1, 0]

    before = transfer()
    (gradient,) = torch.autograd.grad(before, coupling)
    assert gradient.item() == pytest.approx(0.6 * math.sin(0.24), abs=1e-10)
    optimizer.zero_grad()
    (2 - transfer()).backward()
    optimizer.step()
    assert transfer().detach().item() > before.detach().item()
