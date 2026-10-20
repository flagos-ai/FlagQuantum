"""The variational problem as one value, and the reuse it is built out of.

Every assertion here is about *which* pieces belong to one problem, never about
how well an optimizer did. Two of them are bitwise instead of approximate on
purpose: the solver's descent and the core loop must return the same tensor, and
the solver's objective and the core loss must return the same tensor, because
"this module adds a surface and not an arithmetic" is only a claim while the two
paths agree to the last bit. An accuracy assertion would pass just as well if
this module had grown its own optimizer.

Where a reported energy does appear it is compared with an inequality against the
reference start the molecule computed, so the assertion survives a different
BLAS kernel and a different reduction order.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from flagquantum.algorithms import (
    Ansatz,
    CobylaOptimizer,
    Hamiltonian,
    NelderMeadOptimizer,
    VQEResult,
    VQESolver,
    hardware_efficient_parameter_count,
    pauli_term,
    run_vqe,
    uccsd_excitations,
    vqe_loss,
    vqe_objective,
)
from flagquantum.algorithms.molecular import (
    MolecularHamiltonian,
    create_molecular_hamiltonian,
)
from flagquantum.circuit import Circuit
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit


def _molecule() -> MolecularHamiltonian:
    """Return H2 at its equilibrium separation, in a four-spin-orbital space."""

    return create_molecular_hamiltonian(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    )


def _ry_cx_ry(parameters: torch.Tensor) -> Circuit:
    """A three-angle builder this package does not ship as a factory."""

    return (
        Circuit(2, device="cpu", dtype=torch.complex128)
        .gate("ry", 0, theta=parameters[0])
        .gate("cx", (0, 1))
        .gate("ry", 1, theta=parameters[1])
        .gate("ry", 0, theta=parameters[2])
    )


def test_the_solver_descent_is_the_core_loop() -> None:
    """One problem's run returns the tensor the core loop returns.

    Bitwise, so this fails if the solver ever grows a step of its own -- an extra
    evaluation, a different cast, or a re-implemented update -- rather than only
    if it drifts numerically.
    """

    molecule = _molecule()
    solver = VQESolver.for_molecule(molecule, dtype=torch.complex128, steps=5)
    start = torch.zeros(solver.ansatz.parameter_count)

    from_solver = solver.run(start)
    from_core = run_vqe(solver.ansatz, start, solver.operator, steps=5, lr=solver.lr)

    assert isinstance(from_solver, VQEResult)
    assert torch.equal(from_solver.energy, from_core.energy)
    assert torch.equal(from_solver.parameters, from_core.parameters)
    assert from_solver.history == from_core.history


def test_the_solver_objective_is_the_core_loss() -> None:
    """The hand-off objective returns the loss, not a wrapper around it."""

    solver = VQESolver.for_molecule(_molecule(), dtype=torch.complex128)
    angles = torch.tensor([0.11, -0.23, 0.37], dtype=torch.float32)

    handoff = solver.objective()
    direct = vqe_loss(solver.ansatz, angles, solver.operator)

    assert torch.equal(handoff(angles), direct)


def test_the_objective_keeps_the_gradient_the_loss_carries() -> None:
    """A gradient-free caller reads a scalar; an autograd caller still descends it.

    The same callable serves both, so handing the energy to a simplex does not cost
    the differentiability that made it worth reusing. The gradient is asserted to
    be present and finite rather than to have a value, because its value is a
    property of this instance and not of the hand-off.
    """

    objective = vqe_objective(
        Ansatz.hardware_efficient(2, 1, rotations=("ry",)),
        Hamiltonian([pauli_term(1.0, "Z", (0,)), pauli_term(1.0, "X", (0,))]),
    )
    angles = torch.tensor([0.3, -0.7], dtype=torch.float32, requires_grad=True)

    value = objective(angles)
    value.backward()
    plain = objective(torch.tensor([0.3, -0.7], dtype=torch.float32))

    assert value.grad_fn is not None
    assert angles.grad is not None
    assert bool(torch.isfinite(angles.grad).all())
    assert torch.equal(value.detach(), plain)


def test_the_ansatz_count_is_the_count_its_builder_accepts() -> None:
    """The declared count and the builder's own length agree, from one source.

    Both halves are derived from ``uccsd_excitations``, so a count that disagreed
    would be a second enumeration of the excitation set rather than a typo.
    """

    ansatz = Ansatz.uccsd(4, 2)
    declared = uccsd_excitations(2, 4).parameter_count

    assert ansatz.parameter_count == declared
    ansatz(torch.zeros(declared))
    with pytest.raises(ValueError):
        ansatz(torch.zeros(declared - 1))
    with pytest.raises(ValueError):
        ansatz(torch.zeros(declared + 1))


def test_the_hardware_efficient_count_follows_the_rotations_it_applies() -> None:
    """Widening the rotation set widens the count, which widens the builder."""

    narrow = Ansatz.hardware_efficient(3, 2, rotations=("ry",))
    wide = Ansatz.hardware_efficient(3, 2, rotations=("ry", "rz"))

    assert narrow.parameter_count == hardware_efficient_parameter_count(3, 2, ("ry",))
    assert wide.parameter_count == 2 * narrow.parameter_count
    wide(torch.zeros(wide.parameter_count))
    with pytest.raises(ValueError):
        wide(torch.zeros(narrow.parameter_count))


def test_an_ansatz_this_package_does_not_ship_is_configuration() -> None:
    """A caller's own builder is an ``Ansatz``, with no registration step.

    This is the pluggability the row's baseline gap names, and it is a
    constructor rather than a registry: there is no table to add a name to, so
    there is also no table that could disagree with the builder about its length.
    """

    ansatz = Ansatz(name="ry_cx_ry", parameter_count=3, builder=_ry_cx_ry)
    operator = Hamiltonian(
        [
            pauli_term(1.0, "Z", (0,)),
            pauli_term(1.0, "Z", (1,)),
            pauli_term(1.0, "X", (0,)),
        ]
    )
    solver = VQESolver(operator=operator, ansatz=ansatz, steps=20, lr=0.1)

    result = solver.run(torch.zeros(3))

    assert solver.ansatz.name == "ry_cx_ry"
    assert result.parameters.shape == (3,)
    assert float(result.energy) < 0.0


def test_a_builder_that_returns_a_state_is_refused_by_name() -> None:
    """The builder's return type is checked, because a tensor is scorable wrongly.

    An expectation reads a bare tensor as a statevector or as a density matrix by
    its shape, and a ``2x2`` batched statevector is read as a density matrix, so a
    builder that skipped its ``Circuit`` would be scored without anything saying
    the two disagreed.
    """

    ansatz = Ansatz(
        name="returns_a_state",
        parameter_count=1,
        builder=lambda angles: torch.zeros(4, dtype=torch.complex128),
    )

    with pytest.raises(TypeError, match="returns_a_state"):
        ansatz(torch.zeros(1))


def test_the_energy_handoff_refuses_a_foreign_operator() -> None:
    """An ``Observable`` is refused with the conversion named in the message.

    The two Pauli-sum types are not interchangeable: one is what a run requests and
    the other is what these loops evaluate, so a caller who passes the first is
    told which method converts it rather than getting a type error from inside a
    term loop.
    """

    with pytest.raises(TypeError, match="from_observable"):
        vqe_objective(
            Ansatz.uccsd(4, 2),
            _molecule().hamiltonian,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Ansatz"):
        vqe_objective(
            _ry_cx_ry,
            Hamiltonian([pauli_term(1.0, "Z", (0,))]),  # type: ignore[arg-type]
        )


def test_the_angles_are_one_flat_vector_of_the_declared_length() -> None:
    """A nested sequence and a wrong count are each refused, and neither is flattened."""

    objective = vqe_objective(
        Ansatz.uccsd(4, 2),
        Hamiltonian.from_observable(_molecule().hamiltonian),
    )

    with pytest.raises(ValueError, match="one flat vector"):
        objective(torch.zeros(3, 1))
    with pytest.raises(ValueError, match="takes 3 angles"):
        objective(torch.zeros(4))


def test_a_gradient_free_optimizer_reaches_below_the_reference_start() -> None:
    """A simplex consumes the hand-off objective with no adapter at all.

    The assertion is a strict inequality against the Hartree-Fock energy the
    molecule computed, not a distance from the exact one, so it does not depend on
    how many evaluations this build's simplex spends.
    """

    molecule = _molecule()
    solver = VQESolver.for_molecule(molecule, dtype=torch.complex128)
    optimizer = NelderMeadOptimizer(maxiter=200)

    result = optimizer.minimize(
        solver.objective(), torch.zeros(solver.ansatz.parameter_count)
    )

    assert optimizer.evaluations > solver.ansatz.parameter_count
    assert float(result.value) < molecule.hf_energy


def test_the_constrained_optimizer_reports_an_optimum_it_could_not_reach() -> None:
    """A cap on one angle leaves a feasible answer above the exact ground energy.

    The constrained member of the family is the one the autograd entry points
    cannot stand in for, and the point of the assertion is that the run reports
    both halves: it is feasible, and its energy is worse than the unconstrained
    one. A silent penalty would report only the first.
    """

    molecule = _molecule()
    solver = VQESolver.for_molecule(molecule, dtype=torch.complex128)

    capped = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
        solver.objective(),
        torch.zeros(solver.ansatz.parameter_count),
        constraints=[lambda angles: 0.1 - angles[2]],
    )
    unconstrained = solver.run(torch.zeros(solver.ansatz.parameter_count))

    assert capped.feasible is True
    assert capped.residual == 0.0
    # The angle is float32, so the cap is compared with the rounding of 0.1 into
    # that dtype rather than with the decimal literal.
    assert float(capped.parameters[2]) <= 0.1 + torch.finfo(torch.float32).eps
    assert float(capped.value) > float(unconstrained.energy)


def test_the_operator_is_configuration_too() -> None:
    """Varying one field with ``dataclasses.replace`` leaves the rest identical.

    The optimizer and the step budget are the two pieces a caller sweeps, so they
    are fields rather than arguments: replacing them cannot silently rebuild the
    operator or the ansatz.
    """

    solver = VQESolver.for_molecule(_molecule(), dtype=torch.complex128)
    wider = dataclasses.replace(solver, steps=3, lr=0.2)

    assert wider.operator is solver.operator
    assert wider.ansatz is solver.ansatz
    assert (wider.steps, wider.lr) == (3, 0.2)
    assert len(wider.run(torch.zeros(3)).history) == 3


def test_for_molecule_takes_the_electron_count_from_the_molecule() -> None:
    """The ansatz's space and the operator's space are the molecule's, not arguments.

    There is deliberately no ``n_electrons`` parameter on ``for_molecule``: the
    count comes from the object that also holds the operator, so the two cannot be
    chosen independently.
    """

    molecule = _molecule()
    solver = VQESolver.for_molecule(molecule, dtype=torch.complex128)

    assert (
        solver.ansatz.parameter_count
        == uccsd_excitations(
            molecule.n_electrons, 2 * molecule.n_orbitals
        ).parameter_count
    )
    assert solver.operator.n_qubits == 2 * molecule.n_orbitals
    assert len(solver.operator.terms) == len(molecule.hamiltonian.terms)


def test_for_molecule_refuses_a_value_that_carries_no_electron_count() -> None:
    """The constructor is about the molecule object, not about any Pauli sum.

    A bare ``Observable`` has an operator and no electron count, so building a
    UCCSD ansatz from it would have to guess the count the ansatz is built from.
    """

    with pytest.raises(TypeError, match="MolecularHamiltonian"):
        VQESolver.for_molecule(_molecule().hamiltonian)  # type: ignore[arg-type]


def test_from_observable_carries_every_term_and_no_offset_field() -> None:
    """The conversion is total, including the identity term the mapping adds.

    A mapped electronic Hamiltonian's constant shift is a term with no factors
    here rather than a separate offset, so a caller who reads ``n_terms`` sees the
    whole operator.
    """

    molecule = _molecule()
    operator = Hamiltonian.from_observable(molecule.hamiltonian)

    assert len(operator.terms) == len(molecule.hamiltonian.terms)
    assert any(not term.ops for term in operator.terms)
    assert operator.n_qubits == 2 * molecule.n_orbitals


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("name", "", TypeError),
        ("name", "   ", TypeError),
        ("parameter_count", 0, ValidationError),
        ("parameter_count", -1, ValidationError),
        ("parameter_count", 2.5, TypeError),
        ("builder", 3, TypeError),
    ],
)
def test_an_ansatz_refuses_configuration_it_cannot_honour(
    field: str, value: object, expected: type[Exception]
) -> None:
    """Each refusal names a value that a variational loop could not run."""

    kwargs: dict[str, object] = {
        "name": "ansatz",
        "parameter_count": 2,
        "builder": _ry_cx_ry,
    }
    kwargs[field] = value

    with pytest.raises(expected):
        Ansatz(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("steps", 0, ValidationError),
        ("steps", 1.5, TypeError),
        ("lr", 0.0, ValidationError),
        ("lr", -0.1, ValidationError),
        ("lr", float("nan"), ValidationError),
        ("lr", "fast", TypeError),
        ("optimizer_factory", 7, TypeError),
    ],
)
def test_a_solver_refuses_configuration_it_cannot_honour(
    field: str, value: object, expected: type[Exception]
) -> None:
    """A zero step count and a non-positive rate each return the start as an optimum."""

    kwargs: dict[str, object] = {
        "operator": Hamiltonian([pauli_term(1.0, "Z", (0,))]),
        "ansatz": Ansatz(name="one_angle", parameter_count=1, builder=_ry_cx_ry),
    }
    kwargs[field] = value

    with pytest.raises(expected):
        VQESolver(**kwargs)  # type: ignore[arg-type]


def test_the_package_surface_names_the_module_the_names_come_from() -> None:
    """Each name resolves to the module that owns it, and to no other."""

    assert Ansatz.__module__ == "flagquantum.algorithms.solvers"
    assert VQESolver.__module__ == "flagquantum.algorithms.solvers"
    assert vqe_objective.__module__ == "flagquantum.algorithms.solvers"
