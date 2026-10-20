"""Variational solvers whose operator, ansatz and optimizer are configuration.

The entry points in :mod:`flagquantum.algorithms.core` take a circuit builder, a
start vector and an operator as three separate arguments at every call.  That is
the shape one loop wants.  A caller who runs the same problem from several
starts, or who wants the same energy inside an optimizer whose calling
convention is ``minimize(objective, start)``, has to hold the three together and
rebuild the same closure each time -- and the closure is where an operator and
an ansatz drift apart, because nothing in either one names the other.

This module binds the ansatz into one value, :class:`Ansatz`, and the whole
problem into one value, :class:`VQESolver`, which is the surface shape the
baseline's solver library has: the ansatz and the optimizer are configuration
rather than arguments.  Nothing here re-implements the arithmetic beside it.
:meth:`VQESolver.run` is :func:`~flagquantum.algorithms.core.run_vqe`,
:meth:`VQESolver.objective` is :func:`~flagquantum.algorithms.core.vqe_loss`,
and the ansatz factories are
:func:`~flagquantum.algorithms.chemistry.uccsd_ansatz` and
:func:`~flagquantum.algorithms.core.hardware_efficient_ansatz` together with
their parameter counts.  What this module decides is only which pieces belong to
one problem, and it decides that once instead of once per call site.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from ..circuit import Circuit
from ..errors import ValidationError
from .chemistry import uccsd_ansatz, uccsd_excitations
from .core import (
    Hamiltonian,
    OptimizerFactory,
    VQEResult,
    hardware_efficient_ansatz,
    hardware_efficient_parameter_count,
    run_vqe,
    vqe_loss,
)
from .molecular import MolecularHamiltonian


@dataclass(frozen=True)
class Ansatz:
    """One variational ansatz as a value: a builder and its parameter count.

    The count is carried rather than asked of the builder, because a builder is a
    function of one angle vector and nothing else: it cannot say how many angles
    it needs without being called, and calling it at a guessed length raises from
    inside the gate sequence instead of at the boundary.  So the count is a field,
    and the two are kept consistent at construction rather than at each call.

    An instance is any callable of one flat angle vector that returns the circuit
    those angles prepare, so an ansatz this package does not ship is configuration
    too::

        Ansatz(name="ry_cx_ry", parameter_count=3, builder=my_builder)

    There is no name-to-factory table, deliberately.  A table would be a second
    place that knows how many angles an ansatz takes, and the count above would
    then have two sources.  :meth:`uccsd` and :meth:`hardware_efficient` are
    constructors of this type instead, so a caller who wants another ansatz
    supplies it rather than registering it.

    Examples:
        The two factories are the ansätze this package already ships, and the
        count each reports is the count its own builder accepts:

        >>> from flagquantum.algorithms.solvers import Ansatz
        >>> Ansatz.uccsd(4, 2).parameter_count
        3
        >>> Ansatz.hardware_efficient(2, 1).parameter_count
        4
    """

    name: str
    parameter_count: int
    builder: Callable[[torch.Tensor], Circuit]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("an ansatz name must be a non-empty string")
        if isinstance(self.parameter_count, bool) or not isinstance(
            self.parameter_count, int
        ):
            raise TypeError(
                "an ansatz parameter count must be an integer, and "
                f"{self.parameter_count!r} was given"
            )
        if self.parameter_count < 1:
            raise ValidationError(
                "an ansatz must take at least one angle: a variational loop over "
                "no parameters has nothing to optimize, and "
                f"{self.name!r} declared {self.parameter_count}"
            )
        if not callable(self.builder):
            raise TypeError(
                f"the builder of ansatz {self.name!r} must be callable and "
                f"{type(self.builder).__name__} was given"
            )

    def __call__(self, parameters: torch.Tensor) -> Circuit:
        """Return the circuit this ansatz prepares at ``parameters``.

        The returned object is checked rather than assumed.  A builder that
        returns a bare state tensor is refused by name, because
        :meth:`~flagquantum.algorithms.core.Hamiltonian.expectation` reads a
        tensor as a statevector or as a density matrix by its shape -- a batched
        statevector whose batch size equals its width is read as a density matrix
        -- so a builder that produced the wrong one would be scored without
        anything reporting the confusion.

        Raises:
            TypeError: If the builder returns something other than a
                :class:`~flagquantum.Circuit`.
        """

        circuit = self.builder(parameters)
        if not isinstance(circuit, Circuit):
            raise TypeError(
                f"the builder of ansatz {self.name!r} must return a flagquantum "
                f"Circuit and returned {type(circuit).__name__}; a bare tensor is "
                "refused because the expectation reads one as a statevector or as "
                "a density matrix by its shape rather than on the caller's word"
            )
        return circuit

    @classmethod
    def uccsd(
        cls,
        n_qubits: int,
        n_electrons: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Ansatz:
        """Return the UCCSD ansatz of this electron count as configuration.

        The parameter count comes from
        :func:`~flagquantum.algorithms.chemistry.uccsd_excitations` over the same
        two numbers the builder is given, so the count and the circuit cannot
        disagree about the excitation set.  ``dtype`` and ``device`` are carried
        into the builder, so one object fixes the arithmetic as well as the
        shape: a solver whose reference energy is compared against a
        double-precision classical calculation needs ``dtype=torch.complex128``
        here, and the default follows the runtime configuration.

        Args:
            n_qubits: The number of spin orbitals, which must be even.
            n_electrons: The number of electrons, at most ``n_qubits``.
            dtype: The circuit's complex dtype, or ``None`` for the runtime
                configuration's.
            device: The circuit's device, or ``None`` for the runtime
                configuration's.

        Raises:
            ValueError: If the counts are refused by
                :func:`~flagquantum.algorithms.chemistry.uccsd_excitations`.
        """

        excitations = uccsd_excitations(n_electrons, n_qubits)

        def builder(parameters: torch.Tensor) -> Circuit:
            return uccsd_ansatz(
                n_qubits, n_electrons, parameters, dtype=dtype, device=device
            )

        return cls(
            name="uccsd",
            parameter_count=excitations.parameter_count,
            builder=builder,
        )

    @classmethod
    def hardware_efficient(
        cls,
        n_qubits: int,
        layers: int,
        *,
        rotations: Sequence[str] = ("ry", "rz"),
        entanglement: str = "linear",
        device: torch.device | str = "cpu",
    ) -> Ansatz:
        """Return the layered hardware-efficient ansatz as configuration.

        ``rotations`` widens the parameter count, so it is passed once here and
        reaches both the count and the builder through the same tuple.  The
        remaining arguments are
        :func:`~flagquantum.algorithms.core.hardware_efficient_ansatz`'s own.

        Args:
            n_qubits: The number of qubits and angles per rotation block.
            layers: The number of rotation-then-entanglement blocks.
            rotations: The single-qubit rotations applied in each block.
            entanglement: ``"linear"``, ``"circular"``, or ``"none"``.
            device: The circuit's device.

        Raises:
            ValueError: If ``entanglement`` is not one of the three values.
        """

        count = hardware_efficient_parameter_count(n_qubits, layers, rotations)

        def builder(parameters: torch.Tensor) -> Circuit:
            return hardware_efficient_ansatz(
                n_qubits,
                layers,
                parameters,
                rotations=rotations,
                entanglement=entanglement,
                device=device,
            )

        return cls(name="hardware_efficient", parameter_count=count, builder=builder)


def vqe_objective(
    ansatz: Ansatz, operator: Hamiltonian
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return this ansatz's energy over this operator as one plain callable.

    The four VQE entry points beside this module drive their optimizer
    themselves, and the optimizer they drive is a ``torch.optim`` one, so the
    gradient-free units the package ships -- ``CobylaOptimizer``,
    ``SPSAOptimizer`` and ``NelderMeadOptimizer`` -- cannot be supplied in that
    slot: an SPSA step is a rate-limited update from two objective values rather
    than a ``torch.optim`` optimizer, and a simplex is a whole run released at
    once rather than one either.  The route to them is this objective, the same
    way :func:`~flagquantum.algorithms.variational.maxcut_objective` is the route
    to them for the QAOA half.

    The return value takes one flat vector of ``ansatz.parameter_count`` angles
    and returns the scalar energy at them.  It is differentiable, so
    ``torch.autograd`` reaches it, and it is also readable with ``float()``, so a
    unit that spends several evaluations per step instead of a backward pass can
    be handed it just as well.  The length is checked on every call rather than
    once, because the vector the caller passes is the vector the check is about.

    Args:
        ansatz: The ansatz whose energy is evaluated.
        operator: The Pauli sum the energy is taken against.

    Returns:
        A callable of one flat angle vector returning one scalar tensor.

    Raises:
        TypeError: If ``ansatz`` or ``operator`` is of the wrong type.

    Examples:
        Hand the energy to the constrained derivative-free optimizer, which is
        the one member of that family the autograd entry points cannot stand in
        for.  H2's UCCSD ansatz has one doubly-excited angle, and capping it
        below the value the unconstrained minimum wants leaves the energy above
        the exact ground state -- the run reports that it is feasible, and the
        excess is the price of the cap:

        >>> import torch
        >>> from flagquantum.algorithms import CobylaOptimizer
        >>> from flagquantum.algorithms.core import Hamiltonian
        >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
        >>> from flagquantum.algorithms.solvers import Ansatz, vqe_objective
        >>> molecule = create_molecular_hamiltonian(
        ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        ... )
        >>> operator = Hamiltonian.from_observable(molecule.hamiltonian)
        >>> objective = vqe_objective(Ansatz.uccsd(4, 2), operator)
        >>> result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
        ...     objective,
        ...     torch.zeros(3),
        ...     constraints=[lambda angles: 0.1 - angles[2]],
        ... )
        >>> result.feasible, round(float(result.parameters[2]), 6)
        (True, 0.1)
        >>> 0.004 < float(result.value) - molecule.fci_energy < 0.010
        True
    """

    if not isinstance(ansatz, Ansatz):
        raise TypeError(
            "vqe_objective takes a flagquantum Ansatz and received "
            f"{type(ansatz).__name__}"
        )
    if not isinstance(operator, Hamiltonian):
        raise TypeError(
            "vqe_objective takes a flagquantum algorithms Hamiltonian and "
            f"received {type(operator).__name__}; convert an Observable with "
            "Hamiltonian.from_observable first"
        )

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(parameters, dtype=torch.float32)
        if vector.ndim != 1:
            raise ValueError(
                "the angles must be one flat vector of "
                f"{ansatz.parameter_count} values, and a tensor with "
                f"{vector.ndim} dimensions was given; a nested sequence is "
                "refused rather than flattened because its row-major order would "
                "decide which angle reaches which gate"
            )
        if vector.numel() != ansatz.parameter_count:
            raise ValueError(
                f"ansatz {ansatz.name!r} takes {ansatz.parameter_count} angles and "
                f"{vector.numel()} were given"
            )
        return vqe_loss(ansatz, vector, operator)

    return objective


@dataclass(frozen=True)
class VQESolver:
    """One variational ground-state problem, with its optimizer as configuration.

    The three pieces a variational eigensolver spends are bound here once: the
    operator, the ansatz, and the optimizer that walks the ansatz's angles.  A
    :class:`~flagquantum.algorithms.core.VQEResult` comes back from
    :meth:`run`, and the same problem's energy comes back from
    :meth:`objective` as a plain callable for the optimizers that do not fit the
    ``torch.optim`` protocol.

    The optimizer is held as the factory
    :func:`~flagquantum.algorithms.core.run_vqe` already takes -- a callable of
    ``(parameters, lr=...)`` returning a ``torch.optim`` optimizer -- so
    ``dataclasses.replace`` is the way to vary it without touching the rest::

        solver = VQESolver.for_molecule(molecule)
        wider = dataclasses.replace(solver, steps=400, lr=0.05)

    Examples:
        Configure the problem once and run it from the reference start.  The
        energy the run reaches is the FCI energy of the same molecule, which the
        molecule object computed by diagonalizing the determinant space rather
        than by simulating anything:

        >>> import torch
        >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
        >>> from flagquantum.algorithms.solvers import VQESolver
        >>> molecule = create_molecular_hamiltonian(
        ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        ... )
        >>> solver = VQESolver.for_molecule(molecule, dtype=torch.complex128)
        >>> result = solver.run(torch.zeros(solver.ansatz.parameter_count))
        >>> abs(float(result.energy) - molecule.fci_energy) < 1e-5
        True
    """

    operator: Hamiltonian
    ansatz: Ansatz
    optimizer_factory: OptimizerFactory = torch.optim.Adam
    lr: float = 0.05
    steps: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.operator, Hamiltonian):
            raise TypeError(
                "a solver's operator must be a flagquantum algorithms Hamiltonian "
                f"and {type(self.operator).__name__} was given"
            )
        if not isinstance(self.ansatz, Ansatz):
            raise TypeError(
                "a solver's ansatz must be a flagquantum Ansatz and "
                f"{type(self.ansatz).__name__} was given"
            )
        if not callable(self.optimizer_factory):
            raise TypeError(
                "a solver's optimizer factory must be callable and "
                f"{type(self.optimizer_factory).__name__} was given; the "
                "gradient-free optimizers are reached through objective() and "
                "their own minimize() instead"
            )
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError(
                f"a solver's step count must be an integer, and {self.steps!r} was "
                "given"
            )
        if self.steps < 1:
            raise ValidationError(
                "a solver must take at least one step: a factory run and no "
                f"update returns the start it was given, and {self.steps} was "
                "declared"
            )
        if isinstance(self.lr, bool) or not isinstance(self.lr, Real):
            raise TypeError(
                f"a solver's learning rate must be a real number, and {self.lr!r} "
                "was given"
            )
        rate = float(self.lr)
        if not torch.isfinite(torch.tensor(rate)):
            raise ValidationError(
                f"a solver's learning rate must be finite, and {rate} was given"
            )
        if rate <= 0.0:
            raise ValidationError(
                "a solver's learning rate must be positive: a zero rate leaves the "
                f"angles where they started, and {rate} was given"
            )

    def run(self, initial_parameters: torch.Tensor | Sequence[float]) -> VQEResult:
        """Descend this ansatz from ``initial_parameters`` and return the record.

        The descent is :func:`~flagquantum.algorithms.core.run_vqe`, including its
        refusal of an objective that carries no gradient: a builder that ignores
        the angles it is handed is reported by name rather than reaching torch's
        own ``does not require grad`` error.

        Args:
            initial_parameters: One flat angle vector of
                ``ansatz.parameter_count`` values.  The default start is the
                caller's, deliberately: the angles that prepare a reference
                determinant are a property of the ansatz and the problem, not of
                the optimizer.

        Raises:
            ValueError: If the objective carries no gradient in the parameters.
        """

        return run_vqe(
            self.ansatz,
            initial_parameters,
            self.operator,
            steps=self.steps,
            lr=self.lr,
            optimizer_factory=self.optimizer_factory,
        )

    def objective(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return this problem's energy as one plain callable.

        This is the hand-off to the optimizers whose calling convention is
        ``minimize(objective, start)``.  It is
        :func:`vqe_objective` bound to this solver's own ansatz and operator, so
        the operator cannot be swapped for another while the ansatz stays.

        Returns:
            A callable of one flat angle vector returning one scalar tensor.

        Examples:
            A constraint on the angles is expressed as a callable that must stay
            non-negative, and the constrained optimum is reported with the
            largest violation it ended on rather than as an unconstrained answer:

            >>> import torch
            >>> from flagquantum.algorithms import CobylaOptimizer
            >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
            >>> from flagquantum.algorithms.solvers import VQESolver
            >>> molecule = create_molecular_hamiltonian(
            ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
            ... )
            >>> solver = VQESolver.for_molecule(molecule)
            >>> capped = CobylaOptimizer(rhobeg=0.5, rhoend=1e-6).minimize(
            ...     solver.objective(),
            ...     torch.zeros(solver.ansatz.parameter_count),
            ...     constraints=[lambda angles: 0.1 - angles[2]],
            ... )
            >>> capped.feasible, float(capped.value) > molecule.fci_energy
            (True, True)
        """

        return vqe_objective(self.ansatz, self.operator)

    @classmethod
    def for_molecule(
        cls,
        molecule: MolecularHamiltonian,
        *,
        optimizer_factory: OptimizerFactory = torch.optim.Adam,
        lr: float = 0.05,
        steps: int = 100,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> VQESolver:
        """Return the UCCSD solver of an active space, from the object that holds it.

        The point of this constructor is that the electron count and the operator
        come from the same object, so the two cannot be chosen independently: a
        UCCSD ansatz built for two electrons over four spin orbitals is the
        ansatz of a molecule with two electrons in a four-spin-orbital active
        space, and there is no argument here that could say otherwise.  The
        operator is ``molecule.hamiltonian`` converted by
        :meth:`~flagquantum.algorithms.core.Hamiltonian.from_observable`, so it
        carries the mapping's constant term and no truncation.

        The active space is the space the ansatz is built over, not the space the
        molecule occupies: ``molecule.n_orbitals`` is the active count after any
        frozen core, which is the count ``uccsd_excitations`` is asked for.

        Args:
            molecule: The active-space Hamiltonian, its electron count and the
                energies around it.
            optimizer_factory: The ``torch.optim`` factory the descent uses.
            lr: The learning rate handed to the factory.
            steps: The number of optimizer steps, at least one.
            dtype: The circuit's complex dtype, or ``None`` for the runtime
                configuration's.  ``torch.complex128`` makes the reference energy
                agree with the molecule's own double-precision integrals.
            device: The circuit's device, or ``None`` for the runtime
                configuration's.

        Raises:
            TypeError: If ``molecule`` is not a
                :class:`~flagquantum.algorithms.molecular.MolecularHamiltonian`.
            ValueError: If the operator acts on more spin orbitals than the
                active space holds, which is the two objects disagreeing about
                the space rather than a matter of taste.

        Examples:
            The ansatz comes back already sized to the molecule it was built
            from, and there is no electron count to pass that could disagree with
            the operator beside it:

            >>> import torch
            >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
            >>> from flagquantum.algorithms.solvers import VQESolver
            >>> molecule = create_molecular_hamiltonian(
            ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
            ... )
            >>> solver = VQESolver.for_molecule(molecule)
            >>> (solver.ansatz.name, solver.ansatz.parameter_count)
            ('uccsd', 3)
        """

        if not isinstance(molecule, MolecularHamiltonian):
            raise TypeError(
                "VQESolver.for_molecule takes a MolecularHamiltonian and received "
                f"{type(molecule).__name__}; a bare Observable carries no electron "
                "count for the ansatz to be built from"
            )
        operator = Hamiltonian.from_observable(molecule.hamiltonian)
        n_qubits = 2 * molecule.n_orbitals
        if operator.n_qubits > n_qubits:
            raise ValidationError(
                f"the molecule's operator reaches qubit {operator.n_qubits - 1} "
                f"while its active space holds {n_qubits} spin orbitals, so the "
                "operator and the ansatz describe different spaces"
            )
        return cls(
            operator=operator,
            ansatz=Ansatz.uccsd(
                n_qubits, molecule.n_electrons, dtype=dtype, device=device
            ),
            optimizer_factory=optimizer_factory,
            lr=lr,
            steps=steps,
        )


__all__ = ["Ansatz", "VQESolver", "vqe_objective"]
