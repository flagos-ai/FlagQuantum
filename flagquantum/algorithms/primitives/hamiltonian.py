"""General Hamiltonian construction and evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module as _import_public_namespace
from typing import Literal

import torch

from flagquantum.circuit import Circuit
from flagquantum.core.operators.observable import Observable
from flagquantum.simulation.mps.state import MPSState
from flagquantum.simulation.pauli import (
    infer_n_qubits_from_dense_state,
    pauli_product_density_expectation,
    pauli_product_operator,
    pauli_product_statevector_expectation,
)
from flagquantum.simulation.tensor_network.state import TensorNetworkState

_PAULI_NAMES = {"i", "x", "y", "z"}


def _as_qubit_tuple(qubits: Iterable[int] | int | None) -> tuple[int, ...]:
    if qubits is None:
        return ()
    if isinstance(qubits, int):
        return (qubits,)
    return tuple(int(qubit) for qubit in qubits)


def _normalize_pauli(
    pauli: str | Mapping[int, str],
    qubits: Iterable[int] | int | None,
) -> tuple[tuple[int, str], ...]:
    if isinstance(pauli, Mapping):
        items = tuple((int(qubit), str(name).lower()) for qubit, name in pauli.items())
    else:
        qubit_tuple = _as_qubit_tuple(qubits)
        if len(pauli) != len(qubit_tuple):
            raise ValueError("Pauli string length must match qubits length.")
        items = tuple(
            (qubit, name.lower())
            for qubit, name in zip(qubit_tuple, pauli, strict=True)
        )

    normalized = []
    seen = set()
    for qubit, name in items:
        if name not in _PAULI_NAMES:
            raise ValueError("Pauli operators must be one of I, X, Y, or Z.")
        if qubit in seen:
            raise ValueError("A Hamiltonian term cannot repeat a qubit.")
        seen.add(qubit)
        if name != "i":
            normalized.append((qubit, name))
    return tuple(sorted(normalized))


def _coefficient_tensor(
    coefficient: float | complex | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(coefficient, complex) or (
        isinstance(coefficient, torch.Tensor) and coefficient.is_complex()
    ):
        dtype = torch.promote_types(dtype, torch.complex64)
    return torch.as_tensor(coefficient, dtype=dtype, device=device)


@dataclass(frozen=True)
class HamiltonianTerm:
    """One weighted Pauli product term."""

    coefficient: float | complex | torch.Tensor
    pauli: str | Mapping[int, str]
    qubits: tuple[int, ...] = ()

    def __init__(
        self,
        coefficient: float | complex | torch.Tensor,
        pauli: str | Mapping[int, str],
        qubits: Iterable[int] | int | None = None,
    ) -> None:
        self.ops: tuple[tuple[int, str], ...]
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "pauli", pauli)
        object.__setattr__(self, "qubits", _as_qubit_tuple(qubits))
        object.__setattr__(self, "ops", _normalize_pauli(pauli, self.qubits))

    @property
    def max_qubit(self) -> int:
        if not self.ops:
            return -1
        return max(qubit for qubit, _ in self.ops)

    def expectation(
        self, target: Circuit | MPSState | TensorNetworkState | torch.Tensor
    ) -> torch.Tensor:
        """Evaluate this term on a circuit, MPS, tensor-network, or dense state."""

        if isinstance(target, Circuit):
            base = (
                target.state().real.new_ones(target.bsz)
                if not self.ops
                else target.expectation_ps(
                    x=[qubit for qubit, name in self.ops if name == "x"],
                    y=[qubit for qubit, name in self.ops if name == "y"],
                    z=[qubit for qubit, name in self.ops if name == "z"],
                )
            )
            coeff = _coefficient_tensor(
                self.coefficient,
                dtype=base.dtype,
                device=base.device,
            )
            return torch.real(coeff * base)

        if isinstance(target, MPSState):
            base = (
                target.tensors[0].real.new_ones(target.bsz)
                if not self.ops
                else target.expectation_ps(
                    x=[qubit for qubit, name in self.ops if name == "x"],
                    y=[qubit for qubit, name in self.ops if name == "y"],
                    z=[qubit for qubit, name in self.ops if name == "z"],
                )
            )
            coeff = _coefficient_tensor(
                self.coefficient, dtype=base.dtype, device=base.device
            )
            return torch.real(coeff * base)

        if isinstance(target, TensorNetworkState):
            # Contract this term's bra-operator-ket network. The state's own
            # `expectation_z` materializes a dense state below its observable
            # threshold; a caller that selected the tensor-network representation
            # must not be routed through that path for a Hamiltonian energy.
            # `expectation_ps` also answers the empty product, so a constant term
            # needs no special case. A sum of terms goes through
            # `TensorNetworkState.expectation_hamiltonian` instead, which
            # contracts the whole sum as one operator network.
            base = target.expectation_ps(
                x=[qubit for qubit, name in self.ops if name == "x"],
                y=[qubit for qubit, name in self.ops if name == "y"],
                z=[qubit for qubit, name in self.ops if name == "z"],
            )
            coeff = _coefficient_tensor(
                self.coefficient, dtype=base.dtype, device=base.device
            )
            return torch.real(coeff * base)

        tensor = target
        if tensor.ndim >= 2 and tensor.shape[-1] == tensor.shape[-2]:
            n_qubits = infer_n_qubits_from_dense_state(tensor)
            base = (
                tensor.real.new_ones(
                    tensor.shape[0] if tensor.ndim == 3 else 1,
                )
                if not self.ops
                else pauli_product_density_expectation(tensor, self.ops, n_qubits)
            )
        else:
            n_qubits = infer_n_qubits_from_dense_state(tensor)
            base = (
                tensor.real.new_ones(
                    tensor.shape[0] if tensor.ndim == 2 else 1,
                )
                if not self.ops
                else pauli_product_statevector_expectation(tensor, self.ops, n_qubits)
            )
        coeff = _coefficient_tensor(
            self.coefficient, dtype=base.dtype, device=base.device
        )
        return torch.real(coeff * base)


class Hamiltonian:
    """A differentiable weighted sum of Pauli products."""

    def __init__(self, terms: Iterable[HamiltonianTerm]) -> None:
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("Hamiltonian requires at least one term.")

    @classmethod
    def from_observable(cls, observable: Observable) -> "Hamiltonian":
        """Return an :class:`~flagquantum.observables.Observable` as this type.

        The two types describe the same object -- a real linear combination of
        Pauli products -- and they are not interchangeable because they answer
        different questions.  An ``Observable`` is what a circuit run *requests*:
        it belongs to ``fq.expectation(observable)`` and travels with a
        :class:`~flagquantum.observables.OutputRequest` into ``fq.run``.  This type
        is what the loops in this module *evaluate*: it takes a circuit, an MPS
        state, a statevector or a density matrix directly, and it can differentiate
        the result by ``adjoint`` as well as by autograd.  So the conversion runs
        in one direction only, into the richer evaluating form, and a caller who
        already holds an ``Observable`` -- from
        :func:`~flagquantum.algorithms.molecular.create_molecular_hamiltonian`, from
        :func:`~flagquantum.observables.jordan_wigner`, or from Pauli arithmetic
        like ``0.5 * (fq.X(0) @ fq.X(1))`` -- needs it to reach
        :func:`run_vqe`, :func:`run_adapt_vqe` and :func:`vqe_objective` at all.

        Nothing is approximated and no term is dropped: every coefficient and every
        Pauli factor is carried across, including the identity term a mapped
        electronic Hamiltonian uses for its constant shift, which lands here as a
        term with no factors rather than as a separate offset field.

        Args:
            observable: The Pauli sum to convert.

        Returns:
            The same Pauli sum in the form this module's loops evaluate.

        Raises:
            TypeError: If ``observable`` is not an
                :class:`~flagquantum.observables.Observable`.

        Examples:
            A mapped molecule's operator reaches the VQE objective, and the
            energy at the ansatz's zero angles -- where UCCSD prepares the
            Hartree-Fock determinant exactly -- is the Hartree-Fock energy of the
            same molecule.  The two come from different arithmetic, an integral
            engine against a circuit simulation, so the agreement is a check
            rather than a tautology:

            >>> from flagquantum.algorithms.chemistry import uccsd_ansatz
            >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
            >>> molecule = create_molecular_hamiltonian(
            ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
            ... )
            >>> operator = Hamiltonian.from_observable(molecule.hamiltonian)
            >>> builder = lambda angles: uccsd_ansatz(
            ...     4, 2, angles, dtype=torch.complex128
            ... )
            >>> reference = float(vqe_loss(builder, torch.zeros(3), operator))
            >>> abs(reference - molecule.hf_energy) < 1e-12
            True
        """

        if not isinstance(observable, Observable):
            raise TypeError(
                "Hamiltonian.from_observable takes a flagquantum Observables "
                f"value and received {type(observable).__name__}"
            )
        return cls(
            HamiltonianTerm(
                term.coefficient,
                {int(qubit): name for qubit, name in term.factors},
            )
            for term in observable.terms
        )

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def n_qubits(self) -> int:
        return 1 + max((term.max_qubit for term in self.terms), default=-1)

    def expectation(
        self,
        target: Circuit | MPSState | TensorNetworkState | torch.Tensor,
        *,
        differentiation: Literal["autograd", "adjoint"] = "autograd",
    ) -> torch.Tensor:
        if differentiation not in {"autograd", "adjoint"}:
            raise ValueError("differentiation must be 'autograd' or 'adjoint'")
        if differentiation == "adjoint":
            if not isinstance(target, Circuit):
                raise TypeError("adjoint differentiation requires a Circuit target")
            if target.bsz != 1:
                raise ValueError("adjoint differentiation currently requires bsz=1")
            observable_terms: list[tuple[float, tuple[int, ...]]] = []
            constant = 0.0
            for term in self.terms:
                coefficient = term.coefficient
                if isinstance(coefficient, torch.Tensor):
                    if coefficient.numel() != 1 or coefficient.requires_grad:
                        raise ValueError(
                            "adjoint differentiation requires constant scalar coefficients"
                        )
                    coefficient = coefficient.detach().item()
                coefficient_value = complex(coefficient)
                if coefficient_value.imag != 0.0:
                    raise ValueError(
                        "adjoint differentiation requires real Hamiltonian coefficients"
                    )
                qubits = tuple(qubit for qubit, name in term.ops if name == "z")
                if len(qubits) != len(term.ops) or len(qubits) > 2:
                    raise ValueError(
                        "adjoint differentiation currently supports only Z and ZZ terms"
                    )
                if not qubits:
                    constant += coefficient_value.real
                else:
                    observable_terms.append((coefficient_value.real, qubits))
            if not observable_terms:
                raise ValueError(
                    "adjoint differentiation requires at least one Z or ZZ term"
                )
            from flagquantum.runtime.executors.statevector.reverse import (
                execute_torch_distributed_statevector_reverse,
            )

            result = execute_torch_distributed_statevector_reverse(
                target,
                observable_terms=observable_terms,
                device=target.device,
            )
            return result.value + result.value.new_tensor(constant)
        if isinstance(target, MPSState):
            z_coefficients: dict[int, float | complex | torch.Tensor] = {}
            zz_coefficients: dict[int, float | complex | torch.Tensor] = {}
            chain_compatible = True
            for term in self.terms:
                ops = term.ops
                if len(ops) == 1 and ops[0][1] == "z":
                    qubit = ops[0][0]
                    z_coefficients[qubit] = (
                        z_coefficients.get(qubit, 0.0) + term.coefficient
                    )
                elif (
                    len(ops) == 2
                    and ops[0][1] == "z"
                    and ops[1][1] == "z"
                    and ops[1][0] == ops[0][0] + 1
                ):
                    qubit = ops[0][0]
                    zz_coefficients[qubit] = (
                        zz_coefficients.get(qubit, 0.0) + term.coefficient
                    )
                else:
                    chain_compatible = False
                    break
            if chain_compatible:
                return target.expectation_z_zz_chain(
                    z_coefficients=z_coefficients,
                    zz_coefficients=zz_coefficients,
                )
        if isinstance(target, TensorNetworkState):
            return target.expectation_hamiltonian(self)
        values = [term.expectation(target) for term in self.terms]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    def matrix(
        self,
        *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Materialize a dense operator for small-system diagnostics."""

        if self.n_qubits > 12:
            raise ValueError("dense Hamiltonian diagnostics are limited to 12 qubits")
        dimension = 1 << self.n_qubits
        result = torch.zeros(dimension, dimension, dtype=dtype, device=device)
        for term in self.terms:
            coefficient = torch.as_tensor(term.coefficient, dtype=dtype, device=device)
            result = result + coefficient * pauli_product_operator(
                term.ops, self.n_qubits, dtype=dtype, device=device
            )
        return result

    def ground_energy(self) -> torch.Tensor:
        """Return the exact small-system ground energy."""

        eigenvalues: torch.Tensor = torch.linalg.eigvalsh(self.matrix())
        return eigenvalues.min()


def pauli_term(
    coefficient: float | complex | torch.Tensor,
    pauli: str | Mapping[int, str],
    qubits: Iterable[int] | int | None = None,
) -> HamiltonianTerm:
    """Create a weighted Pauli product term."""

    return HamiltonianTerm(coefficient, pauli, qubits)


__all__ = ("HamiltonianTerm", "Hamiltonian", "pauli_term")

for _identity in (
    _as_qubit_tuple,
    _normalize_pauli,
    _coefficient_tensor,
    HamiltonianTerm,
    Hamiltonian,
    pauli_term,
):
    _identity.__module__ = "flagquantum.algorithms.core"
del _identity

_public_namespace = _import_public_namespace("flagquantum.algorithms.core")
for _name in (
    "_as_qubit_tuple",
    "_normalize_pauli",
    "_coefficient_tensor",
    "HamiltonianTerm",
    "Hamiltonian",
    "pauli_term",
):
    setattr(_public_namespace, _name, globals()[_name])
del _public_namespace, _name
