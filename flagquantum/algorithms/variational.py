"""The QAOA solver: a cost operator, an ansatz, and the optimization between them.

The variational algorithms package already carries every piece a Variational
Quantum Eigensolver needs -- :func:`~flagquantum.algorithms.core.run_vqe` drives
a caller's ansatz builder against a caller's Hamiltonian, and
:func:`~flagquantum.algorithms.core.qaoa_circuit` and
:func:`~flagquantum.algorithms.core.qaoa_loss` build and score a QAOA circuit.
What it did not carry was the thing that makes those pieces one solver for a
problem: an entry point that takes a *problem* -- a graph -- and returns fitted
angles.  This module is that entry point, and it is deliberately small.

**The cost operator and the cost layer have to be one object.**  A QAOA circuit is
built from a list of weighted ``ZZ`` edges and an objective is evaluated against
some operator.  Nothing in the circuit or in the objective relates the two, so a
caller can build the ansatz for a graph and score it against a different
Hamiltonian, or weight the edges one way and the operator another, and every
number that comes back is then a number about a problem nobody posed.
:func:`maxcut_hamiltonian` and :func:`run_qaoa` exist so that the edges are
checked once and reach both sides through the same tuple.

**The uniform superposition is a stationary point, and this module says so
instead of hiding it.**  Every layer of the ansatz acts on the state the layer
before it produced, and the first cost and mixer both act on
``H^⊗n |0...0>``, which is an eigenstate of every ``ZZ`` and every ``X``.  The
measured gradient there is exactly ``0.0`` for every parameter at every layer
count, so a solver whose default start is the origin returns the origin and
reports it as an optimum.  The start is therefore a required argument of
:func:`run_qaoa` and the saddle is stated in
:data:`QAOA_LIMITATIONS`, which the returned :class:`QAOAResult` carries.

**What is not here.**  There is no cost Hamiltonian for a general QUBO or Ising
instance, no constraint or penalty term, no warm start from a classical
heuristic, no shot-based or hardware execution path, no bitstring sampling and no
ranking of sampled cuts, no multi-start or basin-hopping wrapper, and no
gradient-free optimizer route: the update is ``torch.optim`` over an autograd
graph.  A constrained problem is built from
:func:`~flagquantum.algorithms.core.qaoa_circuit` and a Hamiltonian the caller
assembles.  Each of those absences is an item in :data:`QAOA_LIMITATIONS`.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from .core import Hamiltonian, OptimizerFactory, pauli_term, qaoa_loss

__all__ = [
    "QAOA_ASSUMPTIONS",
    "QAOA_LIMITATIONS",
    "QAOAResult",
    "maxcut_hamiltonian",
    "run_qaoa",
]

#: What has to hold for a run's reported energy to describe the caller's graph.
QAOA_ASSUMPTIONS: tuple[str, ...] = (
    "The objective is the expectation of maxcut_hamiltonian's operator over the "
    "circuit qaoa_circuit builds from the same edges, so the ansatz and the "
    "objective cannot describe two different graphs.",
    "The cost layer applies one exp(-i * gamma * w * Z_i Z_j) rotation per edge "
    "per layer, and the mixer applies one exp(-i * beta * X_i) rotation per qubit "
    "per layer, which is what makes gamma and beta the only angles a weighted "
    "MaxCut instance needs.",
    "Each layer spends one gamma and one beta, in that order, in one flat "
    "parameter vector; that vector is what the result returns and what the "
    "circuit consumes.",
    "Every gradient is the exact autograd gradient of a statevector expectation "
    "of a program with no noise channel, so no shot noise and no "
    "finite-difference error enters an update.",
)

#: What a run does not establish, however small its reported energy is.
QAOA_LIMITATIONS: tuple[str, ...] = (
    "The uniform superposition the first layer starts from is a stationary point "
    "of this objective: the measured gradient at the zero start is exactly 0.0 "
    "for every parameter at every layer count, so a zero start returns the zero "
    "start with an unchanged objective and the caller has to choose a nonzero "
    "one. Nothing here perturbs it away, because which basin is reachable is a "
    "property of the instance and not of the solver.",
    "The parameters, the circuit angles, and the objective are float32, because "
    "the circuit builder casts every angle to float32. A float64 start is "
    "downcast rather than refused, so the reported energy is float32-accurate "
    "and not float64-accurate.",
    "The optimizer is whatever torch.optim exposes through optimizer_factory, so "
    "the parameter update needs an autograd graph. The gradient-free units this "
    "package ships, SPSAOptimizer and NelderMeadOptimizer, cannot be handed to "
    "this solver even though a shot-based or hardware objective would need them.",
    "Only the weighted MaxCut cost is offered. There is no cost Hamiltonian for "
    "an arbitrary QUBO or Ising instance and no constraint, penalty, or slack "
    "term, so a constrained problem has to be built from qaoa_circuit, "
    "qaoa_loss, and a Hamiltonian the caller assembles.",
    "The expectation is exact and single-device, so the qubit count is bounded by "
    "what the local statevector engine holds. No shot-based path, no trajectory "
    "noise, no distributed execution, and no hardware submission is used here.",
    "The result carries the fitted angles and the energy at them, and nothing "
    "about the cut itself. Sampling the optimized circuit, decoding a bitstring, "
    "and comparing a sampled cut against the optimum are all the caller's steps, "
    "and no cut value is returned or ranked here.",
    "The energy is an expectation and not a bound. It says what the ansatz "
    "achieves, and because QAOA at a finite layer count is not exact, a value "
    "above the operator's ground energy is the normal outcome rather than "
    "evidence of a mistake.",
    "The history records what the objective took at each step, and it is not a "
    "descent. Adam, the default, takes a step of the learning rate's size "
    "regardless of how flat the objective is, so the trajectory can rise and can "
    "leave the vicinity of a good point; only the final value is the result.",
    "A step count below one and a parameter vector whose length is not twice the "
    "layer count are refused here, which run_vqe does not do: that entry point "
    "predates this one, and it returns the starting point for a zero step count.",
)


@dataclass(frozen=True)
class QAOAResult:
    """The point one ``run_qaoa`` call reached, and the trajectory it took.

    The angles are held as the one flat vector the optimizer owned, in the
    layer-major order the circuit consumes them: all ``layers`` gammas, then all
    ``layers`` betas.  :attr:`gammas` and :attr:`betas` expose the two halves, so
    a caller can rebuild the optimized circuit with
    :func:`~flagquantum.algorithms.core.qaoa_circuit` without slicing the vector
    themselves and without this unit having to guess the convention back.
    """

    parameters: torch.Tensor
    energy: torch.Tensor
    history: tuple[float, ...]
    assumptions: tuple[str, ...] = QAOA_ASSUMPTIONS
    limitations: tuple[str, ...] = QAOA_LIMITATIONS

    def __post_init__(self) -> None:
        if self.parameters.ndim != 1:
            raise ValueError(
                "parameters must be one flat vector, and a tensor with "
                f"{self.parameters.ndim} dimensions was given"
            )
        if self.parameters.numel() % 2:
            raise ValueError(
                f"parameters holds {self.parameters.numel()} values; each QAOA "
                "layer spends one gamma and one beta, so the count must be even"
            )
        if self.parameters.numel() == 0:
            raise ValueError(
                "parameters must hold one gamma and one beta per layer, and none "
                "were given"
            )
        if self.energy.numel() != 1:
            raise ValueError(
                f"energy must be one scalar, and {self.energy.numel()} values "
                "were given"
            )

    @property
    def n_steps(self) -> int:
        """The number of optimizer steps the run took."""

        return len(self.history)

    @property
    def layers(self) -> int:
        """The number of cost and mixer layers, one gamma and one beta each."""

        return self.parameters.numel() // 2

    @property
    def gammas(self) -> torch.Tensor:
        """The cost-layer angles, one per layer, in layer order."""

        return self.parameters[: self.layers]

    @property
    def betas(self) -> torch.Tensor:
        """The mixer angles, one per layer, in layer order."""

        return self.parameters[self.layers :]


def _checked_endpoint(n_qubits: int, index: int, endpoint: object) -> int:
    """Return one edge endpoint as a qubit index of the graph, or refuse it."""

    if isinstance(endpoint, bool) or not isinstance(endpoint, int):
        raise ValueError(
            f"edge {index} must name qubit indices, and {endpoint!r} was given"
        )
    if not 0 <= endpoint < n_qubits:
        raise ValueError(
            f"edge {index} names qubit {endpoint}, and the graph has "
            f"{n_qubits} qubit{'s' if n_qubits != 1 else ''}"
        )
    return endpoint


def _checked_cut_edges(
    n_qubits: int, edges: Iterable[tuple[int, int] | tuple[int, int, float]]
) -> tuple[tuple[int, int, float], ...]:
    """Return a graph's edges as checked ``(source, target, weight)`` triples.

    This is the validation the cost operator and the solver share.  A repeated
    undirected pair is refused rather than summed because ``(1, 0)`` and ``(0, 1)``
    are one cut edge, so a caller who lists both has almost certainly generated
    the edge set twice; summing would then count that edge twice in the objective
    while the cost layer applied its rotation twice as well, and the two errors
    would reinforce instead of cancelling.
    """

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    normalized: list[tuple[int, int, float]] = []
    seen: dict[tuple[int, int], int] = {}
    for index, edge in enumerate(edges):
        entries = tuple(edge)
        if len(entries) not in (2, 3):
            raise ValueError(
                f"edge {index} has {len(entries)} entries; a weighted MaxCut edge "
                "is (source, target) or (source, target, weight)"
            )
        source = _checked_endpoint(n_qubits, index, entries[0])
        target = _checked_endpoint(n_qubits, index, entries[1])
        if source == target:
            raise ValueError(
                f"edge {index} joins qubit {source} to itself; Z on one qubit "
                "squares to the identity, so a self-loop is a constant shift "
                "rather than a cut"
            )
        weight = 1.0 if len(entries) == 2 else entries[2]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(
                f"edge {index} weight must be a real number, and {weight!r} was "
                "given"
            )
        weight_value = float(weight)
        if not math.isfinite(weight_value):
            raise ValueError(
                f"edge {index} weight must be finite, and {weight!r} was given"
            )
        pair = (source, target) if source < target else (target, source)
        if pair in seen:
            raise ValueError(
                f"edges {seen[pair]} and {index} are the same undirected pair "
                f"{pair}; list each cut edge once, because a repeated edge is "
                "counted twice by the objective and rotated twice by the cost layer"
            )
        seen[pair] = index
        normalized.append((source, target, weight_value))
    if not normalized:
        raise ValueError(
            "a MaxCut instance needs at least one edge, because a graph with no "
            "edges has no cut to maximize and every assignment is an optimum"
        )
    return tuple(normalized)


def maxcut_hamiltonian(
    n_qubits: int, edges: Iterable[tuple[int, int] | tuple[int, int, float]]
) -> Hamiltonian:
    """Build the weighted MaxCut cost ``sum over edges of w_ij * Z_i Z_j``.

    This is the operator :func:`~flagquantum.algorithms.core.qaoa_circuit`
    implements, with the same weights and the same edge order.  Its cost layer
    applies ``exp(-i * gamma * w_ij * Z_i Z_j)`` once per edge per layer, so a
    circuit built from some edge list and an energy evaluated against an operator
    built from another are two different problems; pairing both sides through one
    checked edge tuple is what keeps them one.

    **What it minimizes.**  On a computational basis state the operator's value is
    ``W - 2 * (weight of the edges cut by that state)``, where ``W`` is the sum of
    all weights, because each edge contributes ``+w`` when its endpoints agree and
    ``-w`` when they differ.  Minimizing it therefore maximizes the weighted cut,
    with ``W - 2 * (largest weighted cut)`` the value at the optimum, and a
    negative weight inverts that one edge's preference rather than being
    meaningless.  No constant term is included, because a constant shifts every
    energy by the same amount and changes no angle; the energy this operator
    reports is the cost and not the cut.

    **The relation to the QUBO form is affine, and that is why this operator
    exists rather than the QUBO one being reused.**  On every computational basis
    state, with ``m`` the number of unit-weight edges,
    :func:`flagquantum.algorithms.qubo.max_cut_qubo` followed by
    :func:`flagquantum.algorithms.qubo.qubo_to_ising` gives a Hamiltonian whose
    value is ``(value_of_this_one - m) / 2``.  The two therefore have the same
    minimizers, so either answers "which cut is largest", but they do not have the
    same angles: the QUBO form both rescales the coefficients by ``1/2`` and adds
    a constant, the constant changes no angle while the rescaling changes which
    ``gamma`` is a good one, and a QAOA sweep therefore needs the operator whose
    coefficients are the ones the cost layer rotates by.

    Args:
        n_qubits: The number of graph nodes, and the number of qubits the cost
            layer will act on.  Positive, and every endpoint is checked against it.
        edges: The undirected edges, as ``(source, target)`` pairs, which carry
            weight ``1.0``, or as ``(source, target, weight)`` triples.

    Returns:
        One ``ZZ`` term per edge, in the order the edges were given.

    Raises:
        ValueError: If ``n_qubits`` is not a positive integer, if an edge is
            neither a pair nor a triple, if an endpoint is not a qubit of the
            graph or is the same qubit twice, if a weight is not a finite real
            number, if the same undirected pair appears twice, or if there is no
            edge at all.

    Examples:
        A triangle's cost reaches ``-1.0``: with three unit-weight edges the value
        on a basis state is ``3 - 2 * cut``, so the best cut of the triangle
        yields ``3 - 4``.

        >>> from flagquantum.algorithms.variational import maxcut_hamiltonian
        >>> float(maxcut_hamiltonian(3, ((0, 1), (1, 2), (2, 0))).ground_energy())
        -1.0

        A weighted edge keeps its weight, and pairs and triples may be mixed:

        >>> weighted = maxcut_hamiltonian(3, ((0, 1, 0.5), (1, 2)))
        >>> [term.coefficient for term in weighted.terms]
        [0.5, 1.0]
    """

    normalized = _checked_cut_edges(n_qubits, edges)
    return Hamiltonian(
        [
            pauli_term(weight, "ZZ", (source, target))
            for source, target, weight in normalized
        ]
    )


def run_qaoa(
    n_qubits: int,
    edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    initial_parameters: torch.Tensor | Sequence[float],
    *,
    steps: int = 100,
    lr: float = 0.05,
    optimizer_factory: OptimizerFactory = torch.optim.Adam,
    device: torch.device | str = "cpu",
) -> QAOAResult:
    """Optimize a weighted MaxCut QAOA ansatz against its own cost operator.

    This is the QAOA entry point beside
    :func:`~flagquantum.algorithms.core.run_vqe`.  It takes a problem, a starting
    point, and an optimizer factory, and returns the fitted angles, the objective
    at them, and one value per step.  The ansatz is
    :func:`~flagquantum.algorithms.core.qaoa_circuit`, the operator is
    :func:`maxcut_hamiltonian` over exactly the edges given here, and the objective
    is :func:`~flagquantum.algorithms.core.qaoa_loss` against that operator, so the
    circuit and the energy cannot disagree about the graph.

    **A zero start is a stationary point, so the start is a required argument.**
    The first layer's cost and mixer both act on the uniform superposition, which
    is an eigenstate of every ``ZZ`` and every ``X``; the measured gradient there
    is exactly ``0.0`` for every parameter at every layer count, and a zero start
    therefore returns the zero start with the objective unchanged.  Nothing here
    perturbs it away, because a default displacement chosen by this unit would be
    a claim about the caller's instance rather than a property of QAOA.  The
    saddle is stated in :data:`QAOA_LIMITATIONS` and is reachable from the
    returned result.

    Args:
        n_qubits: The number of graph nodes and qubits.  Positive.
        edges: The undirected edges as ``(source, target)`` pairs, which carry
            weight ``1.0``, or as ``(source, target, weight)`` triples.
        initial_parameters: The starting angles as one flat vector of
            ``2 * layers`` values, the gammas first and then the betas.  Cast to
            float32; a vector of odd length, or one that is not one-dimensional,
            is refused rather than truncated or flattened.
        steps: The number of optimizer steps.  At least one, because a run with
            no step returns the starting point and would report it as an optimum.
        lr: The learning rate handed to ``optimizer_factory``.  This unit does not
            inspect it, so a factory that ignores it may be given any value.
        optimizer_factory: Builds a ``torch.optim`` optimizer from one parameter
            group and ``lr``, defaulting to Adam.  The update is autograd-based,
            so a gradient-free optimizer cannot be supplied here.
        device: The device the parameters, the circuit, and the objective live on.

    Returns:
        The fitted angles, the objective value at them, one value per step, and
        the assumptions and limitations the run rests on.

    Raises:
        ValueError: If ``n_qubits``, the edges, ``steps``, or the parameter vector
            fail the checks described above.

    Examples:
        Reach a triangle's optimum, whose cost is ``-1.0``, from one layer at a
        small nonzero start:

        >>> import torch
        >>> from flagquantum.algorithms.variational import run_qaoa
        >>> edges = ((0, 1), (1, 2), (2, 0))
        >>> result = run_qaoa(3, edges, torch.full((2,), 0.05), steps=100)
        >>> result.layers, result.n_steps
        (1, 100)
        >>> round(float(result.energy), 4)
        -0.9993

        The zero start stays exactly where it is, which is the documented trap.
        The uniform superposition is an eigenstate of every cost and mixer term,
        so the gradient there is zero at every layer count and not merely small:

        >>> flat = run_qaoa(3, edges, torch.zeros(6), steps=5)
        >>> flat.layers
        3
        >>> flat.history
        (0.0, 0.0, 0.0, 0.0, 0.0)
        >>> bool(flat.energy == 0.0)
        True
    """

    normalized = _checked_cut_edges(n_qubits, edges)
    if isinstance(steps, bool):
        raise ValueError(
            "steps must be a whole number of optimization steps, and a bool is "
            "not a step count"
        )
    try:
        step_count = operator.index(steps)
    except TypeError:
        raise ValueError(
            f"steps must be a whole number of optimizer steps, and {steps!r} was "
            "given"
        ) from None
    if step_count < 1:
        raise ValueError(
            "steps must be at least one, because a run with no step returns the "
            "starting point and reports it as an optimum"
        )

    parameters = torch.as_tensor(initial_parameters, dtype=torch.float32, device=device)
    if parameters.ndim != 1:
        raise ValueError(
            "initial_parameters must be one flat vector of 2 * layers values, the "
            f"gammas and then the betas, and a tensor with {parameters.ndim} "
            "dimensions was given; a nested sequence is refused rather than "
            "flattened because its row-major order would interleave the two "
            "halves instead of stacking them"
        )
    if parameters.numel() == 0:
        raise ValueError(
            "initial_parameters must hold one gamma and one beta per QAOA layer, "
            "and none were given"
        )
    if parameters.numel() % 2:
        raise ValueError(
            f"initial_parameters holds {parameters.numel()} values; each QAOA "
            "layer spends one gamma and one beta, so the count must be even"
        )
    layers = parameters.numel() // 2

    controlled = parameters.detach().clone().requires_grad_(True)
    objective = maxcut_hamiltonian(n_qubits, normalized)
    optimizer = optimizer_factory([controlled], lr=lr)
    history: list[float] = []

    def value() -> torch.Tensor:
        return qaoa_loss(
            n_qubits,
            normalized,
            controlled[:layers],
            controlled[layers:],
            objective,
            device=device,
        )

    for _ in range(step_count):
        optimizer.zero_grad()
        loss = value()
        torch.autograd.backward(loss)
        optimizer.step()
        history.append(float(loss.detach()))

    return QAOAResult(
        parameters=controlled.detach().clone(),
        energy=value().detach(),
        history=tuple(history),
    )
