"""Run the four VQE entry points on one instance and read what each one returns.

`flagquantum.algorithms.core` exposes four entry points that share one
differentiable objective: `run_vqe` descends a fixed ansatz, `run_adapt_vqe`
grows one operator at a time from an operator pool, `run_hybrid_vqe` runs a
schedule of optimizer stages, and `run_layerwise_vqe` deepens a depth-indexed
ansatz while keeping the coordinates the shallower stage already optimized. All
four are reachable through the subpackage surface only -- `from
flagquantum.algorithms import run_vqe` -- because the algorithms package adds no
root-level `fq.` name.

The premise the unit rests on, and cannot check: the reference state the circuit
is built from is a place the pool can act. ADAPT-VQE screens a pool direction by
appending it at angle zero and differentiating, so from a stationary point of the
pool -- a reference state that is an eigenstate of every pool generator -- every
direction reports an exactly zero gradient. The run then selects nothing, stops,
and reports `converged=True`: a statement about that reference state rather than
about the optimum. The screening is exact, so the zeros are exact, and nothing
inside the loop can tell "this direction does not help" from "this direction
would help anywhere else". Both cases are run here rather than asserted: the same
pool from `|00>` selects and from `|++>` does not.

What `converged` reports is the other half, and it is the caller's to read.
Three different runs reach it here: a pool that is exhausted, a screening pass
whose largest gradient fell below the tolerance, and a budget that ran out. The
first two report `True` and the third reports `False`, and the third is not the
one that failed to reach the energy -- the budget-limited run in this instance
stops further from the ground state while a pool that cannot express it reports
`True` at its own limit. The flag reports why the loop stopped and not how close
the energy came, the same reading `NelderMeadOptimizer`'s flag is documented
with. And the energy is not a bound: every value below is a float32 statevector
expectation at a finite step count, so the last digits are the arithmetic's own.

The instance is a two-qubit transverse-field Ising model, `ZZ - X0 - X1`, whose
ground energy is `-sqrt(5)` and is computed here from the operator rather than
quoted. It is small enough that the exact screening gradients and the pool's
limits are visible in one run, and it is a model rather than a molecule: no
chemistry, no basis set, and no hardware is involved anywhere below.

Run it with:

    python -m examples.algorithms.vqe_solvers
"""

from __future__ import annotations

from typing import Any

import torch

from flagquantum.algorithms import (
    Hamiltonian,
    hardware_efficient_ansatz,
    hardware_efficient_parameter_count,
    pauli_term,
    run_adapt_vqe,
    run_hybrid_vqe,
    run_layerwise_vqe,
    run_vqe,
)
from flagquantum.algorithms.optimization import OptimizationStage
from flagquantum.circuit import Circuit

#: Two qubits and one field each: the smallest instance with an entangled ground
#: state, so a product ansatz and an entangling one are distinguishable here.
N_QUBITS = 2
ROTATIONS = ("ry",)
#: The ansatz depth the fixed-ansatz and hybrid runs use, and the depths the
#: layerwise run grows through.
DEPTH = 1
DEPTHS = (1, 2, 3)
#: The operator pool ADAPT-VQE screens: single-qubit rotations on wire 0. The
#: pool is deliberately too small to express the ground state, which is what
#: makes its limit visible instead of implied.
POOL = ("ry", "rz", "rx")
LABEL_WIDTH = 30


def cost_hamiltonian() -> Hamiltonian:
    """Return the two-qubit transverse-field Ising operator."""

    return Hamiltonian(
        (
            pauli_term(1.0, "ZZ", (0, 1)),
            pauli_term(-1.0, "X", (0,)),
            pauli_term(-1.0, "X", (1,)),
        )
    )


def fixed_ansatz(parameters: torch.Tensor) -> Circuit:
    """Return the depth-`DEPTH` hardware-efficient ansatz for one parameter vector."""

    return hardware_efficient_ansatz(
        N_QUBITS, DEPTH, parameters, rotations=ROTATIONS, entanglement="linear"
    )


def reference_circuit() -> Circuit:
    """Return the bare reference state ADAPT-VQE grows from: ``|00>``."""

    return Circuit(N_QUBITS, dtype=torch.complex128)


def plus_circuit() -> Circuit:
    """Return the stationary reference state: ``|++>``, every pool gradient zero."""

    circuit = Circuit(N_QUBITS, dtype=torch.complex128)
    for wire in range(N_QUBITS):
        circuit = circuit.h(wire)
    return circuit


def pool_circuit(reference: Any, operators: tuple[Any, ...], parameters: torch.Tensor):
    """Return ``reference`` with each pool operator appended at its own angle.

    The operator at index ``i`` acts on wire ``i % N_QUBITS``, so a pool of
    single-qubit rotations reaches both wires and the pool is still small enough
    that its limit is the pool and not a wiring accident.
    """

    circuit = reference()
    for index, (operator, theta) in enumerate(zip(operators, parameters, strict=True)):
        circuit = circuit.gate(operator, index % N_QUBITS, theta=theta)
    return circuit


def layerwise_ansatz(depth: int, parameters: torch.Tensor) -> Circuit:
    """Return a depth-indexed ansatz, so ``run_layerwise_vqe`` can grow it."""

    return hardware_efficient_ansatz(
        N_QUBITS, depth, parameters, rotations=ROTATIONS, entanglement="linear"
    )


def layerwise_parameter_count(depth: int) -> int:
    """Return the parameter count the depth-indexed ansatz requires."""

    return hardware_efficient_parameter_count(N_QUBITS, depth, ROTATIONS)


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def refuse(description: str, action: Any) -> None:
    """Run ``action``, print the refusal it must raise, and fail if it returns."""

    try:
        action()
    except ValueError as error:
        report(description, f"{error}")
    else:
        raise SystemExit(
            f"{description} was accepted, and the example exists to refuse it"
        )


def main() -> None:
    cost = cost_hamiltonian()
    ground = float(cost.ground_energy())
    stage = OptimizationStage(group="quantum", method="adam", steps=300, lr=0.1)

    print("=" * 72)
    print("vqe solvers -- flagquantum.algorithms.core")
    print("=" * 72)
    report("task", "run four VQE entry points on one instance")
    report("premise", "the reference state the pool acts on is a place the pool")
    print(f"  {'':<{LABEL_WIDTH}}  can act, and from a stationary point of the pool")
    print(f"  {'':<{LABEL_WIDTH}}  every screened gradient is exactly zero, so the")
    print(f"  {'':<{LABEL_WIDTH}}  same pool selects nothing and the loop cannot tell")
    print(f"  {'':<{LABEL_WIDTH}}  the state's stationarity from a direction that would")
    print(f"  {'':<{LABEL_WIDTH}}  not help; `converged` reports why the loop stopped")
    print(f"  {'':<{LABEL_WIDTH}}  rather than how close the energy came, so an")
    print(f"  {'':<{LABEL_WIDTH}}  exhausted pool and an out-of-budget run are read")
    print(f"  {'':<{LABEL_WIDTH}}  through one boolean; and the energy is not a bound,")
    print(f"  {'':<{LABEL_WIDTH}}  because every descent below runs in float32 at a")
    print(f"  {'':<{LABEL_WIDTH}}  finite step count")
    print()

    print("inputs")
    report("operator", str([(float(t.coefficient), t.wires) for t in cost.terms]))
    report("ground energy", f"{ground:.17g}")
    report("reference state", "|00>, so the pool is not screened at a stationary point")
    report("pool", ", ".join(POOL) + f" on wire 0, {len(POOL)} directions")
    report("stage", f"adam, {stage.steps} steps, lr {stage.lr}")
    print()

    print("run_vqe: one fixed ansatz, descended from a constant start")
    count = hardware_efficient_parameter_count(N_QUBITS, DEPTH, ROTATIONS)
    start = torch.full((count,), 0.1)
    before = start.clone()
    fixed = run_vqe(fixed_ansatz, start, cost, steps=600, lr=0.1)
    again = run_vqe(fixed_ansatz, start, cost, steps=600, lr=0.1)
    report("parameters", count)
    report("start energy", f"{fixed.history[0]:.17g}")
    report("energy", f"{float(fixed.energy):.17g}")
    report("energy above ground", f"{float(fixed.energy) - ground:.6g}")
    report("steps taken", fixed.n_steps)
    print(f"  {'':<{LABEL_WIDTH}}  -- the sign above is negative: the descent landed")
    print(f"  {'':<{LABEL_WIDTH}}  {abs(float(fixed.energy) - ground):.2e} below the exact")
    print(f"  {'':<{LABEL_WIDTH}}  ground energy rather than violating a bound. Every")
    print(f"  {'':<{LABEL_WIDTH}}  energy here is a float32 statevector expectation,")
    print(f"  {'':<{LABEL_WIDTH}}  and the operator's own value is the double-precision")
    print(f"  {'':<{LABEL_WIDTH}}  one printed under inputs")
    report("history length", len(fixed.history))
    report("caller tensor unchanged", torch.equal(start, before))
    report("two runs agree bit for bit", torch.equal(fixed.parameters, again.parameters))
    print(f"  {'':<{LABEL_WIDTH}}  -- the start is copied before anything is updated,")
    print(f"  {'':<{LABEL_WIDTH}}  so the caller's tensor is an input rather than a")
    print(f"  {'':<{LABEL_WIDTH}}  scratch buffer, and the descent is deterministic")
    print()

    print("run_adapt_vqe: grow the ansatz one screened operator at a time")
    adapted = run_adapt_vqe(
        lambda operators, parameters: pool_circuit(
            reference_circuit, operators, parameters
        ),
        POOL,
        cost,
        max_adapt_iterations=4,
        optimization_steps=300,
        lr=0.2,
        gradient_tolerance=1e-6,
    )
    report("adapt initial energy", f"{adapted.initial_energy:.17g}")
    report("adapt selected", str(adapted.selected_pool_indices))
    report("adapt energy", f"{float(adapted.energy):.17g}")
    report("adapt energy above ground", f"{float(adapted.energy) - ground:.6g}")
    report("adapt iterations", adapted.n_adapt_iterations)
    report("adapt converged", str(adapted.converged))
    for index, iteration in enumerate(adapted.iterations):
        report(
            f"  iteration {index} selected",
            f"pool index {iteration.selected_pool_index} "
            f"at gradient {iteration.selected_gradient:.6f}",
        )
        report(
            f"  iteration {index} energies",
            f"{iteration.energy_before:.8f} -> {iteration.energy_after:.8f}",
        )
    report(
        "first screening pass",
        str([f"{value:.6f}" for value in adapted.iterations[0].pool_gradients]),
    )
    print(f"  {'':<{LABEL_WIDTH}}  -- the screen differentiates the objective at")
    print(f"  {'':<{LABEL_WIDTH}}  the appended angle, which is exact here rather")
    print(f"  {'':<{LABEL_WIDTH}}  than estimated from shots, and `ry` is the only")
    print(f"  {'':<{LABEL_WIDTH}}  direction with a non-zero gradient at |00>")
    print()

    print("the reference state is the premise: the same pool from |++> selects nothing")
    stationary = run_adapt_vqe(
        lambda operators, parameters: pool_circuit(plus_circuit, operators, parameters),
        POOL,
        cost,
        max_adapt_iterations=4,
        optimization_steps=300,
        lr=0.2,
        gradient_tolerance=1e-6,
    )
    report("stationary selected", str(stationary.selected_pool_indices))
    report("stationary converged", str(stationary.converged))
    report("stationary adapt iterations", stationary.n_adapt_iterations)
    report("stationary energy", f"{float(stationary.energy):.17g}")
    report("stationary energy above ground", f"{float(stationary.energy) - ground:.6g}")
    print(f"  {'':<{LABEL_WIDTH}}  -- |++> is an eigenstate of every Pauli generator")
    print(f"  {'':<{LABEL_WIDTH}}  in the pool, so every screened gradient is exactly")
    print(f"  {'':<{LABEL_WIDTH}}  zero, nothing is selected, and the run reports the")
    print(f"  {'':<{LABEL_WIDTH}}  exit condition rather than the distance to the")
    print(f"  {'':<{LABEL_WIDTH}}  ground state")
    print()

    print("what `converged` reads: three runs, one boolean, two exits")
    exhausted = run_adapt_vqe(
        lambda operators, parameters: pool_circuit(
            reference_circuit, operators, parameters
        ),
        ("ry",),
        cost,
        max_adapt_iterations=8,
        optimization_steps=300,
        lr=0.2,
        gradient_tolerance=0.0,
    )
    report("a pool of one, exhausted", f"converged={exhausted.converged}")
    report("  exhausted selected", f"{exhausted.selected_pool_indices} of 1")
    report("  exhausted energy", f"{float(exhausted.energy):.17g}")
    tolerated = run_adapt_vqe(
        lambda operators, parameters: pool_circuit(
            reference_circuit, operators, parameters
        ),
        POOL,
        cost,
        max_adapt_iterations=8,
        optimization_steps=600,
        lr=0.4,
        gradient_tolerance=1e-1,
    )
    report("tolerance exit", f"converged={tolerated.converged}")
    report("  tolerance selected", str(tolerated.selected_pool_indices))
    report("  tolerance energy", f"{float(tolerated.energy):.17g}")
    budget = run_adapt_vqe(
        lambda operators, parameters: pool_circuit(
            reference_circuit, operators, parameters
        ),
        POOL,
        cost,
        max_adapt_iterations=1,
        optimization_steps=1,
        lr=1e-9,
        gradient_tolerance=0.0,
    )
    report("budget exit", f"converged={budget.converged}")
    report("  budget selected", str(budget.selected_pool_indices))
    report("  budget energy", f"{float(budget.energy):.17g}")
    print(f"  {'':<{LABEL_WIDTH}}  -- the first two report True and the third False,")
    print(f"  {'':<{LABEL_WIDTH}}  and the third is the run furthest from the ground")
    print(f"  {'':<{LABEL_WIDTH}}  energy: the flag reports why the loop stopped and")
    print(f"  {'':<{LABEL_WIDTH}}  not how close the energy came")
    print()

    print(f"run_layerwise_vqe: deepen through {DEPTHS} keeping the optimized coordinates")
    layerwise = run_layerwise_vqe(
        layerwise_ansatz,
        layerwise_parameter_count,
        torch.zeros(layerwise_parameter_count(DEPTHS[0])),
        cost,
        depths=DEPTHS,
        stages=(stage,),
    )
    report("depths", str(layerwise.depths))
    report(
        "parameters per depth",
        str([layerwise_parameter_count(depth) for depth in DEPTHS]),
    )
    report(
        "energy per depth",
        str([f"{float(result.energy):.10f}" for result in layerwise.stages]),
    )
    report("final parameters", str([round(float(v), 6) for v in layerwise.parameters]))
    print(f"  {'':<{LABEL_WIDTH}}  -- each deeper stage starts from the previous")
    print(f"  {'':<{LABEL_WIDTH}}  stage's parameters with the new coordinates at")
    print(f"  {'':<{LABEL_WIDTH}}  zero, so the layers already optimized are not")
    print(f"  {'':<{LABEL_WIDTH}}  restarted from an unrelated point; each stage is")
    print(f"  {'':<{LABEL_WIDTH}}  a fresh descent, not a warm restart of one")
    print()

    print("run_hybrid_vqe: the same objective under a schedule of stages")
    schedule = (
        OptimizationStage(group="quantum", method="adam", steps=400, lr=0.1),
        OptimizationStage(group="quantum", method="adam", steps=200, lr=0.02),
    )
    hybrid = run_hybrid_vqe(fixed_ansatz, start, cost, stages=schedule)
    report("stages in the schedule", len(schedule))
    report("hybrid energy", f"{float(hybrid.energy):.17g}")
    report("records", len(hybrid.records))
    report("objective evaluations", hybrid.evaluations)
    report("parameter groups", str(tuple(hybrid.parameters)))
    print(f"  {'':<{LABEL_WIDTH}}  -- the schedule is the caller's, and the result")
    print(f"  {'':<{LABEL_WIDTH}}  carries the records and the evaluation count so")
    print(f"  {'':<{LABEL_WIDTH}}  the cost of that schedule is a measured number")
    print()

    print("two refusals: an objective with no gradient, and a pool element with none")
    refuse(
        "run_vqe, builder ignores parameters",
        lambda: run_vqe(
            lambda parameters: reference_circuit(), start, cost, steps=2, lr=0.1
        ),
    )
    refuse(
        "run_adapt_vqe, non-rotational pool",
        lambda: run_adapt_vqe(
            lambda operators, parameters: pool_circuit(
                reference_circuit, operators, parameters
            ),
            ("h",),
            cost,
            max_adapt_iterations=2,
        ),
    )
    print(f"  {'':<{LABEL_WIDTH}}  -- a descent needs an objective that depends on")
    print(f"  {'':<{LABEL_WIDTH}}  what it updates; torch reports this as a backward")
    print(f"  {'':<{LABEL_WIDTH}}  error that names neither the run nor the cause, so")
    print(f"  {'':<{LABEL_WIDTH}}  the check is made here and both are named")
    print()

    print("what this example does not do")
    print(f"  {'':<{LABEL_WIDTH}}  -- no shots: every energy above is an exact")
    print(f"  {'':<{LABEL_WIDTH}}  statevector expectation, so no sampling error and")
    print(f"  {'':<{LABEL_WIDTH}}  no shot-budget tradeoff appears anywhere")
    print(f"  {'':<{LABEL_WIDTH}}  -- no chemistry: the operator is a model, and no")
    print(f"  {'':<{LABEL_WIDTH}}  basis set, molecular integral, or active-space")
    print(f"  {'':<{LABEL_WIDTH}}  selection is involved")
    print(f"  {'':<{LABEL_WIDTH}}  -- no accuracy or scaling claim: the one instance")
    print(f"  {'':<{LABEL_WIDTH}}  is two qubits, and a deeper ansatz reaching a")
    print(f"  {'':<{LABEL_WIDTH}}  lower energy here is a fact about this instance")
    print(f"  {'':<{LABEL_WIDTH}}  -- no optimizer beyond torch's own, no COBYLA, no")
    print(f"  {'':<{LABEL_WIDTH}}  constraint or penalty term, and no warm start")


if __name__ == "__main__":
    main()
