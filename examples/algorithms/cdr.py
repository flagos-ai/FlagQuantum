"""Fit the noise's effect on Clifford circuits, and read the residual beside it.

`flagquantum.algorithms.cdr` is reachable through the subpackage surface only --
`from flagquantum.algorithms.cdr import run_cdr` -- because the algorithms
package adds no root-level `fq.` name beyond the re-exports it declares.

The premise the unit rests on: over the region the training circuits span, the
ideal expectation is affine in the noisy one. The training circuits are built
from the target by snapping each `rx`, `ry`, `rz`, `phase` and `u1` rotation to
its nearest quarter turn and rewriting it into named Clifford gates, so the
premise that makes the method cheap on hardware -- every training circuit is a
stabilizer circuit -- is enforced here and never exploited. The ideal values come
from exact density simulation of those circuits, which costs exactly what
simulating the target costs. Nothing in this example is cheaper than the target
it corrects, and no simulation-cost, capacity or scaling claim follows from the
fact that the training circuits happen to be Clifford.

Two things make the fit's honesty visible. Every training circuit is accepted by
`flagquantum.simulation.stabilizer.require_clifford_program` before it is
measured, so a program the rewrite cannot reach is refused by the engine rather
than snapped approximately. And the largest absolute residual the fit left is
reported, because it is the only diagnostic of a premise that did not hold; when
the training set has two points the residual is reported as absent rather than as
zero, since two points determine a line and the zero would be an arithmetic
identity presented as a check.

`docs/guides/ALGORITHMS.md` carries the per-unit boundary.

Run it with:

    python -m examples.algorithms.cdr
"""

from __future__ import annotations

import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_cdr
from flagquantum.algorithms.cdr import CDR_SNAP_OPCODES, clifford_variants
from flagquantum.circuit import Circuit
from flagquantum.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.simulation.density_matrix import density_matrix_from_ir

CX_PROBABILITY = 0.06
H_PROBABILITY = 0.03
DTYPE = torch.complex128
LABEL_WIDTH = 26


def target_scalar() -> Circuit:
    """One rotation and one entangler, observed as a single Pauli."""
    return Circuit(2, dtype=DTYPE).h(0).ry(0, 0.6).cx(0, 1)


def target_sum() -> Circuit:
    """Two rotations, so the training set has three points and a real residual."""
    return Circuit(2, dtype=DTYPE).h(0).ry(0, 0.7).ry(1, 0.4).cx(0, 1)


def z0() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "z", (0,))])


def z_plus_x() -> Hamiltonian:
    return Hamiltonian(
        [HamiltonianTerm(1.0, "z", (0,)), HamiltonianTerm(1.0, "x", (1,))]
    )


def model(*, with_h: bool) -> NoiseModel:
    rules = NoiseModel()
    rules.add("cx", two_qubit_depolarizing_channel(CX_PROBABILITY, dtype=DTYPE))
    if with_h:
        rules.add("h", depolarizing_channel(H_PROBABILITY, dtype=DTYPE))
    return rules


def exact(circuit: Circuit, observable: Hamiltonian) -> float:
    density = density_matrix_from_ir(circuit, bsz=1, device="cpu", dtype=DTYPE)
    return float(torch.as_tensor(observable.expectation(density)).reshape(-1)[0])


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def main() -> None:
    print("=" * 72)
    print("clifford data regression -- flagquantum.algorithms.cdr")
    print("=" * 72)
    report("task", "fit the noise's effect, do not invert the model")
    report("method", "snap each rotation to its nearest quarter turn")
    report("premise", "the ideal expectation is affine in the noisy one over")
    print(f"  {'':<{LABEL_WIDTH}}  the region the training circuits span, and")
    print(f"  {'':<{LABEL_WIDTH}}  that relation is not checkable from the")
    print(f"  {'':<{LABEL_WIDTH}}  measurements alone. The training circuits")
    print(f"  {'':<{LABEL_WIDTH}}  are Clifford, so the premise that would make")
    print(f"  {'':<{LABEL_WIDTH}}  this cheap on hardware is enforced but never")
    print(f"  {'':<{LABEL_WIDTH}}  exploited: every ideal value here comes from")
    print(f"  {'':<{LABEL_WIDTH}}  exact density simulation, at the target's own")
    print(f"  {'':<{LABEL_WIDTH}}  cost, and the estimate carries no error bound")
    print()

    print("the training set is a rewrite of the target")
    snapped = clifford_variants(target_sum())
    report("snappable rotations", list(CDR_SNAP_OPCODES))
    report("variants", [variant.name for variant in snapped])
    report(
        "angle shifts",
        [f"{variant.angle_shift:.6f}" for variant in snapped],
    )
    report(
        "nearest variant",
        [instruction.name for instruction in snapped[0].circuit.instructions],
    )
    print(f"  {'':<{LABEL_WIDTH}}  the nearest variant is the target with each")
    print(f"  {'':<{LABEL_WIDTH}}  rotation replaced by the word for the quarter")
    print(f"  {'':<{LABEL_WIDTH}}  turn it is closest to; each further variant")
    print(f"  {'':<{LABEL_WIDTH}}  moves exactly one site to its other turn. The")
    print(f"  {'':<{LABEL_WIDTH}}  angle shift is how far that site travelled, so")
    print(f"  {'':<{LABEL_WIDTH}}  a program already on the grid shifts by zero")
    print()

    print("a two-point fit, which is an identity rather than a check")
    circuit = target_scalar()
    observable = z0()
    noiseless = exact(circuit, observable)
    scalar = run_cdr(circuit, observable, noise_model=model(with_h=False), dtype=DTYPE)
    report("observable", "1.0 * z(0)")
    report("exact value", f"{noiseless:.15f}")
    report("training points", [(p.ideal, p.noisy) for p in scalar.fit.points])
    report("slope", f"{scalar.fit.slope:.12f}")
    report("intercept", f"{scalar.fit.intercept:.3e}")
    report("residual", scalar.fit.max_residual)
    report("degrees of freedom", scalar.fit.degrees_of_freedom)
    report("unmitigated", f"{scalar.unmitigated:.15f}")
    report("mitigated", f"{scalar.estimate:.15f}")
    report("unmitigated error", f"{abs(scalar.unmitigated - noiseless):.3e}")
    report("mitigated error", f"{abs(scalar.estimate - noiseless):.3e}")
    print(f"  {'':<{LABEL_WIDTH}}  the slope is the reciprocal of the shrinkage")
    print(f"  {'':<{LABEL_WIDTH}}  the model applied to the second training point,")
    print(f"  {'':<{LABEL_WIDTH}}  so it is a reading of the model rather than of a")
    print(f"  {'':<{LABEL_WIDTH}}  device. Two points determine a line, so the")
    print(f"  {'':<{LABEL_WIDTH}}  agreement above is arithmetic: the residual is")
    print(f"  {'':<{LABEL_WIDTH}}  reported as absent, not as zero, because nothing")
    print(f"  {'':<{LABEL_WIDTH}}  here checked the premise")
    print()

    print("a three-point fit, where the residual is a real diagnostic")
    circuit = target_sum()
    observable = z_plus_x()
    noiseless = exact(circuit, observable)
    sum_result = run_cdr(
        circuit, observable, noise_model=model(with_h=True), dtype=DTYPE
    )
    report("observable", "1.0 * z(0) + 1.0 * x(1)")
    report("exact value", f"{noiseless:.15f}")
    report("training points", [(p.ideal, p.noisy) for p in sum_result.fit.points])
    report(
        "distinct noisy values", len({round(p.noisy, 9) for p in sum_result.fit.points})
    )
    report("slope", f"{sum_result.fit.slope:.12f}")
    report("intercept", f"{sum_result.fit.intercept:.3e}")
    report("residual", f"{sum_result.fit.max_residual:.6e}")
    report("degrees of freedom", sum_result.fit.degrees_of_freedom)
    report("unmitigated", f"{sum_result.unmitigated:.15f}")
    report("mitigated", f"{sum_result.estimate:.15f}")
    report("unmitigated error", f"{abs(sum_result.unmitigated - noiseless):.3e}")
    report("mitigated error", f"{abs(sum_result.estimate - noiseless):.3e}")
    print(f"  {'':<{LABEL_WIDTH}}  three points leave one degree of freedom, so the")
    print(f"  {'':<{LABEL_WIDTH}}  residual above is a measurement rather than an")
    print(f"  {'':<{LABEL_WIDTH}}  identity, and it is not zero: the affine premise")
    print(f"  {'':<{LABEL_WIDTH}}  is close but not exact, the correction closes most")
    print(f"  {'':<{LABEL_WIDTH}}  of the gap rather than all of it, and the residual")
    print(f"  {'':<{LABEL_WIDTH}}  is what says so instead of the estimate being")
    print(f"  {'':<{LABEL_WIDTH}}  reported as the noiseless value")
    print()

    print("the same target with one noise source, where the premise holds")
    exact_line = run_cdr(
        circuit,
        observable,
        noise_model=NoiseModel().add(
            "cx", two_qubit_depolarizing_channel(CX_PROBABILITY, dtype=DTYPE)
        ),
        dtype=DTYPE,
    )
    report("residual", f"{exact_line.fit.max_residual:.3e}")
    report("mitigated error", f"{abs(exact_line.estimate - noiseless):.3e}")
    print(f"  {'':<{LABEL_WIDTH}}  the same three training circuits under one noise")
    print(f"  {'':<{LABEL_WIDTH}}  source fall on a line to the complex128 floor, and")
    print(f"  {'':<{LABEL_WIDTH}}  the residual reports that rather than being")
    print(f"  {'':<{LABEL_WIDTH}}  decorative. Nothing in the estimate changed; what")
    print(f"  {'':<{LABEL_WIDTH}}  changed is that the premise was checked and held")
    print()

    print("the readout boundary")
    confused = model(with_h=False)
    confused.add_readout(0, ReadoutError(((0.95, 0.05), (0.05, 0.95))))
    try:
        run_cdr(target_scalar(), z0(), noise_model=confused, dtype=DTYPE)
    except ValueError as error:
        report("readout rule refused", f"{error}")
    else:
        raise SystemExit(
            "a model declaring a readout rule was measured, and the example exists to "
            "show that Tr(O rho) is read before measurement, so the rule is refused "
            "rather than silently unused"
        )
    print()

    print("a program this unit cannot train on")
    entangling = Circuit(3, dtype=DTYPE).h(0).ry(0, 0.6).ccx(0, 1, 2)
    try:
        run_cdr(entangling, z0(), noise_model=model(with_h=False), dtype=DTYPE)
    except Exception as error:
        report("ccx refused", f"{type(error).__name__} -- {error}")
    else:
        raise SystemExit(
            "a program with no snappable rotation was trained on, and the example "
            "exists to show that the rewrite, not a second gate list, is what decides "
            "which operations this unit can reach"
        )
    print()

    print("take away")
    print("  the correction is a fitted line, so it is worth exactly what the")
    print("  premise is worth. A mitigated value close to the ideal says the")
    print("  relation held over the region the training circuits spanned, which is")
    print("  a statement about that region before it is a statement about a device.")
    print("  The residual is the only warning when it did not hold, and it is")
    print("  reported as absent rather than as zero when the fit had no freedom")
    print("  left to leave one. What is beside it: zero-noise extrapolation scales")
    print("  a declared channel, and probabilistic error cancellation inverts one.")
    print("  What is not here: readout-error mitigation, and any error bound, any")
    print("  confidence interval, and any claim that the training circuits being")
    print("  Clifford saved this path any simulation.")


if __name__ == "__main__":
    main()
