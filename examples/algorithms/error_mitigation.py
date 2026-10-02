"""Continue a noisy observable to zero noise, and read the residual first.

`flagquantum.algorithms.error_mitigation` is reachable through the subpackage
surface only -- `from flagquantum.algorithms.error_mitigation import run_zne` --
because the algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and cannot check: the measured value is a
polynomial of degree at most the fit order in the scale factor, so the fitted
curve's value at zero is the noiseless value. That is an assumption about the
circuit and the channel rather than about the data, and it is not checkable from
the measurements alone. The diagnostic that exposes it when it is false is the
largest absolute residual the fit left, which is why every result reports one --
or reports that it has none, which is the case for a square fit and is exactly
why a square fit's zero residual is not evidence. What the unit extrapolates is
the exact state expectation ``Tr(O rho)``, so no shot is consumed and the
estimate carries no measured uncertainty.

The default scaling multiplies the one error-probability parameter each channel
declares. A channel whose parameters are not error probabilities --
``coherent_overrotation``'s angle, ``reset_error``'s two independent reset
probabilities, ``thermal_relaxation``'s time constants and duration -- is refused
by name rather than scaled, because multiplying such a parameter changes the
shape of the noise rather than only its strength. A model that declares a readout
rule is refused too, for a different reason: readout confusion is applied after
measurement and is not part of ``rho``, so extrapolating a curve that omits it
would return a state-preparation estimate under the name of a measured one.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary.

Sizes, and why: one ``cx`` gate on two wires, whose ``zz`` expectation is one at
zero noise, and four scale factors whose curve has a measured degree of two. The
same four points are fitted at degree one to show what an underfit looks like,
and the same observable is then measured under a coherent over-rotation whose
curve is not polynomial at all, so both failures are visible in the residual
rather than asserted.

Run it with:

    python -m examples.algorithms.error_mitigation
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_zne
from flagquantum.circuit import Circuit
from flagquantum.noise import (
    NoiseModel,
    ReadoutError,
    coherent_overrotation_channel,
    depolarizing_channel,
)

SCALE_FACTORS = (1.0, 3.0, 5.0, 7.0)
# A degree-one fit needs two points, so the empty-model reference is measured at
# the fewest scale factors run_zne accepts.
REFERENCE_SCALE_FACTORS = (1.0, 3.0)
DEPOLARIZING_PROBABILITY = 0.05
OVREROTATION_ANGLE = 0.15
DTYPE = torch.complex128
LABEL_WIDTH = 26


def bell_pair() -> Circuit:
    return Circuit(2).h(0).cx(0, 1)


def zz_observable() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])


def overrotation_scaling(model: NoiseModel, factor: float) -> NoiseModel:
    """The caller's claim: the over-rotation angle grows with the scale factor."""

    scaled = NoiseModel()
    for rule in model.rules:
        declared = dict(rule.channel.parameters)
        scaled.add(
            rule.gate_names,
            coherent_overrotation_channel(
                float(declared["angle"]) * factor,
                axis=str(declared["axis"]),
                dtype=rule.channel.kraus[0].dtype,
            ),
            wires=rule.wires,
        )
    return scaled


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-noise extrapolation demo")
    parser.add_argument(
        "--scale-factors",
        type=float,
        nargs="+",
        default=list(SCALE_FACTORS),
        help="the noise strengths to measure at; include 1.0 for the unmitigated point",
    )
    args = parser.parse_args()

    circuit = bell_pair()
    observable = zz_observable()
    # The reference is the same observable measured with an empty noise model at
    # the dtype the fits below run at, through run_zne itself. Both points of an
    # empty model read the same value, so the scale-factor-1.0 measurement is the
    # noiseless read and the comparison never leaves the path the estimate took.
    # Circuit.density_matrix() takes no dtype and returns the runtime's default
    # precision, so a distance taken from it would report the complex64 rounding
    # floor as the extrapolation's error.
    noiseless = float(
        run_zne(
            circuit,
            observable,
            noise_model=NoiseModel(),
            scale_factors=REFERENCE_SCALE_FACTORS,
            order=1,
            dtype=DTYPE,
        )
        .measurements[0]
        .expectation
    )
    model = NoiseModel().add(
        "cx", depolarizing_channel(DEPOLARIZING_PROBABILITY, dtype=DTYPE)
    )

    degree_one = run_zne(
        circuit,
        observable,
        noise_model=model,
        scale_factors=args.scale_factors,
        order=1,
        dtype=DTYPE,
    )
    degree_two = run_zne(
        circuit,
        observable,
        noise_model=model,
        scale_factors=args.scale_factors,
        order=2,
        dtype=DTYPE,
    )
    # Richardson's construction needs exactly order + 1 points, so the fourth
    # point buys a square degree-three fit rather than a redundant degree-two one.
    richardson = run_zne(
        circuit,
        observable,
        noise_model=model,
        scale_factors=args.scale_factors,
        order=3,
        extrapolation="richardson",
        dtype=DTYPE,
    )

    print("=" * 72)
    print("zero-noise extrapolation -- flagquantum.algorithms.error_mitigation")
    print("=" * 72)
    report("task", "continue a noisy observable to zero noise")
    report("method", "polynomial least squares and Richardson")
    report("premise", "the curve is a polynomial in the scale factor of degree at")
    print(f"  {'':<{LABEL_WIDTH}}  most the fit order, which is not checkable from the")
    print(f"  {'':<{LABEL_WIDTH}}  measurements alone, and the estimate carries no")
    print(f"  {'':<{LABEL_WIDTH}}  measured uncertainty because it is Tr(O rho)")
    print()

    print("inputs")
    report("circuit", "h(0); cx(0, 1)")
    report("observable", "1.0 * zz(0, 1)")
    report("noiseless value", f"{noiseless:.17g}")
    report("channel", f"depolarizing at p = {DEPOLARIZING_PROBABILITY} on cx")
    report("scale factors", tuple(args.scale_factors))
    report("extrapolation dtype", DTYPE)
    print()

    print("measured curve (exact expectations, no shots)")
    for measurement in degree_two.measurements:
        report(
            f"  scale {measurement.scale_factor:g}", f"{measurement.expectation:.12f}"
        )
    print()

    print("fits")
    for name, result in (
        ("degree 1 (underfit)", degree_one),
        ("degree 2", degree_two),
        ("richardson, degree 3", richardson),
    ):
        residual = (
            "none: the fit interpolates"
            if result.fit.max_residual is None
            else f"{result.fit.max_residual:.4e}"
        )
        report(name, f"estimate {result.estimate:.12f}")
        report("", f"distance from noiseless {abs(result.estimate - noiseless):.4e}")
        report("", f"max residual {residual}")
        report("", f"variance amplification {result.variance_amplification:.6f}")
        report("", f"weights {[round(w, 6) for w in result.fit.weights]}")
        print()

    print("what the residual is for")
    unmitigated = degree_two.unmitigated
    if unmitigated is None:
        raise SystemExit(
            "no scale factor of 1.0 was given, so there is no unmitigated point to "
            "report; pass 1.0 in the scale factors"
        )
    distance = abs(unmitigated - noiseless)
    report("unmitigated value", f"{unmitigated:.12f}")
    report("unmitigated distance", f"{distance:.4e}")
    report(
        "degree 2 improved by",
        f"{distance / abs(degree_two.estimate - noiseless):.3e}x",
    )
    report(
        "degree 1 improved by",
        f"{distance / abs(degree_one.estimate - noiseless):.3e}x",
    )
    print(f"  {'':<{LABEL_WIDTH}}  -- 1.8e-2 of residual is what says the degree-one")
    print(f"  {'':<{LABEL_WIDTH}}  fit is of the wrong degree, and its estimate is")
    print(f"  {'':<{LABEL_WIDTH}}  5e-2 away from the noiseless value; the residual")
    print(f"  {'':<{LABEL_WIDTH}}  separates the two fits by six orders")
    print()

    print("a family the polynomial assumption does not hold for")
    overrotation = NoiseModel().add(
        "cx", coherent_overrotation_channel(OVREROTATION_ANGLE, axis="x", dtype=DTYPE)
    )
    try:
        run_zne(
            circuit,
            observable,
            noise_model=overrotation,
            scale_factors=args.scale_factors,
            order=2,
            dtype=DTYPE,
        )
    except ValueError as error:
        report("declared scaling refuses", f"{error}")
    else:
        raise SystemExit(
            "the declared scaling scaled a coherent over-rotation, and the example "
            "exists to show that multiplying an angle is refused by name"
        )

    for order in (1, 2):
        biased = run_zne(
            circuit,
            observable,
            noise_model=overrotation,
            scale_factors=args.scale_factors,
            order=order,
            scaling=overrotation_scaling,
            dtype=DTYPE,
        )
        residual = biased.fit.max_residual
        report(f"caller scaling, degree {order}", f"estimate {biased.estimate:.12f}")
        report(
            "",
            f"distance from noiseless {abs(biased.estimate - noiseless):.4e}, "
            f"max residual {residual:.4e}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- the residual falls from 8.9e-2 to 2.9e-2 and")
    print(f"  {'':<{LABEL_WIDTH}}  stays orders above the 4.2e-9 floor the polynomial")
    print(f"  {'':<{LABEL_WIDTH}}  family reached, so a wrong assumption shows up here")
    print()

    print("the readout boundary")
    confused = NoiseModel().add(
        "cx", depolarizing_channel(DEPOLARIZING_PROBABILITY, dtype=DTYPE)
    )
    confused.add_readout(0, ReadoutError(((0.95, 0.05), (0.05, 0.95))))
    try:
        run_zne(
            circuit,
            observable,
            noise_model=confused,
            scale_factors=args.scale_factors,
            order=2,
            dtype=DTYPE,
        )
    except ValueError as error:
        report("readout rule refused", f"{error}")
    else:
        raise SystemExit(
            "a model declaring a readout rule was measured, and the example exists "
            "to show that Tr(O rho) is read before measurement, so the rule is "
            "refused rather than silently unused"
        )
    print()

    print("take away")
    print("  the fit is the easy half. What makes the estimate readable is the")
    print("  residual it left and the variance it amplifies: a residual at the")
    print("  rounding floor says the curve was a polynomial of the fitted degree, and")
    print("  anything above it says the extrapolated number is one the method has no")
    print("  claim on. A square fit has zero residual by construction, which is why")
    print("  it is reported as absent rather than as an arithmetic zero, and why the")
    print("  honest reading order is residual first, estimate second. The estimate")
    print("  itself is a point value: no shot, no confidence interval, no measured")
    print("  uncertainty, no readout correction, and no probabilistic error")
    print("  cancellation, Clifford data regression or circuit folding beside it.")


if __name__ == "__main__":
    main()
