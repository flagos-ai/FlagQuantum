"""Invert a declared Pauli channel, and read the cost beside the estimate.

`flagquantum.algorithms.pec` is reachable through the subpackage surface only --
`from flagquantum.algorithms.pec import run_pec` -- because the algorithms
package adds no root-level `fq.` name.

The premise the unit rests on: the noise the program experiences is exactly the
channel the model declares, at the location the model declares it. Under that
premise a Pauli channel's inverse is a finite signed combination of Pauli words,
which the unit builds from the channel's Pauli transfer matrix and inserts after
the channel it inverts, so the composite map is the identity and the observable
read off the corrected program is the noiseless one. That is a statement about
the declared model, not about a device: an error the model does not carry -- a
miscalibrated gate, leakage, drift between the declaration and the run --
survives the inversion untouched. The readout boundary is a different one: the
observable is read as ``Tr(O rho)``, before measurement, so a model that declares
a readout rule is refused rather than measured with the rule silently unused.

Two things make the price visible. `gamma`, the absolute sum of the weights at a
location, is at least one and composes multiplicatively over locations; an
implementation that sampled the same combination would pay `gamma**2` in shots,
because a quasi-probability carries the sign of every weight into its variance.
And this path pays none of that: it evaluates every term exactly, so it consumes
no shots, reports no confidence interval, and its `gamma**2` is arithmetic about
a sampled implementation rather than a measurement of this one.

A channel is inverted only if its Pauli transfer matrix is diagonal, which is
what makes it a Pauli channel, and only if no transfer eigenvalue reaches zero,
which is what makes the inverse exist at all. Both boundaries are refused by
name rather than approximated, and each refusal quotes the magnitude it measured.

Sizes, and why: two noisy locations on two wires, whose `zz` expectation is one
at zero noise and whose unmitigated read is far enough from one to be worth
correcting. One location at a time is reported beside it, because the correction
composes per location and the cost of the whole is the product of the parts.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary.

Run it with:

    python -m examples.algorithms.pec
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_pec
from flagquantum.algorithms.pec import pauli_twirl_decomposition
from flagquantum.circuit import Circuit
from flagquantum.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_channel,
    bit_flip_channel,
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    reset_error_channel,
    thermal_relaxation_channel,
    two_qubit_depolarizing_channel,
)

NOISE_PROBABILITY = 0.1
DTYPE = torch.complex128
LABEL_WIDTH = 26


def bell_pair() -> Circuit:
    return Circuit(2, dtype=DTYPE).h(0).cx(0, 1)


def zz_observable() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probabilistic error cancellation demo"
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=NOISE_PROBABILITY,
        help="the bit-flip probability the inverted channel declares",
    )
    args = parser.parse_args()

    circuit = bell_pair()
    observable = zz_observable()
    model = NoiseModel().add("cx", bit_flip_channel(args.probability, dtype=DTYPE))
    result = run_pec(circuit, observable, noise_model=model, dtype=DTYPE)
    noiseless = float(observable.expectation(circuit))
    single = run_pec(
        circuit,
        observable,
        noise_model=NoiseModel().add(
            "cx", bit_flip_channel(args.probability, dtype=DTYPE), wires=(0,)
        ),
        dtype=DTYPE,
    )

    print("=" * 72)
    print("probabilistic error cancellation -- flagquantum.algorithms.pec")
    print("=" * 72)
    report("task", "invert a declared Pauli channel, not the circuit")
    report("method", "the channel's Pauli transfer matrix, inverted exactly")
    report("premise", "the noise is exactly the channel the model declares, at")
    print(
        f"  {'':<{LABEL_WIDTH}}  the location the model declares it, so an undeclared"
    )
    print(f"  {'':<{LABEL_WIDTH}}  error survives the inversion. This path consumes no")
    print(
        f"  {'':<{LABEL_WIDTH}}  shots, so its gamma**2 is a sampled implementation's"
    )
    print(f"  {'':<{LABEL_WIDTH}}  cost rather than a measurement of this one")
    print()

    print("inputs")
    report("circuit", "h(0); cx(0, 1)")
    report("observable", "1.0 * zz(0, 1)")
    report("noiseless value", f"{noiseless:.17g}")
    report("channel", f"bit_flip at p = {args.probability} on cx")
    report("locations", [f"{loc.channel_name}{loc.wires}" for loc in result.locations])
    report("estimate dtype", DTYPE)
    print()

    print("the correction")
    report("unmitigated", f"{result.unmitigated:.15f}")
    report("mitigated", f"{result.estimate:.15f}")
    report("unmitigated error", f"{abs(result.unmitigated - noiseless):.3e}")
    report("mitigated error", f"{abs(result.estimate - noiseless):.3e}")
    report("gamma", f"{result.gamma:.12f}")
    report(
        "sampling overhead",
        f"{result.sampling_overhead:.12f}  (gamma**2, a sampled run's cost)",
    )
    report("exact programs", f"{result.executions} = {result.term_count} terms + 1")
    report("shots consumed", 0)
    print(f"  {'':<{LABEL_WIDTH}}  the mitigated error above is the density")
    print(f"  {'':<{LABEL_WIDTH}}  simulation's own floor for a channel stored in")
    print(f"  {'':<{LABEL_WIDTH}}  this dtype, not a device measurement: the only")
    print(f"  {'':<{LABEL_WIDTH}}  error left is the arithmetic's, because the")
    print(f"  {'':<{LABEL_WIDTH}}  declared noise was removed exactly")
    print()

    print("the cost composes per location")
    report("one location", f"gamma {single.gamma:.12f}, {single.term_count} terms")
    report("two locations", f"gamma {result.gamma:.12f}, {result.term_count} terms")
    report("product of the parts", f"{single.gamma**2:.12f}")
    print(f"  {'':<{LABEL_WIDTH}}  a model naming one wire costs what it declares and")
    print(
        f"  {'':<{LABEL_WIDTH}}  no more, so the price is the declared model's before"
    )
    print(f"  {'':<{LABEL_WIDTH}}  it is the program's")
    print()

    print("what the inverse is worth, by channel family")
    for name, channel in (
        ("bit_flip", bit_flip_channel(args.probability, dtype=DTYPE)),
        ("phase_flip", phase_flip_channel(args.probability, dtype=DTYPE)),
        ("depolarizing", depolarizing_channel(args.probability, dtype=DTYPE)),
        ("phase_damping", phase_damping_channel(args.probability, dtype=DTYPE)),
        (
            "two_qubit_depolarizing",
            two_qubit_depolarizing_channel(args.probability, dtype=DTYPE),
        ),
    ):
        decomposition = pauli_twirl_decomposition(channel)
        report(
            name,
            f"gamma {decomposition.gamma:.9f}, "
            f"words {len(decomposition.pauli_words)}, "
            f"off-diagonal {decomposition.largest_offdiagonal:.1e}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  the off-diagonal figure is what decides admission,")
    print(
        f"  {'':<{LABEL_WIDTH}}  and every entry above is exactly zero: these transfer"
    )
    print(
        f"  {'':<{LABEL_WIDTH}}  matrices are diagonal, and at this dtype there is no"
    )
    print(f"  {'':<{LABEL_WIDTH}}  rounding left to hide a diagonal one behind")
    print()

    print("channels the inverse does not exist for, refused by name")
    for name, channel in (
        (
            "amplitude_damping",
            amplitude_damping_channel(args.probability, dtype=DTYPE),
        ),
        (
            "coherent_overrotation",
            coherent_overrotation_channel(0.2, axis="x", dtype=DTYPE),
        ),
        ("reset_error", reset_error_channel(0.05, dtype=DTYPE)),
        (
            "thermal_relaxation",
            thermal_relaxation_channel(0.05, 0.05, 1.0, dtype=DTYPE),
        ),
    ):
        try:
            pauli_twirl_decomposition(channel)
        except ValueError as error:
            report(name, f"refused -- {error}")
        else:
            raise SystemExit(
                f"{name} was admitted, and the example exists to show that a channel "
                "whose Pauli transfer matrix is not diagonal has no finite signed "
                "Pauli inverse to find"
            )
    print()

    print("a channel whose inverse is undefined rather than merely expensive")
    try:
        pauli_twirl_decomposition(bit_flip_channel(0.5, dtype=DTYPE))
    except ValueError as error:
        report("bit_flip at 0.5", f"refused -- {error}")
    else:
        raise SystemExit(
            "bit_flip at one half has a zero transfer eigenvalue and was admitted; "
            "the example exists to show that the refusal is about the inverse "
            "existing, not about the arithmetic being hard"
        )
    print()

    print("the readout boundary")
    confused = NoiseModel().add("cx", bit_flip_channel(args.probability, dtype=DTYPE))
    confused.add_readout(0, ReadoutError(((0.95, 0.05), (0.05, 0.95))))
    try:
        run_pec(circuit, observable, noise_model=confused, dtype=DTYPE)
    except ValueError as error:
        report("readout rule refused", f"{error}")
    else:
        raise SystemExit(
            "a model declaring a readout rule was measured, and the example exists to "
            "show that Tr(O rho) is read before measurement, so the rule is refused "
            "rather than silently unused"
        )
    print()

    print("take away")
    print("  the inversion is exact and the price is in the reporting. A mitigated")
    print("  value close to the ideal says the declared channel was inverted, which")
    print("  is a statement about the model: the same run against a device whose")
    print("  error is not the declared one returns the device's error, now carrying")
    print("  a signed weight. gamma is what the correction costs and gamma**2 is what")
    print("  a sampled implementation of it would cost, and neither is a measurement")
    print("  because this path consumes no shots. What is beside it: zero-noise")
    print("  extrapolation scales a channel instead of inverting it. What is not")
    print("  here: Clifford data regression and readout-error mitigation.")


if __name__ == "__main__":
    main()
