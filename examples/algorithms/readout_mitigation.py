"""Correct a measured distribution for the readout confusion a model declares.

`flagquantum.algorithms.readout_mitigation` is reachable through the subpackage
surface only -- `from flagquantum.algorithms import plan_readout_mitigation` --
because the algorithms package adds no root-level `fq.` name beyond the
re-exports it declares.

The premise this unit rests on has two halves, and the script prints both. The
first is that the confusion matrix a model declares **is** the device's readout
behaviour: a miscalibrated, drifting, or correlated-in-a-way-the-model-does-not-
carry readout error survives the inversion untouched, so a corrected
distribution that lands on the ideal one is a statement about the model before
it is a statement about a device. The second is that the correction pays for
itself in shots. Inverting a stochastic matrix is an amplification, the
amplification is reported as a number, and the standard error of anything read
off the corrected vector grows with it -- so the unit is **not** a free
improvement and it says so rather than returning a clean-looking vector.

What makes the output honest is that it is under no obligation to be a
distribution. The inverse of a stochastic matrix has negative entries whenever
the confusion is invertible and not a permutation, so a corrected vector
routinely carries negative mass. Clipping that mass would return a different
vector under the same name, so the mass is summed, reported beside the vector,
and `is_physical` states whether the result happened to land in the simplex
rather than asserting that it must.

`docs/guides/ALGORITHMS.md` carries the per-unit boundary.

Run it with:

    python -m examples.algorithms.readout_mitigation
"""

from __future__ import annotations

import torch

from flagquantum.algorithms import (
    plan_readout_mitigation,
    run_readout_mitigation,
    run_readout_mitigation_counts,
)
from flagquantum.circuit import Circuit
from flagquantum.noise import (
    CorrelatedReadoutError,
    NoiseModel,
    ReadoutError,
    ReadoutRule,
)
from flagquantum.runtime import ExecutionOptions, run

CONFUSION = 0.08
CORRELATED = 0.06
SHOTS = 4000
DTYPE = torch.complex128
LABEL_WIDTH = 26


def circuit() -> Circuit:
    """One excitation on qubit 1, so the ideal distribution is a point mass."""
    return Circuit(2, dtype=DTYPE).x(1)


def exact_distribution(program: Circuit) -> torch.Tensor:
    """Read the runtime's exact computational-basis distribution."""
    state = run(program, options=ExecutionOptions()).state[0]
    return torch.as_tensor((state.conj() * state).real, dtype=torch.float64)


def symmetric(probability: float) -> ReadoutError:
    return ReadoutError(
        ((1.0 - probability, probability), (probability, 1.0 - probability))
    )


def correlated_matrix(probability: float) -> CorrelatedReadoutError:
    """A two-qubit confusion that is not the product of its own marginals."""
    return CorrelatedReadoutError(
        (
            (1.0 - 2 * probability, probability, probability, 0.0),
            (probability, 1.0 - 2 * probability, 0.0, probability),
            (probability, 0.0, 1.0 - 2 * probability, probability),
            (0.0, probability, probability, 1.0 - 2 * probability),
        )
    )


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def main() -> None:
    print("=" * 72)
    print("readout-error mitigation -- flagquantum.algorithms.readout_mitigation")
    print("=" * 72)
    report("task", "undo a declared readout confusion on a measured vector")
    report("method", "invert the block, one block per readout rule")
    report(
        "premise",
        "the confusion the model declares is the device's readout",
    )
    print(f"  {'':<{LABEL_WIDTH}}  behaviour, and the correction pays for itself")
    print(f"  {'':<{LABEL_WIDTH}}  in shots: inverting a stochastic matrix is an")
    print(f"  {'':<{LABEL_WIDTH}}  amplification, and the standard error of any")
    print(f"  {'':<{LABEL_WIDTH}}  value read off the corrected vector grows with")
    print(f"  {'':<{LABEL_WIDTH}}  it, so the correction is not a free improvement.")
    print(f"  {'':<{LABEL_WIDTH}}  A miscalibrated or drifting readout error survives")
    print(f"  {'':<{LABEL_WIDTH}}  the inversion untouched, so landing on the ideal")
    print(f"  {'':<{LABEL_WIDTH}}  distribution here says what the model is, not what")
    print(f"  {'':<{LABEL_WIDTH}}  the device is")
    print()

    print("one qubit, corrected onto the exact distribution")
    ideal = exact_distribution(circuit())
    model = NoiseModel(readout_rules=[ReadoutRule((1,), symmetric(CONFUSION))])
    plan = plan_readout_mitigation(model, n_qubits=2)
    measured = model.apply_readout_probabilities(ideal, n_wires=2)
    corrected = run_readout_mitigation(measured, plan)
    report("confusion on qubit 1", f"{CONFUSION}")
    report("blocks", [list(block.qubits) for block in plan.blocks])
    report("covered qubits", list(plan.covered_qubits))
    report("uncovered qubits", list(plan.uncovered_qubits))
    report("ideal", [f"{value:.6f}" for value in ideal.tolist()])
    report("measured", [f"{value:.6f}" for value in measured.tolist()])
    report("corrected", [f"{value:.12f}" for value in corrected.probabilities.tolist()])
    report("largest error", f"{float((corrected.probabilities - ideal).abs().max()):.3e}")
    print(f"  {'':<{LABEL_WIDTH}}  qubit 0 carries no rule, so it is left exactly")
    print(f"  {'':<{LABEL_WIDTH}}  where it was and is recorded as uncovered: the")
    print(f"  {'':<{LABEL_WIDTH}}  correction is the identity there rather than a")
    print(f"  {'':<{LABEL_WIDTH}}  guessed confusion. The error above is the")
    print(f"  {'':<{LABEL_WIDTH}}  float64 floor of inverting one 2 by 2 matrix")
    print()

    print("a correlated pair is inverted as one block, not two")
    pair = NoiseModel(
        readout_rules=[ReadoutRule((0, 1), correlated_matrix(CORRELATED))]
    )
    pair_plan = plan_readout_mitigation(pair, n_qubits=2)
    pair_ideal = exact_distribution(Circuit(2, dtype=DTYPE).x(1))
    pair_measured = pair.apply_readout_probabilities(pair_ideal, n_wires=2)
    pair_corrected = run_readout_mitigation(pair_measured, pair_plan)
    report("blocks", [list(block.qubits) for block in pair_plan.blocks])
    report("block width", pair_plan.blocks[0].probabilities.shape[0])
    report(
        "smallest singular value",
        f"{pair_plan.blocks[0].smallest_singular_value:.6f}",
    )
    report(
        "largest error",
        f"{float((pair_corrected.probabilities - pair_ideal).abs().max()):.3e}",
    )
    print(f"  {'':<{LABEL_WIDTH}}  the pair does not factorize, so splitting it")
    print(f"  {'':<{LABEL_WIDTH}}  into two one-qubit rules would invert a different")
    print(f"  {'':<{LABEL_WIDTH}}  operator; the block is inverted whole and the")
    print(f"  {'':<{LABEL_WIDTH}}  block's own condition number is what bounds the")
    print(f"  {'':<{LABEL_WIDTH}}  amplification on those two qubits")
    print()

    print("what the correction costs, and how the cost composes")
    report("one-qubit amplification", f"{plan.sampling_overhead:.9f}")
    report("pair amplification", f"{pair_plan.sampling_overhead:.9f}")
    report("pair condition number", f"{pair_plan.condition_number:.9f}")
    report("shots", SHOTS)
    report(
        "standard error bound",
        f"{run_readout_mitigation(measured, plan, shots=SHOTS).standard_error_bound:.9f}",
    )
    print(f"  {'':<{LABEL_WIDTH}}  the reported amplification is the whole map's")
    print(f"  {'':<{LABEL_WIDTH}}  induced 1-norm, which for a map that is the")
    print(f"  {'':<{LABEL_WIDTH}}  tensor product of disjoint blocks is the")
    print(f"  {'':<{LABEL_WIDTH}}  **product** of the blocks' own -- equal to the")
    print(f"  {'':<{LABEL_WIDTH}}  worst block's only when there is one block. The")
    print(f"  {'':<{LABEL_WIDTH}}  standard error bound is that amplification over the")
    print(f"  {'':<{LABEL_WIDTH}}  square root of the shot count, so a device whose")
    print(f"  {'':<{LABEL_WIDTH}}  readout is confused on many qubits is paid for in")
    print(f"  {'':<{LABEL_WIDTH}}  shots rather than hidden")
    print()

    print("a corrected vector is not obliged to be a distribution")
    skewed = NoiseModel(readout_rules=[ReadoutRule((0,), symmetric(0.2))])
    skewed_plan = plan_readout_mitigation(skewed, n_qubits=1)
    skewed_result = run_readout_mitigation(
        torch.tensor([0.9, 0.1], dtype=torch.float64), skewed_plan
    )
    report(
        "skewed measured",
        [f"{value:.6f}" for value in skewed_result.observed.tolist()],
    )
    report(
        "skewed corrected",
        [f"{value:.6f}" for value in skewed_result.probabilities.tolist()],
    )
    report("negative mass", f"{skewed_result.negative_mass:.6f}")
    report("total variation", f"{skewed_result.total_variation:.6f}")
    report("is_physical", skewed_result.is_physical)
    print(f"  {'':<{LABEL_WIDTH}}  the corrected vector sums to one and holds a")
    print(f"  {'':<{LABEL_WIDTH}}  negative entry: the inverse of a stochastic matrix")
    print(f"  {'':<{LABEL_WIDTH}}  is not stochastic. Clipping the negative mass would")
    print(f"  {'':<{LABEL_WIDTH}}  return a different vector under the same name, so")
    print(f"  {'':<{LABEL_WIDTH}}  it is summed and reported beside the vector, and")
    print(f"  {'':<{LABEL_WIDTH}}  is_physical states which side of the simplex this")
    print(f"  {'':<{LABEL_WIDTH}}  one landed on")
    print()

    print("a histogram is keyed the way the runtime keys it")
    counts_plan = plan_readout_mitigation(
        NoiseModel(readout_rules=[ReadoutRule((1,), symmetric(CONFUSION))]),
        n_qubits=2,
    )
    histogram = {"01": 3680, "00": 320}
    from_counts = run_readout_mitigation_counts(histogram, counts_plan)
    report("histogram", histogram)
    report("total shots", from_counts.shots)
    report(
        "counts corrected",
        [f"{value:.6f}" for value in from_counts.probabilities.tolist()],
    )
    report("standard error bound", f"{from_counts.standard_error_bound:.9f}")
    print(f"  {'':<{LABEL_WIDTH}}  the key's first character is qubit 0, which is the")
    print(f"  {'':<{LABEL_WIDTH}}  convention the flat index space uses: the string")
    print(f"  {'':<{LABEL_WIDTH}}  '01' is qubit 1 set and its flat index is 1. The")
    print(f"  {'':<{LABEL_WIDTH}}  histogram is normalized by its own total and the")
    print(f"  {'':<{LABEL_WIDTH}}  total travels into the result, so the bound above")
    print(f"  {'':<{LABEL_WIDTH}}  is quoted for the shots this histogram actually")
    print(f"  {'':<{LABEL_WIDTH}}  holds rather than for a count the caller forgot")
    print()

    print("what is refused, by name")
    half = NoiseModel(readout_rules=[ReadoutRule((0,), symmetric(0.5))])
    try:
        plan_readout_mitigation(half, n_qubits=1)
    except Exception as error:
        report("a maximally confused qubit", f"{type(error).__name__} -- {error}")
    else:
        raise SystemExit(
            "a maximally confused qubit was inverted, and the example exists to show "
            "that a block whose singular value reaches the floor is refused by name "
            "rather than inverted into weights no shot count can support"
        )
    wider = 11
    identity = CorrelatedReadoutError(
        tuple(
            tuple(1.0 if row == column else 0.0 for column in range(2**wider))
            for row in range(2**wider)
        )
    )
    try:
        plan_readout_mitigation(
            NoiseModel(readout_rules=[ReadoutRule(tuple(range(wider)), identity)]),
            n_qubits=wider,
        )
    except Exception as error:
        report("a block of 11 qubits", f"{type(error).__name__} -- {error}")
    else:
        raise SystemExit(
            "an eleven-qubit correlated block was inverted, and the example exists to "
            "show that the ceiling is a memory statement that is enforced"
        )
    try:
        run_readout_mitigation_counts({"0": 1}, counts_plan)
    except Exception as error:
        report("a one-character histogram", f"{type(error).__name__} -- {error}")
    else:
        raise SystemExit(
            "a histogram of the wrong width was accepted, and the example exists to "
            "show that the width is the plan's and the two cannot disagree"
        )
    print(f"  {'':<{LABEL_WIDTH}}  each refusal names what it measured rather than")
    print(f"  {'':<{LABEL_WIDTH}}  what it expected: the singular value and the floor")
    print(f"  {'':<{LABEL_WIDTH}}  for the first, the block width and the matrix size")
    print(f"  {'':<{LABEL_WIDTH}}  for the second, the two widths for the third")
    print()

    print("take away")
    print("  the correction is exact whenever the declared confusion is the device's")
    print("  confusion, and that is the whole of the claim. It is not free: the")
    print("  amplification is reported as a number, it composes multiplicatively over")
    print("  the device's readout blocks, and it shows up as a wider error bar rather")
    print("  than as a better answer. The corrected vector is not a density matrix")
    print("  and is not forced to be one -- its negative mass is measured and")
    print("  reported. What is beside it: zero-noise extrapolation scales a declared")
    print("  channel, probabilistic error cancellation inverts one, and Clifford data")
    print("  regression fits the noise's effect on training circuits. What is not")
    print("  here: any smoothing or maximum-likelihood refit that would restore")
    print("  positivity, any tensored approximation for a block wider than the")
    print("  ceiling, and any correction of a confusion the model does not carry.")


if __name__ == "__main__":
    main()
