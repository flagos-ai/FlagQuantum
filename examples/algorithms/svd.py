"""Read a matrix's singular values off the phase of its Hermitian embedding.

`flagquantum.algorithms.svd` is reachable through the subpackage surface only --
`from flagquantum.algorithms.svd import estimate_singular_values` -- because the
algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and does not meet, and it has two halves that are
both load-bearing. **The input model is assumed**: the cited algorithm's cost is
counted in queries to a structure that returns the matrix's entries, and against
that count the state the estimation is applied to is assumed to be preparable.
Neither is present here -- the matrix is an ordinary tensor, its embedding and
that embedding's exponential are dense classical objects, and the input state is
built from the singular vectors a classical `torch.linalg.svd` returns, the very
decomposition the readout estimates. **And a block encoding is not free to
read**: a readout that post-selects the ancilla succeeds with probability
`||(A/alpha)|psi>||**2` on a normalised input, whose greatest value over inputs is
`(||A||/alpha)**2`, and where `alpha` is much larger than `||A||` that probability
is exponentially small. `docs/guides/ALGORITHMS.md` carries the per-unit boundary
in its "Singular values by phase estimation" section.

Sizes, and why: the matrix is the guide's own 2x2 instance, at six counting
wires, 20000 shots and sampling seed 11, so a reader who runs this script and
then reads the guide sees the same numbers. The matrix is 2x2 because the
embedding is twice as wide as the matrix and every form of the phase unitary is a
dense gate on it, and the unit is bounded at four rows and four columns. The
sample is the guide's rather than a smaller one because the readout is a grid
value the sample decides. The script then spends one more run on the width's own
boundary, at the narrowest width the caller may ask for.

Run it with:

    python -m examples.algorithms.svd
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms.svd import estimate_singular_values

MATRIX = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
N_COUNTING_WIRES = 6
# The one-wire runs. The register then holds two counter values and one of them is
# the refused half, so the first run returns alpha itself; the second is the same
# width at a sample of one shot, where the mode is the single draw the sample made
# and a draw that lands on the refused counter value raises.
ONE_WIRE_SHOTS = 1024
ONE_WIRE_SAMPLE_SHOTS = 1
ONE_WIRE_SEED = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Singular values by phase estimation")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    exact = [float(value) for value in torch.linalg.svdvals(MATRIX)]
    result = estimate_singular_values(
        MATRIX,
        n_counting_wires=N_COUNTING_WIRES,
        shots=args.shots,
        seed=args.seed,
    )
    mode = max(result.distribution, key=result.distribution.__getitem__)
    top_three = sorted(result.distribution.items(), key=lambda item: -item[1])[:3]
    one_wire = estimate_singular_values(
        MATRIX, n_counting_wires=1, shots=ONE_WIRE_SHOTS, seed=args.seed
    )
    # Two counter values at one wire, and the accepted one is the mode: the other is
    # the refused half, which is the value that reads above alpha.
    accepted = max(one_wire.distribution, key=one_wire.distribution.__getitem__)
    refused = next(key for key in one_wire.distribution if key != accepted)

    print("=" * 72)
    print("singular values by phase estimation -- flagquantum.algorithms.svd")
    print("=" * 72)
    print(f"  {'task':<18}: read a matrix's singular values off the phase of its")
    print(f"  {'':<18}  Hermitian embedding's exponential")
    print(f"  {'execution':<18}: local statevector sampling, CPU")
    print(f"  {'premise':<18}: the input model is assumed, not met -- the matrix, its")
    print(f"  {'':<18}  embedding and that exponential are dense classical objects,")
    print(f"  {'':<18}  and the input state is built from the singular vectors a")
    print(f"  {'':<18}  torch.linalg.svd returns, the decomposition the readout")
    print(f"  {'':<18}  estimates; and a block encoding is not free to read -- the")
    print(f"  {'':<18}  post-selected readout succeeds with probability")
    print(f"  {'':<18}  ||(A/alpha)|psi>||**2 at most (||A||/alpha)**2")
    print()
    print("inputs")
    print(f"  {'matrix':<18}: {MATRIX.tolist()}")
    print(f"  {'shots':<18}: {args.shots}")
    print(f"  {'seed':<18}: {args.seed}")
    print(f"  {'counting wires':<18}: {result.n_counting_wires}")
    print()
    print("result")
    print(f"  {'exact singular values':<18}: {[round(value, 6) for value in exact]}")
    print(f"  {'readout':<18}: {round(result.dominant_singular_value, 6)}")
    print(f"  {'readout share':<18}: {round(result.dominant_share, 4)}")
    print(f"  {'resolution':<18}: {round(result.resolution, 6)}")
    print(f"  {'alpha':<18}: {round(result.alpha, 6)}")
    largest = max(exact)
    print(f"  {'within(largest)':<18}: {result.within(largest)}")
    print()
    print("counting register")
    print(f"  {'mode':<18}: {mode}")
    print(
        f"  {'top counter values':<18}: "
        f"{[(key, round(share, 5)) for key, share in top_three]}"
    )
    print()
    print("one counting wire")
    print(f"  {'one-wire readout':<18}: {round(one_wire.dominant_singular_value, 6)}")
    print(f"  {'one-wire alpha':<18}: {round(one_wire.alpha, 6)}")
    equal_to_alpha = one_wire.dominant_singular_value == one_wire.alpha
    print(f"  {'readout equals alpha':<18}: {equal_to_alpha}")
    print(f"  {'':<18}  -- the only value a one-wire run can return")
    print(
        f"  {'refused counter':<18}: {refused!r}, carrying "
        f"{round(one_wire.distribution[refused], 4)} of the sample at "
        f"{ONE_WIRE_SHOTS} shots"
    )
    print(f"  {'one-shot sample':<18}: {ONE_WIRE_SAMPLE_SHOTS} shot, seed {ONE_WIRE_SEED}")
    try:
        estimate_singular_values(
            MATRIX,
            n_counting_wires=1,
            shots=ONE_WIRE_SAMPLE_SHOTS,
            seed=ONE_WIRE_SEED,
        )
    except ValueError as error:
        print(f"  {'raised':<18}: {error}")
    else:
        raise SystemExit(
            f"the one-shot run at seed {ONE_WIRE_SEED} returned a value, and the "
            "example exists to show that a one-wire run whose mode lands in the "
            "refused half raises"
        )
    print()
    print("take away")
    print("  the readout is the counting register's mode, scaled by alpha and read at")
    print("  the register's own resolution. The unit does not claim the mode reports")
    print("  the largest singular value, and no error bound, confidence interval or")
    print("  shot-selection rule is computed or reported. The width is the caller's:")
    print("  at one wire the register holds two counter values, one of them refused,")
    print("  so the only value it can return is alpha itself -- and a sample that")
    print("  lands the mode in the refused half raises, which is an outcome of the")
    print("  sample rather than a precondition on the arguments.")


if __name__ == "__main__":
    main()
