"""Absorb one terminal mode while recording finite-Fock photon counts.

The measured mode retains its original coordinates in vacuum. The detector
efficiency controls registration, not absorption or propagation loss. Conditional
densities are explicit example analysis; sample_terminal still returns indices.
"""

from __future__ import annotations

import math

import torch

import flagquantum.lindblad as fql
from examples.boson_interference import interference_plan
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal


def absorptive_instrument(
    cutoff: int, efficiency: float, *, mode: int = 0
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Declare square vacuum-reset branches, one Kraus matrix per incoming n.

    Examples:
        >>> instrument = absorptive_instrument(3, 0.65)
        >>> list(instrument)
        ['0', '1', '2']
    """
    if type(cutoff) is not int or cutoff not in (3, 4):
        raise ValueError("this reference uses cutoff 3 or 4")
    if (
        type(efficiency) not in (int, float)
        or not math.isfinite(efficiency)
        or not 0 <= efficiency <= 1
    ):
        raise ValueError("efficiency must be a finite real scalar in [0,1]")
    if type(mode) is not int or mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    identity = torch.eye(cutoff, dtype=torch.complex128)
    instrument = {}
    for observed in range(cutoff):
        operators = []
        for incoming in range(observed, cutoff):
            if efficiency == 0:
                weight = float(observed == 0)
            elif efficiency == 1:
                weight = float(observed == incoming)
            else:
                weight = (
                    math.comb(incoming, observed)
                    * efficiency**observed
                    * (1 - efficiency) ** (incoming - observed)
                )
            reset = torch.zeros_like(identity)
            reset[0, incoming] = math.sqrt(weight)
            operators.append(
                torch.kron(reset, identity)
                if mode == 0
                else torch.kron(identity, reset)
            )
        instrument[str(observed)] = tuple(operators)
    return instrument


def main() -> None:
    for cutoff in (3, 4):
        for rate in (0.0, 0.2):
            plan = interference_plan(rate, cutoff=cutoff)
            evolution = fql.run(plan)
            replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
            assert (
                evolution.density_matrices is not None
                and replay.density_matrices is not None
            )
            assert torch.equal(evolution.density_matrices, replay.density_matrices)
            density = evolution.density_matrices[-1]
            for efficiency in (0.0, 0.65, 1.0):
                instrument = absorptive_instrument(cutoff, efficiency)
                rng = torch.get_rng_state()
                detected = sample_terminal(density, instrument, shots=4096, seed=7)
                repeated = sample_terminal(
                    replay.density_matrices[-1],
                    absorptive_instrument(cutoff, efficiency),
                    shots=4096,
                    seed=7,
                )
                assert isinstance(detected.value, torch.Tensor) and isinstance(
                    repeated.value, torch.Tensor
                )
                assert torch.equal(detected.value, repeated.value)
                assert torch.equal(rng, torch.get_rng_state())
                print(f"N={cutoff}, propagation loss={rate}, efficiency={efficiency}")
                print(
                    "Count probabilities:",
                    [round(float(p), 8) for p in detected.metadata["probabilities"]],
                )
                print(
                    "Observed counts:",
                    torch.bincount(detected.value, minlength=cutoff).tolist(),
                )
                for label, operators in instrument.items():
                    branch = SuperOperator.from_kraus(operators).apply(density)
                    original_trace = float(branch.trace().real)
                    # Only the declared finite test family has this positive-weight separation.
                    # Residuals below resolution are reported, never promoted to a state.
                    if original_trace <= 1e-8:
                        print(
                            f"Count {label}: residual weight={original_trace:.2e}; no conditional state"
                        )
                        continue
                    conditional = branch / original_trace
                    torch.testing.assert_close(
                        conditional[cutoff:],
                        torch.zeros_like(conditional[cutoff:]),
                        atol=1e-10,
                        rtol=0,
                    )
                    remaining = conditional[:cutoff, :cutoff]
                    mean = float(
                        remaining.diag().real
                        @ torch.arange(cutoff, dtype=torch.float64)
                    )
                    print(
                        f"Count {label}: mode 0 absorbed; conditional mode 1 occupation={mean:.8f}"
                    )
    print("Exact numeric evolution/sample replay and private Torch RNG: True")
    print("Finite vacuum-reset reference; no removed-mode or detector-event result")


if __name__ == "__main__":
    main()
