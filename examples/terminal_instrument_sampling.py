"""Sample declared presence and finite-Fock QND outcomes through Runtime.

The presence reference assumes efficiency 0.8 and zero false positives. The
finite-Fock reference is QND number detection, not a destructive photodetector.
Labels describe observed outcomes, rather than hidden loss truth or events.
"""

from __future__ import annotations

import torch

from examples.terminal_instrument_reference import fock_reference, presence_reference
from flagquantum.runtime import sample_terminal


def main() -> None:
    """Execute both explicit instruments and replay their private seeds."""

    global_rng = torch.get_rng_state()
    for name, build in (("presence", presence_reference), ("fock", fock_reference)):
        density, branches = build()
        result = sample_terminal(density, branches, shots=4096, seed=7)
        replay = sample_terminal(
            density,
            branches,
            shots=4096,
            seed=result.metadata["terminal_instrument"]["effective_seed"],
        )
        assert isinstance(result.value, torch.Tensor)
        assert isinstance(replay.value, torch.Tensor)
        print(f"{name} labels:", tuple(result.metadata["outcome_labels"]))
        print(
            f"{name} probabilities:",
            [round(p, 8) for p in result.metadata["probabilities"]],
        )
        print(f"{name} indices shape:", tuple(result.value.shape))
        print(f"{name} private-seed replay:", torch.equal(result.value, replay.value))
    print("global RNG unchanged:", torch.equal(global_rng, torch.get_rng_state()))


if __name__ == "__main__":
    main()
