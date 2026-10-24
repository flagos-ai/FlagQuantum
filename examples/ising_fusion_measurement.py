"""Observe ideal Ising pair charge without confusing sampling and state error.

This is a local CPU four-sigma, total-vacuum model. Terminal sampling returns
indices, not conditional states: the explicit Kraus maps below explain the QND
back-action. No hardware, feedback, gradients or universal computation is claimed.
"""

from __future__ import annotations

from typing import Literal

import torch

import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal


def fusion_state(initial: Literal["1", "psi"]) -> torch.Tensor:
    """Execute a nontrivial braid word and verify its saved-plan replay."""
    program = (
        IsingBraidProgram(initial_channel=initial)
        .exchange(1)
        .exchange(0, clockwise=True)
    )
    selected = fq.plan(program)
    result = fq.run(selected)
    replay = fq.run(fq.ExecutionPlan.from_json(selected.to_json()))
    assert result.state is not None and replay.state is not None
    assert torch.equal(result.state, replay.state)
    state = result.state[0]
    assert isinstance(state, torch.Tensor)
    return state


def main() -> None:
    shots = 20_000
    initial_channels: tuple[Literal["1", "psi"], ...] = ("1", "psi")
    for initial in initial_channels:
        state = fusion_state(initial)
        program = IsingBraidProgram(initial_channel=initial)
        projectors = [program.projector(label) for label in ("1", "psi")]
        instrument = {
            label: (p,) for label, p in zip(("1", "psi"), projectors, strict=True)
        }
        before_rng = torch.random.get_rng_state().clone()
        sampled = sample_terminal(state, instrument, shots=shots, seed=7)
        repeated = sample_terminal(state, instrument, shots=shots, seed=7)
        assert isinstance(sampled.value, torch.Tensor)
        assert isinstance(repeated.value, torch.Tensor)
        assert torch.equal(sampled.value, repeated.value)
        assert torch.equal(before_rng, torch.random.get_rng_state())

        branches = [SuperOperator.from_kraus([p]).apply(state) for p in projectors]
        probabilities = torch.stack([torch.trace(branch).real for branch in branches])
        unread = SuperOperator.from_kraus(projectors).apply(state)
        torch.testing.assert_close(unread, sum(branches), atol=1e-12, rtol=0)
        for outcome, (projector, branch, probability) in enumerate(
            zip(projectors, branches, probabilities, strict=True)
        ):
            conditional = branch / probability
            torch.testing.assert_close(conditional, projector, atol=1e-12, rtol=0)
            qnd = SuperOperator.from_kraus([projector]).apply(conditional)
            torch.testing.assert_close(qnd, conditional, atol=1e-12, rtol=0)
            observed = sample_terminal(conditional, instrument, shots=32, seed=7)
            assert isinstance(observed.value, torch.Tensor)
            assert torch.all(observed.value == outcome)

        frequencies = torch.bincount(sampled.value, minlength=2).double() / shots
        standard_error = torch.sqrt(probabilities * (1 - probabilities) / shots)
        assert torch.all(
            abs(frequencies - probabilities) <= 6 * standard_error + 2 / shots
        )
        numerical_error = float(torch.max(abs(probabilities - 0.5)))
        assert numerical_error < 1e-12
        print(f"Initial channel: {initial}")
        print("Born probability error:", f"{numerical_error:.2e}")
        print("Input coherence magnitude:", round(float(abs(state[0, 1])), 12))
        print("Exact probabilities:", [round(p, 12) for p in probabilities.tolist()])
        print("Sample frequencies:", [round(p, 5) for p in frequencies.tolist()])
        print("Binomial standard error:", round(float(standard_error[0]), 6))
        print("Unread coherence magnitude:", round(float(abs(unread[0, 1])), 12))
        print("QND conditional charge preserved: True")
        print("Exact plan and private-seed replay: True")
        print("Caller RNG unchanged: True")


if __name__ == "__main__":
    main()
