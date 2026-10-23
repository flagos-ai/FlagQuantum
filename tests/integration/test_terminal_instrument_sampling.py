"""Executable user path for the admitted Runtime measurement entry."""

from __future__ import annotations

import pytest
import torch

import flagquantum.lindblad as fql
from examples.rydberg_native_dynamics import native_pair_plan
from examples.terminal_instrument_reference import presence_reference
from examples.terminal_instrument_sampling import main
from flagquantum.runtime import sample_terminal

pytestmark = pytest.mark.integration


def test_presence_and_fock_user_example(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert capsys.readouterr().out.splitlines() == [
        "presence labels: ('no_flag', 'flag')",
        "presence probabilities: [0.68, 0.32]",
        "presence indices shape: (4096,)",
        "presence private-seed replay: True",
        "fock labels: ('0', '1', '2')",
        "fock probabilities: [0.6, 0.3, 0.1]",
        "fock indices shape: (4096,)",
        "fock private-seed replay: True",
        "global RNG unchanged: True",
    ]


@pytest.mark.parametrize("batch", [None, 2])
def test_native_plan_density_feeds_observed_flags_without_hidden_loss_truth(
    batch: int | None,
) -> None:
    plan = native_pair_plan(intervals=16, batch_size=batch)
    encoded = plan.to_json()
    evolution = fql.run(plan)
    assert evolution.density_matrices is not None
    _, single_atom = presence_reference()
    pair = {
        f"{left}/{right}": (torch.kron(a[0], b[0]),)
        for left, a in single_atom.items()
        for right, b in single_atom.items()
    }
    detection = sample_terminal(
        evolution.density_matrices[-1], pair, shots=2048, seed=17
    )
    assert detection.metadata["outcome_labels"] == [
        "no_flag/no_flag",
        "no_flag/flag",
        "flag/no_flag",
        "flag/flag",
    ]
    probabilities = torch.tensor(
        detection.metadata["probabilities"], dtype=torch.float64
    )
    quiet = probabilities[..., 0]
    torch.testing.assert_close(
        quiet, evolution.expectation("no_flag")[-1], atol=1e-10, rtol=0
    )
    assert bool((quiet > evolution.expectation("survival")[-1] + 1e-3).all())
    if batch == 2:
        # The second actor pair is physically lost, but an inefficient detector
        # still returns no flag with probability (1 - 0.8)**2.
        assert quiet[1].item() == pytest.approx(0.04, abs=1e-10)
    assert plan.to_json() == encoded
    assert plan.to_dict()["version"] == "1.7"
