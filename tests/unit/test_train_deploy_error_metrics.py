"""Separate optimizer accuracy from a finite-shot deployment observation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from examples import train_parameterized_circuit_then_deploy as example
from flagquantum.remote.qpu.contracts import DeploymentResult, ProviderTaskHandle
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.unit


def _reported(stdout: str, label: str) -> float:
    line = next(line for line in stdout.splitlines() if line.startswith(label))
    return float(line.split(":", 1)[1])


@pytest.mark.parametrize(
    "counts", [{"10": 1024}, {"01": 1, "10": 1023}], ids=["ground", "rare-shot"]
)
def test_training_gap_does_not_depend_on_the_sampled_counts(
    counts: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fetch = InMemoryRemoteTarget.fetch_result

    def observed(
        provider: InMemoryRemoteTarget, handle: ProviderTaskHandle
    ) -> DeploymentResult:
        # Both observations are possible for the approximate trained state.
        return replace(fetch(provider, handle), counts=counts)

    monkeypatch.setattr(InMemoryRemoteTarget, "fetch_result", observed)
    example.main()
    stdout = capsys.readouterr().out
    energy = _reported(stdout, "training_energy:")
    gap = _reported(stdout, "  Gap to ground state:")

    assert gap == pytest.approx(energy + 1.2, abs=1e-10)
    assert abs(energy + 1.2) < 1e-6
    assert abs(gap) < 1e-6
    sampled_gap = _reported(stdout, "  Sampled deployment gap:")
    assert sampled_gap == pytest.approx(0.4 * counts.get("01", 0) / 1024, abs=5e-8)


def test_example_sampling_is_replayable_without_changing_the_callers_rng(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observations = []
    with torch.random.fork_rng(devices=[]):
        for caller_seed in (91, 92):
            torch.manual_seed(caller_seed)
            before = torch.random.get_rng_state().clone()
            example.main()
            assert torch.equal(before, torch.random.get_rng_state())
            stdout = capsys.readouterr().out
            observations.append(
                next(
                    line
                    for line in stdout.splitlines()
                    if line.startswith("deployment_counts:")
                )
            )
    assert observations[0] == observations[1]
