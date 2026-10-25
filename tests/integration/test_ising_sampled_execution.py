"""One native sampled trajectory through public planning, execution and replay."""

from __future__ import annotations

from typing import Literal
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize("suffix", [0, 1])
def test_saved_conditional_trajectory_without_replanning(
    initial: Literal["1", "psi"], suffix: int
) -> None:
    source = IsingBraidProgram(initial_channel=initial).exchange(1).measure_pair()
    source.exchange(suffix, clockwise=True)
    selected = fq.plan(
        source,
        options=fq.ExecutionOptions(shots=1, seed=17),
        outputs=fq.expectation(source.projector("psi"), name="psi"),
    )
    encoded = selected.to_json()
    restored = fq.ExecutionPlan.from_json(encoded)
    assert selected.schema_version == "1.3" and selected.analysis is None
    with patch(
        "flagquantum.runtime._ising_plan.plan_ising",
        side_effect=AssertionError("saved plans must not replan"),
    ):
        direct = fq.run(selected)
        replay = fq.run(restored)
    assert selected.to_json() == encoded == restored.to_json()
    event = direct.measurement("mid_circuit_measurement")
    index = int(event.value[0, 0])
    if suffix == 0:
        expected = torch.diag(
            torch.tensor([index == 0, index == 1], dtype=torch.complex128)
        )
    else:
        # Inverse middle exchange maps the selected charge to equal populations
        # with opposite imaginary coherences for the two observed charges.
        coherence = (2 * index - 1) * 0.5j
        expected = torch.tensor(
            [[0.5, coherence], [coherence.conjugate(), 0.5]], dtype=torch.complex128
        )
    torch.testing.assert_close(direct.state[0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        direct.expectation("psi")[0], expected[1, 1].real, atol=1e-12, rtol=0
    )
    assert torch.equal(direct.state, replay.state)
    assert torch.equal(event.value, replay.measurement(-1).value)
    assert direct.samples is None and len(direct.measurements) == 2
    assert direct.runtime["distribution_semantics"] == "single_device_fast_path"
    assert direct.runtime["scalability_claim_allowed"] is False
