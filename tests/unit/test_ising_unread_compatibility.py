"""Freeze braid-only artifacts before proposal 089 adds an unread channel.

These are real production consumers. The fixture copies only the missing Ising
1.1 artifacts from the immutable pre-implementation capture; existing circuit,
Rydberg and Lindblad fixture authorities remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir.program.physical_codec import native_program_to_dict
from flagquantum.experimental.simulation import IsingBraidProgram

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize("word", [(), ((1, False), (1, True)), ((1, False), (0, True))])
def test_braid_only_sources_and_plans_retain_exact_pre_observation_bytes(
    initial: Literal["1", "psi"], word: tuple[tuple[int, bool], ...]
) -> None:
    capture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/ising_braid_v1_1_captures.json"
        ).read_text()
    )
    key = "ising_" + initial + "_" + str(word)
    program = IsingBraidProgram(initial_channel=initial)
    for left, clockwise in word:
        program.exchange(left, clockwise=clockwise)
    outputs = fq.expectation(program.projector("psi"), name="psi")
    plan = fq.plan(program, outputs=outputs)
    encoded = capture["plans"][key]
    assert hashlib.sha256(encoded.encode()).hexdigest() == capture["plan_sha256"][key]
    assert plan.to_json() == encoded
    source = native_program_to_dict(program.to_ir())
    assert source == capture["sources"][key]
    assert source["version"] == json.loads(encoded)["version"] == "1.1"
    restored = fq.ExecutionPlan.from_json(encoded)
    assert restored.identity == plan.identity
    assert restored.to_json() == encoded
    direct = fq.run(program, outputs=outputs)
    replay = fq.run(restored)
    assert direct.state is not None and replay.state is not None
    assert direct.state.shape == replay.state.shape == (1, 2, 2)
    assert direct.state.dtype == replay.state.dtype == torch.complex128
    assert torch.equal(direct.state, replay.state)
    assert torch.equal(direct.expectation("psi"), replay.expectation("psi"))
