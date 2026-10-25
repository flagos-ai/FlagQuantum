"""Real old-plan consumers captured before proposal 091 implementation."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir.program.physical_codec import native_program_to_dict
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.runtime import sample_terminal

pytestmark = pytest.mark.unit
CAPTURE = Path(__file__).parents[1] / "fixtures/ising_unread_v1_2_captures.json"


@pytest.mark.parametrize(
    "key", [f"{c}-{p}-{s}" for c in ("1", "psi") for p in (1, 2) for s in (0, 1)]
)
def test_exact_unread_artifacts_and_real_replay(key: str) -> None:
    capture = json.loads(CAPTURE.read_text())["native_1_2"][key]
    initial, prefix, suffix = key.split("-")
    source = IsingBraidProgram(
        initial_channel="1" if initial == "1" else "psi"
    ).exchange(1)
    if prefix == "2":
        source.exchange(0, clockwise=True)
    source.dephase_pair().exchange(int(suffix))
    plan = fq.plan(source, outputs=fq.expectation(source.projector("psi"), name="psi"))
    encoded = capture["plan_json"]
    assert hashlib.sha256(encoded.encode()).hexdigest() == capture["plan_sha256"]
    assert plan.to_json() == encoded
    assert native_program_to_dict(source.to_ir()) == capture["source"]
    restored = fq.ExecutionPlan.from_json(encoded)
    assert restored.to_json() == encoded and restored.identity == plan.identity
    direct, replay = fq.run(plan), fq.run(restored)
    assert direct.state is not None and replay.state is not None
    assert direct.state.shape == (1, 2, 2) and direct.state.dtype == torch.complex128
    assert torch.equal(direct.state, replay.state)
    assert torch.equal(direct.expectation("psi"), replay.expectation("psi"))


def test_terminal_seed_behavior_retains_its_original_record() -> None:
    source = IsingBraidProgram().exchange(1)
    observed = sample_terminal(
        fq.run(source).state,
        {c: [source.projector(c)] for c in ("1", "psi")},
        shots=1,
        seed=17,
    )
    assert isinstance(observed.value, torch.Tensor)
    capture = json.loads(CAPTURE.read_text())["terminal"]
    repeat = sample_terminal(
        fq.run(source).state,
        {c: [source.projector(c)] for c in ("1", "psi")},
        shots=1,
        seed=17,
    )
    assert isinstance(repeat.value, torch.Tensor)
    assert torch.equal(observed.value, repeat.value)
    assert dict(observed.metadata) == dict(repeat.metadata)
    metadata = dict(observed.metadata)
    old_metadata = dict(capture["metadata"])
    captured_build = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and old_metadata["terminal_instrument"]["torch_version"]
        == str(torch.__version__)
    )
    if captured_build:
        # Retain the original exact checks on the captured host profile.
        assert observed.value.tolist() == capture["value"]
        assert metadata == old_metadata
    # Captured JSON bytes are exact above. Evolved physical weights on a
    # different host follow the approved physics budget, not decimal identity.
    torch.testing.assert_close(
        torch.tensor(metadata.pop("probabilities"), dtype=torch.float64),
        torch.tensor(old_metadata.pop("probabilities"), dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    correction = metadata.pop("probability_correction")
    old_correction = old_metadata.pop("probability_correction")
    assert 0 <= correction <= 1e-10
    assert abs(correction - old_correction) <= 1e-12
    # Only the reviewed Torch build promises the captured random label. Other
    # builds retain their truthful version field and same-environment replay.
    old_instrument = dict(old_metadata["terminal_instrument"])
    old_instrument["torch_version"] = str(torch.__version__)
    old_metadata["terminal_instrument"] = old_instrument
    assert metadata == old_metadata
