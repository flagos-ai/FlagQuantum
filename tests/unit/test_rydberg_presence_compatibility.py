"""Original artifacts stay readable and byte-preserving throughout proposal 092."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.train_rydberg_controls import control_plan, moving_pair_program
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.core.rydberg import RydbergProgram
from flagquantum.errors import SerializationError
from flagquantum.runtime.execution_plan_contract import ExecutionPlanContractError

pytestmark = pytest.mark.unit
CAPTURE = Path(__file__).parents[1] / "fixtures/rydberg_presence_legacy_captures.json"


def _single() -> RydbergProgram:
    source = RydbergProgram(
        atoms=(("a", "home"),),
        sites={"home": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.4,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    source.pulse(
        "a",
        start=0.0,
        duration=0.5,
        amplitude=1.2,
        detuning=0.0,
        phase=0.0,
        channel="laser",
    )
    source.pulse(
        "a",
        start=0.5,
        duration=0.5,
        amplitude=0.0,
        detuning=0.7,
        phase=0.0,
        channel="laser",
    )
    return source


@pytest.mark.parametrize(
    "key", ["single_atom", "moving_pair", "braid", "unread", "sampled"]
)
def test_old_source_construction_and_codec_bytes(key: str) -> None:
    capture = json.loads(CAPTURE.read_text())
    sources = {
        "single_atom": _single(),
        "moving_pair": moving_pair_program(),
        "braid": IsingBraidProgram(),
        "unread": IsingBraidProgram().dephase_pair(),
        "sampled": IsingBraidProgram().measure_pair(),
    }
    saved = capture["sources"][key]
    source = sources[key]
    encoded = json.dumps(
        native_program_to_dict(source.to_ir()), sort_keys=True, separators=(",", ":")
    )
    assert encoded == saved
    restored = native_program_from_dict(json.loads(saved))
    assert (
        json.dumps(
            native_program_to_dict(restored), sort_keys=True, separators=(",", ":")
        )
        == saved
    )


@pytest.mark.parametrize(
    "key",
    ["single_atom_native", "pair_native", "braid", "unread", "sampled", "circuit"],
)
def test_old_plan_exact_replay_and_host_scoped_reconstruction(key: str) -> None:
    capture = json.loads(CAPTURE.read_text())
    saved = capture["plans"][key]
    if key in {"single_atom_native", "pair_native"}:
        restored = fql.LindbladPlan.from_json(saved)
        assert restored.to_json() == saved and json.loads(saved)["version"] == "1.7"
        if key == "single_atom_native":
            source = _single()
            fresh = fql.plan(
                source,
                source.basis_state(("g",)),
                torch.linspace(0, 1, 129, dtype=torch.float64),
                method="runge-kutta",
                return_density_matrices=True,
            )
        else:
            fresh = control_plan(
                torch.tensor([0.8, 1.2], dtype=torch.float64), intervals=8
            )
        profile = capture["capture_profile"]
        if (platform.system(), platform.machine(), str(torch.__version__)) == (
            profile["system"],
            profile["machine"],
            profile["torch"],
        ):
            assert fresh.to_json() == saved
        # On every host, stored bytes are exact; generated physical matrices use
        # the predeclared integration budget, not host-specific decimal spelling.
        a, b = fql.run(restored), fql.run(fresh)
        torch.testing.assert_close(a.populations, b.populations, atol=1e-10, rtol=0)
        if a.density_matrices is not None:
            assert b.density_matrices is not None
            torch.testing.assert_close(
                a.density_matrices, b.density_matrices, atol=1e-10, rtol=0
            )
        with pytest.raises(TypeError):
            fq.run(restored)
    else:
        if key == "sampled" and capture["capture_profile"]["torch"] != str(
            torch.__version__
        ):
            with pytest.raises(ExecutionPlanContractError) as caught:
                fq.ExecutionPlan.from_json(saved)
            assert caught.value.reason_code == "environment_incompatible"
            return
        restored_root = fq.ExecutionPlan.from_json(saved)
        assert restored_root.to_json() == saved
        if key == "circuit":
            fresh_root = fq.plan(fq.Circuit(1).h(0))
        else:
            s = IsingBraidProgram()
            if key == "unread":
                s.dephase_pair()
            elif key == "sampled":
                s.measure_pair()
            fresh_root = fq.plan(
                s,
                options=(
                    fq.ExecutionOptions(shots=1, seed=17) if key == "sampled" else None
                ),
            )
        assert fresh_root.to_json() == saved
        assert torch.equal(fq.run(restored_root).state, fq.run(fresh_root).state)


def test_production_sampled_source_and_reader_are_not_pretended() -> None:
    assert callable(RydbergProgram.measure_presence)
    payload = json.loads(json.loads(CAPTURE.read_text())["plans"]["single_atom_native"])
    payload["version"] = "1.8"
    from flagquantum.errors import SerializationError
    from flagquantum.lindblad._plan import _identity

    payload["identity"] = _identity(payload)
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(payload)


def test_disposable_recipe_is_not_a_production_plan_and_seals_rng() -> None:
    from flagquantum.runtime.execution_plan_contract import canonical_hash
    from tests.unit.rydberg_presence_reference import ReferenceSource, plan, read, run
    from tests.unit.test_rydberg_presence_contract import _program

    source = _program(ReferenceSource, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    recipe = plan(
        source,
        source.basis_state(("g",)),
        torch.linspace(0, 1, 65, dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=17),
    )
    payload = json.loads(recipe.to_json())
    assert payload["schema"] == "test.rydberg_presence_recipe"
    assert payload["native_plan"]["version"] == "1.7"
    assert payload["native_plan"]["program"]["version"] == "1.0"
    assert run(recipe).plan is None
    for reader, exception in (
        (fq.ExecutionPlan.from_json, ExecutionPlanContractError),
        (fql.LindbladPlan.from_json, SerializationError),
    ):
        with pytest.raises(exception):
            reader(recipe.to_json())
    payload["rng_torch_version"] = "incompatible-test-build"
    payload["identity"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "identity"}
    )
    with pytest.raises(ExecutionPlanContractError) as caught:
        read(json.dumps(payload))
    assert caught.value.reason_code == "environment_incompatible"
