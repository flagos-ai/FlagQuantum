"""Exact synthesis namespace/input scope and fail-closed checker mutations."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.services import synthesis
from tools.public_api_snapshot import _validate_authorized_synthesis_run

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _contract():
    return json.loads(
        (ROOT / "contracts/synthesis-program-run-v1-candidate.json").read_text()
    )


def test_namespace_reexports_one_authoritative_services_implementation() -> None:
    contract = _contract()
    assert _validate_authorized_synthesis_run(contract) == []
    assert tuple(synth.__all__) == tuple(contract["namespace_exports"])
    assert all(
        getattr(synth, name) is getattr(synthesis, name) for name in synth.__all__
    )
    assert synth.Program.__module__ == "flagquantum.services.synthesis"
    assert importlib.util.find_spec("flagquantum.services._synthesis") is None
    assert set(synth.__all__).isdisjoint(fq.__all__)
    namespace = {}
    exec("from flagquantum.synth import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(synth.__all__)


@pytest.mark.parametrize(
    "change",
    [
        "authorization",
        "release",
        "trial",
        "signature",
        "owner",
        "namespace",
        "exports",
        "flag_type",
        "overrides",
        "unknown",
    ],
)
def test_admission_checker_rejects_unapproved_contract_expansion(change) -> None:
    contract = copy.deepcopy(_contract())
    if change == "authorization":
        contract["implementation_authorized"] = False
    elif change == "release":
        contract["release_authorized"] = True
    elif change == "trial":
        contract["human_trial_records"] = ["unverified"]
    elif change == "signature":
        contract["run_signature"] = contract["run_signature"].replace(
            "SynthesisProgram", "object"
        )
    elif change == "owner":
        contract["type_owner"] = "flagquantum.services._synthesis.Program"
    elif change == "namespace":
        contract["namespace"] = "flagquantum.runtime"
    elif change == "exports":
        contract["namespace_exports"].append("run")
    elif change == "flag_type":
        contract["release_authorized"] = 0
    elif change == "overrides":
        contract["non_none_overrides_rejected"].remove("noise_model")
    else:
        contract["additional_types"] = ["object"]
    assert _validate_authorized_synthesis_run(contract)
