"""Intentional breaking fixtures for the approved terminal measurement boundary."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime as runtime
from flagquantum.runtime.result import MeasurementResult
from tools import public_api_snapshot

pytestmark = pytest.mark.unit


@pytest.fixture
def admission(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    contract = json.loads(public_api_snapshot.EXECUTION_RESULT_CONTRACT.read_text())[
        "terminal_instrument_sampling"
    ]
    contract["implementation_required"] = False
    monkeypatch.setattr(runtime, "__all__", tuple(contract["namespace_prefix"]))
    monkeypatch.setattr(
        runtime,
        "_EXPORTS",
        {
            name: target
            for name, target in runtime._EXPORTS.items()
            if name != "sample_terminal"
        },
    )
    monkeypatch.delattr(runtime, "sample_terminal", raising=False)
    return contract


def declared_entry(
    density_matrix: torch.Tensor,
    instrument: Mapping[str, Sequence[torch.Tensor]],
    *,
    shots: int,
    seed: int | None = None,
) -> MeasurementResult:
    raise AssertionError("API checking must inspect the signature, not execute a draw")


def test_current_runtime_matches_its_admitted_delivery_stage() -> None:
    contract = json.loads(public_api_snapshot.EXECUTION_RESULT_CONTRACT.read_text())[
        "terminal_instrument_sampling"
    ]
    assert hasattr(runtime, "sample_terminal") is contract["implementation_required"]
    assert not hasattr(fq, "sample_terminal")
    assert public_api_snapshot._validate_terminal_instrument_admission(contract) == []
    assert public_api_snapshot.validate() == ()


def test_activation_requires_the_exact_runtime_only_entry(
    admission: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admission["implementation_required"] = True
    assert public_api_snapshot._validate_terminal_instrument_admission(admission)
    monkeypatch.setattr(runtime, "__all__", (*runtime.__all__, "sample_terminal"))
    monkeypatch.setattr(runtime, "sample_terminal", declared_entry, raising=False)
    assert public_api_snapshot._validate_terminal_instrument_admission(admission) == []


@pytest.mark.parametrize("placeholder", [None, declared_entry])
def test_pending_admission_refuses_a_published_placeholder(
    admission: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    placeholder: Any,
) -> None:
    monkeypatch.setattr(runtime, "sample_terminal", placeholder, raising=False)
    errors = public_api_snapshot._validate_terminal_instrument_admission(admission)
    assert any("must not publish an API stub" in error for error in errors)


def test_root_alias_is_not_admitted_even_when_runtime_entry_is_correct(
    admission: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admission["implementation_required"] = True
    monkeypatch.setattr(runtime, "__all__", (*runtime.__all__, "sample_terminal"))
    monkeypatch.setattr(runtime, "sample_terminal", declared_entry, raising=False)
    monkeypatch.setattr(fq, "sample_terminal", declared_entry, raising=False)
    assert any(
        "must not enter the stable root" in error
        for error in public_api_snapshot._validate_terminal_instrument_admission(
            admission
        )
    )


@pytest.mark.parametrize("change", ["drop", "reorder", "unknown"])
def test_old_runtime_exports_cannot_be_removed_reordered_or_extended(
    admission: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    names = runtime.__all__
    changed = {
        "drop": names[:-1],
        "reorder": (*names[1:], names[0]),
        "unknown": (*names, "unapproved_entry"),
    }[change]
    monkeypatch.setattr(runtime, "__all__", changed)
    assert any(
        "Runtime exports differ" in error
        for error in public_api_snapshot._validate_terminal_instrument_admission(
            admission
        )
    )


@pytest.mark.parametrize(
    "signature",
    [
        "(density_matrix, instrument, shots=1, seed=None)",
        "(density_matrix, instrument, *, shots, seed=0)",
        "(state, instrument, *, shots, seed=None)",
    ],
)
def test_signature_drift_cannot_be_hidden_by_a_contract_edit(
    admission: dict[str, Any], monkeypatch: pytest.MonkeyPatch, signature: str
) -> None:
    admission["implementation_required"] = True
    admission["signature"] = signature
    monkeypatch.setattr(runtime, "__all__", (*runtime.__all__, "sample_terminal"))
    monkeypatch.setattr(runtime, "sample_terminal", declared_entry, raising=False)
    assert any(
        "signature differs" in error
        for error in public_api_snapshot._validate_terminal_instrument_admission(
            admission
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unreviewed.instrument"),
        ("version", "2.0"),
        ("fields", ["outcome_labels", "probabilities", "hidden_loss_truth"]),
        ("instrument_fields", ["schema", "version"]),
        ("branch_fields", ["label", "kraus", "conditional_state"]),
    ],
)
def test_metadata_drift_is_rejected_before_implementation(
    admission: dict[str, Any], field: str, value: Any
) -> None:
    admission["result_metadata"][field] = value
    assert public_api_snapshot._validate_terminal_instrument_admission(admission)


def test_metadata_key_order_is_not_an_unapproved_new_promise(
    admission: dict[str, Any],
) -> None:
    for field in ("fields", "instrument_fields", "branch_fields"):
        admission["result_metadata"][field].reverse()
    assert public_api_snapshot._validate_terminal_instrument_admission(admission) == []


@pytest.mark.parametrize(
    "change",
    [
        {"implementation_required": 0},
        {"namespace": "flagquantum"},
        {"entry": "simple_sample"},
        {"result": {"kind": "new_measurement_result"}},
    ],
)
def test_activation_owner_and_result_envelope_are_not_extensible(
    admission: dict[str, Any], change: dict[str, Any]
) -> None:
    altered = copy.deepcopy(admission)
    altered.update(change)
    assert public_api_snapshot._validate_terminal_instrument_admission(altered)


def test_missing_admission_cannot_silently_disable_the_guard() -> None:
    assert public_api_snapshot._validate_terminal_instrument_admission(None)
