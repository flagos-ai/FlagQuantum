"""Real matrix requests fail before unsupported planning, execution or submission."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.remote import jobs
from flagquantum.remote.compute._native_job import NativeJiudingJobClient
from flagquantum.remote.compute.jiuding import JiudingClient

pytestmark = pytest.mark.unit


def _unexpected(*args: object, **kwargs: object) -> None:
    pytest.fail("unsupported matrix request reached planning, execution or submission")


@pytest.mark.parametrize("entry", [fq.run, fq.plan])
def test_root_circuit_paths_refuse_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    entry: Callable[..., object],
) -> None:
    monkeypatch.setattr("flagquantum.runtime.execution.run", _unexpected)
    monkeypatch.setattr("flagquantum.runtime.planner.plan", _unexpected)
    with pytest.raises(CapabilityError, match="Lindblad"):
        entry(fq.Circuit(1), outputs=fq.expectation(torch.eye(2)))


@pytest.mark.parametrize("target", ["quafu:test", "azure:test"])
def test_root_qpu_routes_refuse_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    monkeypatch.setattr("flagquantum.remote.qpu.execution.QuafuProvider", _unexpected)
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", _unexpected
    )
    with pytest.raises(CapabilityError, match="flagquantum.lindblad"):
        fq.run(
            fq.Circuit(1),
            outputs=fq.expectation(torch.eye(2)),
            target=target,
            shots=100,
        )


def test_receipt_request_refuses_before_journal_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "QuafuProvider", _unexpected)
    receipt = tmp_path / "job.json"
    with pytest.raises(CapabilityError, match="Lindblad"):
        jobs.submit(
            fq.Circuit(1),
            target="quafu:test",
            outputs=fq.expectation(torch.eye(2)),
            shots=100,
            receipt=receipt,
        )
    assert not list(tmp_path.iterdir())


def test_native_batch_refuses_before_workspace_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(NativeJiudingJobClient)
    monkeypatch.setattr(client, "workspace", _unexpected)
    with pytest.raises(CapabilityError, match="Lindblad"):
        client.submit_program(
            fq.Circuit(1),
            target="jiuding:cpu",
            image="test",
            outputs=fq.expectation(torch.eye(2)),
        )


@pytest.mark.parametrize("receipt", [False, True])
def test_managed_and_explicit_program_paths_refuse_before_target_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: bool,
) -> None:
    client = object.__new__(JiudingClient)
    monkeypatch.setattr(client, "_resolve_compute_target", _unexpected)
    with pytest.raises(CapabilityError, match="Lindblad"):
        client.submit_program(
            fq.Circuit(1),
            target="jiuding:cpu",
            image="test",
            outputs=fq.expectation(torch.eye(2)),
            receipt=tmp_path / "job.json" if receipt else None,
        )
    assert not list(tmp_path.iterdir())


def test_resident_path_refuses_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(JiudingClient)
    monkeypatch.setattr(client, "_execute", _unexpected)
    with pytest.raises(CapabilityError, match="Lindblad"):
        client.run(
            fq.Circuit(1), target="jiuding:cpu", outputs=fq.expectation(torch.eye(2))
        )


def test_request_types_are_resolvable_without_loading_torch_on_root_import() -> None:
    script = """
import sys
import flagquantum as fq
assert 'torch' not in sys.modules
from typing import get_type_hints
from flagquantum.observables import OutputRequest, expectation
import torch
assert fq.expectation is expectation
assert get_type_hints(expectation)['observable'] == fq.Observable | torch.Tensor
assert get_type_hints(OutputRequest)['observable'] == fq.Observable | torch.Tensor | None
assert not any(n == 'cudaq' or n.startswith('cudaq.') for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        check=True,
        timeout=30,
    )


def test_nonnumeric_bit_storage_is_not_an_expectation_matrix() -> None:
    with pytest.raises(TypeError, match="numeric"):
        fq.expectation(torch.zeros((2, 2), dtype=torch.uint8).view(torch.bits8))


def test_ragged_nested_storage_is_refused_without_inspecting_sizes() -> None:
    matrix = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
    with pytest.raises(TypeError, match="ordinary numeric dense tensor"):
        fq.expectation(matrix)


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [("type", "Observable | None"), ("default", "unexpected"), ("kw_only", True)],
)
def test_existing_api_checker_detects_unapproved_request_field_drift(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    replacement: object,
) -> None:
    from dataclasses import fields

    from tools import public_api_snapshot

    observable = next(
        field for field in fields(fq.OutputRequest) if field.name == "observable"
    )
    monkeypatch.setattr(observable, attribute, replacement)
    assert (
        "OutputRequest fields differ from approved proposal 080"
        in public_api_snapshot.validate()
    )
