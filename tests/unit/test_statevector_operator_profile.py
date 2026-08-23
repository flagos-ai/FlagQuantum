from __future__ import annotations

from types import SimpleNamespace

import torch

from flagquantum import Circuit
from flagquantum.runtime import execution
from flagquantum.runtime.capabilities import (
    CapabilityPreflightReport,
    load_operator_profile,
)
from flagquantum.runtime.operator_probes import (
    preflight_statevector_local_p0,
    probe_operator_profile,
)


def test_statevector_local_p0_profile_is_packaged_and_stably_hashed() -> None:
    first = load_operator_profile("statevector_local_p0")
    second = load_operator_profile("statevector_local_p0")

    assert first == second
    assert first.profile_hash == second.profile_hash
    assert len(first.profile_hash) == 64
    assert first.representation == "statevector"
    assert first.distribution == "local"
    assert {item.operator for item in first.requirements} >= {
        "aten::bmm",
        "aten::reshape",
        "aten::permute",
        "aten::cos",
        "aten::real",
        "aten::imag",
        "aten::abs",
    }


def test_statevector_local_p0_cpu_reference_probe_passes_complex64() -> None:
    report = preflight_statevector_local_p0(
        device="cpu",
        dtype=torch.complex64,
        provider="pytorch_cpu_test",
        refresh=True,
    )

    assert report.supported
    assert report.required_dtypes == ("complex64",)
    assert len(report.evidence_ids) == 21


def test_probe_evidence_is_bound_to_profile_hash_and_device() -> None:
    profile = load_operator_profile("statevector_local_p0")
    evidence = probe_operator_profile(
        profile,
        device="cpu",
        dtype="complex64",
        provider="pytorch_cpu_test",
        refresh=True,
    )

    assert len(evidence) == len(profile.requirements)
    assert {item.profile_hash for item in evidence} == {profile.profile_hash}
    assert {item.device_type for item in evidence} == {"cpu"}
    assert all(item.is_verified for item in evidence)


def test_cpu_statevector_does_not_enter_flagos_preflight(monkeypatch) -> None:
    monkeypatch.setattr(execution, "resolve_device", lambda device: torch.device("cpu"))

    report = execution._preflight_flagos_statevector(
        Circuit(1).to_ir(), {"device": "cpu"}
    )

    assert report is None


def test_flagos_statevector_preflight_uses_platform_provider(monkeypatch) -> None:
    profile = load_operator_profile("statevector_local_p0")
    expected = CapabilityPreflightReport(
        profile=profile.name,
        profile_hash=profile.profile_hash,
        device_type="flagos",
        required_dtypes=("complex64",),
        supported=True,
        evidence_ids=("evidence",),
        blockers=(),
    )
    captured: dict[str, object] = {}

    def fake_preflight(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        execution,
        "resolve_device",
        lambda device: SimpleNamespace(type="flagos"),
    )
    monkeypatch.setattr(
        "flagquantum.runtime.operator_probes.preflight_statevector_local_p0",
        fake_preflight,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.platforms.get_platform_runtime",
        lambda name: SimpleNamespace(
            identity=lambda: SimpleNamespace(provider="torch_fl")
        ),
    )

    report = execution._preflight_flagos_statevector(
        Circuit(1).to_ir(), {"device": "flagos:0", "dtype": "complex64"}
    )

    assert report is expected
    assert captured["provider"] == "torch_fl"
    assert captured["dtype"] == "complex64"
