from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from flagquantum.kernels.catalog import (
    EVIDENCE,
    KernelRequest,
    match_kernel_implementations,
)

pytestmark = pytest.mark.unit


def _request(**overrides: Any) -> KernelRequest:
    values: dict[str, Any] = {
        "semantic_id": "statevector.apply.matrix_1q.local",
        "device": "cuda",
        "dtype": "complex64",
        "layout": "flat_statevector",
        "direction": "forward",
        "addressing": ("local",),
    }
    values.update(overrides)
    return KernelRequest(**values)


def _candidate_ids(request: KernelRequest) -> tuple[str, ...]:
    result = match_kernel_implementations(request)
    return tuple(
        candidate.implementation.implementation_id for candidate in result.candidates
    )


def test_forward_request_returns_all_compatible_implementations_in_catalog_order():
    result = match_kernel_implementations(_request())

    assert result.semantic_known is True
    assert result.matched is True
    assert _candidate_ids(_request()) == (
        "FQKI-TRITON-SV-001-A",
        "FQKI-TRITON-SV-001-B",
        "FQKI-FLAGTREE-SV-001-A",
    )
    assert not result.rejections
    assert all(
        candidate.evidence.implementation_id
        == candidate.implementation.implementation_id
        for candidate in result.candidates
    )


def test_backward_request_rejects_forward_only_implementation() -> None:
    result = match_kernel_implementations(_request(direction="backward"))

    assert tuple(
        candidate.implementation.implementation_id for candidate in result.candidates
    ) == ("FQKI-TRITON-SV-001-B",)
    assert result.rejections[0].implementation.implementation_id == (
        "FQKI-TRITON-SV-001-A"
    )
    assert tuple(item.code for item in result.rejections[0].mismatches) == (
        "direction",
    )


def test_internal_fallback_does_not_advertise_a_cpu_specialization() -> None:
    request = _request(
        semantic_id="statevector.apply.rx_rz_sequence.local",
        device="cpu",
    )
    result = match_kernel_implementations(request)

    assert result.matched is False
    assert tuple(item.code for item in result.rejections[0].mismatches) == ("device",)
    assert result.rejections[0].implementation.internal_fallback is True


def test_provider_filter_is_capability_filter_not_priority_policy() -> None:
    triton = match_kernel_implementations(_request(providers=("triton",)))
    flagtree = match_kernel_implementations(_request(providers=("flagtree",)))

    assert len(triton.candidates) == 2
    assert tuple(
        rejection.implementation.implementation_id for rejection in triton.rejections
    ) == ("FQKI-FLAGTREE-SV-001-A",)
    assert tuple(
        candidate.implementation.implementation_id for candidate in flagtree.candidates
    ) == ("FQKI-FLAGTREE-SV-001-A",)
    assert len(flagtree.rejections) == 2
    assert all(
        tuple(item.code for item in rejection.mismatches) == ("provider",)
        for rejection in flagtree.rejections
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"dtype": "complex128"}, "dtype"),
        ({"layout": "interleaved_statevector"}, "layout"),
        ({"addressing": ("distributed",)}, "addressing"),
        ({"minimum_maturity": "stable"}, "maturity"),
    ],
)
def test_declared_capability_mismatches_are_explained(
    overrides: dict[str, object], code: str
) -> None:
    result = match_kernel_implementations(_request(**overrides))

    assert not result.candidates
    assert len(result.rejections) == 3
    assert all(
        code in {item.code for item in rejection.mismatches}
        for rejection in result.rejections
    )


def test_unknown_semantic_is_distinct_from_known_but_unsupported() -> None:
    result = match_kernel_implementations(
        _request(semantic_id="statevector.apply.unknown.local")
    )

    assert result.semantic_known is False
    assert result.matched is False
    assert not result.candidates
    assert not result.rejections


def test_missing_evidence_rejects_only_the_affected_implementation() -> None:
    result = match_kernel_implementations(_request(), evidence=EVIDENCE[1:])

    assert tuple(
        candidate.implementation.implementation_id for candidate in result.candidates
    ) == (
        "FQKI-TRITON-SV-001-B",
        "FQKI-FLAGTREE-SV-001-A",
    )
    assert tuple(item.code for item in result.rejections[0].mismatches) == ("evidence",)


def test_incomplete_evidence_does_not_make_an_implementation_selectable() -> None:
    incomplete = replace(EVIDENCE[1], gradient_tests=())
    result = match_kernel_implementations(
        _request(direction="backward"),
        evidence=(EVIDENCE[0], incomplete, *EVIDENCE[2:]),
    )

    assert not result.candidates
    assert {item.code for item in result.rejections[1].mismatches} == {"evidence"}
    evidence_mismatch = result.rejections[1].mismatches[0]
    assert evidence_mismatch.requested == ("correctness", "gradient")
    assert evidence_mismatch.available == ("correctness",)


def test_duplicate_evidence_is_rejected() -> None:
    duplicate = replace(EVIDENCE[0], evidence_id="FQKE-TRITON-SV-001-Z")

    with pytest.raises(ValueError, match="duplicate evidence for implementation"):
        match_kernel_implementations(_request(), evidence=(*EVIDENCE, duplicate))


def test_request_rejects_duplicate_capability_filters() -> None:
    with pytest.raises(ValueError, match="providers must be unique"):
        _request(providers=("triton", "triton"))

    with pytest.raises(ValueError, match="addressing entries must be unique"):
        _request(addressing=("local", "local"))
