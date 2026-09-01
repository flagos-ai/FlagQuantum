"""Documentation contracts for explicitly experimental capability APIs."""

from __future__ import annotations

import flagquantum as fq
from tools.docs_source_of_truth import capability_api_is_available


def test_experimental_capability_api_must_be_importable_and_scoped() -> None:
    stable = set(fq.__all__)

    assert capability_api_is_available(
        fq, stable, "experimental.simulation.run_tebd", "experimental"
    )
    assert not capability_api_is_available(
        fq, stable, "experimental.simulation.run_tebd", "development_evidence"
    )
    assert not capability_api_is_available(
        fq, stable, "experimental.does_not_exist", "experimental"
    )
    assert capability_api_is_available(fq, stable, "run", "production_supported")


def test_experimental_capability_resolution_does_not_depend_on_root_attribute(
    monkeypatch,
) -> None:
    stable = set(fq.__all__)
    monkeypatch.delattr(fq, "experimental", raising=False)

    assert capability_api_is_available(
        fq, stable, "experimental.simulation.run_tebd", "experimental"
    )
