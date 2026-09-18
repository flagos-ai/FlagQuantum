"""The stage 2a names are reachable from the qec package root."""

from __future__ import annotations

import pytest

import flagquantum.qec as qec

pytestmark = pytest.mark.unit


def test_stage_two_names_are_published() -> None:
    for name in (
        "DemError",
        "DemSample",
        "DetectorErrorModel",
        "PhenomenologicalNoise",
    ):
        assert name in qec.__all__
        assert hasattr(qec, name)


def test_stage_one_names_are_still_published() -> None:
    for name in ("Pauli", "CodeCheck", "RepetitionCode", "build_memory_circuit"):
        assert name in qec.__all__


def test_frozen_names_are_unchanged() -> None:
    for name in (
        "Decoder",
        "StreamingDecoder",
        "RepetitionLookupDecoder",
        "RepetitionStreamingLookupDecoder",
        "RepetitionTemporalDecoder",
        "run_repetition_memory_experiment",
        "run_repetition_memory_noise_sweep",
    ):
        assert name in qec.__all__
