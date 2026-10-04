"""The state-mode vocabulary stays the one authority on what a mode names.

A state mode is a payload's own word for the execution path that produced it, so
the audit reads capabilities out of these sets and the release gates admit a
payload by them. A set that drifted from ``_backend_family``, or that grew a
member belonging to another capability, would silently reclassify evidence; these
tests read the sets against the classifier they are consumed by rather than
against a second copy of the names.
"""

from __future__ import annotations

import pytest

from flagquantum.runtime.audit import vocabulary
from flagquantum.runtime.audit.validation_helpers import _backend_family

pytestmark = pytest.mark.unit

FAMILIES = {
    "statevector": vocabulary.STATEVECTOR_STATE_MODES,
    "mps": vocabulary.MPS_STATE_MODES,
    "tensor_network": vocabulary.TENSOR_NETWORK_STATE_MODES,
}


def test_every_state_mode_is_classified_as_the_family_that_names_it() -> None:
    for family, modes in FAMILIES.items():
        assert modes, f"the {family} vocabulary must name at least one mode"
        for mode in modes:
            assert _backend_family({"state_mode": mode}) == family
            # The classifier reads `mode` as the fallback key, so a payload that
            # states its mode there is classified the same way.
            assert _backend_family({"mode": mode}) == family


def test_the_families_do_not_share_a_state_mode() -> None:
    names = sorted(FAMILIES)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert FAMILIES[left].isdisjoint(FAMILIES[right]), (
                f"{left} and {right} must not both claim one state mode"
            )


def test_the_tensor_network_vocabulary_names_the_canonical_modes() -> None:
    assert frozenset(
        {"distributed_tensor_network", "jax_sharded_tensor_network", "tensor_network"}
    ) == vocabulary.TENSOR_NETWORK_STATE_MODES
    # The abbreviation is recognized by the classifier because tracked artifacts
    # and comparison payloads use it, but every vocabulary names canonical modes
    # and leaves abbreviations to the classifier, as the sibling sets do.
    assert "tn" not in vocabulary.TENSOR_NETWORK_STATE_MODES
    assert "sv" not in vocabulary.STATEVECTOR_STATE_MODES


def test_a_descriptive_mode_is_classified_without_being_a_vocabulary_member() -> None:
    # Tracked artifacts state the contraction shape rather than the canonical
    # name, so the classifier recognizes the family prefix while the vocabulary
    # stays a closed set a release payload can be checked against.
    described = "distributed_tensor_network_amplitudes"
    assert described not in vocabulary.TENSOR_NETWORK_STATE_MODES
    assert _backend_family({"state_mode": described}) == "tensor_network"
    assert _backend_family({"state_mode": "unknown_mode"}) == "unknown"
    assert _backend_family({}) == "unknown"


def test_the_state_mode_vocabularies_are_published() -> None:
    for name in (
        "STATEVECTOR_STATE_MODES",
        "MPS_STATE_MODES",
        "TENSOR_NETWORK_STATE_MODES",
    ):
        assert name in vocabulary.__all__
        assert isinstance(getattr(vocabulary, name), frozenset)
    assert set(vocabulary.__all__) == {
        name for name in vars(vocabulary) if name.isupper() and not name.startswith("_")
    }
