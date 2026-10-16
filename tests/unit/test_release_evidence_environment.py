from pathlib import Path

import pytest

from tools.check_release_evidence_environment import readiness_errors

pytestmark = pytest.mark.unit

RELEASE_ROOT = Path("/repo/benchmarks/results/scalability")


def _seal(destination: str) -> tuple[str, ...]:
    """Run the preflight with a complete environment and one destination.

    Everything except the destination is satisfied, so a non-empty error list is
    about where the seal would land and nothing else.
    """

    return readiness_errors(
        commit="a" * 40,
        signing_key_present=True,
        device_uuids=("GPU-0", "GPU-1"),
        world_size=2,
        seal_destination=Path(destination),
        release_root=RELEASE_ROOT,
    )


def test_release_evidence_preflight_accepts_complete_environment():
    assert (
        readiness_errors(
            commit="a" * 40,
            signing_key_present=True,
            device_uuids=("GPU-0", "GPU-1"),
            world_size=2,
        )
        == ()
    )


def test_release_evidence_preflight_fails_closed_without_authority_or_hardware():
    errors = readiness_errors(
        commit="",
        signing_key_present=False,
        device_uuids=("GPU-0",),
        world_size=2,
    )
    assert len(errors) == 3
    assert any("Git commit" in error for error in errors)
    assert any("FQ_EVIDENCE_SIGNING_KEY" in error for error in errors)
    assert any("GPU UUIDs" in error for error in errors)


def test_a_candidate_destination_is_accepted_when_the_release_directory_is_empty():
    assert _seal("/repo/benchmarks/results/smoke/release_candidates/mps/leg.json") == ()


def test_a_destination_inside_the_release_directory_is_refused_when_it_is_empty():
    """The state that made the old rule fail open.

    A release directory that holds nothing yet is exactly the state a first
    campaign seals in, so a rule phrased as "the release directory already
    contains promoted JSON" admitted the one seal it existed to prevent.
    """

    errors = _seal("/repo/benchmarks/results/scalability/leg.json")
    assert len(errors) == 1
    assert "inside the release directory" in errors[0]


def test_a_candidate_destination_is_accepted_when_the_directory_holds_promoted_json():
    """The state that made the old rule fail closed.

    A promoted release directory is not emptied again, so refusing to seal while
    it holds JSON refused every later seal for the life of the branch. The
    destination is what matters, and this one is outside the release directory.
    """

    assert _seal("/repo/benchmarks/results/smoke/release_candidates/tn/leg.json") == ()


def test_a_destination_spelled_through_the_release_directory_is_refused():
    """A lexical prefix test would accept this path and seal into the release
    directory, so both sides are resolved before they are compared."""

    errors = _seal("/repo/benchmarks/results/scalability/../scalability/leg.json")
    assert len(errors) == 1
    assert "inside the release directory" in errors[0]


def test_a_destination_that_merely_starts_with_the_release_directory_name_is_accepted():
    """A neighbouring directory is not the release directory, and a check that
    compared path strings without a separator would call it one."""

    assert _seal("/repo/benchmarks/results/scalability_candidates/leg.json") == ()


def test_a_destination_that_resolves_into_the_release_directory_is_refused(tmp_path):
    release = tmp_path / "scalability"
    release.mkdir()
    link = tmp_path / "candidate"
    link.symlink_to(release, target_is_directory=True)
    errors = readiness_errors(
        commit="a" * 40,
        signing_key_present=True,
        device_uuids=("GPU-0",),
        world_size=1,
        seal_destination=link / "leg.json",
        release_root=release,
    )
    assert len(errors) == 1
    assert "inside the release directory" in errors[0]


def test_the_environment_check_alone_names_no_destination():
    """The standalone command validates the environment; a caller that is not
    sealing anywhere has no destination to have got wrong."""

    assert (
        readiness_errors(
            commit="a" * 40,
            signing_key_present=True,
            device_uuids=("GPU-0", "GPU-1"),
            world_size=2,
            seal_destination=None,
            release_root=RELEASE_ROOT,
        )
        == ()
    )
