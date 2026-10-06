"""Contract for the predicate that decides whether an anchor may be read here.

Every compiler anchor in this directory records the readings of one of Qiskit's
own transpiler passes, and every one of those files now asks
``require_certified_lane`` whether the release that answered is one this
repository certifies. That makes the predicate load-bearing for the whole anchor
family, so its own failure modes are tested here rather than only through the
anchors it guards.

The failure mode worth pinning is that this guard could be *neutered* rather than
satisfied. A policy that cannot be parsed, or a version string that cannot be read
as a lane, would -- if the predicate were written the other way round -- turn
every anchor comparison into a pass. Both directions are asserted: a certified
lane is accepted, and an unreadable or uncertified one is refused rather than
quietly passed. The predicate reads ``dependency-policy.toml`` on every call
precisely so that a lane bump moves these anchors with it, and the tests below
substitute the policy file to reach the branches a correct policy never takes.
"""

from __future__ import annotations

import pytest

from tests.benchmark_contract import qiskit_lane

pytestmark = pytest.mark.benchmark_contract


_POLICY_TEMPLATE = """\
[tested]
qiskit = {lanes}
"""


def _policy(tmp_path, lanes: str):
    """Point the predicate at a substitute policy and return the path.

    The predicate reads its policy from a module-level path, so substituting that
    path is the only way to reach the branches a correct policy never takes. The
    substitute is written per call and never cached, matching how the real policy
    is read.
    """

    path = tmp_path / "dependency-policy.toml"
    path.write_text(_POLICY_TEMPLATE.format(lanes=lanes), encoding="utf-8")
    return path


def test_a_patch_release_and_a_prerelease_both_read_as_their_lane() -> None:
    """The certified unit is the lane, so the patch component must not be read."""

    assert qiskit_lane.lane_of("2.5.2") == "2.5"
    assert qiskit_lane.lane_of("2.0.0") == "2.0"
    assert qiskit_lane.lane_of("2.0.0rc1") == "2.0"
    assert qiskit_lane.lane_of("2.5.2+local.1") == "2.5"


def test_a_version_that_is_not_a_lane_is_returned_unchanged() -> None:
    """Unreadable readings must not be mapped onto a lane they did not come from.

    Returning the input unchanged is what makes the caller fail: the result cannot
    equal any declared lane, so an anchor taken on an instrument nobody can name
    is reported instead of being compared.
    """

    for version in ("nightly", "2", "", "x.y.z", "two.five"):
        assert qiskit_lane.lane_of(version) == version


def test_being_equal_to_its_own_lane_reading_does_not_make_a_lane() -> None:
    """The policy check may not be written as equality with ``lane_of``.

    ``lane_of`` is the identity on anything it cannot read, so ``"latest"`` equals
    its own result and would be accepted as a lane by a round-trip check -- which
    is exactly how a lane list that cannot be parsed would stop being refused. A
    patch release must not be accepted either, since the certified unit is the
    lane rather than the patch.
    """

    for unreadable in ("latest", "2", "x.y", "2."):
        assert qiskit_lane.lane_of(unreadable) == unreadable
        assert not qiskit_lane.is_lane(unreadable)
    for lane in ("2.0", "2.5", "10.12"):
        assert qiskit_lane.is_lane(lane)
    for patch in ("2.5.2", "2.5.0rc1", "2.5.2+local.1"):
        assert not qiskit_lane.is_lane(patch)


def test_the_live_policy_declares_lanes_and_they_are_what_the_anchors_compare() -> None:
    """The repository's own policy must be readable in the form the anchors use."""

    lanes = qiskit_lane.certified_lanes()

    assert lanes, "the repository must certify at least one Qiskit lane"
    assert all(lane == qiskit_lane.lane_of(lane) for lane in lanes), lanes


def test_every_patch_release_on_a_certified_lane_is_accepted() -> None:
    """Certifying a lane certifies its patch releases, which is the point of it."""

    for lane in qiskit_lane.certified_lanes():
        for version in (lane, f"{lane}.0", f"{lane}.2", f"{lane}.0rc1"):
            qiskit_lane.require_certified_lane(version, recording="a test")


def test_a_truthy_but_uncertified_reading_is_refused() -> None:
    """The guard must not be satisfiable by the field merely being non-empty.

    This is the mutation that matters. Asserting only that the recorded version is
    truthy passes on every host, which is what the anchors did before they were
    made to read a lane; the guard has to fail on a release the repository never
    measured on, and it has to say which release and which anchor.
    """

    with pytest.raises(pytest.fail.Exception, match="not on a lane"):
        qiskit_lane.require_certified_lane("1.2.4", recording="this anchor")


def test_a_policy_that_cannot_be_read_refuses_rather_than_passes(
    tmp_path, monkeypatch
) -> None:
    """An unparseable lane list must fail closed instead of certifying everything."""

    for lanes in ("[]", '"2.0"', '["2.0", "2.5.2"]', '["latest"]', '["2"]', '["2.x"]'):
        monkeypatch.setattr(qiskit_lane, "_POLICY", _policy(tmp_path, lanes))
        with pytest.raises(AssertionError, match="must declare|must list"):
            qiskit_lane.certified_lanes()
