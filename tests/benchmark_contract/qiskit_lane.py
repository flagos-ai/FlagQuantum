"""Which Qiskit releases a benchmark-contract anchor may be read on.

A Qiskit anchor records the readings of one of Qiskit's own transpiler passes, so
those readings are only reproducible on a Qiskit the repository certifies. Which
releases those are is declared once, in ``dependency-policy.toml`` under
``[tested].qiskit`` as a list of ``major.minor`` lanes, and it is read here on
every call rather than copied into a test. A lane bump therefore moves these
assertions with it instead of leaving a second, stale source of truth behind.

**The certified unit is the lane, not the patch release.** Measured over the
whole compiler anchor family, every reading is identical on ``2.0.0`` and
``2.0.3``, and identical on ``2.5.0`` and ``2.5.2``; readings move only when the
lane moves. Pinning a patch release therefore does not pin a measurement, it pins
whichever machine the person who wrote the anchor happened to have.

The checks below are fail-closed in the direction that matters: a version that
cannot be read as a lane is returned unchanged, so it matches no certified lane
and the caller fails rather than passing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_POLICY = REPOSITORY_ROOT / "dependency-policy.toml"


def lane_of(version: str) -> str:
    """Return the ``major.minor`` lane of a ``qiskit.__version__`` reading.

    ``"2.5.2"`` and ``"2.5.0rc1"`` both give ``"2.5"``. Anything that does not
    begin with two numeric components is returned unchanged, which is not a lane
    and so cannot match one: a version this function cannot read is a failure at
    the call site rather than a silent pass here.
    """

    head = version.split("+", 1)[0].split(".", 2)
    if len(head) >= 2 and all(part.isdigit() for part in head[:2]):
        return f"{head[0]}.{head[1]}"
    return version


def certified_lanes() -> tuple[str, ...]:
    """The Qiskit lanes this repository certifies, in policy order.

    Raises ``AssertionError`` if the policy does not declare them in the
    ``major.minor`` form these anchors are compared on, because a lane list that
    cannot be parsed would otherwise turn every comparison below into a pass.
    """

    policy = tomllib.loads(_POLICY.read_text(encoding="utf-8"))
    lanes = policy.get("tested", {}).get("qiskit")
    assert isinstance(lanes, list) and lanes, (
        f"{_POLICY.name} must declare [tested].qiskit as a non-empty list of "
        f"version lanes; read {lanes!r}"
    )
    assert all(lane == lane_of(lane) for lane in lanes), (
        f"[tested].qiskit must list major.minor lanes, not ranges or patch "
        f"releases; read {lanes!r}"
    )
    return tuple(lanes)


def require_certified_lane(version: str, *, recording: str) -> None:
    """Fail unless ``version`` is a Qiskit release on a certified lane.

    ``recording`` names what was read, so the failure says which anchor is being
    compared with which instrument. The failure is deliberate and is not a skip.
    The only lane that installs Qiskit here is CI's ``qiskit-optional`` job, which
    collects none of these files, so a skip would fire nowhere that reports and
    the only place the comparison would ever run is a developer machine on a
    release nobody certified -- which is exactly how a stale anchor stays
    invisible. A host outside the certified lanes is not comparable with these
    readings, and that is a result about the host, not silence about the anchor.
    """

    lanes = certified_lanes()
    if lane_of(version) in lanes:
        return
    pytest.fail(
        f"{recording} records readings taken on Qiskit {version}, which is not on "
        f"a lane this repository certifies ({', '.join(lanes)}, from "
        f"{_POLICY.name} [tested].qiskit). These readings are not reproducible "
        f"here. Install a certified lane to read this anchor.",
        pytrace=False,
    )
