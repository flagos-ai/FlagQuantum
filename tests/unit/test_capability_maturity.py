import hashlib
import json
import subprocess
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tools.check_capability_maturity import (
    CODE_VERSION_ORIGIN_FIELD,
    EXPECTED_LEVELS,
    REQUIRED_USER_FIELDS,
    aggregate_claim_value,
    claim_values,
    code_version_errors,
    maturity_errors,
)
from tools.evidence_provenance import (
    ORIGIN_PRODUCING_HOST_HISTORY,
    ORIGIN_REPOSITORY_HISTORY,
    ProvenanceUnavailableError,
    revision_resolves,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]

#: A full 40-character revision. It is shaped like a commit and names none.
FABRICATED_REVISION = "0" * 40


def _require_repository_history() -> None:
    """Skip when this checkout cannot resolve a revision against its history.

    Performance claims pin the revision their artifact was produced from, and
    resolving that pin needs the commits themselves. A depth-1 clone cannot tell
    a revision it never fetched from one the repository never contained, so the
    checked-in matrix is validated in the full-history quality job instead.
    """

    try:
        revision_resolves(FABRICATED_REVISION, ROOT)
    except ProvenanceUnavailableError as error:
        pytest.skip(f"the capability matrix revision pins need full history: {error}")


def test_repository_capability_maturity_matrix_is_valid():
    _require_repository_history()
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    assert maturity_errors(data, ROOT) == ()


def test_levels_are_explicit_and_not_collapsed():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    assert tuple(data["levels"]) == EXPECTED_LEVELS
    assert (
        data["capabilities"]["sharded_mps_training"]["level"] == "development_evidence"
    )
    assert data["capabilities"]["tensor_network_training"]["level"] == "experimental"
    assert data["capabilities"]["local_statevector"]["level"] == "production_supported"
    assert data["capabilities"]["ir"]["level"] == "release_certified"


def test_release_certification_fails_closed_without_release_evidence():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["local_statevector"]["level"] = "release_certified"
    errors = maturity_errors(data, ROOT)
    assert "local_statevector: release_certified requires release_gate" in errors
    assert "local_statevector: release_certified requires release_artifact" in errors


def test_every_capability_has_user_discovery_metadata():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    for capability in data["capabilities"].values():
        assert set(REQUIRED_USER_FIELDS) <= set(capability)


def test_internal_experiments_may_have_no_public_api() -> None:
    _require_repository_history()
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    capability = data["capabilities"]["split_real_imag_statevector_p5_autograd_bridge"]
    assert capability["level"] == "experimental"
    assert capability["public_apis"] == []
    assert maturity_errors(data, ROOT) == ()


def test_supported_capability_requires_a_public_api() -> None:
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["local_statevector"]["public_apis"] = []
    assert any(
        "local_statevector: supported or certified capability requires a public API"
        in error
        for error in maturity_errors(data, ROOT)
    )


def test_user_discovery_links_fail_closed_when_missing():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["ir"]["quick_start"] = "examples/does_not_exist.py"
    errors = maturity_errors(data, ROOT)
    assert "ir: quick_start path does not exist: examples/does_not_exist.py" in errors


def test_performance_claim_binds_artifact_hash_code_environment_and_maturity():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    assert claim["artifact_sha256"]
    assert claim["code_version"]
    assert claim["environment"]
    assert claim["maturity"] == "development_evidence"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_sha256", "0" * 64, "artifact_sha256 mismatch"),
        ("code_version", "unknown", "code_version does not match artifact commit"),
        ("maturity", "experimental", "claim maturity must equal capability level"),
    ),
)
def test_performance_claim_identity_fails_closed(field, value, message):
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim[field] = value
    assert any(message in error for error in maturity_errors(data, ROOT))


def test_performance_claim_fails_closed_when_artifact_or_environment_is_missing():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim["artifact"] = "benchmarks/results/local/does-not-exist.json"
    errors = maturity_errors(data, ROOT)
    assert any("artifact does not exist" in error for error in errors)

    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim["artifact"] = "README.md"
    errors = maturity_errors(data, ROOT)
    assert any("artifact must be under benchmarks/results" in error for error in errors)

    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim["environment"] = []
    assert any("missing environment" in error for error in maturity_errors(data, ROOT))


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _minimal_matrix(root: Path, code_version: str) -> dict:
    """Build the smallest matrix that reaches the performance-claim checks.

    ``root`` is also the repository the pin is resolved against, so the caller
    decides whether it contains ``code_version``.
    """

    artifact = Path("benchmarks/results/local/claim.json")
    payload = {"commit": code_version, "value": 1}
    written = root / artifact
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(json.dumps(payload), encoding="utf-8")
    for name in ("quick_start.md", "documentation.md", "focused_tests.py"):
        (root / name).write_text("", encoding="utf-8")
    return {
        "schema": "flagquantum_capability_maturity_v2",
        "levels": {
            level: {"rank": rank, "required_evidence": []}
            for rank, level in enumerate(EXPECTED_LEVELS)
        },
        "capabilities": {
            "sharded_mps_training": {
                "title": "Sharded MPS training",
                "summary": "Train a sharded MPS.",
                "category": "simulation_and_training",
                "user_goals": ["Train a sharded MPS"],
                "public_apis": [],
                "runtime_modes": ["single_process"],
                "hardware": ["cpu"],
                "gradient_support": ["not_applicable"],
                "distribution_semantics": ["sharded_across_ranks"],
                "quick_start": "quick_start.md",
                "documentation": "documentation.md",
                "level": "development_evidence",
                "owner": "simulation",
                "limitations": "Development evidence only.",
                "focused_tests": "focused_tests.py",
                "development_artifact": "documentation.md",
                "performance_claims": [
                    {
                        "id": "claim",
                        "title": "Recorded claim",
                        "artifact": str(artifact),
                        "artifact_sha256": hashlib.sha256(
                            written.read_bytes()
                        ).hexdigest(),
                        "code_version": code_version,
                        "maturity": "development_evidence",
                        "scope": "The recorded workload only.",
                        "environment_limitations": "Development host.",
                        "evidence_checks": {"value": 1},
                        "metrics": [
                            {
                                "label": "Value",
                                "selector": "value",
                                "format": "integer",
                            }
                        ],
                        "environment": [
                            {
                                "label": "Value",
                                "selector": "value",
                                "format": "integer",
                            }
                        ],
                    }
                ],
            }
        },
    }


def _commit_in(root: Path, name: str) -> str:
    """Create one commit in ``root`` and return its revision."""

    (root / name).write_text(name, encoding="utf-8")
    _git(root, "add", name)
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        name,
    )
    return _git(root, "rev-parse", "HEAD")


def test_performance_claim_code_version_must_resolve_in_repository_history(tmp_path):
    """A faithful transcription of a revision that names no commit is not proof.

    The matrix and the artifact both carry the pin, so agreeing with each other
    proves only that the value was copied. Resolution is what a reader needs:
    this repository contains the revision, or the claim says where it came from.
    """

    _git(tmp_path, "init")
    data = _minimal_matrix(tmp_path, FABRICATED_REVISION)
    errors = maturity_errors(data, tmp_path)
    assert errors == (
        f"sharded_mps_training/claim code_version {FABRICATED_REVISION} names no "
        f"commit in this repository; record {CODE_VERSION_ORIGIN_FIELD} as "
        f'"{ORIGIN_PRODUCING_HOST_HISTORY}" if the pin comes from the producing '
        "host's history",
    )

    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim[CODE_VERSION_ORIGIN_FIELD] = ORIGIN_PRODUCING_HOST_HISTORY
    assert maturity_errors(data, tmp_path) == ()

    claim[CODE_VERSION_ORIGIN_FIELD] = "somewhere_else"
    assert maturity_errors(data, tmp_path) == (
        f"sharded_mps_training/claim {CODE_VERSION_ORIGIN_FIELD} 'somewhere_else' is "
        "not supported; use one of 'repository_history', 'producing_host_history'",
    )


def test_performance_claim_code_version_of_a_real_revision_needs_no_disclosure(
    tmp_path,
):
    """A pin the repository contains is traceable on its own."""

    _git(tmp_path, "init")
    head = _commit_in(tmp_path, "recorded.txt")
    data = _minimal_matrix(tmp_path, head)
    assert maturity_errors(data, tmp_path) == ()
    assert (
        data["capabilities"]["sharded_mps_training"]["performance_claims"][0][
            "code_version"
        ]
        == head
    )


def test_repository_performance_claims_declare_their_code_version_origin():
    """Every claim either resolves here or says why it cannot."""

    _require_repository_history()
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    for capability in data["capabilities"].values():
        for claim in capability.get("performance_claims", ()):
            assert (
                code_version_errors(
                    claim["code_version"],
                    claim.get(CODE_VERSION_ORIGIN_FIELD, ORIGIN_REPOSITORY_HISTORY),
                    label=claim["id"],
                    root=ROOT,
                )
                == ()
            )


def test_performance_claim_code_version_must_be_a_full_revision():
    assert code_version_errors("9d56a6e", ORIGIN_REPOSITORY_HISTORY, label="claim") == (
        "claim requires a full hexadecimal code_version, not '9d56a6e'",
    )


def test_performance_claim_code_version_fails_closed_when_history_is_unavailable(
    tmp_path,
):
    """An unusable checkout is not evidence that the pin is bad."""

    errors = code_version_errors(
        FABRICATED_REVISION,
        ORIGIN_REPOSITORY_HISTORY,
        label="claim",
        root=tmp_path,
    )
    assert len(errors) == 1
    assert "could not be checked" in errors[0]
    assert "names no commit" not in errors[0]


def test_performance_claim_code_version_must_be_reachable_from_a_ref(tmp_path):
    """A commit only a deleted branch pointed at is not a pin a reader obtains.

    The producing checkout holds that commit, so resolution alone passes while a
    clone that fetches this repository's refs never receives it. Recording
    evidence at the head of the branch that records it produces exactly this pin,
    which is why a claim is held to published history rather than to the object
    database it happened to be written in.
    """

    _git(tmp_path, "init")
    _commit_in(tmp_path, "recorded.txt")
    branch = _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD")
    _git(tmp_path, "checkout", "-q", "-b", "evidence")
    orphan = _commit_in(tmp_path, "swept.txt")
    _git(tmp_path, "checkout", "-q", branch)
    _git(tmp_path, "branch", "-D", "evidence")
    _git(tmp_path, "reflog", "expire", "--expire=now", "--all")

    assert revision_resolves(orphan, tmp_path) is True
    assert code_version_errors(
        orphan,
        ORIGIN_REPOSITORY_HISTORY,
        label="claim",
        root=tmp_path,
    ) == (
        f"claim code_version {orphan} resolves here but no ref reaches it, so a "
        "clone that fetches this repository does not obtain it; re-record the "
        "claim at a revision published history reaches",
    )


def _shallow_clone(tmp_path: Path) -> Path:
    """Clone a two-commit repository at depth one, so only its tip resolves."""

    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init")
    _commit_in(origin, "first.txt")
    _commit_in(origin, "second.txt")

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "--depth", "1", origin.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
        text=True,
    )
    return shallow


def test_performance_claim_code_version_fails_closed_in_a_shallow_clone(tmp_path):
    """A shallow clone cannot tell a missing pin from one it never fetched."""

    shallow = _shallow_clone(tmp_path)
    errors = code_version_errors(
        FABRICATED_REVISION,
        ORIGIN_REPOSITORY_HISTORY,
        label="claim",
        root=shallow,
    )
    assert len(errors) == 1
    assert "shallow clone" in errors[0]


def test_shallow_clone_refs_are_not_read_as_unpublished_history(tmp_path):
    """A fetched tip is not evidence that every other pin went unpublished.

    The tip of a shallow clone resolves, and asking whether a published ref
    reaches a revision would then answer from the refs that were never fetched
    and report a valid pin as orphaned. The checkout says it cannot answer.
    """

    shallow = _shallow_clone(tmp_path)
    tip = _git(shallow, "rev-parse", "HEAD")
    assert revision_resolves(tip, shallow) is True
    errors = code_version_errors(
        tip,
        ORIGIN_REPOSITORY_HISTORY,
        label="claim",
        root=shallow,
    )
    assert len(errors) == 1
    assert "could not be checked" in errors[0]
    assert "no ref reaches it" not in errors[0]


def test_performance_claim_code_version_origin_is_case_sensitive_and_validated():
    errors = code_version_errors(
        FABRICATED_REVISION,
        ORIGIN_PRODUCING_HOST_HISTORY.upper(),
        label="claim",
        root=ROOT,
    )
    assert len(errors) == 1
    assert "is not supported" in errors[0]


def test_claim_selector_aggregates_artifact_values():
    values = claim_values({"ranks": [{"value": 2}, {"value": 3}]}, "ranks[].value")
    assert aggregate_claim_value(values, "max") == 3
