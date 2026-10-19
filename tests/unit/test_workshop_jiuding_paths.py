"""Personal workshop identity and storage behavior."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
WORKSHOP = Path(__file__).resolve().parents[2] / "workshops/flagos2026"
spec = importlib.util.spec_from_file_location(
    "paths", WORKSHOP / "scripts/jiuding_paths.py"
)
assert spec and spec.loader
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FLAGQUANTUM_WORKSHOP_INSTANCE", raising=False)
    home, project = tmp_path / "home/alice", tmp_path / "project"
    home.mkdir(parents=True)
    project.mkdir()
    table = tmp_path / "mountinfo"
    table.write_text(
        f"1 0 0:1 /ps/p/alice {home} rw - nfs nfs rw\n"
        f"2 0 0:2 /ps/p/PUBLIC {project} rw - nfs nfs rw\n"
    )

    def resolve(uid="alice", owner="alice", records=None):
        def request(path, body, headers, *, method):
            assert (path, body, headers, method) == (
                "/api/v1/users/token/userinfo",
                None,
                {},
                "GET",
            )
            return {"data": {"id": uid}}

        client = SimpleNamespace(
            _auth=lambda: {},
            _request=request,
            workspace=lambda: {
                "creatorId": owner,
                "projsetId": "ps",
                "projId": "p",
                "storageInfo": (
                    records
                    if records is not None
                    else [
                        {"directoryType": 2, "mountPointPath": ""},
                        {"directoryType": 4, "mountPointPath": ""},
                    ]
                ),
            },
        )
        return paths.personal_directory(
            client,
            mountinfo=table,
            home_root=home.parent,
            project=project / "flagquantum",
        )

    return home, project, table, resolve


def test_default_mounts_and_account_identity(storage) -> None:
    home, project, _, resolve = storage
    assert resolve() == home / "flagquantum-workspace/alice"
    assert resolve("bob") == project / "flagquantum/bob"
    assert not (project / "flagquantum/alice").exists()


def test_image_instance_preserves_earlier_user_workspace(
    storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, resolve = storage
    original = resolve()
    (original / "user.txt").write_text("keep")
    monkeypatch.setenv("FLAGQUANTUM_WORKSHOP_INSTANCE", "pr621-20261012")
    fresh = resolve()
    assert fresh == original / "instances/pr621-20261012"
    assert not (fresh / "user.txt").exists()
    assert (original / "user.txt").read_text() == "keep"


def test_unsafe_image_instance_writes_nothing(
    storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, project, _, resolve = storage
    monkeypatch.setenv("FLAGQUANTUM_WORKSHOP_INSTANCE", "../another-user")
    with pytest.raises(RuntimeError, match="image-defined"):
        resolve()
    assert not list(home.iterdir()) and not list(project.iterdir())


@pytest.mark.parametrize(
    "uid", [None, True, "", "../bob", "/root", "a/b", ".", "..", "a" * 129]
)
def test_invalid_identity_writes_nothing(storage, uid) -> None:
    home, project, _, resolve = storage
    with pytest.raises(RuntimeError, match="safe platform ID"):
        resolve(uid)
    assert not list(home.iterdir()) and not list(project.iterdir())


@pytest.mark.parametrize("key", ["mountPointPath", "mountPath", "mountPoint", "path"])
def test_explicit_mount_and_symlink_rejection(storage, key: str) -> None:
    _, project, _, resolve = storage
    destination = resolve(records=[{key: str(project)}])
    assert destination == project / "flagquantum/alice"
    destination.rename(project / "saved")
    destination.symlink_to(project / "saved", target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        resolve(records=[{key: str(project)}])


@pytest.mark.parametrize(
    "records", [[], [{"path": "/"}], [{"directoryType": 99, "mountPointPath": ""}]]
)
def test_unverified_configuration_stops(storage, records) -> None:
    _, _, _, resolve = storage
    with pytest.raises(RuntimeError, match="No writable personal"):
        resolve(records=records)


def test_other_users_bind_root_is_not_a_personal_mount(storage) -> None:
    home, _, table, resolve = storage
    table.write_text(table.read_text().replace("/ps/p/alice", "/ps/p/other"))
    with pytest.raises(RuntimeError, match="No writable personal"):
        resolve(records=[{"directoryType": 2, "mountPointPath": ""}])
    assert not list(home.iterdir())


def test_two_users_preserve_notebooks_outputs_and_receipts(
    tmp_path: Path, storage
) -> None:
    _, _, _, resolve = storage
    template = tmp_path / "template"
    (template / "notebooks").mkdir(parents=True)
    (template / "notebooks/lab.ipynb").write_text("template")
    (template / "notebooks_cn").mkdir()
    (template / "notebooks_cn/lab.ipynb").write_text("localized template")
    (template / "outputs").mkdir()
    (template / "outputs/old.txt").write_text("private")
    (template / ".env").write_text("secret")
    for uid in ("alice", "bob"):
        directory = resolve(uid)
        workshop = paths.prepare_workshop(template, directory)
        assert not (workshop / ".env").exists() and not (workshop / "outputs").exists()
        notebook = workshop / "notebooks/lab.ipynb"
        notebook.write_text(uid)
        localized = workshop / "notebooks_cn/lab.ipynb"
        assert localized.read_text() == "localized template"
        localized.write_text(f"{uid} localized edits")
        (workshop / "outputs").mkdir()
        pointer = workshop / "outputs/latest.txt"
        script, receipt = paths.stage_job(uid, directory)
        receipt.write_text(uid)
        pointer.write_text(str(receipt))
        paths.prepare_workshop(template, directory)
        assert notebook.read_text() == uid and script.read_text() == uid
        assert localized.read_text() == f"{uid} localized edits"
        assert Path(pointer.read_text()).read_text() == uid
        assert script.parent.stat().st_mode & 0o777 == 0o700
        assert paths.stage_job("another", directory)[0].parent != script.parent


def test_notebook_blocks_template_and_prepares_personal_results(
    tmp_path: Path, storage, monkeypatch
) -> None:
    _, _, _, resolve = storage
    directory = resolve()
    monkeypatch.setitem(
        sys.modules,
        "jiuding_paths",
        SimpleNamespace(
            personal_directory=lambda client: directory,
            prepare_workshop=paths.prepare_workshop,
            stage_job=paths.stage_job,
        ),
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.syspath_prepend(str(WORKSHOP / "scripts"))
    monkeypatch.setitem(
        sys.modules,
        "flagquantum.remote.compute.jiuding",
        SimpleNamespace(JiudingClient=lambda: object()),
    )
    monkeypatch.setenv("JIUDING_AK", "test-presence")
    exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda p: p == Path("/etc/accesskey") or exists(p)
    )
    template = tmp_path / "flagos2026"
    cwd = template / "notebooks"
    cwd.mkdir(parents=True)
    notebook = json.loads((WORKSHOP / "notebooks/05_jiuding_jobs.ipynb").read_text())
    cells = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    locator = next(c for c in cells if "search_roots" in c)
    monkeypatch.chdir(cwd)
    with pytest.raises(RuntimeError, match="Personal copy is ready"):
        exec(locator, {})
    assert not (template / "outputs").exists()
    monkeypatch.chdir(directory / "workshop/flagos2026/notebooks")
    scope = {}
    exec(locator, scope)

    assert scope["OUTPUTS"].is_relative_to(directory)
    assert not list(scope["OUTPUTS"].glob("*.receipt.json"))
