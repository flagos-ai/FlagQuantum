"""Complete protobuf build source, generated outputs and check failure behavior."""

import csv
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import process, protobuf_command
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = REFERENCE / "dev_tools/build-protos.sh"
GENERATOR = REFERENCE / "protobuf-build-deps"
PLUGINS = REFERENCE / "mypy-profile-deps"


def environment(binary: Path | None = None) -> dict[str, str]:
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(GENERATOR), str(PLUGINS), str(ROOT)]),
        PATH=os.pathsep.join(
            [str(binary or GENERATOR / "bin"), str(PLUGINS / "bin"), os.environ["PATH"]]
        ),
    )


def generated(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*_pb2.py*"))
        if path.suffix in {".py", ".pyi"}
    }


def test_source_script_and_all_eight_schema_blobs() -> None:
    selected = {"Q-0041"}
    count = 0
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            schema = row["path"].startswith("qualtran/protos/") and row[
                "path"
            ].endswith(".proto")
            if row["id"] in selected or schema:
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                count += 1
    assert count == 9
    distributions = {
        d.metadata["Name"].lower(): d.version
        for d in importlib.metadata.distributions(path=[str(GENERATOR)])
    }
    assert distributions == {"grpcio": "1.76.0", "grpcio-tools": "1.76.0"}
    lock = (REFERENCE / "dev_tools/requirements/envs/dev.env.txt").read_text()
    for name, version in distributions.items():
        assert f"{name}=={version}\n" in lock


@pytest.fixture
def schema_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "source with spaces"
    (root / "dev_tools").mkdir(parents=True)
    shutil.copyfile(SOURCE, root / "dev_tools/build-protos.sh")
    shutil.copytree(
        REFERENCE / "qualtran/protos",
        root / "qualtran/protos",
        ignore=shutil.ignore_patterns("*_pb2*", "__pycache__"),
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=15)
    return root


def run_native(
    root: Path, output: Path, schemas: list[str], check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.development.protobuf_command",
            "--schema-root",
            str(root),
            "--output",
            str(output),
            *(["--check"] if check else []),
            *schemas,
        ],
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_complete_original_and_native_generation(
    schema_checkout: Path, tmp_path: Path
) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "python"
    executable.write_text(
        f"#!{sys.executable}\nimport os,sys\nos.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
    )
    executable.chmod(0o755)
    source = subprocess.run(
        ["bash", str(schema_checkout / "dev_tools/build-protos.sh")],
        cwd=tmp_path,
        env=environment(binary),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert source.returncode == 0, source.stdout + source.stderr
    schemas = sorted(
        str(p.relative_to(schema_checkout))
        for p in schema_checkout.glob("qualtran/protos/*.proto")
    )
    output = tmp_path / "native generated"
    native = run_native(schema_checkout, output, schemas)
    assert native.returncode == 0, native.stdout + native.stderr
    original = generated(schema_checkout)
    assert len(original) == 16
    assert generated(output) == original
    assert run_native(schema_checkout, output, schemas, check=True).returncode == 0
    paths = sorted(p for p in output.rglob("*_pb2.py*"))
    times = {p: p.stat().st_mtime_ns for p in paths}
    assert run_native(schema_checkout, output, schemas).returncode == 0
    assert times == {p: p.stat().st_mtime_ns for p in paths}
    changed = output / "qualtran/protos/args_pb2.pyi"
    changed.write_text("stale output\n")
    failed = run_native(schema_checkout, output, schemas, check=True)
    assert failed.returncode == 1
    assert "qualtran/protos/args_pb2.pyi" in failed.stdout
    assert changed.read_text() == "stale output\n"
    assert run_native(schema_checkout, output, schemas).returncode == 0
    assert generated(output) == original


@pytest.mark.parametrize(
    "scenario",
    ["equal", "content", "missing", "extra", "same_count_names", "empty", "failure"],
)
@pytest.mark.parametrize("check", [False, True])
def test_generator_check_apply_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, check: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.proto").write_text('syntax = "proto3";')
    output = tmp_path / "output"
    output.mkdir()
    (output / "sample_pb2.py").write_bytes(b"old" if scenario == "content" else b"new")
    if scenario == "missing":
        (output / "sample_pb2.py").unlink()
    if scenario in {"extra", "same_count_names"}:
        (output / "stale_pb2.py").write_bytes(b"stale")
    if scenario == "same_count_names":
        (output / "sample_pb2.py").unlink()
    cache = output / "__pycache__"
    cache.mkdir()
    (cache / "sample_pb2.pyc").write_bytes(b"irrelevant cache")
    before = generated(output)
    staging = []

    def compiler(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs == dict(
            cwd=source, check=False, timeout=1200, log_run_to_stderr=False
        )
        assert arguments[:5] == [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I=.",
            arguments[4],
        ]
        stage = Path(arguments[4].removeprefix("--python_out="))
        assert arguments[5:] == [f"--mypy_out={stage}", "sample.proto"]
        staging.append(stage)
        if scenario != "empty":
            (stage / "sample_pb2.py").write_bytes(b"new")
        return subprocess.CompletedProcess(arguments, 7 if scenario == "failure" else 0)

    monkeypatch.setattr(process, "run", compiler)
    result = protobuf_command.build_protobuf(
        source, output, [Path("sample.proto")], check=check
    )
    expected = (
        7
        if scenario == "failure"
        else int(scenario == "empty" or (check and scenario != "equal"))
    )
    assert result == expected
    assert all(not stage.exists() for stage in staging)
    if check or scenario in {"empty", "failure"}:
        assert generated(output) == before
    else:
        assert (output / "sample_pb2.py").read_bytes() == b"new"
        if scenario in {"extra", "same_count_names"}:
            assert (output / "stale_pb2.py").exists()


@pytest.mark.parametrize("schemas", [[], ["wrong.txt"], ["../escape.proto"]])
def test_invalid_input_precedes_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schemas: list[str]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "wrong.txt").touch()
    (tmp_path / "escape.proto").touch()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid schema must fail before the compiler")

    monkeypatch.setattr(process, "run", forbidden)
    with pytest.raises(ValueError):
        protobuf_command.build_protobuf(
            source, tmp_path / "output", [Path(p) for p in schemas]
        )


@pytest.mark.parametrize("portable_arrays", [False, True])
@pytest.mark.parametrize(
    "scenario", ["equal", "content", "missing", "extra", "failure"]
)
def test_original_check_mode_and_shell_dependency(
    schema_checkout: Path, tmp_path: Path, portable_arrays: bool, scenario: str
) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    child = binary / "python"
    child.write_text(
        f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
        "output = Path(next(a for a in sys.argv if a.startswith('--python_out=')).split('=',1)[1])\n"
        "path = output / 'qualtran/protos/sample_pb2.py'\npath.parent.mkdir(parents=True,exist_ok=True)\n"
        "path.write_text('new')\nraise SystemExit(17 if os.environ['FQ_TEST_SCENARIO']=='failure' else 0)\n"
    )
    child.chmod(0o755)
    target = schema_checkout / "qualtran/protos/sample_pb2.py"
    if scenario != "missing":
        target.write_text("old" if scenario == "content" else "new")
    if scenario == "extra":
        (target.parent / "stale_pb2.pyi").write_text("stale")
    subprocess.run(["git", "add", "."], cwd=schema_checkout, check=True, timeout=15)
    env = environment(binary) | {"FQ_TEST_SCENARIO": scenario}
    if portable_arrays:
        init = tmp_path / "arrays.sh"
        init.write_text(
            'mapfile() {\n local variable="$2" line quoted index=0\n'
            " while IFS= read -r line; do\n"
            "  printf -v quoted '%q' \"$line\"\n"
            '  eval "$variable[$index]=$quoted"\n'
            "  index=$((index+1))\n done\n}\n"
        )
        env["BASH_ENV"] = str(init)
    result = subprocess.run(
        ["/bin/bash", str(schema_checkout / "dev_tools/build-protos.sh"), "--check"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    try:
        if scenario == "failure":
            assert result.returncode == 17
        elif portable_arrays:
            assert result.returncode == int(scenario != "equal"), result.stderr
        else:
            version = subprocess.check_output(
                ["/bin/bash", "--version"], text=True, timeout=15
            )
            if "version 3." in version:
                assert result.returncode == 0
                assert result.stderr.count("mapfile: command not found") == 2
                assert "up to date" in result.stderr
            else:
                assert result.returncode == int(scenario != "equal"), result.stderr
        if target.exists():
            assert target.read_text() == ("old" if scenario == "content" else "new")
    finally:
        for line in result.stderr.splitlines():
            if line.startswith("Using temporary directory "):
                directory = Path(line.removeprefix("Using temporary directory "))
                assert directory.parent == Path("/tmp") and directory.name.startswith(
                    "nqcr-build-protos."
                )
                if directory.exists():
                    shutil.rmtree(directory)


def test_timeout_cleans_staging_without_touching_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.proto").touch()
    output = tmp_path / "output"
    output.mkdir()
    (output / "sample_pb2.py").write_bytes(b"original")
    stages = []

    def timeout(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        stage = Path(arguments[4].removeprefix("--python_out="))
        stages.append(stage)
        (stage / "sample_pb2.py").write_bytes(b"partial")
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

    monkeypatch.setattr(process, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        protobuf_command.build_protobuf(source, output, [Path("sample.proto")])
    assert all(not stage.exists() for stage in stages)
    assert (output / "sample_pb2.py").read_bytes() == b"original"


@pytest.mark.parametrize("arguments,status", [(["--help"], 0), (["unknown"], 2)])
def test_original_and_native_cli_argument_status(
    arguments: list[str], status: int
) -> None:
    for command in (
        ["/bin/bash", str(SOURCE), *arguments],
        [sys.executable, "-m", "tools.development.protobuf_command", *arguments],
    ):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(),
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == status
        assert bool(result.stdout) if status == 0 else bool(result.stderr)
