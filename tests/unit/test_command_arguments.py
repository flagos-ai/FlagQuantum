"""Compare complete source argument helpers and their real CLI consumers."""

import importlib
import json
import math
import os
import struct
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest

from flagquantum.ecosystem._command_arguments import (
    bool_argument,
    check_unknown_arguments,
    enum_argument,
    find_argument,
    float_argument,
    int64_argument,
    open_file_argument,
    require_argument,
)
from flagquantum.ecosystem._command_numbers import parse_c_float, parse_int64

pytestmark = pytest.mark.unit


def observe(
    operation: str, name: str, default: str, low: str, high: str, argv: list[str]
) -> object:
    if operation == "file":
        path = Path("argument-file")
        path.unlink(missing_ok=True)
        if high != "absent":
            path.write_bytes(b"abc\r\n")
        fallback = BytesIO() if default == "DEFAULT" else None
        try:
            with open_file_argument(name, fallback, low, argv) as opened:
                is_default = opened is fallback
                if high == "write":
                    opened.write(b"Z")
                    opened.flush()
            return is_default, path.read_bytes() if path.exists() else b""
        finally:
            if fallback is not None:
                fallback.close()
            path.unlink(missing_ok=True)
    if operation == "find":
        return find_argument(name, argv)
    if operation == "require":
        return require_argument(name, argv)
    if operation == "bool":
        return bool_argument(name, argv)
    if operation == "int":
        return int64_argument(name, int(default), int(low), int(high), argv)
    if operation in ("float", "double"):
        precision: Literal[32, 64] = 32 if operation == "float" else 64
        return float_argument(
            name, float(default), float(low), float(high), argv, precision=precision
        )
    if operation == "enum":
        return enum_argument(
            name,
            None if default == "NONE" else default,
            {"": 10, "test": 20, "rest": 30},
            argv,
        )
    assert operation == "unknown"
    check_unknown_arguments(
        ["--a", "--b"],
        ["--old"],
        "program",
        None if default == "NONE" else default,
        argv,
    )
    return None


def test_complete_original_scalar_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    count = 0
    with (
        Path(__file__)
        .with_name("data")
        .joinpath("command_arguments.jsonl")
        .open() as stream
    ):
        for line in stream:
            record: object = json.loads(line)
            assert isinstance(record, list) and len(record) == 2
            case, expected = record
            assert isinstance(case, list) and len(case) == 6
            operation, name, default, low, high, argv = case
            assert all(isinstance(value, str) for value in case[:5])
            assert isinstance(argv, list) and all(
                isinstance(word, str) for word in argv
            )
            assert isinstance(expected, str)
            kind, _, payload = expected.partition(" ")
            if kind in ("invalid", "range"):
                exception = ValueError if kind == "invalid" else KeyError
                with pytest.raises(exception) as error:
                    observe(operation, name, default, low, high, argv)
                if kind == "invalid":
                    assert str(error.value) == bytes.fromhex(payload).decode(), case
            else:
                actual = observe(operation, name, default, low, high, argv)
                if kind in ("missing", "ok"):
                    assert actual is None, case
                elif kind == "text":
                    assert actual == bytes.fromhex(payload).decode(), case
                elif kind == "file":
                    identity, _, content = payload.partition(" ")
                    assert actual == (bool(int(identity)), bytes.fromhex(content)), case
                elif kind == "bool":
                    assert actual is bool(int(payload)), case
                elif kind == "int":
                    assert actual == int(payload), case
                else:
                    assert kind == "float" and isinstance(actual, float)
                    original = float.fromhex(payload)
                    if math.isnan(original):
                        assert math.isnan(actual) and math.copysign(
                            1, actual
                        ) == math.copysign(1, original), case
                    else:
                        assert struct.pack("!d", actual) == struct.pack(
                            "!d", original
                        ), case
            count += 1
    assert count == 2756


def test_reference_identity_and_c_string_boundary() -> None:
    value = object()
    choices = {"ok": value}
    assert enum_argument("--value", "ok", choices, ["program"]) is value
    assert enum_argument("--value", None, choices, ["program", "--value=ok"]) is value
    assert find_argument("--value\0ignored", ["program", "--value=ok\0ignored"]) == "ok"
    assert parse_c_float("1.25\0ignored", 64) == 1.25
    with pytest.raises(ValueError):
        parse_int64("")


def test_direct_float32_rounding_preserves_source_witness() -> None:
    text = "1.0000000596046447753906250000000001"
    direct = parse_c_float(text, 32)
    assert direct.hex() == "0x1.0000020000000p+0"
    assert struct.unpack("f", struct.pack("f", float(text)))[0] == 1.0
    assert parse_c_float(text, 64).hex() == "0x1.0000010000000p+0"


def test_file_default_and_failure_ownership(tmp_path: Path) -> None:
    default = BytesIO(b"caller")
    with open_file_argument("--in", default, "rb", ["program"]) as stream:
        assert stream is default and stream.read() == b"caller"
    assert not default.closed
    for argv, message in [
        (["program"], "Missing command line argument"),
        (["program", "--in="], "can't be empty"),
        (["program", "--in=" + str(tmp_path / "missing")], "Failed to open"),
    ]:
        with (
            pytest.raises(ValueError, match=message),
            open_file_argument("--in", None, "rb", argv),
        ):
            pytest.fail("A failing file acquisition yielded a stream")
    path = tmp_path / "source"
    path.write_bytes(b"abc\r\n")
    with open_file_argument(
        "--in", None, "rb", ["program", "--in=" + str(path)]
    ) as opened:
        assert opened.read() == b"abc\r\n"
    assert opened.closed
    with (
        pytest.raises(RuntimeError, match="consumer"),
        open_file_argument(
            "--in", None, "rb", ["program", "--in", str(path)]
        ) as opened,
    ):
        raise RuntimeError("consumer")
    assert opened.closed
    with open_file_argument(
        "--in", None, "re", ["program", "--in", str(path)]
    ) as opened:
        assert opened.read() == b"abc\r\n"
        assert not os.get_inheritable(opened.fileno())
    assert opened.closed


@pytest.mark.parametrize(
    "mode",
    [
        "r",
        "rb",
        "r+",
        "rb+",
        "r+b",
        "w",
        "wb",
        "w+",
        "wb+",
        "a",
        "ab",
        "a+",
        "ab+",
        "wx",
        "w+x",
        "wx+",
    ],
)
def test_file_standard_modes_and_lifetime(mode: str, tmp_path: Path) -> None:
    path = tmp_path / "source"
    if "x" not in mode:
        path.write_bytes(b"abc")
    with open_file_argument(
        "--in", None, mode, ["program", "--in=" + str(path)]
    ) as stream:
        if "r" in mode and "+" not in mode:
            assert stream.read() == b"abc"
        else:
            stream.write(b"Z")
    assert stream.closed
    expected = (
        b"abc"
        if mode in ("r", "rb")
        else (
            b"abcZ"
            if mode.startswith("a")
            else b"Zbc" if mode.startswith("r") else b"Z"
        )
    )
    assert path.read_bytes() == expected


@pytest.mark.parametrize("command", ["kmx_sample", "kmx_spin", "kmx_sampled_counts"])
def test_real_commands_reuse_integer_conversion(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("flagquantum.ecosystem." + command)
    source = tmp_path / "input.kmx"
    source.write_text("X q0\n")
    calls: list[str] = []
    original = parse_int64

    def conversion(text: str) -> int:
        calls.append(text)
        return original(text)

    monkeypatch.setattr(module, "parse_int64", conversion)
    option = "--shots"
    argv = ["--in", str(source), "--out", str(tmp_path / "output"), option, "1__2"]
    if command == "kmx_spin":
        argv = [
            "--in",
            str(source),
            "--out",
            str(tmp_path / "output"),
            "--shots",
            "1__2",
            "--mode",
            "64",
        ]
    assert module.main(argv) == 0
    assert "1__2" in calls


def test_host_mode_consumer_failure_closes_stream(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    with (
        pytest.raises(RuntimeError, match="consumer"),
        open_file_argument(
            "--in", None, "re", ["program", "--in", str(path)]
        ) as opened,
    ):
        assert opened.read(1) == b"a"
        raise RuntimeError("consumer")
    assert opened.closed


def test_host_mode_transfer_failure_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.ecosystem._command_arguments import _open_host_c_file

    path = tmp_path / "source"
    path.write_bytes(b"abc")
    descriptors: list[int] = []

    def fail_transfer(descriptor: int, mode: str) -> None:
        descriptors.append(descriptor)
        raise RuntimeError("transfer")

    monkeypatch.setattr(os, "fdopen", fail_transfer)
    with pytest.raises(RuntimeError, match="transfer"):
        _open_host_c_file(str(path), "re")
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_host_mode_duplicate_failure_closes_c_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    descriptors: list[int] = []

    def fail_duplicate(descriptor: int) -> int:
        descriptors.append(descriptor)
        raise OSError("duplicate")

    monkeypatch.setattr(os, "dup", fail_duplicate)
    with (
        pytest.raises(ValueError, match="Failed to open"),
        open_file_argument("--in", None, "re", ["program", "--in", str(path)]),
    ):
        pytest.fail("Failed descriptor duplication yielded a stream")
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
