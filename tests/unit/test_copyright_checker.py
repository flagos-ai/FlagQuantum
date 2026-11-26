"""Complete source checker, original suite and native plugin entry comparisons."""

import csv
import hashlib
import io
import sys
import types
from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    package = types.ModuleType("_development_reference")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load_source("pylint_copyright_checker")
    monkeypatch.setitem(
        sys.modules, package.__name__ + ".pylint_copyright_checker", module
    )
    return module


@pytest.fixture
def notice() -> bytes:
    path = REFERENCE / "dev_tools/qualtran_dev_tools/pylint_copyright_checker.py"
    return b"\n".join(path.read_bytes().splitlines()[:13]) + b"\n"


def messages(owner: Any, data: bytes, *, enabled: bool = True) -> list[Any]:
    from astroid import parse
    from pylint.testutils import UnittestLinter

    linter = UnittestLinter()
    checker = owner.CopyrightChecker(linter)
    node = parse("")
    with (
        patch.object(linter, "is_message_enabled", return_value=enabled),
        patch.object(node, "stream", return_value=io.BytesIO(data)),
    ):
        checker.process_module(node)
    return list(linter.release_messages())


def test_frozen_source_and_exact_optional_versions() -> None:
    remaining = {"Q-0084", "Q-0085"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in remaining:
                continue
            data = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                == row["git_blob"]
            )
            remaining.remove(row["id"])
    assert not remaining
    assert metadata.version("pylint") == "3.3.9"
    assert metadata.version("astroid") == "3.3.11"
    locked = REFERENCE / "dev_tools/requirements/envs/dev.env.txt"
    data = locked.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == "f6a3c34f107384d381a8ea3f2a91a02ed1ae98b9"
    )


def test_unchanged_original_suite_for_both_providers(source: Any) -> None:
    from tools.development import copyright_checker as native

    suite = load_source("pylint_copyright_checker_test").CopyrightCheckerTestSuite
    for owner in [source, native]:
        case_type = type(
            "ProviderCase", (suite,), {"CHECKER_CLASS": owner.CopyrightChecker}
        )
        for name in ["missing", "wrong", "shorter", "longer"]:
            case = case_type()
            case.setup_method()
            getattr(case, "test_" + name + "_copyright")()
        for prefix in ["", "#!/usr/bin/env python\n", "#!/usr/bin/env/python\n\n"]:
            case = case_type()
            case.setup_method()
            case.test_correct_copyright(prefix)


def test_every_header_character_and_whitespace(source: Any, notice: bytes) -> None:
    from tools.development import copyright_checker as native

    for prefix in [b"", b"#!/usr/bin/python\n", b"#!/bin/python\n\n \t\n"]:
        for suffix in [b"", b"import os\n", b"\n# Additional retained attribution.\n"]:
            for header in [notice, notice.replace(b"\n", b" \t\r\n")]:
                data = prefix + header + suffix
                assert messages(source, data) == messages(native, data) == []
    for index, char in enumerate(notice):
        if char in b"\r\n":
            continue
        data = notice[:index] + (b"!" if char != 33 else b"?") + notice[index + 1 :]
        assert messages(native, data) == messages(source, data), index
    for year in [b"2000", b"2099", b"20ab", b"1999", b"2100"]:
        data = notice.replace(b"2023", year)
        assert messages(native, data) == messages(source, data)
    for line in range(13):
        parts = notice.splitlines(keepends=True)
        parts[line] = parts[line].rstrip(b"\n") + b" extra\n"
        assert messages(native, b"".join(parts)) == messages(source, b"".join(parts))
    assert (
        messages(native, b"invalid", enabled=False)
        == messages(source, b"invalid", enabled=False)
        == []
    )


def test_truncated_notice_source_bug_and_native_rejection(
    source: Any, notice: bytes
) -> None:
    from tools.development import copyright_checker as native

    parts = notice.splitlines(keepends=True)
    for length in range(13):
        data = b"".join(parts[:length])
        assert messages(source, data) == []
        actual = messages(native, data)
        assert len(actual) == 1
        assert actual[0].msg_id == "wrong-or-nonexistent-copyright-notice"
        assert actual[0].line == length + 1
        assert actual[0].col_offset == 0
    for data in [b"#!/usr/bin/python\n", b"#!/bin/python\n\n"]:
        assert messages(source, data) == []
        assert len(messages(native, data)) == 1


def test_register_and_real_pylint_plugin(
    source: Any, notice: bytes, tmp_path: Path
) -> None:
    from pylint.lint import PyLinter, Run
    from pylint.reporters.text import TextReporter

    from tools.development import copyright_checker as native

    for owner in [source, native]:
        linter = PyLinter()
        owner.register(linter)
        assert any(
            isinstance(checker, owner.CopyrightChecker)
            for checker in linter.get_checkers()
        )
    for name, data, expected in [
        ("valid", notice + b"value = 1\n", 0),
        ("invalid", b"value = 1\n", 8),
        ("truncated", notice.splitlines(keepends=True)[0], 8),
        (
            "disabled",
            b"# pylint: disable=wrong-or-nonexistent-copyright-notice\nvalue = 1\n",
            0,
        ),
    ]:
        path = tmp_path / (name + ".py")
        path.write_bytes(data)
        output = io.StringIO()
        result = Run(
            [
                "--rcfile=/dev/null",
                "--persistent=n",
                "--load-plugins=tools.development.copyright_checker",
                "--disable=all",
                "--enable=wrong-or-nonexistent-copyright-notice",
                "--score=n",
                "--reports=n",
                str(path),
            ],
            reporter=TextReporter(output),
            exit=False,
        )
        assert result.linter.msg_status == expected, output.getvalue()
        assert ("wrong-or-nonexistent-copyright-notice" in output.getvalue()) == bool(
            expected
        )
