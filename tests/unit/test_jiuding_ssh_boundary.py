"""Managed artifact transfers preserve binary subprocess contracts."""

import subprocess
from unittest.mock import Mock

import pytest

from flagquantum.remote.compute._managed_program import _run_ssh

pytestmark = pytest.mark.unit


def test_managed_ssh_preserves_binary_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client._ssh_base.return_value = ["ssh", "test-host"]
    response = subprocess.CompletedProcess(["ssh"], 0, b"\xffoutput", b"")
    run = Mock(return_value=response)
    monkeypatch.setattr(subprocess, "run", run)

    result = _run_ssh(client, ["cat", "a b"], input=b"\x00\xff", timeout=60)

    assert result is response
    run.assert_called_once_with(
        ["ssh", "test-host", "cat 'a b'"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
        input=b"\x00\xff",
    )


def test_managed_ssh_decodes_failed_binary_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client._ssh_base.return_value = ["ssh", "test-host"]
    response = subprocess.CompletedProcess(["ssh"], 1, b"", b"failure\xff")
    monkeypatch.setattr(subprocess, "run", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="artifact transfer failed: failure"):
        _run_ssh(client, ["cat", "missing"])
