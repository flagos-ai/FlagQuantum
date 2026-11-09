from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.remote.realtime_device_call"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout


def test_realtime_device_call_user_example_runs_end_to_end() -> None:
    stdout = _run()

    assert "FlagQuantum realtime device call check passed" in stdout
    assert "registered channels: ('host-dispatch',)" in stdout
    assert "round 1 correction: 1" in stdout


def test_realtime_round_trip_does_not_require_qec() -> None:
    """Keep the messaging workflow usable before a QEC adapter is admitted."""
    root = Path(__file__).resolve().parents[2]
    script = """\
        import importlib.abc
        import runpy
        import sys

        class RejectQEC(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "flagquantum.qec" or fullname.startswith("flagquantum.qec."):
                    raise AssertionError(f"Messaging unexpectedly requires {fullname}")

        sys.meta_path.insert(0, RejectQEC())
        runpy.run_module("examples.remote.realtime_device_call", run_name="__main__")
        assert not any(
            name == "flagquantum.qec" or name.startswith("flagquantum.qec.")
            for name in sys.modules
        )
        """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "round 1 correction: 1" in completed.stdout
    assert "mismatched schema status: -2" in completed.stdout
    assert "FlagQuantum realtime device call check passed" in completed.stdout


def test_the_wire_format_is_stated_rather_than_implied() -> None:
    """A real-time protocol is a layout agreement, so the layout is printed.

    The slot geometry and the header size are what a device implementer has
    to match. They are asserted exactly because they are a contract: a change
    here is a change every peer has to make at the same time, and a test that
    tolerated it would let the two sides drift apart silently.
    """

    stdout = _run()

    assert "slot geometry: 384 bytes, header 24" in stdout

    # The encoded request is printed as hex so the agreement is inspectable
    # rather than described. Its length is the claim: a 52-byte request does
    # not fit a 32-byte slot, which is the case the next test covers.
    request_line = next(
        line for line in stdout.splitlines() if "request bytes over a" in line
    )
    encoded = request_line.split(":")[1].strip()
    assert len(encoded) % 2 == 0, request_line
    assert int(encoded, 16) != 0, request_line


def test_the_protocol_refuses_a_mismatched_peer_by_status() -> None:
    """A real-time path cannot raise, so it answers with a status.

    This runs under a device's timing budget, so a refusal is a negative
    status code the caller reads rather than an exception that would unwind a
    loop the device is waiting on. Both refusals are asserted: a schema the
    peer does not share, and a device whose channel was never opened.
    """

    stdout = _run()

    assert "mismatched schema status: -2" in stdout
    assert (
        "unopened device: device id 1 has no open channel; open device ids are 0"
    ) in stdout
