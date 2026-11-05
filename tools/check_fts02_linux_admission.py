"""Check the Linux execution boundary before a separately approved CPU run."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import threading
from importlib.metadata import version
from pathlib import Path

from tools import fts02_native_boundary as boundary
from tools.fts02_process_budget import (
    Reservation,
    isolation_identity,
    runtime_identity,
)
from tools.fts02_process_budget import (
    check_mounts as check_mounts,
)
from tools.launch_fts02_cpu import Evidence

CAPACITY = 32 * 1024 * 1024


def _check_native_runtime() -> None:
    """Require optional SDKs without installing, skipping or falling back."""
    try:
        import pymatching
        import stim
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "native admission requires Stim and PyMatching runtime dependencies"
        ) from error
    import torch

    torch.set_num_threads(1)
    if torch.ones(1).item() != 1 or not stim.Circuit("M 0").num_measurements:
        raise RuntimeError("native runtime smoke failed")
    pymatching.Matching()


def native_probe(evidence: Path, reservation: Reservation) -> None:
    if not hasattr(os, "posix_fallocate"):
        raise RuntimeError("native admission requires posix_fallocate")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    lib = ctypes.CDLL("libseccomp.so.2")
    for name, expected in (
        (b"clone", errno.EPERM),
        (b"clone3", errno.ENOSYS),
        (b"setsid", errno.EPERM),
        (b"setpgid", errno.EPERM),
    ):
        number = lib.seccomp_syscall_resolve_name(name)
        ctypes.set_errno(0)
        flags = signal.SIGCHLD if name == b"clone" else 0
        if name in (b"clone", b"clone3"):
            reservation.charge("admission", "native_denial")
        result = libc.syscall(ctypes.c_long(number), flags, 0, 0, 0, 0)
        if result == 0 and name in (b"clone", b"clone3"):
            os._exit(96)
        if result > 0 and name in (b"clone", b"clone3"):
            os.waitpid(result, 0)
        if result != -1 or ctypes.get_errno() != expected:
            raise RuntimeError(f"unexpected syscall result for {name!r}")
    if hasattr(libc, "fork"):
        ctypes.set_errno(0)
        reservation.charge("admission", "native_denial")
        result = libc.fork()
        if result == 0:
            os._exit(96)
        if result > 0:
            os.waitpid(result, 0)
        if result != -1 or ctypes.get_errno() != errno.EPERM:
            raise RuntimeError("native fork was not denied")
    status = Path("/proc/self/status").read_text()
    if "Seccomp:\t2\n" not in status or "NoNewPrivs:\t1\n" not in status:
        raise RuntimeError("expected a loaded seccomp filter and no-new-privileges")

    completed: list[bool] = []
    thread = threading.Thread(target=lambda: completed.append(True))
    thread.start()
    thread.join(timeout=5)
    if completed != [True] or thread.is_alive():
        raise RuntimeError("thread creation failed")

    _check_native_runtime()
    # Verify an actual capacity refusal in a disposable file. It is removed
    # on either outcome. The requested allocation exceeds the whole mount.
    trial = evidence / "fts02-admission-capacity-probe"
    with trial.open("xb") as stream:
        try:
            try:
                os.posix_fallocate(stream.fileno(), 0, CAPACITY + 4096)
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
            else:
                raise RuntimeError("evidence allocation exceeded approved capacity")
        finally:
            trial.unlink()
    print(
        json.dumps(
            {
                "status": "native_admission_pass",
                "python": sys.version.split()[0],
                "torch": version("torch"),
                "stim": version("stim"),
                "pymatching": version("pymatching"),
            }
        )
    )


def _partial_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    raise TypeError("unsupported timeout output type")


def launch_child(
    reservation: Reservation,
    output: Evidence,
    role: str,
    command: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Charge once; retain bounded partial output and preserve timeout cleanup."""
    reservation.charge("admission", role)
    try:
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            preexec_fn=boundary.install,
        )
    except subprocess.TimeoutExpired as error:
        try:
            for channel, data in (("stdout", error.stdout), ("stderr", error.stderr)):
                output.write(
                    reservation.directory / f"admission-{role}-timeout.{channel}",
                    _partial_bytes(data),
                    diagnostic=True,
                )
        except (OSError, RuntimeError, ValueError, TypeError) as write_error:
            raise write_error from error
        raise
    output.write(
        reservation.directory / f"admission-{role}.log",
        (result.stdout + result.stderr).encode(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--native-child", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or os.geteuid() == 0:
        parser.error("requires a non-root Linux admission namespace")
    evidence = args.evidence.resolve()
    if any(c.isspace() or c == "\\" for c in str(evidence)):
        parser.error("evidence mount path must be unescaped in mountinfo")
    reservation = Reservation(
        args.reservation.resolve(), args.revision, runtime_identity()
    )
    if reservation.directory.parent != evidence:
        parser.error("reservation must be on the evidence mount")
    if not args.native_child:
        reservation.claim("admission")
    identity = isolation_identity(evidence)
    if args.native_child:
        native_probe(evidence, reservation)
        return

    output = Evidence(reservation)

    native = launch_child(
        reservation,
        output,
        "native",
        [
            sys.executable,
            "-B",
            "-m",
            "tools.check_fts02_linux_admission",
            "--native-child",
            "--reservation",
            str(reservation.directory),
            "--evidence",
            str(evidence),
            "--revision",
            args.revision,
        ],
    )
    native.check_returncode()
    git = [
        "/usr/bin/git",
        "-c",
        "core.fsmonitor=false",
    ]
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "LC_ALL": "C",
    }
    for command, expected in (
        (["rev-parse", "HEAD"], args.revision),
        (["status", "--porcelain", "--untracked-files=no"], ""),
    ):
        result = launch_child(reservation, output, "git", [*git, *command], env=env)
        result.check_returncode()
        if result.stdout.strip() != expected:
            raise RuntimeError("unexpected Git provenance")
    # Reserve the alias's descendant request before permitting the negative check.
    reservation.charge("admission", "unexpected")
    denied = launch_child(
        reservation,
        output,
        "negative",
        [*git, "-c", "alias.fts02-test=!exit 97", "fts02-test"],
        env=env,
    )
    if denied.returncode in (0, 97) or "Operation not permitted" not in denied.stderr:
        raise RuntimeError("Git descendant creation did not fail with EPERM")
    if isolation_identity(evidence) != identity:
        raise RuntimeError("isolation changed during admission")
    reservation.stamp_isolation(identity)
    reservation.update(flag="admission_passed")
    print(native.stdout, end="")
    print("git_provenance=pass git_descendant=EPERM evidence_mount=pass")


if __name__ == "__main__":
    main()
