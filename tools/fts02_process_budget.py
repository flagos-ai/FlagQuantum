"""Creation accounting for the six FTS-02 CPU points."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

POINTS = ("w2-x", "w2-z", "w3", "w6-16", "w6-32", "w6-64")


def point_limits(point: str) -> dict[str, int]:
    if point not in POINTS:
        raise ValueError(f"unknown FTS-02 point: {point}")
    provenance = 0 if point.startswith("w2-") else 2
    return {
        "watchdog": 1,
        "target": 1,
        "admission": 1,
        "final": 1,
        "regular": (30_002 if point == "w3" else 12_002) - provenance,
        "cleanup": 6,
        "diagnostic": 4,
        "provenance": provenance,
    }


class CreationDeniedError(RuntimeError):
    """The requested process role has no remaining capacity."""


@dataclass
class CreationBudget:
    """Count attempts before creation, including attempts that raise."""

    point: str
    _attempts: dict[str, int] = field(default_factory=dict, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _limits: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self._limits = (
            {"coordinator": 1} if self.point == "campaign" else point_limits(self.point)
        )

    def charge(self, role: str) -> None:
        with self._lock:
            count = self._attempts.get(role, 0)
            if count >= self._limits.get(role, 0):
                raise CreationDeniedError(
                    f"{self.point}/{role}: creation limit reached"
                )
            self._attempts[role] = count + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._attempts)


def is_git_provenance(argv: object) -> bool:
    """Identify the two exact provenance requests made by the producers."""
    return isinstance(argv, (tuple, list)) and tuple(argv) in (
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain", "--untracked-files=no"),
    )


class TargetCreationGuard:
    """Audit Python process requests in a target before dispatch.

    Install before producer imports. Native creation and Git descendants
    require a separate containment boundary before live admission.
    """

    def __init__(self, point: str, root: str) -> None:
        self.budget = CreationBudget(point)
        self.root = root

    def __call__(self, event: str, args: tuple[object, ...]) -> None:
        if event == "subprocess.Popen":
            executable, argv, cwd, _environment = args
            if executable == "git" and cwd == self.root and is_git_provenance(argv):
                self.budget.charge("provenance")
                return
            raise CreationDeniedError(f"{self.budget.point}: unapproved target command")
        if event in {
            "os.exec",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.spawn",
            "os.system",
            "pty.spawn",
            "_posixsubprocess.fork_exec",
        }:
            raise CreationDeniedError(
                f"{self.budget.point}: unapproved creation via {event}"
            )


def run_points(
    launch: Callable[[str, CreationBudget], int],
) -> tuple[int, dict[str, dict[str, int]]]:
    """Dispatch the fixed points through an injected watchdog launcher."""
    attempts: dict[str, dict[str, int]] = {}
    for point in POINTS:
        budget = CreationBudget(point)
        budget.charge("watchdog")
        try:
            result = launch(point, budget)
        finally:
            attempts[point] = budget.snapshot()
        if result:
            return result, attempts
    return 0, attempts


# The six in-process watchdog slots fund admission, not additional workers.
CAPACITY = 90_097
EVIDENCE_CAPACITY = 32 * 1024 * 1024
CONTROL_RESERVE = 64 * 1024
ISOLATION_RESERVE = 4 * 1024


def reservation_limits() -> dict[str, int]:
    """Allocate every slot once, including external coordinator precharges."""
    limits = {
        "campaign/coordinator": 1,
        "admission/coordinator": 1,
        "admission/native": 1,
        "admission/git": 2,
        "admission/negative": 1,
        "admission/unexpected": 1,
        "admission/native_denial": 3,
    }
    for point in POINTS:
        limits.update(
            {
                f"{point}/{role}": cap
                for role, cap in point_limits(point).items()
                if role != "watchdog"
            }
        )
    limits["w3/regular"] -= 3
    assert sum(limits.values()) == CAPACITY
    return limits


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("invalid environment lock")
    return dict(value)


def runtime_identity() -> dict[str, str]:
    """Bind the executable and complete installed distribution versions."""
    import hashlib
    import sys
    from importlib.metadata import distributions
    from pathlib import Path

    identity = {
        "python": sys.version,
        "executable_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
    }
    for distribution in distributions():
        name = distribution.metadata["Name"]
        if not name:
            raise ValueError("distribution without a name")
        key = "package/" + name.lower().replace("_", "-")
        if key in identity:
            raise ValueError("duplicate installed distribution")
        identity[key] = distribution.version
    for name in ("torch", "stim", "pymatching"):
        if "package/" + name not in identity:
            raise ValueError(f"required runtime is missing: {name}")
    return identity


@dataclass
class Reservation:
    """Persist counts before OS calls; stale locks and interrupted claims refuse reuse.

    This ledger is for trusted, sequential local tools, not a security boundary
    against a malicious same-user process. The enclosing read-only namespace and
    capped evidence filesystem remain mandatory native admission prerequisites.
    """

    directory: Path
    revision: str
    environment: dict[str, str]

    @staticmethod
    def create(
        directory: Path, revision: str, environment: dict[str, str]
    ) -> Reservation:
        """Called by the existing operator BEFORE starting admission's coordinator."""
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("a full source revision is required")
        _string_map(environment)
        if not environment:
            raise ValueError("an approved dependency lock is required")
        directory.mkdir()
        reservation = Reservation(directory, revision, dict(environment))
        reservation._save({"admission/coordinator": 1}, {"prepared"}, 0)
        return reservation

    def _save(self, attempts: dict[str, int], flags: set[str], evidence: int) -> None:
        import json
        import os

        payload = json.dumps(
            {
                "revision": self.revision,
                "environment": self.environment,
                "attempts": attempts,
                "flags": sorted(flags),
                "evidence": evidence,
                "capacity": CAPACITY,
            },
            sort_keys=True,
        ).encode()
        if len(payload) * 2 > CONTROL_RESERVE - ISOLATION_RESERVE:
            raise ValueError("reservation exceeds control-file reserve")
        temporary = self.directory / "next.json"
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.directory / "reservation.json")
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _read(self) -> tuple[dict[str, int], set[str], int]:
        import json

        path = self.directory / "reservation.json"
        if path.is_symlink() or path.stat().st_size > CONTROL_RESERVE:
            raise ValueError("invalid reservation file")
        value = json.loads(path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("revision") != self.revision
            or _string_map(value.get("environment")) != self.environment
            or value.get("capacity") != CAPACITY
        ):
            raise ValueError("reservation identity mismatch")
        attempts = value.get("attempts")
        flags = value.get("flags")
        evidence = value.get("evidence")
        limits = reservation_limits()
        if (
            not isinstance(attempts, dict)
            or not all(
                isinstance(k, str) and type(v) is int and 0 <= v <= limits.get(k, -1)
                for k, v in attempts.items()
            )
            or not isinstance(flags, list)
            or not all(isinstance(f, str) for f in flags)
            or type(evidence) is not int
            or not 0 <= evidence <= EVIDENCE_CAPACITY
        ):
            raise ValueError("invalid reservation counters")
        return dict(attempts), set(flags), evidence

    def update(
        self,
        *,
        role: str | None = None,
        flag: str | None = None,
        require: str | None = None,
        bytes_count: int = 0,
        byte_limit: int = EVIDENCE_CAPACITY - 1024 * 1024,
    ) -> None:
        """Commit an attempt, transition or byte charge before its operation."""
        import os

        lock = self.directory / "active.lock"
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            attempts, flags, evidence = self._read()
            if "closed" in flags or (require is not None and require not in flags):
                raise CreationDeniedError("reservation phase is unavailable")
            if flag is not None:
                if flag in flags:
                    raise CreationDeniedError("reservation phase was already consumed")
                flags.add(flag)
            if role is not None:
                count = attempts.get(role, 0)
                if count >= reservation_limits().get(role, 0):
                    raise CreationDeniedError(f"creation limit reached: {role}")
                attempts[role] = count + 1
            if bytes_count < 0 or (
                bytes_count > 0
                and evidence + bytes_count > min(byte_limit, EVIDENCE_CAPACITY)
            ):
                raise CreationDeniedError("evidence capacity exhausted")
            self._save(attempts, flags, evidence + bytes_count)
        finally:
            os.close(descriptor)
            lock.unlink()

    def claim(self, phase: str) -> None:
        """Consume a precharged coordinator once, before running its body."""
        if phase not in ("admission", "campaign"):
            raise ValueError("invalid phase")
        attempts, _, _ = self._read()
        if attempts.get(f"{phase}/coordinator") != 1:
            raise CreationDeniedError("coordinator was not charged before creation")
        self.update(
            flag=f"entered/{phase}",
            require="prepared" if phase == "admission" else "admission_passed",
        )

    def precharge_campaign(self) -> None:
        """The existing operator calls this BEFORE its one campaign launch."""
        self.update(
            role="campaign/coordinator",
            flag="campaign_reserved",
            require="admission_passed",
        )

    def charge(self, point: str, role: str) -> None:
        phase = "admission" if point == "admission" else "campaign"
        self.update(role=f"{point}/{role}", require=f"entered/{phase}")

    def stamp_isolation(self, identity: dict[str, str]) -> None:
        """Persist admitted namespace/mount identity before enabling a campaign."""
        import json
        import os

        _string_map(identity)
        if not identity:
            raise ValueError("an isolation identity is required")
        payload = json.dumps(identity, sort_keys=True).encode()
        if len(payload) > ISOLATION_RESERVE:
            raise ValueError("isolation identity exceeds control reserve")
        with (self.directory / "isolation.json").open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def verify_isolation(self, identity: dict[str, str]) -> None:
        """Refuse missing/changed admission identity before any campaign child."""
        import json

        path = self.directory / "isolation.json"
        if path.is_symlink() or path.stat().st_size > ISOLATION_RESERVE:
            raise ValueError("invalid admitted isolation identity")
        if not identity or _string_map(json.loads(path.read_text())) != identity:
            raise ValueError("admitted isolation identity changed")

    def summary(self) -> dict[str, object]:
        attempts, flags, evidence = self._read()
        used = sum(attempts.values())
        return {
            "capacity_locked": CAPACITY,
            "creation_attempts": used,
            "successful_os_creations": None,
            "successful_os_creations_note": "attempts include failures; not inferred from charges",
            "unused_capacity": CAPACITY - used,
            "attempts": attempts,
            "flags": sorted(flags),
            "charged_evidence_bytes": evidence,
            "reusable_capacity": 0,
        }


def check_mounts(mountinfo: str, evidence: Path, capacity: int) -> None:
    """Require a dedicated capped tmpfs and read-only remaining mounts."""
    found = False
    for line in mountinfo.splitlines():
        mount, filesystem = line.split(" - ", 1)
        fields = mount.split()
        mountpoint = fields[4]
        if mountpoint == str(evidence):
            if (
                found
                or fields[3] != "/"
                or filesystem.split()[0] != "tmpfs"
                or "rw" not in fields[5].split(",")
            ):
                raise RuntimeError("evidence must be a single writable tmpfs mount")
            found = True
        elif "ro" not in fields[5].split(","):
            raise RuntimeError(f"unexpected writable mount: {mountpoint}")
    if not found or not 0 < capacity <= EVIDENCE_CAPACITY:
        raise RuntimeError("evidence tmpfs exceeds its approved capacity")


def isolation_identity(evidence: Path) -> dict[str, str]:
    """Read and validate current isolation without creating a process or mount."""
    import hashlib
    import os
    import sys

    if sys.platform != "linux" or os.geteuid() == 0:
        raise RuntimeError("requires a non-root Linux admission namespace")
    mountinfo = Path("/proc/self/mountinfo").read_text()
    filesystem = os.statvfs(evidence)
    capacity = filesystem.f_frsize * filesystem.f_blocks
    check_mounts(mountinfo, evidence, capacity)
    return {
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "network_namespace": os.readlink("/proc/self/ns/net"),
        "mountinfo_sha256": hashlib.sha256(mountinfo.encode()).hexdigest(),
        "evidence_path": str(evidence.resolve()),
        "evidence_device": str(evidence.stat().st_dev),
        "capacity_bytes": str(capacity),
        "uid": str(os.geteuid()),
    }
