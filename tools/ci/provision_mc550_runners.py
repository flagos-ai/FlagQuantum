"""Provision the authorized sixteen-slot MC550 CPU pool on one Ubuntu host.

Run as root with a directory containing downloads.json. Registration is separate;
the installer does not accept or persist a GitHub administrator credential.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

ROOT = Path("/liuwei01/flagquantum-runners")


def run(*args: str, timeout: int = 120, **kwargs: object) -> None:
    subprocess.run(args, check=True, timeout=timeout, **kwargs)


def download(row: dict[str, str]) -> Path:
    target = ROOT / "packages" / row["filename"]
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        run(
            "curl",
            "--fail",
            "--location",
            "--retry",
            "1",
            "--connect-timeout",
            "15",
            "--max-time",
            "1200",
            "--output",
            str(partial),
            row["url"],
            timeout=1230,
        )
        partial.rename(target)
    with target.open("rb") as stream:
        digest = (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )
    if digest != row["sha256"]:
        raise RuntimeError(f"Official SHA256 mismatch: {target.name}")
    os.chown(target, 0, 0)
    target.chmod(0o644)
    print("Verified", target.name, flush=True)
    return target


def cpus_for_slot(slot: int) -> list[int]:
    """Reserve eight physical cores per socket; keep each slot within one NUMA node."""
    if not 1 <= slot <= 16:
        raise ValueError("Expected slot 1 through 16")
    first = (8 if slot <= 8 else 48) + ((slot - 1) % 8) * 4
    cores = list(range(first, first + 4))
    return cores + [core + 80 for core in cores]


def unit(slot: int) -> str:
    name = f"cpu{slot:02d}"
    user = f"fqmc{slot:02d}"
    home = ROOT / "homes" / name
    app = ROOT / "runners" / name
    cpus = " ".join(map(str, cpus_for_slot(slot)))
    return f"""[Unit]
Description=FlagQuantum MC550 CPU runner {name}
Wants=network-online.target
After=network-online.target liuwei01.mount
Requires=liuwei01.mount
StartLimitIntervalSec=0
[Service]
Type=simple
User={user}
Group={user}
Slice=fqmc.slice
WorkingDirectory={app}
ExecStart={app}/bin/Runner.Listener run --startuptype service
Restart=always
RestartSec=10
KillMode=control-group
KillSignal=SIGINT
TimeoutStopSec=90
UMask=0077
CPUQuota=400%
AllowedCPUs={cpus}
CPUWeight=1
Nice=10
MemoryHigh=12G
MemoryMax=16G
MemorySwapMax=0
TasksMax=1024
IOWeight=1
IOSchedulingClass=idle
LimitNOFILE=65536
LimitCORE=0
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
PrivateTmp=yes
PrivateDevices=yes
ProtectSystem=strict
ProtectHome=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
RestrictSUIDSGID=yes
RestrictRealtime=yes
RestrictNamespaces=yes
LockPersonality=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK
ReadWritePaths={home} {app}/_work {app}/_diag
InaccessiblePaths=-/data -/public-nfs -/public-flash -/workspace -/mnt -/media
InaccessiblePaths=-/run/docker.sock -/run/containerd -/run/dbus -/run/systemd/private
Environment=HOME={home}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=LANG=C.UTF-8
Environment=TMPDIR={home}/tmp
Environment=XDG_CACHE_HOME={home}/.cache
Environment=RUNNER_TOOL_CACHE={home}/toolcache
Environment=AGENT_TOOLSDIRECTORY={home}/toolcache
Environment=DOTNET_CLI_HOME={home}
Environment=DOTNET_EnableDiagnostics=0
Environment=PYTHONNOUSERSITE=1
Environment=PIP_CONFIG_FILE=/dev/null
Environment=PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
Environment=PIP_REQUIRE_VIRTUALENV=true
Environment=FQ_CPU_WHEEL_CACHE={ROOT}/tools/public-wheels
Environment=PIP_FIND_LINKS={ROOT}/tools/public-wheels
Environment=FQ_MC550_SLOT={name}
Environment=OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
Environment=CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
[Install]
WantedBy=multi-user.target
"""


def main() -> None:
    metadata = json.loads(Path(sys.argv[1]).read_text())
    topology = subprocess.check_output(
        ["lscpu", "-p=CPU,CORE,SOCKET"], text=True, timeout=10
    )
    rows = [
        tuple(map(int, row.split(",")))
        for row in topology.splitlines()
        if not row.startswith("#")
    ]
    mapping = {cpu: (core, socket) for cpu, core, socket in rows}
    if len(mapping) != 160 or any(mapping[c] != mapping[c + 80] for c in range(80)):
        raise RuntimeError("CPU topology differs from the inspected MC550 hosts")
    for subdir in [
        "packages",
        "tools",
        "tools/public-wheels",
        "admin",
        "runners",
        "homes",
    ]:
        (ROOT / subdir).mkdir(parents=True, exist_ok=True)
    (ROOT / "admin").chmod(0o700)
    cache = ROOT / "tools/python-toolcache"
    env = dict(
        os.environ,
        AGENT_TOOLSDIRECTORY=str(cache),
        PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple",
        PIP_CONFIG_FILE="/dev/null",
        PIP_REQUIRE_VIRTUALENV="false",
    )
    for row in metadata["python"]:
        archive = download(row)
        if not (cache / "Python" / row["version"] / "x64.complete").exists():
            stage = ROOT / "admin" / row["version"]
            stage.mkdir(exist_ok=True)
            run("tar", "xzf", str(archive), "--no-same-owner", "-C", str(stage))
            runtime_env = dict(
                env,
                LD_LIBRARY_PATH=str(cache / "Python" / row["version"] / "x64/lib"),
            )
            run("bash", "setup.sh", cwd=stage, env=runtime_env, timeout=300)
        print("Python ready", row["version"], flush=True)
    run("chmod", "-R", "go-w", str(cache))
    archive = download(metadata["runner"])
    for slot in range(1, 17):
        name, user = f"cpu{slot:02d}", f"fqmc{slot:02d}"
        home, app = ROOT / "homes" / name, ROOT / "runners" / name
        try:
            account = pwd.getpwnam(user)
        except KeyError:
            run(
                "useradd",
                "--no-create-home",
                "--home-dir",
                str(home),
                "--shell",
                "/usr/sbin/nologin",
                user,
            )
            account = pwd.getpwnam(user)
        if account.pw_dir != str(home) or account.pw_uid == 0:
            raise RuntimeError(f"Unexpected existing account: {user}")
        for directory in [
            home,
            home / "tmp",
            home / ".cache",
            home / "toolcache",
            home / "toolcache/Python",
            app / "_work",
            app / "_diag",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            os.chown(directory, account.pw_uid, account.pw_gid)
            directory.chmod(0o700)
        if not (app / "bin/Runner.Listener").exists():
            run("tar", "xzf", str(archive), "--no-same-owner", "-C", str(app))
        for row in metadata["python"]:
            target = home / "toolcache/Python" / row["version"]
            if not target.exists():
                target.symlink_to(cache / "Python" / row["version"])
        run(str(app / "bin/Runner.Listener"), "--version", timeout=20)
        Path(f"/etc/systemd/system/fqmc-{name}.service").write_text(unit(slot))
    run("systemctl", "daemon-reload")
    print(
        "Prepared sixteen isolated services; registration and startup are separate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
