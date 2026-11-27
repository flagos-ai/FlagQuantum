"""Write an auditable snapshot of the currently installed environment."""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
from importlib.metadata import distributions
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version


def audit_requirement(name: str, version: str) -> str:
    """Return a pinned requirement without a repository-specific local suffix."""

    return f"{canonicalize_name(name)}=={Version(version).public}"


def installed_audit_requirements() -> tuple[str, ...]:
    """Return deterministic pins for installed third-party distributions."""

    requirements = {
        audit_requirement(name, distribution.version)
        for distribution in distributions()
        if (name := distribution.metadata.get("Name"))
        and canonicalize_name(name) != "flagquantum"
    }
    return tuple(sorted(requirements))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    requirements = installed_audit_requirements()
    args.output.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    print(f"wrote {len(requirements)} installed dependency pins to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
