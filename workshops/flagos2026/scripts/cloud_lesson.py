"""Locate personal teaching files; never submit or execute quantum programs."""

from __future__ import annotations

import os
from pathlib import Path


def workshop_directory(start: Path) -> Path:
    """Find the workshop and keep platform users out of the common template."""
    start = start.resolve()
    workshop = next(
        (
            root / "flagos2026"
            for root in (start, *start.parents)
            if (root / "flagos2026/notebooks").is_dir()
        ),
        None,
    )
    if workshop is None:
        raise RuntimeError(
            "Cannot locate flagos2026/notebooks. Open the deployed workshop first."
        )
    if Path("/etc/accesskey").exists() or os.getenv("JIUDING_AK"):
        from jiuding_paths import personal_directory, prepare_workshop

        from flagquantum.remote.compute.jiuding import JiudingClient

        personal = prepare_workshop(workshop, personal_directory(JiudingClient()))
        if not start.is_relative_to(personal):
            raise RuntimeError(
                f"Personal copy is ready. Open {personal / 'notebooks'}; "
                "the common template is unchanged."
            )
        workshop = personal
    return workshop
