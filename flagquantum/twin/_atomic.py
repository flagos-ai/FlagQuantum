"""Create-once writes shared by the Twin artifacts.

Every Twin artifact is saved the same way: created exclusively with owner-only
permissions, idempotent when the same artifact is saved twice, and refusing to
replace different or invalid content. Nine writers previously carried that
skeleton each, and two of them had drifted to ``Path.open("x")``, which takes
the process umask and wrote world-readable files where the others wrote
owner-only. One implementation keeps the nine from drifting apart again.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def write_once(
    destination: Path,
    encoded: str,
    *,
    label: str,
    matches: Callable[[Path], bool],
) -> None:
    """Create ``destination`` once and never replace another artifact.

    Saving the same artifact twice is a no-op. An existing file holding an
    invalid artifact, or a different one, is left exactly as it was.

    Args:
        destination: File to create. The file is created with mode ``0o600``
            and is never opened for replacement.
        encoded: Complete serialized artifact, including its trailing newline.
        label: Artifact name used in the failure messages, such as
            ``"Twin evidence"``.
        matches: Reports whether the artifact already at the destination is the
            same one being written. Raises ``ValueError`` when that file is not
            a valid artifact. Called only when the destination already exists,
            so nothing it does is paid for by a first save.

    Raises:
        ValueError: If the destination already holds an invalid or different
            artifact, or if it cannot be written.
    """

    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError as exists_error:
        try:
            same_artifact = matches(destination)
        except ValueError as error:
            raise ValueError(
                f"Refusing to replace invalid {label} at {destination}"
            ) from error
        if not same_artifact:
            raise ValueError(
                f"Refusing to replace different {label} at {destination}"
            ) from exists_error
    except OSError as error:
        raise ValueError(f"Cannot write {label} to {destination}") from error
