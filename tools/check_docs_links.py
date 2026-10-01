#!/usr/bin/env python3
"""Verify that relative documentation links and heading anchors resolve."""

from __future__ import annotations

import argparse
import re
import subprocess
import urllib.parse
from collections.abc import Iterator, Sequence
from pathlib import Path

INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>\n]*>|[^)\s]+)")
FENCE = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Targets that leave the repository, or that address the current document.
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "ftp://", "data:", "tel:")


def github_slug(heading: str) -> str:
    """Return the anchor GitHub derives from a heading's visible text."""
    text = heading.strip().lower()
    text = re.sub(r"\[\]`*", "", text)
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def heading_slugs(text: str) -> set[str]:
    """Collect anchors for one document, including GitHub's duplicate suffixes."""
    without_fences: list[str] = []
    inside = False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            without_fences.append(line)
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for line in without_fences:
        match = HEADING.match(line)
        if match is None:
            continue
        base = github_slug(match.group(2))
        if not base:
            continue
        index = counts.get(base, 0)
        counts[base] = index + 1
        slugs.add(base if index == 0 else f"{base}-{index}")
    return slugs


def link_targets(text: str) -> Iterator[tuple[int, str]]:
    """Yield the line number and target of every inline link outside fences."""
    inside = False
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            inside = not inside
            continue
        if inside:
            continue
        for match in INLINE_LINK.finditer(line):
            raw = match.group(1)
            yield number, raw[1:-1] if raw.startswith("<") else raw


def anchor_errors(path: Path, line: int, target: Path, fragment: str) -> list[str]:
    if target.suffix != ".md":
        return []
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if fragment in heading_slugs(text):
        return []
    return [f"{path}:{line}: anchor does not resolve: #{fragment}"]


def violations(path: Path) -> list[str]:
    """Report broken relative links and anchors declared by one document."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    errors: list[str] = []
    root = path.parent
    for line, target in link_targets(text):
        if target.startswith(EXTERNAL_PREFIXES):
            continue
        decoded = urllib.parse.unquote(target)
        relative, _, fragment = decoded.partition("#")
        if not relative:
            if fragment and fragment not in heading_slugs(text):
                errors.append(f"{path}:{line}: anchor does not resolve: #{fragment}")
            continue
        resolved = Path(relative) if relative.startswith("/") else root / relative
        resolved = Path(str(resolved).split("?", 1)[0])
        if not resolved.exists():
            errors.append(f"{path}:{line}: link target does not exist: {target}")
            continue
        if fragment:
            errors.extend(anchor_errors(path, line, resolved, fragment))
    return errors


def repository_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return tuple(
        root / name
        for name in sorted(set(result.stdout.decode("utf-8").split("\0")))
        if name.endswith(".md")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paths = args.files or repository_files(root)
    errors: list[str] = []
    for path in paths:
        errors.extend(violations(path))
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked {len(paths)} markdown files; all local links and anchors resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
