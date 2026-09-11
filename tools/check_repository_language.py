"""Find Han-script text in repository files without printing their contents."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

HAN = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002ffff\U00030000-\U000323af]"
)


def json_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from json_strings(key)
            yield from json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)


def violations(path: Path) -> list[str]:
    errors = []
    if HAN.search(path.as_posix()):
        errors.append(f"{path}: filename contains Han-script characters")
    if not path.is_file():
        return errors
    data = path.read_bytes()
    if b"\0" in data:
        return errors
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors
    for number, line in enumerate(text.splitlines(), 1):
        if HAN.search(line):
            errors.append(f"{path}:{number}: Han-script text")
    if path.suffix in {".json", ".ipynb"} and not HAN.search(text):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return errors
        if any(HAN.search(item) for item in json_strings(value)):
            errors.append(f"{path}: JSON contains escaped Han-script text")
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
        if name
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paths = args.files or repository_files(root)
    failed = 0
    for path in paths:
        errors = violations(path)
        if errors:
            failed += 1
            for error in errors:
                print(error)
    print(f"Checked {len(paths)} paths; {failed} files contain Han-script text.")
    print("Binary images and non-UTF-8 files require separate visual review.")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
