"""Enforce English content with a scoped Chinese teaching-prose exception."""

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
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHINESE_NOTEBOOK_NAMES = {
    "01_first_circuit.ipynb",
    "02_quantum_training.ipynb",
    "03_hybrid_model.ipynb",
    "04_one_problem_three_simulators.ipynb",
    "05_jiuding_jobs.ipynb",
    "06_quafu_hardware.ipynb",
}


def chinese_notebook_violations(path: Path, text: str) -> list[str]:
    """Exclude only teaching prose; keep all executable and recorded data checked."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return [f"{path}: invalid localized notebook JSON"]
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        return [f"{path}: invalid localized notebook cells"]
    for cell in document["cells"]:
        if not isinstance(cell, dict):
            return [f"{path}: invalid localized notebook cell"]
        if cell.get("cell_type") == "markdown":
            source = cell.get("source")
            if not isinstance(source, str) and not (
                isinstance(source, list)
                and all(isinstance(line, str) for line in source)
            ):
                return [f"{path}: invalid localized Markdown source"]
            prose = source if isinstance(source, str) else "".join(source)
            fence = ""
            for line in prose.splitlines():
                stripped = line.lstrip()
                if fence and HAN.search(line):
                    return [f"{path}: Han-script text in localized code example"]
                closing = rf"{re.escape(fence[0])}{{{len(fence)},}}\s*" if fence else ""
                if fence and re.fullmatch(closing, stripped):
                    fence = ""
                elif not fence:
                    opening = re.match(r"(`{3,}|~{3,})", stripped)
                    if opening:
                        fence = opening.group(1)
            del cell["source"]
    if any(HAN.search(item) for item in json_strings(document)):
        return [f"{path}: Han-script text outside localized Markdown source"]
    return []


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
    localized_directory = REPOSITORY_ROOT / "workshops/flagos2026/notebooks_cn"
    if (
        path.resolve().parent == localized_directory
        and path.name in CHINESE_NOTEBOOK_NAMES
    ):
        return errors + chinese_notebook_violations(path, text)
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
    root = REPOSITORY_ROOT
    paths = args.files or repository_files(root)
    failed = 0
    for path in paths:
        errors = violations(path)
        if errors:
            failed += 1
            for error in errors:
                print(error)
    print(f"Checked {len(paths)} paths; {failed} files violate the language policy.")
    print("Binary images and non-UTF-8 files require separate visual review.")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
