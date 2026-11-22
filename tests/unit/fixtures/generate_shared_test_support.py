"""Observe complete unchanged shared test support with explicit source parents."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

SOURCE_BLOBS = {
    "test.test.cc": "c3e1aabfeb966a78eb9a48eb86a2b12e5ecc6d73",
    "test.test.h": "5fd12089cb99f6e277a063bc96eb4df76c273c8f",
}


def regenerate(source: Path, gtest: Path, output: Path) -> None:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen shared test support differs for {name}")
    with tempfile.TemporaryDirectory(prefix="fq-shared-test-") as directory:
        root = Path(directory)
        binary = root / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source),
                "-I" + str(gtest / "include"),
                str(Path(__file__).with_name("shared_test_oracle.cc")),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

        def run(*args: str, cwd: Path = root) -> list[str]:
            result = subprocess.run(
                [str(binary), *args],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            )
            return result.stdout.splitlines()

        vectors = [
            [],
            [0],
            [1],
            [0xFFFFFFFF],
            [1 << 32],
            [(1 << 64) - 1],
            [0, 1, 2, 3],
            list(range(8)),
            list(range(625)),
        ]
        seeds = [
            [words, list(map(int, run("seed", *map(str, words))))] for words in vectors
        ]
        independent = [
            [seed, list(map(int, run("independent", str(seed))))]
            for seed in (0, 1, 1084, (1 << 64) - 1)
        ]
        lookups = []
        for directories in range(8):
            for files in range(8):
                scenario = root / f"layout-{directories}-{files}"
                cwd = scenario / "cwd"
                cwd.mkdir(parents=True)
                prefixes = (
                    cwd / "kickmix/testdata",
                    cwd / "testdata",
                    scenario / "testdata",
                )
                for index, prefix in enumerate(prefixes):
                    if directories & (1 << index):
                        prefix.mkdir(parents=True)
                        if files & (1 << index):
                            (prefix / "asset").write_bytes(bytes([index]))
                result = run("lookup", "asset", cwd=cwd)
                after = [
                    (
                        (prefix / "asset").read_bytes().hex()
                        if (prefix / "asset").is_file()
                        else None
                    )
                    for prefix in prefixes
                ]
                lookups.append([directories, files, result, after])
        directory_case = root / "directory-case"
        (directory_case / "testdata/asset").mkdir(parents=True)
        directory_result = run("lookup", "asset", cwd=directory_case)
        records = {
            "seed": seeds,
            "independent": independent,
            "lookup": lookups,
            "files": run("files"),
            "directory": directory_result,
        }
        assert run("external") == ["312 64"]
        output.write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gtest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source_root, args.gtest_root, args.output)


if __name__ == "__main__":
    main()
