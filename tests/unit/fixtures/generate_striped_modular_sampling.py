"""Regenerate all striped sampling outputs against unchanged locked sources."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

SOURCE_BLOBS = {
    "mod_rand.cc": "57e02164309dc0490c066f6ebe20598b01e897b0",
    "mod_rand.h": "06d631340d5355c8939d1c19fe5023e30e1f190d",
    "mod_rand.perf.cc": "05bf7b04633ad9df65e94506bc4a4e072915bedf",
    "mod_rand.test.cc": "17065d0635d1b650356296183d802e677c8f9bb7",
    "fixed_width_int.cc": "50a2300a311befb075bd6680287fa3e0e36bacc2",
    "fixed_width_int.h": "7057645bae714c72deb5b60dca85589bd4a9c8a7",
    "word_ops.h": "b0cab29ebcd7e616b6a40e7e4d99eb72bf0c9a45",
}


def regenerate(source_root: Path, corpus: Path, output: Path) -> int:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source_root / "kickmix/util" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source blob mismatch for {name}")
    cases: list[list[int | str]] = []
    with corpus.open() as stream:
        for line in stream:
            record: object = json.loads(line)
            if not isinstance(record, list) or len(record) != 2:
                raise ValueError("Reference record must contain input and output")
            case, expected = record
            if (
                not isinstance(case, list)
                or len(case) != 6
                or not isinstance(expected, str)
            ):
                raise ValueError("Striped reference input must contain six fields")
            if not isinstance(case[1], str) or any(
                type(case[i]) is not int for i in (0, 2, 3, 4, 5)
            ):
                raise ValueError("Striped reference operand field types are invalid")
            cases.append(case)
    with tempfile.TemporaryDirectory(prefix="fq-striped-sampling-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source_root),
                str(Path(__file__).with_name("striped_modular_sampling_oracle.cc")),
                str(source_root / "kickmix/util/mod_rand.cc"),
                str(source_root / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=60,
        )
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, case)) + "\n" for case in cases),
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise ValueError("Source returned an incomplete reference stream")
    with output.open("w") as stream:
        for case, line in zip(cases, lines, strict=True):
            stream.write(json.dumps([case, line], separators=(",", ":")) + "\n")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Recorded {regenerate(args.source_root, args.corpus, args.output)} original observations"
    )


if __name__ == "__main__":
    main()
