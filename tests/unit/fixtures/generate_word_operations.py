"""Record all locked 64-bit word operations from both portable product routes."""

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

SOURCE_BLOB = "b0cab29ebcd7e616b6a40e7e4d99eb72bf0c9a45"
TEST_BLOB = "467a9eef03c6a2fc32199e081395ac9487ddd609"


def generate(source_root: Path, output: Path) -> int:
    for name, expected in (
        ("word_ops.h", SOURCE_BLOB),
        ("word_ops.test.cc", TEST_BLOB),
    ):
        raw = (source_root / "kickmix/util" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source blob mismatch for {name}")
    mask = (1 << 64) - 1
    values = (0, 1, 2, mask // 2, mask - 1, mask)
    cases = [
        (a, b, c, d) for a in values for b in values for c in values for d in values
    ]
    cases.extend(
        (1 << bit, mask ^ (1 << bit), c, mask) for bit in range(64) for c in (0, 1)
    )
    rng = random.Random(1084)
    cases.extend(
        (
            rng.getrandbits(64),
            rng.getrandbits(64),
            rng.getrandbits(64),
            rng.getrandbits(64),
        )
        for _ in range(2048)
    )
    payload = "".join(" ".join(map(str, case)) + "\n" for case in cases)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="fq-word-operations-") as directory:
        for fallback in (False, True):
            binary = Path(directory) / ("fallback" if fallback else "native")
            command = ["clang++", "-std=c++20", "-O1", "-I" + str(source_root)]
            if fallback:
                command.append("-U__SIZEOF_INT128__")
            command.extend(
                [
                    str(Path(__file__).with_name("word_operations_oracle.cc")),
                    "-o",
                    str(binary),
                ]
            )
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            result = subprocess.run(
                [str(binary)],
                input=payload,
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            )
            outputs.append(result.stdout.splitlines())
    if outputs[0] != outputs[1] or len(outputs[0]) != len(cases):
        raise ValueError("Source product routes disagree or returned incomplete output")
    with output.open("w") as stream:
        for case, line in zip(cases, outputs[0], strict=True):
            stream.write(
                json.dumps([*case, *map(int, line.split())], separators=(",", ":"))
                + "\n"
            )
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Recorded {generate(args.source_root, args.output)} unchanged-source observations"
    )


if __name__ == "__main__":
    main()
