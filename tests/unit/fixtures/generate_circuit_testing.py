"""Observe complete unchanged circuit-test helpers through a compiled oracle."""

import argparse
import hashlib
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

from tests.unit.fixtures.generate_command_arguments import quote_wire


def compile_source(source_root: Path, gtest_root: Path, binary: Path) -> None:
    records: object = json.loads(
        Path(__file__).with_name("circuit_testing_source_blobs.json").read_text()
    )
    if not isinstance(records, dict):
        raise ValueError("Invalid frozen circuit helper manifest")
    for name, expected in records.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("Invalid frozen source entry")
        raw = (source_root / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source mismatch for {name}")
    sources = [
        name
        for name in records
        if name.endswith(".cc") and name != "kickmix/util/circuit_testing.test.cc"
    ]
    subprocess.run(
        [
            "clang++",
            "-std=c++20",
            "-O1",
            "-Wl,-dead_strip",
            "-pthread",
            "-I" + str(source_root),
            "-I" + str(gtest_root / "include"),
            "-I" + str(gtest_root),
            str(Path(__file__).with_name("circuit_testing_oracle.cc")),
            *(str(source_root / name) for name in sources),
            str(gtest_root / "src/gtest-all.cc"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )


def regenerate(binary: Path, output: Path) -> int:
    texts = [
        "",
        "a",
        " ",
        "\n",
        " b ",
        "\n        test\n    ",
        "\n        test\n            test2\n    ",
        "\n            test2\n        test\n    ",
        "\n            test2 test3\n        test\n    ",
        "\r",
        "\r\n    a\r\n",
        "  \r  a\n    b",
        "\t  a\n    b",
        "█",
        "é",
        "\0a\n",
    ]
    texts += [
        "".join(chars)
        for size in range(5)
        for chars in itertools.product(" a\n\r", repeat=size)
    ]
    cases = [(op, text) for op in ("undent", "lines") for text in texts]
    cases += [
        ("expect", text)
        for text in texts
        + [
            "q0: -X-",
            "\n    q0: -X-\n",
            "q0: -X-\n",
            "q0: -Y-",
            "\n\nq0: -X-",
            "q0: -X-\nextra",
            "q0: -X-\n\n",
            "q0: -█-",
        ]
    ]
    cases.append(("expect_classical", " b0=0 b0^=1 if b1 push_cond if b0 pop_cond"))
    cases.append(("circuit", ""))
    result = subprocess.run(
        [str(binary)],
        input="".join(op + " " + quote_wire(text) + "\n" for op, text in cases),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    lines = iter(result.stdout.splitlines())
    records: dict[str, list[object]] = {
        "undent": [],
        "lines": [],
        "expect": [],
        "circuit": [],
        "expect_classical": [],
    }
    for op, text in cases:
        if op == "lines":
            count = int(next(lines))
            expected: object = [
                bytes.fromhex(next(lines)).decode() for _ in range(count)
            ]
        elif op == "circuit":
            records[op] = [
                bytes.fromhex(next(lines)).decode(),
                bytes.fromhex(next(lines)).decode(),
                next(lines),
            ]
            continue
        else:
            value = next(lines)
            expected = (
                value
                if op in ("expect", "expect_classical")
                else bytes.fromhex(value).decode()
            )
        records[op].append([text, expected])
    if next(lines, None) is not None:
        raise ValueError("Unexpected trailing source observation")
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gtest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fq-circuit-helper-") as directory:
        binary = Path(directory) / "oracle"
        compile_source(args.source_root, args.gtest_root, binary)
        result = subprocess.run(
            [str(binary), "tests"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        print(result.stdout)
        print(f"Recorded {regenerate(binary, args.output)} original observations")


if __name__ == "__main__":
    main()
