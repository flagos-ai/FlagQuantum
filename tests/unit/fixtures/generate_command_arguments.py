"""Regenerate complete command arguments observations from frozen C++ sources."""

import argparse
import hashlib
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

SOURCE_BLOBS = {
    "kickmix/util/arg_parse.cc": "4c0434f8cf4560e294c31bd7cd23aa6eac6c12d7",
    "kickmix/util/arg_parse.h": "98751e5a407193df83371e73234bec6a32d2ac69",
    "kickmix/util/arg_parse.test.cc": "b4aaedf0dd80752e353cd030e4234f1c22b49ccb",
    "test.test.cc": "c3e1aabfeb966a78eb9a48eb86a2b12e5ecc6d73",
    "test.test.h": "5fd12089cb99f6e277a063bc96eb4df76c273c8f",
}


def quote_wire(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def regenerate(source_root: Path, output: Path) -> int:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source_root / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen command arguments dependency mismatch for {name}")
    cases: list[tuple[str, str, str, str, str, list[str]]] = []
    layouts = [
        [],
        ["--value"],
        ["--value="],
        ["--value", "--"],
        ["--", "--value=3"],
        ["--value=3", "--value=4"],
        ["--value", "-x"],
        ["--value", "-3"],
        ["--value", "-.5"],
        ["--value", ""],
        ["--valuex=3"],
    ]
    texts = [
        "",
        "+",
        "-",
        "0",
        "-0",
        "1",
        "-1",
        "1_",
        "1__2",
        "0_",
        "_1",
        " 1",
        "1 ",
        "1e2",
        "0x1p-1",
        "inf",
        "-inf",
        "nan",
        "nan(payload)",
        "1.0000000596046447753906250000000001",
        "9223372036854775807",
        "9223372036854775808",
        "-9223372036854775808",
        "-9223372036854775809",
        "18446744073709551616",
        "18446744073709551615",
        "18446744073709551617",
        "18446744073709551620",
        "-18446744073709551616",
        "\n2",
        "--a\0ignored",
        "1e-999",
        "1e999",
        "test",
        "rest",
        "bad",
        "\t2",
    ]
    layouts += [["--value=" + text] for text in texts]
    layouts += [["--value", text] for text in texts]
    for operation in ("find", "require", "bool", "int", "float", "double", "enum"):
        for args in layouts:
            bounds = [("42", "-100", "100"), ("-1", "0", "10"), ("9", "11", "10")]
            if operation in ("find", "require", "bool"):
                bounds = [("0", "0", "0")]
            if operation == "enum":
                bounds = [("NONE", "0", "0"), ("test", "0", "0"), ("missing", "0", "0")]
            if operation in ("float", "double"):
                bounds += [("nan", "-inf", "inf"), ("inf", "nan", "nan")]
            if operation == "int":
                bounds += [("0", "-9223372036854775808", "9223372036854775807")]
            for default, low, high in bounds:
                cases.append(
                    (operation, "--value", default, low, high, ["program", *args])
                )
    for length in range(4):
        for words in itertools.product(
            ("--a", "--b=3", "--old", "text", "-3", "--"), repeat=length
        ):
            for mode in ("NONE", "mode"):
                cases.append(
                    ("unknown", "--value", mode, "0", "0", ["program", *words])
                )
    for argv in (
        ["program", "mode"],
        ["program", "mode", "--a"],
        ["program", "mode", "bad"],
    ):
        cases.append(("unknown", "--value", "mode", "0", "0", argv))
    for mode in (
        "r",
        "rb",
        "r+",
        "rb+",
        "r+b",
        "w",
        "wb",
        "w+",
        "wb+",
        "a",
        "ab",
        "a+",
        "ab+",
        "wx",
        "w+x",
        "wx+",
        "re",
        "rbe",
        "r+e",
        "rb+e",
        "r+be",
        "we",
        "wbe",
        "w+e",
        "ae",
        "abe",
        "a+e",
        "wxe",
        "w+xe",
        "wx+e",
        "invalid",
    ):
        for default in ("NONE", "DEFAULT"):
            for action in ("read", "write", "absent"):
                if action == "write" and mode.startswith("r") and "+" not in mode:
                    continue
                for argv in (
                    ["program"],
                    ["program", "--value="],
                    ["program", "--value=argument-file"],
                ):
                    cases.append(("file", "--value", default, mode, action, argv))
    cases.append(
        ("file", "--value", "NONE", "re", "read", ["program", "--value=argument-file"])
    )
    with tempfile.TemporaryDirectory(prefix="fq-command-arguments-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O3",
                "-I" + str(source_root),
                str(Path(__file__).with_name("command_arguments_oracle.cc")),
                str(source_root / "kickmix/util/arg_parse.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        result = subprocess.run(
            [str(binary)],
            cwd=directory,
            input="".join(
                operation
                + " "
                + " ".join(quote_wire(value) for value in (name, default, low, high))
                + " "
                + str(len(args))
                + " "
                + " ".join(quote_wire(arg) for arg in args)
                + "\n"
                for operation, name, default, low, high, args in cases
            ),
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise ValueError(
            "Original argument parser returned an incomplete observation stream"
        )
    with output.open("w") as stream:
        for case, expected in zip(cases, lines, strict=True):
            stream.write(json.dumps([case, expected], separators=(",", ":")) + "\n")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Recorded {regenerate(args.source_root, args.output)} original states")


if __name__ == "__main__":
    main()
