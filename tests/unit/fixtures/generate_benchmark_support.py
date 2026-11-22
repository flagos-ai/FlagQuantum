"""Regenerate complete original adaptive benchmark and CLI observations."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

SOURCE_BLOBS = {
    "main.perf.cc": "a9ce0097d2114a3e06d23bd60d4e379513ffe877",
    "perf.perf.h": "7bc21514b1b9af48533e457abafbb421d24b68fe",
    "kickmix/util/arg_parse.cc": "4c0434f8cf4560e294c31bd7cd23aa6eac6c12d7",
    "kickmix/util/arg_parse.h": "98751e5a407193df83371e73234bec6a32d2ac69",
}


def regenerate(source: Path, output: Path) -> None:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen benchmark source differs for {name}")
    with tempfile.TemporaryDirectory(prefix="fq-benchmark-support-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source),
                str(Path(__file__).with_name("benchmark_support_oracle.cc")),
                str(source / "kickmix/util/arg_parse.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

        def run(arguments: list[str]) -> list[object]:
            result = subprocess.run(
                [str(binary), *arguments],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            return [arguments, result.returncode, result.stdout, result.stderr]

        scalars = [
            0.0,
            1e-18,
            1e-15,
            1e-12,
            1e-9,
            1e-6,
            0.001,
            1.0,
            9.99999,
            10.0,
            99.99999,
            100.0,
            123.456,
            999.99999,
            1000.0,
            1000.000001,
            1e6,
            1e9,
            1e12,
            1e15,
        ]
        scalars += [
            float(f"{mantissa}e{exponent}")
            for exponent in range(-15, 16)
            for mantissa in (1, 9.999, 10, 99.999, 100, 123.456, 999.999)
        ]
        si = [[value, run(["si", repr(value)])[2]] for value in scalars]
        cli = [
            run(arguments)
            for arguments in [
                [],
                ["--only", "timing_alpha"],
                ["--only", "timing_*"],
                ["--only", "*"],
                ["--only", "timing_beta,timing_alpha,timing_beta"],
                ["--only", "empty"],
                ["--only", "missing"],
                ["--only", ""],
                ["--only", "timing_alpha,"],
                ["--only", "timing_alpha", "--only", "empty"],
                ["--unknown"],
                ["--target_seconds", "-1"],
                ["--only", "timing_beta", "--target_seconds", "0"],
                ["--only=timing_beta"],
                ["--", "--only", "empty"],
            ]
        ]
        adaptive = [
            run(["adaptive", target, cost])
            for target, cost in [
                ("0", "1"),
                ("0.001", "1000000"),
                ("0.001", "10000"),
                ("0.001", "1"),
                ("1", "300000000"),
                ("0.1", "13000000"),
                ("0.1", "9000000"),
                ("0.5", "123456"),
                ("1", "0"),
                ("1e16", "1"),
            ]
        ]
        reports = [
            run(
                [
                    "report",
                    repr(seconds),
                    str(repetitions),
                    repr(goal),
                    "ops",
                    "32",
                    "items",
                    "3",
                ]
            )
            for seconds in (0.000000123, 0.002, 0.123456, 12.3456, 123456.0)
            for repetitions in (1, 10, 1001)
            for goal in (-1.0, 1e-6, 0.0002, 1.0, 12345678.0)
        ]
        reports += [
            run(["report", seconds, str(repetitions), "-1", "ops", multiplier])
            for seconds in ("0", "-0", "-1", "nan", "-nan", "inf", "-inf")
            for repetitions in (0, 1)
            for multiplier in ("0", "3", "nan", "inf", "-inf")
        ]
        goals = [run(["goals", str(value)]) for value in (0, 1, 123.456, 1e-100, 1e100)]
        output.write_text(
            json.dumps(
                {
                    "si": si,
                    "cli": cli,
                    "adaptive": adaptive,
                    "reports": reports,
                    "goals": goals,
                    "rates": run(["rates"]),
                    "duplicates": run(["duplicates"]),
                },
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source_root, args.output)


if __name__ == "__main__":
    main()
