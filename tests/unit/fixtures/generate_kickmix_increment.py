"""Regenerate increment stream digests from the locked, unmodified C++ source."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

SOURCE_BLOB = "890b484fb6043d45d8e60a91dd241ab8092f7b21"
SOURCE_COMMIT = "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="locked kickmix directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve() / "src"
    increment = source / "kickmix/gen/adders/gen_iadd1.cc"
    contents = increment.read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(contents)).encode() + b"\0" + contents
    ).hexdigest()
    if blob != SOURCE_BLOB:
        parser.error("Increment reference differs from the locked source blob")
    root = Path(__file__).resolve().parents[3]
    groups = ("build", "circuit", "id", "mem", "util", "py/circuit/ops")
    sources = [
        str(path)
        for group in groups
        for path in sorted((source / "kickmix" / group).glob("*.cc"))
        if not path.name.endswith((".pybind.cc", ".test.cc", ".perf.cc"))
    ]
    cases = []
    with tempfile.TemporaryDirectory(prefix="fq-increment-reference-") as temporary:
        oracle = Path(temporary) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O0",
                "-I" + str(source),
                str(root / "tests/unit/fixtures/kickmix_increment_oracle.cc"),
                str(increment),
                *sources,
                "-o",
                str(oracle),
            ],
            check=True,
            timeout=180,
        )
        for width in range(17):
            for controlled in (False, True):
                for extra in (0, 2):
                    clean = max(0, width - (2 if controlled else 3)) + extra
                    text = subprocess.check_output(
                        [str(oracle), str(width), str(clean), str(int(controlled))],
                        text=True,
                        timeout=5,
                    )
                    lines = [
                        line
                        for line in text.splitlines()
                        if not line.startswith(
                            ("APPEND_TO_REGISTER ", "REGISTER ", "INFO_TAG ", "R ")
                        )
                    ]
                    cases.append(
                        {
                            "width": width,
                            "controlled": controlled,
                            "extra_clean": extra,
                            "sha256": hashlib.sha256(
                                "\n".join(lines).encode()
                            ).hexdigest(),
                            "opcodes": dict(Counter(line.split()[0] for line in lines)),
                        }
                    )
    args.output.write_text(
        json.dumps(
            {
                "source_commit": SOURCE_COMMIT,
                "source_blob": SOURCE_BLOB,
                "normalization": "Register/INFO metadata omitted; R on supplied clean wires is an explicit zero-input precondition. All other operations, operands and feedback slots retained.",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
