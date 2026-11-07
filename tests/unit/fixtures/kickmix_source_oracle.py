"""Build an isolated oracle against unchanged locked C++ source files."""

import hashlib
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def source_oracle(
    source_root: Path, generator: str, blob: str, harness: Path
) -> Iterator[Path]:
    """Validate the generator blob, compile its dependencies and clean temporary output."""
    source = source_root.resolve() / "src"
    implementation = source / "kickmix/gen" / generator
    contents = implementation.read_bytes()
    actual = hashlib.sha1(
        b"blob " + str(len(contents)).encode() + b"\0" + contents
    ).hexdigest()
    if actual != blob:
        raise ValueError("Generator reference differs from the locked source blob")
    groups = ("build", "circuit", "id", "mem", "util", "py/circuit/ops")
    sources = [
        str(path)
        for group in groups
        for path in sorted((source / "kickmix" / group).glob("*.cc"))
        if not path.name.endswith((".pybind.cc", ".test.cc", ".perf.cc"))
    ]
    with tempfile.TemporaryDirectory(prefix="fq-source-oracle-") as temporary:
        oracle = Path(temporary) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O0",
                "-I" + str(source),
                str(harness),
                str(implementation),
                *sources,
                "-o",
                str(oracle),
            ],
            check=True,
            timeout=180,
        )
        yield oracle


def normalized_stream(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Preserve ordered primitive operands and effective classical guard sets."""
    scopes: list[str] = []
    result = []
    for line in text.splitlines():
        if line.startswith(("APPEND_TO_REGISTER ", "REGISTER ", "INFO_TAG ", "R ")):
            continue
        if line.startswith("PUSH_CONDITION if "):
            scopes.append(line.split()[-1])
        elif line == "POP_CONDITION":
            scopes.pop()
        else:
            primitive, conditional, bit = line.partition(" if ")
            result.append(
                (
                    primitive,
                    tuple(sorted(set((*scopes, bit) if conditional else scopes))),
                )
            )
    if scopes:
        raise ValueError("Oracle stream has unclosed condition scopes")
    return result
