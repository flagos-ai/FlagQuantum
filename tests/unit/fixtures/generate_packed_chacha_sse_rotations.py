"""Capture defined SSE rol32 template paths from unchanged locked headers."""

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

_BLOBS = {
    "kickmix/util/chacha.h": "82a1bb8f8c17028aa53bf0dbc368bed6803cf724",
    "kickmix/simd/b128_sse.h": "2821464869d036c5a2e7f30624a54bd0fa7f78d1",
}
_AMOUNTS = (
    32 - ((1 << 31) - 1),
    -256,
    -33,
    -1,
    *range(33),
    33,
    64,
    255,
    256,
    1024,
    (1 << 31) - 1,
)
_CPP = r"""
#include "kickmix/simd/b128_sse.h"
#include "kickmix/util/chacha.h"
#include <array>
#include <cstdint>
#include <iostream>
template<int K> void probe() {
    std::array<uint32_t, 36> values{};
    for (int k = 0; k < 32; k++) values[k] = uint32_t{1} << k;
    values[32] = 0;
    values[33] = UINT32_MAX;
    values[34] = 0x12345678u;
    values[35] = 0xdeadbeefu;
    for (uint32_t v : values) {
        std::array<uint32_t,4> input{v, v ^ 0x55555555u, ~v, v * 0x9e3779b9u};
        kickmix::b128_sse word{_mm_set_epi32(input[3], input[2], input[1], input[0])};
        auto result = kickmix::rol32<K>(word);
        std::cout << K;
        for (auto a : input) std::cout << " " << a;
        for (int k = 0; k < 4; k++) std::cout << " " << result.u32(k);
        std::cout << "\n";
    }
}
int main() { CALLS }
"""


def regenerate(source_root: Path, output: Path) -> int:
    for name, expected in _BLOBS.items():
        raw = (source_root / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen SSE rotation dependency mismatch: {name}")
    arm = platform.machine().lower() in {"arm64", "aarch64"}
    if arm and platform.system() != "Darwin":
        raise RuntimeError(
            "Source SSE rotation capture requires an x86 execution environment"
        )
    arch = ["-arch", "x86_64"] if arm else []
    with tempfile.TemporaryDirectory(prefix="fq-chacha-sse-") as folder:
        source = Path(folder) / "probe.cc"
        binary = Path(folder) / "probe"
        source.write_text(
            _CPP.replace("CALLS", " ".join(f"probe<{k}>();" for k in _AMOUNTS))
        )
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                *arch,
                "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined",
                "-I" + str(source_root),
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        command = ["arch", "-x86_64", str(binary)] if arm else [str(binary)]
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
        if result.stderr:
            raise RuntimeError(
                f"Original SSE rotation emitted diagnostics: {result.stderr}"
            )
    records = [list(map(int, line.split())) for line in result.stdout.splitlines()]
    if len(records) != len(_AMOUNTS) * 36 or any(len(row) != 9 for row in records):
        raise ValueError("Incomplete source SSE rotation observations")
    output.write_text(
        json.dumps(
            {
                "revision": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "frozen_blobs": _BLOBS,
                "ubsan": "no diagnostics",
                "amounts": _AMOUNTS,
                "records": records,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        regenerate(args.source_root, args.output),
        "complete source SSE rotation vectors",
    )
