"""Regenerate complete packed counter observations from frozen C++ headers."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

SOURCE_BLOBS = {
    "simd/b128_polyfill.h": "e96adba41426754826834c33b92f8c667df31749",
    "simd/b128_sse.h": "2821464869d036c5a2e7f30624a54bd0fa7f78d1",
    "simd/b256_avx.h": "5f115d5672ca9db31a0493835f5af7afb0b8dcdf",
    "simd/b256_polyfill.h": "8e38c8052f6bb803c40ae878efa9eaa0f0ba8373",
    "simd/b512_avx.h": "a65a4a2e3c54c37544b7e75b8d20767a191cb36a",
    "simd/b512_polyfill.h": "b69dd1bdbc3dd247b11640dfbce7b8d8392ac818",
    "simd/b64_polyfill.h": "e9fc246c9149e8d83ccc8e6c6b28aab389fb9795",
    "simd/simd.h": "d78f0f06eae47487afea82b4a6367d1f553228d3",
    "simd/simd.perf.h": "ee4a3fac4c4f65de710e63305bcbb19d9ef56c85",
    "simd/simd.test.cc": "516c4cc67e2b41b4fe0914ca4ddd6250a6b50cc8",
    "simd/simd.test.h": "20fe9ccd6bff96f19dc14f629259f41900f1f67d",
    "util/pop_counter.h": "13166ccb667aa4ab7d6da21fb044fa14e1b379c0",
    "util/pop_counter.perf.cc": "acf7500eb70702deb54587ed062d3d495a69aa51",
    "util/pop_counter.test.cc": "255534ca5cb697e9e3f779f77a265110dd0d2aa4",
}


def regenerate(source_root: Path, output: Path, timings: Path) -> int:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source_root / "kickmix" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen packed counter dependency mismatch for {name}")
    cases = [
        (width, seed, repeats, mode, clear)
        for width in (64, 128, 256, 512)
        for seed in (0, 1084)
        for repeats in (0, 1, 31, 32, 33, 63, 64, 65, 500, 10000)
        for mode in range(6)
        for clear in (0, 1)
    ]
    with tempfile.TemporaryDirectory(prefix="fq-population-counter-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O3",
                "-I" + str(source_root),
                str(Path(__file__).with_name("population_counter_oracle.cc")),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, case)) + "\n" for case in cases),
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise ValueError("Original counter returned an incomplete observation stream")
    with output.open("w") as stream:
        for case, expected in zip(cases, lines, strict=True):
            stream.write(json.dumps([case, expected], separators=(",", ":")) + "\n")
    timings.write_text(result.stderr)
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Recorded {regenerate(args.source_root, args.output, args.timings)} original states"
    )


if __name__ == "__main__":
    main()
