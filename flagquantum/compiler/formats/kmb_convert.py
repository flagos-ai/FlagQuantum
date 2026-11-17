"""Convert pinned KMX and KMB files or standard streams."""

import argparse
import sys
from pathlib import Path

from .kmb import decode_kmb, encode_kmb


def main(argv: list[str] | None = None) -> int:
    """Convert a complete payload before opening the output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("kmx2kmb", "kmb2kmx"))
    parser.add_argument("--in", dest="input", type=Path, help="default: stdin")
    parser.add_argument("--out", dest="output", type=Path, help="default: stdout")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read() if args.input is None else args.input.read_bytes()
        if args.direction == "kmx2kmb":
            result = encode_kmb(raw.decode())
        else:
            text = decode_kmb(raw)
            result = (text + "\n").encode() if text else b""
        written = (
            sys.stdout.buffer.write(result)
            if args.output is None
            else args.output.write_bytes(result)
        )
        if written != len(result):
            raise OSError(f"Incomplete write: wrote {written} of {len(result)} bytes")
        if args.output is None:
            sys.stdout.buffer.flush()
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
