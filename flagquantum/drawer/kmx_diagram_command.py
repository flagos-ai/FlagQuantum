# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_diagram.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_diagram.cc

"""Draw physical KMX or KMB circuits as text, HTML or SVG."""

import argparse
import sys
from pathlib import Path

from ..compiler.formats._binary_streams import write_binary_result
from ..compiler.formats.kmx_circuit import read_circuit_file
from .reversible_phase_diagram import render_circuit_text
from .reversible_phase_graphics import render_circuit_html, render_circuit_svg

_RENDER = {
    "text": render_circuit_text,
    "html": render_circuit_html,
    "svg": render_circuit_svg,
}


def main(argv: list[str] | None = None) -> int:
    """Read the circuit and render it before opening the destination."""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("--in", dest="input", type=Path, help="default: stdin")
    parser.add_argument("--out", dest="output", type=Path, help="default: stdout")
    parser.add_argument("--kind", choices=tuple(_RENDER), default="text")
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError(f"Unrecognized command line argument {unknown[0]}")
        data = (
            sys.stdin.buffer.read() if args.input is None else args.input.read_bytes()
        )
        diagram = _RENDER[args.kind](read_circuit_file(data))
        write_binary_result((diagram + "\n").encode(), args.output, sys.stdout.buffer)
    except (argparse.ArgumentError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
