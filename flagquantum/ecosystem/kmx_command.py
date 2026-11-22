# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix main.cc at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Dispatch raw physical-format commands to their existing FlagQuantum owners."""

import sys

_INVOCATION = "python -m flagquantum.ecosystem.kmx_command"

_HELP = f"""    {_INVOCATION} circuit                       # generate a circuit
    {_INVOCATION} diagram                       # make a picture of a circuit
    {_INVOCATION} count_operations              # count instructions in a circuit
    {_INVOCATION} count_operations_sampled      # count instructions executed while simulating a circuit
    {_INVOCATION} kmb2kmx                       # reads a kmb file and writes out an equivalent kmx file
    {_INVOCATION} kmx2kmb                       # reads a kmx file and writes out an equivalent kmb file
    {_INVOCATION} sample                        # simulate a circuit
    {_INVOCATION} spin                          # time how long it takes to simulate a circuit
"""


def main(argv: list[str] | None = None) -> int:
    """Forward one command to its format, drawing or sampling entry point."""
    arguments = [
        word.partition("\0")[0] for word in (sys.argv[1:] if argv is None else argv)
    ]
    command = arguments[0] if arguments else ""
    rest = arguments[1:]
    try:
        if command == "circuit":
            from ..drawer.kmx_generate_command import main as run
        elif command == "count_operations":
            from ..compiler.formats.kmx_operation_counts import main as run
        elif command in ("kmb2kmx", "kmx2kmb"):
            from ..compiler.formats.kmb_convert import main as run

            return run([command, *rest])
        elif command == "count_operations_sampled":
            from .kmx_sampled_counts import main as run
        elif command == "sample":
            from .kmx_sample import main as run
        elif command == "spin":
            from .kmx_spin import main as run
        elif command == "diagram":
            from ..drawer.kmx_diagram_command import main as run
        else:
            prefix = (
                "Unrecognized subcommand." if arguments else "Specify a subcommand."
            )
            sys.stderr.write(prefix + " Known subcommands are:\n" + _HELP)
            return 1
        return run(rest)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
