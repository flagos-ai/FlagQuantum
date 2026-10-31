"""Public native file/text workflow with bounded exact CPU named reading.

Run python -m examples.algorithms.synthesis_source_trial for a small coherent
polynomial, or append a local .qmod path to run the admitted pinned subset.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc main(output x:qnum<2>, output result:qnum) {
    allocate(x);
    hadamard_transform(x);
    result=x*x+1;
}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    args = parser.parse_args()
    if args.source is None:
        program = synth.synthesize(
            SOURCE, source_format="qmod", source_name="polynomial.qmod", max_qubits=24
        )
    else:
        program = synth.synthesize(args.source, max_qubits=24)
    result = fq.run(program)
    widths = program.report["register_widths"]
    assert isinstance(widths, Mapping)
    distribution = program.probabilities(
        result, registers=tuple(widths), max_amplitudes=16_777_216
    )
    print(dict(distribution))
    print(program.report["resources"])


if __name__ == "__main__":
    main()
