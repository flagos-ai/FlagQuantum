"""Inspect amplitudes as well as probabilities for a constant phase gate."""

import flagquantum as fq
from flagquantum import synth


def main() -> None:
    program = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);H(q);PHASE(1,q);}",
        source_format="qmod",
        max_qubits=1,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("q",))))
    print(result.statevector())
    print(program.report["resources"])


if __name__ == "__main__":
    main()
