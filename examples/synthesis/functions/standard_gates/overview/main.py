"""Synthesize and execute every program in the standard-gate tutorial."""

import flagquantum as fq
from examples.synthesis.functions.standard_gates.crx.main import main as crx
from examples.synthesis.functions.standard_gates.cx.main import main as cx
from examples.synthesis.functions.standard_gates.phase.main import main as phase
from examples.synthesis.functions.standard_gates.r.main import main as r
from examples.synthesis.functions.standard_gates.rz.main import main as rz
from examples.synthesis.functions.standard_gates.rzz.main import main as rzz
from examples.synthesis.functions.standard_gates.swap.main import main as swap
from examples.synthesis.functions.standard_gates.u.main import main as u
from examples.synthesis.functions.standard_gates.x.main import main as x
from flagquantum import synth


def run_example() -> None:
    """Reuse the native qfunc definitions rather than embed external source text."""
    for name, function in (
        ("X", x),
        ("RZ", rz),
        ("R", r),
        ("PHASE", phase),
        ("RZZ", rzz),
        ("CX", cx),
        ("CRX", crx),
        ("SWAP", swap),
        ("U", u),
    ):
        program = synth.synthesize(function)
        print("Gate:", name)
        print(program.report["resources"])
        result = fq.run(program)
        print(
            "Probabilities:",
            dict(program.probabilities(result, registers=tuple(program.registers))),
        )


if __name__ == "__main__":
    run_example()
