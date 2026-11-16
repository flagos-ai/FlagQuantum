"""Compare Qmod and native qfunc against the independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.algorithms.phase_estimation.static_qpe.main import estimate
from flagquantum import synth

QMOD = Path(__file__).with_name("source.qmod").read_text(encoding="utf-8")


def main() -> None:
    reference = fq.Circuit(2, dtype=torch.complex128).x(1).h(0)
    reference.any(
        0, 1, matrix=torch.diag(torch.tensor([1, 1, 1, 1j], dtype=torch.complex128))
    ).h(0)
    expected = fq.run(reference).statevector().reshape(-1)
    programs = (
        synth.synthesize(estimate, memory_limit_bytes=1024**2),
        synth.synthesize(QMOD, source_format="qmod", memory_limit_bytes=1024**2),
    )
    for program in programs:
        result = fq.run(program)
        state = result.statevector().reshape(4, -1)
        torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
        assert state[:, 1:].abs().square().sum() < 1e-24
        print(dict(program.probabilities(result, registers=("counter",))))
        print(program.report["resources"])


if __name__ == "__main__":
    main()
