"""Save and inspect a CPU Bell experiment without replay during archive I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="New .fqx destination")
    args = parser.parse_args()
    program = fq.compile(fq.Circuit(2).h(0).cx(0, 1))
    result = fq.run(
        program,
        outputs=fq.counts(),
        options=fq.ExecutionOptions(
            device="cpu", backend="pytorch", shots=128, seed=42
        ),
    )
    fq.save(args.path, program, result=result)
    experiment = fq.load(args.path)
    assert experiment.result is not None
    assert experiment.result.counts == result.counts
    assert sum(experiment.result.counts[0].values()) == 128
    print("Recorded Bell counts preserved; total shots: 128")


if __name__ == "__main__":
    main()
