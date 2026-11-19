"""Retain the Grover workshop's uniform state, two drafts and two solutions."""

from typing import Annotated

import flagquantum as fq
from examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.main import (
    main as solution,
)
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    hadamard_transform,
    qfunc,
)


@qfunc
def uniform(x: Output[QArray[QBit]]) -> None:
    allocate(5, x)
    hadamard_transform(x)


@qfunc
def draft(
    a: Output[Annotated[QNum, 2]],
    b: Output[Annotated[QNum, 2]],
    c: Output[Annotated[QNum, 3]],
) -> None:
    allocate(a)
    allocate(b)
    allocate(c)


def run_workflow() -> dict[str, synth.Program]:
    programs = {"uniform": synth.synthesize(uniform, max_workspace=0)}
    for name, callback in (("draft", draft), ("solution", solution)):
        for objective in ("width", "depth"):
            programs[f"{name}_{objective}"] = synth.synthesize(
                callback, objective=objective, max_workspace=0
            )
    for name, program in programs.items():
        registers = ("x",) if name == "uniform" else ("a", "b", "c")
        print(name, dict(program.probabilities(fq.run(program), registers=registers)))
        print("Resources:", program.report["resources"])
    return programs


def run_example() -> None:
    run_workflow()


if __name__ == "__main__":
    run_example()
