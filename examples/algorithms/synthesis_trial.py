"""Constrained synthesis: high-level arithmetic, named preparation and named output.

Run from the repository root with PYTHONPATH=. No gate design, physical mapping
or bitstring decoding is needed in either ordinary workflow.
"""

import math

import flagquantum as fq
from flagquantum import synth


def polynomial(x: int, y: int) -> int:
    return x * x + y


@synth.qfunc
def declared_polynomial(x: int, y: int) -> int:
    return x**2 + y


def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u


def main() -> None:
    print(
        "premise: unreleased CPU synthesis profile with source-retaining Python and unsigned "
        "arithmetic; finite checked candidates, no optimality guarantee.\n"
    )
    inputs = {"x": synth.UInt(2), "y": synth.UInt(2)}
    basis = synth.synthesize(
        polynomial,
        inputs=inputs,
        input_state=synth.InputState.basis(x=3, y=2),
        max_qubits=18,
    )
    observed = basis.probabilities(
        fq.run(basis),
        registers=("x", "y", "result"),
    )
    assert set(observed) == {(3, 2, 11)}
    assert math.isclose(observed[(3, 2, 11)], 1, abs_tol=1e-10)

    coherent = synth.synthesize(
        polynomial,
        inputs=inputs,
        input_state=synth.InputState.uniform(),
        max_qubits=18,
    )
    joint = coherent.probabilities(
        fq.run(coherent),
        registers=("x", "y", "result"),
    )
    expected = {(x, y, x * x + y) for x in range(4) for y in range(4)}
    assert set(joint) == expected
    assert all(math.isclose(joint[key], 1 / 16, abs_tol=1e-10) for key in expected)

    declared = synth.synthesize(
        declared_polynomial,
        inputs=inputs,
        input_state=synth.InputState.uniform(),
        max_qubits=18,
    )
    assert declared.circuit.instructions == coherent.circuit.instructions
    routed = synth.synthesize(
        pressure,
        inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        max_qubits=9,
        topology="line",
    )
    line = routed.probabilities(fq.run(routed), registers=("x", "y", "result"))
    assert set(line) == {(x, y, pressure(x, y)) for x in range(2) for y in range(2)}
    assert all(math.isclose(p, 0.25, abs_tol=1e-10) for p in line.values())
    print("line:", dict(line))
    print("basis:", dict(observed))
    print("uniform:", dict(joint))
    print("logical resources:", coherent.report["resources"])


if __name__ == "__main__":
    main()
