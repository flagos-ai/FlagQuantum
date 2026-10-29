"""Private CS-04 trial: high-level arithmetic, named preparation and named output.

Run from the repository root with PYTHONPATH=. No gate design, physical mapping
or bitstring decoding is needed in either ordinary workflow.
"""

import math

import flagquantum as fq
from flagquantum.services import _synthesis as synth


def polynomial(x: int, y: int) -> int:
    return x * x + y


@synth.qfunc
def declared_polynomial(x: int, y: int) -> int:
    return x**2 + y


def main() -> None:
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    inputs = {"x": synth.UInt(2), "y": synth.UInt(2)}
    basis = synth.synthesize(
        polynomial,
        inputs=inputs,
        input_state=synth.InputState.basis(x=3, y=2),
        max_qubits=18,
    )
    observed = basis.read_distribution(
        fq.run(basis.circuit, options=options),
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
    joint = coherent.read_distribution(
        fq.run(coherent.circuit, options=options),
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
    print("basis:", dict(observed))
    print("uniform:", dict(joint))
    print("logical resources:", coherent.report["resources"])


if __name__ == "__main__":
    main()
