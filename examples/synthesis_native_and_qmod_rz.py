"""Compare current high-level arithmetic, Qmod migration and a gate reference."""

import cmath

import torch

import flagquantum as fq
from flagquantum import synth

ARITHMETIC_QMOD = """qfunc energy(const x:qnum<2>,const y:qnum<2>,output z:qnum) {
    z=x*x+y;
}
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}"""


def energy(x: int, y: int) -> int:
    return x * x + y


def main() -> None:
    program = synth.synthesize(
        energy,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        max_qubits=24,
        memory_limit_bytes=2**30,
    )
    result = fq.run(program)
    actual = dict(program.probabilities(result, registers=("x", "y", "result")))
    expected = {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    assert actual.keys() == expected.keys()
    assert all(abs(actual[key] - value) < 1e-12 for key, value in expected.items())
    print("Native high-level (x, y, result):", actual)
    print("Native resources:", program.report["resources"])

    migrated = synth.synthesize(
        ARITHMETIC_QMOD, source_format="qmod", memory_limit_bytes=2**30
    )
    imported = dict(
        migrated.probabilities(fq.run(migrated), registers=("x", "y", "result"))
    )
    assert imported.keys() == actual.keys()
    assert all(abs(imported[key] - value) < 1e-12 for key, value in actual.items())
    print("Qmod high-level arithmetic agrees on all 16 named outcomes.")

    # This coherent variant adds H; the pinned original RZ.qmod has no H.
    rotation = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);H(q);RZ(1.9,q);}",
        source_format="qmod",
        memory_limit_bytes=1024**2,
    )
    state = fq.run(rotation).statevector().reshape(-1)
    reference = fq.Circuit(1, dtype=torch.complex128).h(0).rz(0, theta=1.9)
    oracle = (
        torch.tensor([cmath.exp(-0.95j), cmath.exp(0.95j)], dtype=torch.complex128)
        / 2**0.5
    )
    torch.testing.assert_close(state, oracle, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        state, fq.run(reference).statevector().reshape(-1), atol=1e-12, rtol=0
    )
    print("Qmod RZ, Circuit reference and absolute phase oracle agree.")
    print("RZ resources:", rotation.report["resources"])
    print("Unified native qfunc programs with explicit real phase remain a gap.")


if __name__ == "__main__":
    main()
