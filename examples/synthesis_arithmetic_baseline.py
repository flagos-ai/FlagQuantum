"""Read named results and authoritative costs without designing an adder."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import estimate_resources


def energy(x: int, y: int) -> int:
    return x * x + y


QMOD = """qfunc main(output x:qnum<2>, output y:qnum<2>, output result:qnum) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    result=x*x+y;
}"""


def main() -> None:
    program = synth.synthesize(
        energy,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        memory_limit_bytes=64 * 1024**2,
    )
    result = fq.run(program)
    outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
    expected = {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    assert outcomes.keys() == expected.keys()
    assert all(abs(outcomes[key] - value) < 1e-12 for key, value in expected.items())
    resource = estimate_resources(program.circuit)
    print(outcomes)
    print(resource.to_dict())
    print("Workspace:", program.report["workspace_qubits"])
    print("Planned complex128 state buffer:", program.execution.state_bytes, "bytes")
    print("Counts describe this generated circuit; they are not runtime or peak RSS.")


if __name__ == "__main__":
    main()
