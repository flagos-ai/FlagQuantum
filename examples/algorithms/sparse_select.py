"""Run a sparse chemistry SELECT and its gate-level decomposition."""

import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectSparse
from flagquantum.compiler.select_lowering import lower_select_program


def main() -> None:
    circuit = fq.Circuit(11, dtype=torch.complex128)
    circuit.h(0).h(4).h(6).x(8)
    circuit.append(
        SelectSparse.make_on(
            p=(0,),
            q=(1,),
            r=(2,),
            s=(3,),
            alpha=4,
            beta=5,
            flag_1b=6,
            sys=(7, 8, 9, 10),
        )
    )

    sampled = fq.run(
        circuit, options=fq.ExecutionOptions(target="samples", shots=16, seed=5)
    )
    assert sampled.plan is not None
    report = sampled.plan.summary()["select_lowering"]
    print(f"Sampling path: {report['operations'][0]['path']}")
    print(f"Sampling workspace: {report['workspace_qubits']} qubits")
    print()
    print("premise: The dynamic path requires mid-circuit measurement and feedback.")
    print()

    # Target compilation uses this same unitary lowering before gate legalization.
    gate_ir = lower_select_program(circuit.to_ir(), "unitary_ladder")
    gate_circuit = fq.Circuit.from_ir(gate_ir)
    output = gate_circuit.state().reshape(1 << circuit.n_qubits, -1)
    torch.testing.assert_close(output[:, 0], circuit.state().reshape(-1))
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))
    print(
        f"Gate-level SELECT verified with {gate_ir.n_wires - circuit.n_qubits} work qubit."
    )


if __name__ == "__main__":
    main()
