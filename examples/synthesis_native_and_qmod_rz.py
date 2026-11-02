"""Write native phase intent first; compare migration and gate-level baselines."""

import torch

import flagquantum as fq
from flagquantum import synth


def phase(q: int) -> float:
    return 1.9 * (q - 0.5)


def main() -> None:
    program = synth.synthesize(
        phase,
        inputs={"q": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=1,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    result = fq.run(program)
    print(
        "Native high-level probabilities:",
        dict(program.probabilities(result, registers=("q",))),
    )
    print("Native high-level amplitudes:", result.statevector())
    print(program.report["resources"])

    migrated = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);H(q);RZ(1.9,q);}",
        source_format="qmod",
        memory_limit_bytes=1024**2,
    )
    gate_baseline = fq.Circuit(1, dtype=torch.complex128).h(0).rz(0, theta=1.9)
    for reference in (fq.run(migrated), fq.run(gate_baseline)):
        torch.testing.assert_close(
            result.statevector(), reference.statevector(), atol=1e-12, rtol=0
        )
    print("Native intent, Qmod migration and gate baseline agree.")


if __name__ == "__main__":
    main()
