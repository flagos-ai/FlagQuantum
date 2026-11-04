"""Time the same native dynamic SELECT gate sequence in two executors."""

import json
from statistics import median
from time import perf_counter_ns

import cirq
import numpy as np
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.compiler.pauli_select import lower_select_pauli_lcu_to_unary_iteration
from flagquantum.runtime.dynamic.execution import run_dynamic


def convert(program):
    """Benchmark-only conversion of the gates used by unary SELECT."""
    q = cirq.LineQubit.range(program.n_qubits)
    gates = dict(
        x=cirq.X,
        y=cirq.Y,
        z=cirq.Z,
        h=cirq.H,
        s=cirq.S,
        t=cirq.T,
        tdg=cirq.T**-1,
        cx=cirq.CNOT,
        cy=cirq.Y.controlled(),
        cz=cirq.CZ,
    )
    operations = []
    for instruction in program.to_ir().instructions:
        name = instruction.name
        operands = [q[i] for i in instruction.wires]
        if name == "measure":
            op = cirq.measure(
                *operands, key=f"bit{instruction.metadata['classical_bit']}"
            )
        elif name == "reset":
            op = cirq.reset(*operands)
        elif name == "ry":
            op = cirq.ry(float(instruction.params["theta"])).on(*operands)
        else:
            op = gates[name].on(*operands)
        if instruction.metadata.get("conditions"):
            assert len(instruction.metadata["conditions"]) == 1
            bit, value = instruction.metadata["conditions"][0]
            assert value == 1
            op = op.with_classical_controls(f"bit{bit}")
        operations.append(op)
    assert len(operations) == len(program.to_ir().instructions)
    # Preserve one instruction per moment, including repeated measurement keys.
    return cirq.Circuit(cirq.Moment([op]) for op in operations), q


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(
    json.dumps(
        dict(
            cirq=cirq.__version__,
            torch=torch.__version__,
            repeats=3,
            warmups=1,
            threads=1,
        )
    ),
    flush=True,
)
for entries in (4, 8, 16):
    bits = entries.bit_length() - 1
    width = bits + 2
    words = ("XI", "-YZ", "IX", "ZY") * (entries // 4)
    start = perf_counter_ns()
    source = fq.Circuit(width)
    source.append(
        SelectPauliLCU(bits, 2, words).make_on(
            selection=range(bits), target=range(bits, width)
        )
    )
    unary = lower_select_pauli_lcu_to_unary_iteration(source.to_ir().instructions[0])
    program = type(unary)(unary.n_qubits, dtype=torch.complex128)
    static = fq.Circuit(unary.n_qubits, dtype=torch.complex128)
    for i in range(width):
        theta = 0.37 + i * 0.13
        program.ry(i, theta=theta).s(i)
        static.ry(i, theta=theta).s(i)
    program.compose(unary)
    static.compose(source)
    build_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    converted, q = convert(program)
    conversion_ms = (perf_counter_ns() - start) / 1e6
    reference = static.state().numpy().ravel()
    # Every corrected measurement branch must preserve the same complex state.
    for seed in (13, 42, 99):
        actual = (
            cirq.Simulator(dtype=np.complex128, seed=seed)
            .simulate(converted, qubit_order=q)
            .final_state_vector
        )
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=0)

    sampled = converted + cirq.Circuit(cirq.measure(*q, key="out"))
    print(
        json.dumps(
            dict(
                entries=entries,
                build_ms=build_ms,
                conversion_ms=conversion_ms,
                instructions=len(program.to_ir().instructions),
                parity="passed_three_seeds",
            )
        ),
        flush=True,
    )
    for shots in (32, 256, 1024):
        for engine in ("trajectory", "batched", "cirq"):
            times = []
            for repeat in range(4):
                simulator = cirq.Simulator(dtype=np.complex128, seed=42)
                start = perf_counter_ns()
                if engine == "cirq":
                    samples = simulator.run(sampled, repetitions=shots).records["out"][
                        :, 0, :
                    ]
                    assert not samples[:, width:].any()
                else:
                    samples = run_dynamic(
                        program, shots=shots, seed=42, strategy=engine
                    ).final_samples
                    assert not samples[:, width:].any()
                elapsed = (perf_counter_ns() - start) / 1e6
                if repeat:
                    times.append(elapsed)
            print(
                json.dumps(
                    dict(
                        entries=entries,
                        shots=shots,
                        engine=engine,
                        sample_ms=median(times),
                    )
                ),
                flush=True,
            )
