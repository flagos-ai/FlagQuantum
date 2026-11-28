"""Compare CPU sampling of a signed Pauli SELECT with 32 shots."""

import argparse
import json
from importlib.metadata import version
from statistics import median
from time import perf_counter_ns

try:
    import cirq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Install the optional benchmark dependencies listed in README.md."
    ) from exc

import torch
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU as Reference

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.algorithms.primitives.multiplexers.pauli_select import (
    lower_select_pauli_lcu_to_unary_iteration,
)
from flagquantum.runtime.dynamic.execution import run_dynamic


def measure(entries, backend, strategy, shots):
    index = entries.bit_length() - 1
    width = index + 2
    words = ("XI", "-YZ", "IX", "ZY") * (entries // 4)
    samples = []
    for repeat in range(6):
        start = perf_counter_ns()
        if backend == "flagquantum":
            program = fq.Circuit(width)
            op = SelectPauliLCU(index, 2, words).make_on(
                selection=range(index), target=range(index, width)
            )
            program.append(op)
            instruction = program.to_ir().instructions[0]
            bound = perf_counter_ns()
            lowered = lower_select_pauli_lcu_to_unary_iteration(instruction)
            decomposed = perf_counter_ns()
            circuit = type(lowered)(lowered.n_qubits, dtype=torch.complex128)
            for q in range(width):
                circuit.h(q).s(q)
            circuit.compose(lowered)
            ready = perf_counter_ns()
            result = run_dynamic(circuit, shots=shots, seed=42, strategy=strategy)
            finished = perf_counter_ns()
            assert result.final_samples.shape == (shots, circuit.n_qubits)
            assert not result.final_samples[:, width:].any()
        else:
            q = cirq.LineQubit.range(width)
            reference = Reference(
                index,
                2,
                tuple(
                    cirq.DensePauliString(
                        word.removeprefix("-"),
                        coefficient=-1 if word.startswith("-") else 1,
                    )
                    for word in words
                ),
            )
            operation = reference.on_registers(selection=q[:index], target=q[index:])
            bound = perf_counter_ns()
            context = cirq.DecompositionContext(
                cirq.GreedyQubitManager(prefix="work", maximize_reuse=True)
            )
            gates = cirq.Circuit(cirq.decompose(operation, context=context))
            decomposed = perf_counter_ns()
            work = sorted(gates.all_qubits().difference(q))
            circuit = cirq.Circuit(cirq.H.on_each(*q), cirq.S.on_each(*q))
            circuit += gates
            circuit.append(cirq.measure(*q, *work, key="out"))
            simulator = cirq.Simulator(dtype="complex128", seed=42)
            ready = perf_counter_ns()
            result = simulator.run(circuit, repetitions=shots)
            finished = perf_counter_ns()
            assert result.records["out"][:, 0, :].shape == (shots, width + len(work))
            assert not result.records["out"][:, 0, :][:, width:].any()
        if repeat:
            samples.append(
                tuple(
                    (b - a) / 1e6
                    for a, b in (
                        (start, bound),
                        (bound, decomposed),
                        (decomposed, ready),
                        (ready, finished),
                    )
                )
            )
    return dict(
        entries=entries,
        backend=backend,
        strategy=strategy,
        shots=shots,
        **dict(
            zip(
                ("bind_ms", "decompose_ms", "setup_ms", "sample_ms"),
                map(median, zip(*samples, strict=True)),
                strict=True,
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=32)
    args = parser.parse_args()
    assert version("qualtran") == "0.7.0"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(
        json.dumps(
            {
                "qualtran": version("qualtran"),
                "cirq": cirq.__version__,
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )
    for entries in (4, 8, 16):
        for backend, strategy in (
            ("flagquantum", "trajectory"),
            ("flagquantum", "batched"),
            ("qualtran", "cirq"),
        ):
            print(
                json.dumps(measure(entries, backend, strategy, args.shots)), flush=True
            )
