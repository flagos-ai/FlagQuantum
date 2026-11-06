"""Original angle-array generator, exact phase and measured QROM cleanup."""

from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.programmable_rotation_array import (
    programmable_rotation_array_ir,
)
from flagquantum.compiler.registers.source_qrom import source_qrom_workspace_size
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "angles,kappa,gate_name",
    [
        (((1, 3, 2), (4, 0, 3)), 2, "X"),
        (((3, 0, 1, 2), (1, 2, 3, 0), (0, 1, 2, 1)), 3, "Z"),
        (((1,), (2,)), 2, "X"),
        (((1, 2, 3), (4, 0, 1)), 2, "CZ"),
        (((0, 0), (1, 2)), 2, "X"),
        (((1, 2), (0, 0), (1, 2)), 2, "Z"),
        (((0, 0), (0, 0)), 3, "Z"),
    ],
)
def test_pinned_generator_and_complex_branches(angles, kappa, gate_name):
    import cirq
    import numpy as np
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
        ProgrammableRotationGateArray,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    extent = len(angles[0])
    select = tuple(range((extent - 1).bit_length()))
    load = tuple(range(len(select), len(select) + kappa))
    base = getattr(cirq, gate_name)
    target = tuple(
        range(len(select) + kappa, len(select) + kappa + cirq.num_qubits(base))
    )
    candidates = (cirq.CNOT, cirq.SWAP) if len(target) == 2 else (cirq.H, cirq.Y)
    interleavers = candidates[: len(angles) - 1]
    source = ProgrammableRotationGateArray(
        *angles,
        kappa=kappa,
        rotation_gate=base,
        interleaved_unitaries=interleavers,
    )

    def qubits(wires):
        return np.array([cirq.LineQubit(wire) for wire in wires], dtype=object)

    def matrix_gate(gate, wires):
        matrix = cirq.unitary(gate)
        return Instruction("any", wires, matrix=matrix.tolist())

    source_ops = tuple(
        source.decompose_from_registers(
            context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
            selection=qubits(select),
            kappa_load_target=qubits(load),
            rotations_target=qubits(target),
        )
    )
    qroms = tuple(op.gate for op in source_ops if isinstance(op.gate, QROM))
    sizes = tuple(
        source_qrom_workspace_size(0, (len(select),), (tuple(map(int, qrom.data[0])),))
        for qrom in qroms
    )
    work = tuple(range(target[-1] + 1, target[-1] + 1 + max(sizes)))
    bits = tuple(range(len(work)))
    gates = programmable_rotation_array_ir(
        select,
        load,
        target,
        angles,
        work,
        bits,
        rotation=lambda control, wires, power: (
            matrix_gate(
                cirq.ControlledGate(base ** (2.0 ** (-1 - power))), (control, *wires)
            ),
        ),
        interleaved=tuple((matrix_gate(gate, target),) for gate in interleavers),
    )
    ref = []
    qrom_index = 0
    for op in source_ops:
        if isinstance(op.gate, QROM):
            size = sizes[qrom_index]
            qrom_index += 1
            ref.extend(
                source_qrom_gates(
                    op.gate,
                    ({"selection": select} if select else {}) | {"target0_": load},
                    work[:size],
                    bits[:size],
                )
            )
        else:
            ref.append(matrix_gate(op.gate, tuple(q.x for q in op.qubits)))
    assert_source(gates, tuple(ref))
    actual_matrices = [g.matrix for g in gates if g.name == "any"]
    expected_matrices = [g.matrix for g in ref if g.name == "any"]
    for actual, expected_matrix in zip(actual_matrices, expected_matrices, strict=True):
        np.testing.assert_allclose(actual, expected_matrix, atol=2e-12, rtol=0)
    measured = sum(g.name == "measure" for g in gates)
    counts = Counter(g.name for g in gates)
    model = sum(get_cost_value(qrom, QECGatesCost()).and_bloq for qrom in qroms)
    assert counts["t"] + counts["tdg"] == 4 * measured == 4 * model

    # Evaluate source macros on every valid address and preserve cross-address phase.
    width = target[-1] + 1 + len(work) + 1
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for address, spectator in product(range(extent), (0, 1)):
        index = (address << (width - len(select))) | spectator
        dimension = 1 << len(target)
        indices = [index | (j << (len(work) + 1)) for j in range(dimension)]
        state = np.array(
            [1 + (0.6j if spectator else -0.8j) * j for j in range(dimension)]
        ) * np.exp(0.2j * address)
        initial[indices] = torch.as_tensor(state)
        loaded = 0
        for op in source_ops:
            if isinstance(op.gate, QROM):
                loaded ^= int(op.gate.data[0][address])
            elif op.qubits[0].x not in load:
                state = cirq.unitary(op.gate) @ state
            elif loaded & (1 << (kappa - 1 - load.index(op.qubits[0].x))):
                state = cirq.unitary(op.gate)[dimension:, dimension:] @ state
        assert loaded == 0
        expected[indices] = torch.as_tensor(state)
    norm = initial.norm()
    for history in product((0, 1), repeat=measured):
        torch.testing.assert_close(
            forced_branch(initial / norm, gates, history),
            expected / norm / sqrt(2**measured),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"angles": ((),)}, "unsigned angle rows"),
        ({"angles": ((1, -1),)}, "unsigned angle rows"),
        ({"selection": ()}, "source selection width"),
        ({"interleaved": ((),)}, "one interleaver"),
        ({"target": (0,)}, "distinct nonnegative"),
        ({"workspace": (3,), "classical_bits": ()}, "classical slot"),
        (
            {"angles": ((0, 1), (1, 0)), "interleaved": ((Instruction("z", (0,)),),)},
            "interleaver used a wire",
        ),
        (
            {"angles": ((0, 1, 2),), "selection": (0, 3)},
            "clean work wires",
        ),
        (
            {"rotation": lambda *_: (Instruction("z", (0,)),)},
            "outside its control",
        ),
    ],
)
def test_array_rejects_invalid_bindings(overrides, error):
    arguments = dict(
        selection=(0,),
        load=(1,),
        target=(2,),
        angles=((0, 1),),
        workspace=(),
        classical_bits=(),
        rotation=lambda *_: (),
        interleaved=(),
    )
    with pytest.raises(ValueError, match=error):
        programmable_rotation_array_ir(**(arguments | overrides))


def test_append_executes_native_controlled_phase_and_measured_lookup() -> None:
    from dataclasses import dataclass
    from math import pi

    from flagquantum.core.ir import CircuitIR
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    @dataclass(frozen=True)
    class ApplyArray:
        def append_to(self, circuit: DynamicCircuit) -> DynamicCircuit:
            instructions = programmable_rotation_array_ir(
                (0, 1),
                (2,),
                (3,),
                ((0, 1, 1),),
                (4, 5),
                (0, 1),
                rotation=lambda control, target, power: (
                    Instruction(
                        "cphase", (control, *target), {"theta": pi / 2 ** (power + 1)}
                    ),
                ),
                interleaved=(),
            )
            circuit.compose(CircuitIR(6, instructions))
            return circuit

    initial = torch.zeros(64, dtype=torch.complex128)
    initial[4], initial[20], initial[36] = (1 / sqrt(3),) * 3
    expected = initial.clone()
    expected[[20, 36]] *= 1j
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(ApplyArray())
    result = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states,
        expected.expand(8, -1),
        atol=2e-12,
        rtol=0,
    )
