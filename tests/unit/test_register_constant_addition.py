"""Unsigned constant addition against pinned source gates and complex trajectories."""

from collections import Counter
from importlib.metadata import version
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic import add_constant_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit
CASES = [
    (n, k) for n in (1, 2, 3, 5, 8) for k in (0, 1, (1 << n) - 1, -1, -(1 << n) - 3)
]


def layout(width: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        tuple(range(2 * width, 3 * width - 1)),
        tuple(range(40, 40 + width - 1)),
    )


def source_gates(
    width: int, value: int, *, source: Any = None
) -> tuple[tuple[Instruction, ...], Any]:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import AddK

    target, constant, carries, bits = layout(width)
    source = AddK(QUInt(width), value) if source is None else source
    return (
        reference_arithmetic_gates(
            source,
            {"x": target},
            (*constant, *carries),
            dict(zip(carries, bits, strict=True)),
        ),
        source,
    )


def reference_arithmetic_gates(
    source: Any,
    inputs: dict[str, tuple[int, ...]],
    allocations: tuple[int, ...],
    measurement_slots: dict[int, int],
    *,
    cirq_style: bool = False,
    expected_outputs: dict[str, tuple[int, ...]] | None = None,
    reuse_allocations: bool = True,
) -> tuple[Instruction, ...]:
    """Read unmodified pinned arithmetic with explicit quantum/classical binding."""
    pytest.importorskip("qualtran")
    if version("qualtran") != "0.7.0":
        pytest.skip("Reference gate stream requires Qualtran v0.7.0")
    import cirq
    import numpy as np
    from qualtran import CompositeBloq
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.cirq_interop import BloqAsCirqGate

    class Manager(cirq.QubitManager):
        def __init__(self) -> None:
            self.available = list(reversed(allocations))
            self.rank = {wire: i for i, wire in enumerate(allocations)}

        def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            assert dim == 2
            return [cirq.LineQubit(self.available.pop()) for _ in range(n)]

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            return self.qalloc(n, dim)

        def qfree(self, qubits: list[cirq.Qid]) -> None:
            if not reuse_allocations:
                # Preserve dedicated caller output names across disjoint lifetimes.
                return
            self.available.extend(reversed([q.x for q in qubits]))
            if cirq_style:
                # Bloq-to-Cirq may temporarily allocate replacement RIGHT outputs
                # before mapping them onto supplied wires. Recycling these names
                # must not reorder subsequent source-generator allocations.
                self.available = sorted(
                    set(self.available), key=self.rank.__getitem__, reverse=True
                )

    manager = Manager()
    context = cirq.DecompositionContext(manager)
    if cirq_style and hasattr(source, "decompose_from_registers"):
        # Retain original allocation/generator order instead of a DAG export that
        # may reorder independent allocations and change their temporary names.
        quregs = {
            name: np.array([cirq.LineQubit(w) for w in wires])
            for name, wires in inputs.items()
        }
        operation = source.decompose_from_registers(context=context, **quregs)
    else:
        composite = (
            source if isinstance(source, CompositeBloq) else source.decompose_bloq()
        )
        operation, outputs = (
            composite if cirq_style else composite.flatten()
        ).to_cirq_circuit_and_quregs(
            qubit_manager=manager,
            **{
                name: np.array([cirq.LineQubit(w) for w in wires]).reshape(
                    (
                        *source.signature.get_left(name).shape,
                        int(source.signature.get_left(name).bitsize),
                    )
                )
                for name, wires in inputs.items()
            },
        )
        assert {
            name: tuple(q.x for q in wires.flat) for name, wires in outputs.items()
        } == (inputs if expected_outputs is None else expected_outputs)
    names = {
        cirq.X: "x",
        cirq.H: "h",
        cirq.T: "t",
        cirq.T**-1: "tdg",
        cirq.S: "s",
        cirq.CNOT: "cx",
        cirq.CZ: "cz",
        BloqAsCirqGate(CNOT()): "cx",
        BloqAsCirqGate(XGate()): "x",
    }
    upstream = cirq.decompose(
        operation,
        context=context,
        keep=lambda op: op.gate in names
        or cirq.is_measurement(op)
        or isinstance(op.gate, cirq.ResetChannel)
        or isinstance(op, cirq.ClassicallyControlledOperation),
    )
    measurements = {}
    gates = []
    for op in upstream:
        metadata = {}
        if isinstance(op, cirq.ClassicallyControlledOperation):
            (key,) = cirq.control_keys(op)
            metadata["conditions"] = ((measurements[key], 1),)
            op = op.without_classical_controls()
        wires = tuple(q.x for q in op.qubits)
        if cirq.is_measurement(op):
            bit = measurement_slots[wires[0]]
            measurements[cirq.measurement_key_obj(op)] = bit
            metadata = {"is_dynamic": True, "classical_bit": bit}
            name = "measure"
        elif isinstance(op.gate, cirq.ResetChannel):
            metadata = {"is_dynamic": True}
            name = "reset"
        else:
            name = names[op.gate]
        gates.append(Instruction(name, wires, metadata=metadata))
    return tuple(gates)


@pytest.mark.parametrize("width,value", CASES)
def test_literal_source_gates_classical_conditions_and_resources(
    width: int, value: int
) -> None:
    expected, source = source_gates(width, value)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    target, constant, carries, bits = layout(width)
    actual = add_constant_ir(target, value, constant, carries, bits)

    def key(g: Instruction) -> tuple[Any, ...]:
        return (g.name, g.wires, tuple(sorted(g.metadata.items())))

    # The source DAG may commute disjoint operations, but each wire sees the same stream.
    assert Counter(map(key, actual)) == Counter(map(key, expected))
    for wire in (*target, *constant, *carries):
        assert [key(g) for g in actual if wire in g.wires] == [
            key(g) for g in expected if wire in g.wires
        ]
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1
    assert counts["reset"] == width - 1
    if width == 1:
        # The pinned model uses the general Add formula even at width one.
        assert model.clifford == 2 * (value % 2) - 3
        assert counts["x"] + counts["cx"] == 2 * (value % 2) + 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "width,value", [(n, k) for n in (1, 2, 3, 4) for k in (0, 1, (1 << n) - 1, -3)]
)
def test_append_complex_amplitudes_every_input_and_clean_work(
    width: int, value: int
) -> None:
    target, constant, carries, bits = layout(width)
    gates = add_constant_ir(target, value, constant, carries, bits)
    work_count = len(constant) + len(carries)
    n_qubits = width + work_count
    inputs = torch.arange(1 << width, dtype=torch.float64)
    amplitudes = torch.complex(inputs + 1, 2 * inputs - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[:: 1 << work_count] = amplitudes
    expected = torch.zeros_like(state)
    mapped = (torch.arange(1 << width) + value) % (1 << width)
    expected[mapped << work_count] = amplitudes
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), expected.expand(64, -1), rtol=0, atol=1e-12
    )
    if width > 1:
        assert len(
            set(map(tuple, result.classical_bits[:, 40 : 40 + width - 1].tolist()))
        ) == 1 << (width - 1)


def test_invalid_constant_addition_inputs() -> None:
    target, constant, carries, bits = layout(3)
    for value in (8, True):
        with pytest.raises(ValueError, match="Additive constant"):
            add_constant_ir(target, value, constant, carries, bits)
    with pytest.raises(ValueError, match="equal nonempty"):
        add_constant_ir((), 0, (), (), ())
    with pytest.raises(ValueError, match="equal nonempty"):
        add_constant_ir(target, 1, constant[:-1], carries, bits)
    with pytest.raises(ValueError, match="distinct nonnegative"):
        add_constant_ir(target, 1, target, carries, bits)
    with pytest.raises(ValueError, match="classical slots"):
        add_constant_ir(target, 1, constant, carries, (-1, 2))
    with pytest.raises(ValueError, match="clean carry"):
        add_constant_ir(target, 1, constant, (), ())
    with pytest.raises(ValueError, match="distinct classical"):
        add_constant_ir(target, 1, constant, carries, (1, 1))
