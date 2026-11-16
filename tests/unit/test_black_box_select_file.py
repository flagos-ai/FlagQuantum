"""Frozen BlackBoxSelect and SelectOracle layout and execution correspondence."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha1
from importlib.metadata import version
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol

from examples.algorithms.black_box_select_catalog import PackedHubbardSelectExample
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.layout.select_partition import packed_select_layout
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.multiplexers import black_box_select

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    directory = Path(black_box_select.__file__).parent
    for name, sha in (
        ("black_box_select.py", "8cb8d402b17f17952aa3f06b3fca8ed4cc048239"),
        ("black_box_select_test.py", "6bd3dd68d3adef743a4ae6990180951c40d928d0"),
        ("select_base.py", "dc72bbc1a41b7cf499b1273f1269984289c48672"),
    ):
        data = (directory / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == sha
    return black_box_select


def _renumber_measurements(stream: tuple[Instruction, ...]) -> tuple[Instruction, ...]:
    """Align reused classical slots by source measurement occurrence."""
    mapping: dict[int, int] = {}
    result = []
    events = count()
    for op in stream:
        metadata = dict(op.metadata)
        if op.name == "measure":
            measurement = next(events)
            mapping[metadata["classical_bit"]] = measurement
            metadata["classical_bit"] = measurement
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (mapping[slot], value) for slot, value in metadata["conditions"]
            )
        result.append(Instruction(op.name, op.wires, op.params, op.matrix, metadata))
    return tuple(result)


def test_original_hubbard_wrapper_layout_stream_phase_and_resources(
    source: Any,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = source._black_box_select.make()
    op = PackedHubbardSelectExample()
    assert str(original) == "SELECT"
    assert [(r.name, r.bitsize, r.side.value) for r in op.signature] == [
        (r.name, r.bitsize, r.side.value) for r in original.signature
    ]
    ((instance, inputs, outputs),) = original.decompose_bloq().iter_bloqsoqs()
    partition = instance.bloq
    assert type(partition).__name__ == "AutoPartition"
    assert partition.bloq is original.select and not partition.left_only
    assert tuple(inputs) == ("selection", "system") and len(outputs) == 2
    assert [(reg.name, names) for reg, names in partition.partitions] == [
        ("selection", tuple(reg.name for reg in original.select.selection_registers)),
        ("system", ("target",)),
    ]

    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    gates = circuit.to_ir().instructions
    ports: dict[str, tuple[int, ...]] = {
        name: (i,) for i, name in enumerate(partition.partitions[0][1])
    }
    ports["target"] = tuple(range(8, 16))
    work = tuple(range(16, op.qubit_count))
    expected = reference_arithmetic_gates(
        original.select,
        ports,
        work,
        dict(zip(work, work, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
        preserve_components=True,
        scalar_phase_anchor=0,
    )
    # The Majorana routines reserve a parity carrier first. The source Cirq
    # allocator returns that carrier after the two predicate wires. All three
    # are returned before the final spin-up traversal, where allocation agrees.
    phase_start = next(i for i, gate in enumerate(expected) if gate.name == "phase") - 1
    assert (
        phase_start
        == next(i for i, gate in enumerate(gates) if gate.name == "phase") - 1
    )
    prefix, tail = expected[:phase_start], expected[phase_start:]
    rename = dict(zip(work, (*work[1:], work[0]), strict=True))
    remapped = (
        *(
            Instruction(
                g.name,
                tuple(rename.get(w, w) for w in g.wires),
                g.params,
                g.matrix,
                g.metadata,
            )
            for g in prefix
        ),
        *tail,
    )
    assert_source_dependencies(
        _renumber_measurements(gates), _renumber_measurements(remapped)
    )
    native = estimate_component_resources(SelectHubbard(2).source_resource_model())
    cost = get_cost_value(original, QECGatesCost())
    assert {k: v for k, v in asdict(native.gates).items() if v} == cost.asdict()
    assert (
        native.peak_logical_qubits
        == get_cost_value(original, QubitCount())
        == op.qubit_count
    )
    counts = Counter(g.name for g in gates)
    assert (counts["t"] + counts["tdg"], counts["cx"], counts["measure"]) == (
        102,
        177,
        15,
    )

    state = {
        ((index << 8) | ((index * 37 + word * 53) % 256))
        << 3: complex(index + 1, word - 2)
        for index in range(256)
        for word in range(2)
    }
    norm = sum(abs(v) ** 2 for v in state.values()) ** 0.5
    state = {k: v / norm for k, v in state.items()}
    histories = (tuple((i >> shift) & 1 for i in range(15)) for shift in range(4))
    for history in histories:
        output = _branch(
            gates, op.qubit_count, history, dict(state), normalize_measurements=True
        )
        reference = _branch(
            expected, op.qubit_count, history, dict(state), normalize_measurements=True
        )
        assert all(address & 7 == 0 for address in output)
        assert all(
            abs(output.get(k, 0) - reference.get(k, 0)) < 2e-12
            for k in output.keys() | reference.keys()
        )


def test_original_toy_oracle_and_symbolic_packing(source: Any) -> None:
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.multiplexers.black_box_select_test import TestSelectOracle

    original = source.BlackBoxSelect(TestSelectOracle())
    assert original.selection_registers == (SourceRegister("selection", QAny(1)),)
    assert original.target_registers == (SourceRegister("system", QAny(5)),)
    assert tuple(r.dtype for r in original.select.signature) == (QBit(), QAny(5))
    selected = (Register("z", quantum_bit_type()),)
    target = (Register("a", opaque_register_type(5)),)
    signature, layout = packed_select_layout(selected, target)
    assert [(r.name, r.bitsize) for r in signature] == [
        (r.name, r.bitsize) for r in original.signature
    ]
    packed = {"selection": (11,), "system": (8, 7, 4, 2, 6)}
    child, unused = layout.bind(packed)
    assert child == {"z": (11,), "a": (8, 7, 4, 2, 6)}
    assert layout.restore(child, unused) == packed

    n = Symbol("n", integer=True, positive=True)
    signature, layout = packed_select_layout(
        (Register("index", opaque_register_type(n), shape=(2,)),), target
    )
    assert signature[0].bitsize == 2 * n
    with pytest.raises(ValueError, match="concrete widths"):
        layout.bind(packed)


def test_locked_control_and_non_through_packing_boundaries(source: Any) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )

    original = source.BlackBoxSelect(Source(2, 2, control_val=1))
    assert tuple(reg.name for reg in original.control_registers) == ("control",)
    assert tuple(reg.name for reg in original.signature) == ("selection", "system")
    with pytest.raises(ValueError, match="registers do not match given partitions"):
        original.decompose_bloq()
    with pytest.raises(ValueError, match="uncontrolled"):
        packed_select_layout(
            (), (), controls=(Register("control", quantum_bit_type()),)
        )
    for register in (
        Register("classical", classical_bit_type()),
        Register("output", quantum_bit_type(), side=RegisterSide.RIGHT),
    ):
        with pytest.raises(ValueError, match="quantum THRU"):
            packed_select_layout((), (register,))
