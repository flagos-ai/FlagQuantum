"""Compare typed ECC stages with already source-validated physical stages."""

import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from unittest.mock import patch

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register


def pack(
    values: Mapping[str, int], ports: Mapping[str, tuple[int, ...]], width: int
) -> int:
    return sum(
        ((values.get(name, 0) >> (len(wires) - i - 1)) & 1) << (width - wire - 1)
        for name, wires in ports.items()
        for i, wire in enumerate(wires)
    )


def check_stage_channel(
    graph: ProgramRecord,
    original: CircuitIR,
    original_ports: Mapping[str, tuple[int, ...]],
    input_rows: Sequence[Mapping[str, int]],
    *,
    delta: Mapping[str, int],
) -> BoundRegisterProgram:
    """Prove every supplied complex column, every erasure history and zero work.

    Input ownership is explicitly transferred because source arithmetic may
    release and rebind caller bits. Output names follow the actual signature;
    consumed logical inputs and all other private physical storage must clear.
    Original histories are independently certified before replacing their path.
    """
    ports: dict[str, tuple[int, ...]] = {}
    signature = Signature(graph.functions[0].registers or ())
    offset = 0
    for register in signature.lefts():
        size = int(register.bitsize)
        assert register.shape_symbolic == ()
        ports[register.name] = tuple(range(offset, offset + size))
        offset += size
    provisional = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, offset + 200)),
        entry=graph.functions[0].name,
        allow_workspace_outputs=True,
        input_ownership=tuple((w, False) for w in range(offset)),
    )
    width = (
        max(
            (w for event in provisional.events for w in event.wires), default=offset - 1
        )
        + 1
    )
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, width)),
        entry=graph.functions[0].name,
        allow_workspace_outputs=True,
        input_ownership=tuple((w, False) for w in range(offset)),
    )
    output_ports = dict(
        zip((r.name for r in signature.rights()), bound.outputs, strict=True)
    )
    old_inputs = tuple(
        pack(row, original_ports, original.n_wires) for row in input_rows
    )
    raw_inputs = tuple(pack(row, ports, width) for row in input_rows)
    assert len(set(raw_inputs)) == len(input_rows)
    old_rows = tuple(
        tuple(word >> (original.n_wires - i - 1) & 1 for i in range(original.n_wires))
        for word in old_inputs
    )
    old_outputs, old_events = certified_and_channel_map(original.instructions, old_rows)
    expected = []
    for row in old_outputs:
        values = {
            name: sum(row[w] << (len(wires) - i - 1) for i, w in enumerate(wires))
            for name, wires in original_ports.items()
        }
        assert sum(v << (original.n_wires - i - 1) for i, v in enumerate(row)) == pack(
            {name: values[name] for name in output_ports},
            original_ports,
            original.n_wires,
        )
        expected.append(pack(values, output_ports, width))
    inputs = tuple(
        tuple(word >> (width - i - 1) & 1 for i in range(width)) for word in raw_inputs
    )
    actual, events = certified_and_channel_map(bound.circuit.instructions, inputs)
    assert tuple(
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in actual
    ) == tuple(expected)
    assert events - old_events == delta.get("measure", 0)
    measured_delta = Counter(g.name for g in bound.circuit.instructions)
    measured_delta.subtract(g.name for g in original.instructions)
    assert measured_delta == Counter(delta)
    amplitudes = tuple(complex(i + 1, 11 - i) for i in range(len(input_rows)))
    norm = sum(abs(value) ** 2 for value in amplitudes) ** 0.5
    before = {
        word: value / norm for word, value in zip(raw_inputs, amplitudes, strict=True)
    }
    after = {
        word: value / norm for word, value in zip(expected, amplitudes, strict=True)
    }
    assert len(after) == len(input_rows)
    with patch.dict(sys.modules, {"qualtran": None, "cirq": None, "qiskit": None}):
        circuit = DynamicCircuit(width)
        circuit.append(FieldOperation(width, bound.circuit.instructions))
        native = circuit.to_ir()
        assert native.instructions == bound.circuit.instructions
        for seed in (0, 19):
            assert execute_sparse_register(native, before, seed=seed) == pytest.approx(
                after, abs=3e-12
            )
    return bound
