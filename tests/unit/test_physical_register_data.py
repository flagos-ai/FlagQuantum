"""Verify complete source register-data values behind existing format consumers."""

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmb import KmxRegister, encode_kmb
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.formats.kmx_circuit import (
    KmxCircuit,
    read_kmb_circuit,
    read_kmx_circuit,
)
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.primitives.broadcast_binding import broadcast_cx_ir
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)
from flagquantum.simulation.physical_operation_counters import PhysicalOperationCounters

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "register_data.json").read_text()
)
_ROWS: tuple[str, ...] = tuple(_DATA["source_rows"])


def source_values() -> list[tuple[KmxRegister, str, str]]:
    result = []
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] != "value":
            continue
        _, name, words, diagnostic, equalities = fields
        members: list[tuple[Literal["q", "b"], int]] = []
        for text in words.split(","):
            if text:
                word = int(text)
                assert word >> 29 in (1, 2)
                members.append(
                    ("q" if word >> 29 == 2 else "b", word & ((1 << 29) - 1))
                )
        value = KmxRegister(bytes.fromhex(name).decode(), tuple(members))
        result.append((value, bytes.fromhex(diagnostic).decode(), equalities))
    return result


def test_complete_source_fields_equality_and_diagnostics() -> None:
    values = source_values()
    assert len(values) == 20
    for register, diagnostic, equalities in values:
        assert register.diagnostic_text == diagnostic.replace(
            "RegisterData{", "KmxRegister{", 1
        )
        assert (
            "".join(str(int(register == other)) for other, _, _ in values) == equalities
        )
        assert copy.copy(register) == register
        assert copy.deepcopy(register) == register
        builder = KmxCircuitBuilder()
        builder.replace_registers((register,))
        frozen = builder.freeze()
        assert frozen.registers == (register,)
        assert read_kmx_circuit(frozen.to_kmx()) == frozen
        assert read_kmb_circuit(encode_kmb(frozen.to_kmx())) == frozen
        duplicate = copy.copy(builder)
        duplicate.replace_registers((KmxRegister("changed", (("q", 2),)),))
        assert builder.freeze() == frozen
        assert duplicate.freeze().registers != frozen.registers
    source_copy = next(row.split("\t")[1:] for row in _ROWS if row.startswith("copy\t"))
    assert source_copy == ["independent", "1073741825", "0"]
    sentinel = next(
        row.split("\t")[1:] for row in _ROWS if row.startswith("sentinel\t")
    )
    assert sentinel[0] == sentinel[1]
    assert sentinel[2:] == ["0", "0", "536870912"]
    # Raw false is an untyped source default sentinel. Native register members
    # are explicitly quantum or classical IDs and do not encode this sentinel.
    assert OperandIdentity(OperandKind.BOOLEAN, 0) != OperandIdentity(
        OperandKind.CLASSICAL, 0
    )


def test_metadata_replacement_preserves_append_state_and_primitive_counts() -> None:
    gates = broadcast_cx_ir(0, 1)
    amplitudes = torch.tensor([1 + 2j, 3 - 4j, -2 + 1j, 4 + 7j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    circuit = DynamicCircuit(2, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(2, gates))
    program = lower_reversible_phase(circuit.to_ir())
    raw = read_kmx_circuit(encode_kmx(program))
    builder = KmxCircuitBuilder()
    builder.append(raw)
    register = KmxRegister("logical", (("q", 1), ("q", 0), ("q", 1)))
    builder.replace_registers((register,))
    frozen = builder.freeze()
    assert tuple(frozen) == tuple(raw)
    assert frozen.registers == (register,)
    assert len(frozen.records) == len(raw.records)
    assert [r.opcode for r in frozen if r.opcode != "INFO_TAG"] == ["CX"]
    assert [record.opcode for record in frozen] == [record.opcode for record in raw]
    assert read_kmb_circuit(encode_kmb(frozen.to_kmx())) == frozen
    assert KmxCircuit((), tuple(frozen)) == raw
    expected = amplitudes[torch.tensor([0, 1, 3, 2])]
    result = run_dynamic(circuit, shots=2, seed=3, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), atol=1e-12, rtol=0
    )
    assert result.final_states.numel() == 8

    state = PackedBasisState(
        defaultdict(int, {0: 12, 1: 10}), defaultdict(int), [0] * 4
    )
    tally = PhysicalOperationCounters(4)
    apply_packed_basis_records(tuple(frozen), state, lambda: 0, observer=tally.observe)
    assert [state.qubits[0], state.qubits[1]] == [12, 6]
    for lane in range(4):
        counts = dict(
            PhysicalOperationStatistics.from_counts(*tally.snapshot(lane)).items()
        )
        assert counts.pop("CX") == 1
        assert all(name == "INFO_TAG" for name in counts)
