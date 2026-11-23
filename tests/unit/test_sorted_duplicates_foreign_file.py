"""Frozen duplicate objects retain factories, wire libraries and SDK boundaries."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations_with_replacement, product
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
from flagquantum.ecosystem.cirq._qubit_manager import InteropQubitManager
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_register_tensor import native_signature
from tests.unit.test_sorted_duplicates_component import bind_case

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")
sympy = pytest.importorskip("sympy")
FACTORIES = ("_has_duplicates", "_has_duplicates_symb", "_has_duplicates_symb_len")


def original_sdk_primitives(operation: Any) -> tuple[Any, ...]:
    """Expand untouched SDK bodies with sibling work reservations and atomic CCX."""
    manager = InteropQubitManager(
        cirq.GreedyQubitManager(prefix="source_work", maximize_reuse=True)
    )
    manager.manage_qubits(operation.qubits)
    context = cirq.DecompositionContext(manager)
    basis = {cirq.X, cirq.H, cirq.T, cirq.T**-1, cirq.S, cirq.CNOT, cirq.CZ, cirq.CCX}

    def expand(frame: tuple[Any, ...]) -> tuple[Any, ...]:
        leaves = []
        with manager.reserve_qubits(q for op in frame for q in op.qubits):
            for op in frame:
                plain = op.without_classical_controls()
                if plain.gate in basis or isinstance(
                    plain.gate, (cirq.MeasurementGate, cirq.ResetChannel)
                ):
                    leaves.append(op)
                else:
                    children = tuple(
                        cirq.flatten_to_ops(cirq.decompose_once(plain, context=context))
                    )
                    for child in expand(children):
                        leaves.append(
                            child.with_classical_controls(*op.classical_controls)
                            if op.classical_controls
                            else child
                        )
        return tuple(leaves)

    return expand((operation,))


@pytest.mark.parametrize("factory_name", FACTORIES)
@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("signed", (False, True))
def test_actual_factory_libraries_and_native_resolution(
    factory_name: str, controlled: bool, signed: bool
) -> None:
    import attrs
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.lists import has_duplicates as original
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = getattr(original, factory_name).make()
    source = attrs.evolve(
        source,
        dtype=QInt(source.dtype.bitsize) if signed else source.dtype,
        is_controlled=controlled,
    )
    symbolic_length = factory_name.endswith("symb_len")
    for depth in (0, 1, 2):
        if symbolic_length and depth:
            with pytest.raises(ValueError, match="Cannot serialize HasLength"):
                bloqs_to_proto(source, max_depth=depth)
            continue
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert source in bloqs_from_proto(proto)
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    )
    declaration = library.entries[0].declaration
    symbolic = factory_name != "_has_duplicates"

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        assert arguments["l"] == source.l
        assert arguments["dtype"] == registers[0].dtype
        assert arguments["is_controlled"] == controlled
        if symbolic:
            return FunctionRecord(name, (), registers=registers)
        record = sorted_duplicates_program(
            source.l, registers[0].dtype, controlled=controlled
        )
        assert record.functions[0].registers == registers
        record = replace(
            record,
            functions=(replace(record.functions[0], name=name), *record.functions[1:]),
        )
        return _namespace_helpers(record, name)

    resolved = resolve_component_program(library, 0, {declaration.name: factory})
    if symbolic_length:
        assert resolved.functions[0].registers == declaration.registers
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(resolved, ())
        return
    assert resolved.functions[0].registers == declaration.registers
    if symbolic:
        assert declaration.registers is not None
        with pytest.raises(ValueError, match="unsupported executable"):
            sorted_duplicates_program(4, declaration.registers[0].dtype)
    else:
        expected, ports, logical = bind_case(4, 3, signed, controlled)
        actual = bind_register_program(
            resolved, ports, clean=tuple(range(logical, expected.circuit.n_wires))
        )
        assert actual.circuit == expected.circuit and actual.outputs == expected.outputs


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("controlled", (False, True))
def test_unsigned_typed_sdk_complete_stream_and_valid_complex_branches(
    width: int, controlled: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    source = HasDuplicates(2, QUInt(width), controlled)
    logical = 2 * width + 1 + controlled
    imported = import_cirq_register_gate(
        source.on(*reversed(cirq.LineQubit.range(logical))),
        signature=native_signature(source.signature),
    )
    assert imported is not None
    _, ports, _ = bind_case(2, width, False, controlled, spectator=True)
    actual = bind_register_program(
        imported, ports, clean=tuple(range(logical + 1, logical + 1 + width + 8))
    )
    assert actual.outputs == ports
    operation = source.on(*cirq.LineQubit.range(logical))
    leaves = original_sdk_primitives(operation)
    order = list(operation.qubits)
    for leaf in leaves:
        for qubit in leaf.qubits:
            if qubit not in order:
                order.append(qubit)
    reference = import_cirq_feedback(
        cirq.Circuit(cirq.Moment((leaf,)) for leaf in leaves), qubit_order=order
    ).ir
    mapping = (*range(logical), *range(logical + 1, logical + 1 + len(order) - logical))
    from flagquantum.core.ir import Instruction

    expected = tuple(
        Instruction(
            gate.name,
            tuple(mapping[wire] for wire in gate.wires),
            params=gate.params,
            metadata=gate.metadata,
        )
        for gate in reference.instructions
    )

    def wire_history(gates: tuple[Instruction, ...], wire: int) -> tuple[object, ...]:
        return tuple(
            (
                gate.name,
                gate.wires.index(wire),
                tuple((i, q) for i, q in enumerate(gate.wires) if q < logical),
            )
            for gate in gates
            if wire in gate.wires
        )

    # Independent scheduling can change allocation order. Match complete wire
    # histories before asserting every operand, ordered gate and feedback field.
    renaming = dict(zip(range(logical), range(logical), strict=True))
    for wire in sorted({q for gate in expected for q in gate.wires if q > logical}):
        candidates = [
            q
            for q in range(logical + 1, actual.circuit.n_wires)
            if q not in renaming.values()
            and wire_history(expected, wire)
            == wire_history(actual.circuit.instructions, q)
        ]
        assert candidates, wire
        renaming[wire] = candidates[0]
    expected = tuple(
        replace(gate, wires=tuple(renaming[q] for q in gate.wires)) for gate in expected
    )
    assert_source(actual.circuit.instructions, expected)
    assert CircuitIR.from_json(actual.circuit.to_json()) == actual.circuit
    if width > 2:
        return
    total = actual.circuit.n_wires
    work = total - logical - 1
    labels, mapped = [], []
    for left, right in combinations_with_replacement(range(1 << width), 2):
        for flag, ctrl, spectator in product(
            (0, 1), (0, 1) if controlled else (1,), (0, 1)
        ):
            data = ((left << width) | right) << 1 | flag
            output = (data & ~1) | (flag ^ ((left == right) and bool(ctrl)))
            if controlled:
                data, output = (data << 1) | ctrl, (output << 1) | ctrl
            labels.append(((data << 1) | spectator) << work)
            mapped.append(((output << 1) | spectator) << work)
    raw = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    target = torch.zeros_like(initial)
    initial[labels], target[mapped] = amplitudes, amplitudes
    measurements = sum(g.name == "measure" for g in actual.circuit.instructions)
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial, actual.circuit.instructions, history)
        torch.testing.assert_close(
            branch, target / 2 ** (measurements / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, 1 << work)[:, 1:]) == 0


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("controlled", (False, True))
def test_signed_sdk_retains_original_sign_extension_failure(
    width: int, controlled: bool
) -> None:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    source = HasDuplicates(2, QInt(width), controlled)
    operation = source.on(*cirq.LineQubit.range(2 * width + 1 + controlled))
    with pytest.raises(ValueError, match="Wrong number of qubits.*SignExtend"):
        cirq.decompose(
            operation,
            context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
        )
    with pytest.raises(ValueError, match="Wrong number of qubits.*SignExtend"):
        import_cirq_register_gate(
            operation, signature=native_signature(source.signature)
        )
