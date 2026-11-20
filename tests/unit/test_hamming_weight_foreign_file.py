"""Hamming objects retain symbolic widths and sided SDK output ownership."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

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
from flagquantum.compiler.registers.bits.hamming_weight import hamming_weight_ir
from flagquantum.compiler.registers.bits.hamming_weight_component import (
    hamming_weight_program,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
sympy = pytest.importorskip("sympy")


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_hamming_library_keeps_full_symbolic_signature(
    symbolic: bool, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = HammingWeightCompute(sympy.Symbol("n") if symbolic else 3)
    if adjoint:
        source = source.adjoint()
    for depth in (0, 1):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert (
            deserialize_component_library(serialize_component_library(library))
            == library
        )
        assert source in bloqs_from_proto(proto)
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    declarations: list[tuple[Register, ...]] = []

    def declaration(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None and arguments
        declarations.append(registers)
        if symbolic:
            return FunctionRecord(name, (), registers=registers)
        inverse = registers[-1].side.name == "LEFT"
        record = hamming_weight_program(3, adjoint=inverse)
        assert record.functions[0].registers == registers
        record = replace(
            record,
            functions=(replace(record.functions[0], name=name), *record.functions[1:]),
        )
        return _namespace_helpers(record, name)

    resolved = resolve_component_program(
        library, 0, {entry.declaration.name: declaration for entry in library.entries}
    )
    assert resolved.functions[0].registers == library.entries[0].declaration.registers
    assert declarations
    if symbolic:
        registers = library.entries[0].declaration.registers
        assert registers is not None
        dimension = registers[-1].bitsize
        assert isinstance(dimension, sympy.Expr)
        for n in (1, 2, 3, 7, 8, 15, 16):
            assert int(dimension.subs(sympy.Symbol("n"), n)) == n.bit_length()
    else:
        inputs = ((0, 1, 2), (3,), (4, 5)) if adjoint else ((0, 1, 2),)
        clean = () if adjoint else (3, 4, 5)
        ownership = ((3, False), (4, False), (5, False)) if adjoint else ()
        actual = bind_register_program(
            resolved,
            inputs,
            clean=clean,
            allow_workspace_outputs=not adjoint,
            input_ownership=ownership,
        )
        expected = bind_register_program(
            hamming_weight_program(3, adjoint=adjoint),
            inputs,
            clean=clean,
            allow_workspace_outputs=not adjoint,
            input_ownership=ownership,
        )
        assert actual.circuit == expected.circuit and actual.outputs == expected.outputs


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5))
@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_sided_sdk_hamming_stream_complex_state_and_erasure(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    root = HammingWeightCompute(width)
    source = root.adjoint() if adjoint else root
    total = width + int(root.junk_bitsize) + int(root.out_bitsize)
    operation = source.on(*reversed(cirq.LineQubit.range(total)))
    if width == 1 and adjoint:
        with pytest.raises(KeyError):
            source.decompose_bloq()
        with pytest.raises(KeyError):
            import_cirq_register_gate(
                operation, signature=native_signature(source.signature)
            )
        return
    imported = import_cirq_register_gate(
        operation, signature=native_signature(source.signature)
    )
    assert imported is not None
    x = tuple(range(width))
    junk = tuple(range(width, width + int(root.junk_bitsize)))
    out = tuple(range(width + len(junk), total))
    inputs = (x, junk, out) if adjoint else (x,)
    bound = bind_register_program(
        imported,
        inputs,
        clean=() if adjoint else (*junk, *out),
        allow_workspace_outputs=not adjoint,
        input_ownership=(
            tuple((wire, False) for wire in (*junk, *out)) if adjoint else ()
        ),
    )
    actual_x, actual_junk, actual_out = (x, junk, out) if adjoint else bound.outputs
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    expected = hamming_weight_ir(
        actual_x,
        actual_junk,
        actual_out,
        adjoint=adjoint,
        classical_bits=tuple(slots[wire] for wire in actual_junk) if adjoint else (),
    )
    assert_source(bound.circuit.instructions, expected)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    if width > 3:
        return
    raw = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 4 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (total + 1), dtype=torch.complex128)
    initial.reshape(1 << width, -1)[:, :2] = amplitudes.reshape(1 << width, 2)
    computed = forced_branch(
        initial, hamming_weight_ir(actual_x, actual_junk, actual_out), ()
    )
    histories = product((0, 1), repeat=len(junk)) if adjoint else ((),)
    for history in histories:
        result = forced_branch(
            computed if adjoint else initial, bound.circuit.instructions, history
        )
        target = initial if adjoint else computed
        torch.testing.assert_close(
            result, target / 2 ** (len(history) / 2), rtol=0, atol=3e-12
        )
        if adjoint:
            assert torch.count_nonzero(result.reshape(1 << width, -1)[:, 2:]) == 0
