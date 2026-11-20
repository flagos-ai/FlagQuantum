"""Complete negation factories and SDK paths reuse the accepted native owners."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_registers import deserialize_register
from flagquantum.compiler.registers.arithmetic.negation_component import (
    negation_program,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")


@pytest.mark.parametrize("symbolic", (False, True))
def test_actual_factories_full_library_and_owned_native_resolution(
    symbolic: bool,
) -> None:
    from qualtran.bloqs.arithmetic.negate import _negate, _negate_symb
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = (_negate_symb if symbolic else _negate).make()
    for depth, count in ((0, 1), (1, 3), (2, 8 if symbolic else 10)):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert len(library.entries) == count
        assert source == bloqs_from_proto(proto)[0]
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    declaration = library.entries[0].declaration

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None and arguments["dtype"] == registers[0].dtype
        if symbolic:
            return FunctionRecord(name, (), registers=registers)
        graph = negation_program(registers[0].dtype)
        graph = replace(
            graph,
            functions=(replace(graph.functions[0], name=name), *graph.functions[1:]),
        )
        return _namespace_helpers(graph, name)

    resolved = resolve_component_program(library, 0, {declaration.name: factory})
    assert resolved.functions[0].registers == declaration.registers
    if symbolic:
        assert declaration.registers is not None
        with pytest.raises(ValueError):
            negation_program(declaration.registers[0].dtype)
        return
    ports = (tuple(range(8)),)
    clean = tuple(range(8, 23))
    bound = bind_register_program(resolved, ports, clean=clean)
    assert declaration.registers is not None
    direct = bind_register_program(
        negation_program(declaration.registers[0].dtype), ports, clean=clean
    )
    assert bound.circuit == direct.circuit and bound.outputs == direct.outputs


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("width", (1, 2, 3, 5))
def test_actual_sdk_plain_negation_gates_phases_and_work(kind: str, width: int) -> None:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate

    source = Negate(
        {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[kind](width)
    )
    from qualtran.serialization.registers import register_to_proto

    signature = Signature(
        deserialize_register(register_to_proto(r).SerializeToString())
        for r in source.signature
    )
    operation = source.on(*reversed(cirq.LineQubit.range(width)))
    if kind == "signed" and width == 1:
        with pytest.raises(ValueError, match="Too-large"):
            import_cirq_register_gate(operation, signature=signature)
        return
    imported = import_cirq_register_gate(operation, signature=signature)
    assert imported is not None
    ports = (tuple(range(width)),)
    clean = tuple(range(width, 3 * width - 1))
    actual = bind_register_program(imported, ports, clean=clean)
    direct = bind_register_program(
        negation_program(tuple(signature)[0].dtype), ports, clean=clean
    )
    assert actual.outputs == direct.outputs
    assert_source(actual.circuit.instructions, direct.circuit.instructions)
    assert Counter(g.name for g in actual.circuit.instructions) == Counter(
        g.name for g in direct.circuit.instructions
    )
    if width > 3:
        return
    # Place the spectator after all work, so source SDK qubit naming is irrelevant.
    work = len(clean)
    total = width + work + 1
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    raw = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 3 * raw)
    amplitudes /= amplitudes.norm()
    for i, (value, spectator) in enumerate(product(range(1 << width), (0, 1))):
        initial[(value << (work + 1)) | spectator] = amplitudes[i]
        expected[((-value % (1 << width)) << (work + 1)) | spectator] = amplitudes[i]
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, actual.circuit.instructions))
    for history in product((0, 1), repeat=width - 1):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** ((width - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.linalg.vector_norm(branch.reshape(1 << width, -1)[:, 2:]) < 3e-12


@pytest.mark.parametrize("control", (0, 1))
def test_controlled_sdk_preserves_original_unresolved_measured_adjoint(
    control: int,
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate

    source = Negate(QUInt(3)).controlled(CtrlSpec(cvs=control))
    operation = source.on(*cirq.LineQubit.range(4))
    leaves = cirq.decompose(operation)
    unresolved = [
        leaf.gate
        for leaf in leaves
        if getattr(getattr(leaf.gate, "subbloq", None), "uncompute", False)
    ]
    assert unresolved
    with pytest.raises(RecursionError):
        cirq.unitary(unresolved[0])
    with pytest.raises(RecursionError):
        import_cirq_register_gate(
            operation, signature=native_signature(source.signature)
        )
    # Native specialized controls are independently accepted in test_negation_component.


@pytest.mark.parametrize("control", (0, 1))
@pytest.mark.parametrize("signed", (False, True))
def test_controlled_library_routes_to_existing_specialized_native_owner(
    control: int, signed: bool
) -> None:
    from qualtran import CtrlSpec, QInt, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.compiler.formats.legacy_component_arguments import (
        ControlSpecification,
    )

    source = Negate(QInt(2) if signed else QUInt(2)).controlled(CtrlSpec(cvs=control))
    wire = bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert len(library.entries) == 2
    assert serialize_component_library(library) == wire

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert registers is not None
        if "subbloq" in arguments:
            child = arguments["subbloq"]
            spec = arguments["ctrl_spec"]
            assert isinstance(child, FunctionRecord)
            assert child.registers is not None
            assert isinstance(spec, ControlSpecification)
            assert len(spec.clauses) == 1
            assert int(spec.clauses[0][1]) == control
            graph = negation_program(child.registers[0].dtype, control_value=control)
        else:
            assert arguments["dtype"] == registers[0].dtype
            graph = negation_program(registers[0].dtype)
        assert graph.functions[0].registers == registers
        graph = replace(
            graph,
            functions=(replace(graph.functions[0], name=name), *graph.functions[1:]),
        )
        return _namespace_helpers(graph, name)

    resolved = resolve_component_program(
        library, 0, {entry.declaration.name: factory for entry in library.entries}
    )
    ports = ((0,), (1, 2))
    bound = bind_register_program(resolved, ports, clean=(3, 4, 5))
    assert resolved.functions[0].registers is not None
    direct = bind_register_program(
        negation_program(
            resolved.functions[0].registers[-1].dtype, control_value=control
        ),
        ports,
        clean=(3, 4, 5),
    )
    assert bound.circuit == direct.circuit and bound.outputs == direct.outputs
