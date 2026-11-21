"""Complete subtraction factories and SDK consumers use owned native graphs."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import sympy
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
from flagquantum.compiler.registers.arithmetic.subtraction_control import (
    controlled_subtraction_program,
)
from flagquantum.compiler.registers.arithmetic.subtraction_declaration import (
    SubtractFromDeclaration,
    SubtractionDeclaration,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_subtraction_declaration import native_subtraction_declaration

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")


def native_signature(source: Any) -> Signature:
    """Translate original typed ports at the optional SDK boundary."""
    from qualtran.serialization.registers import register_to_proto

    return Signature(
        deserialize_register(register_to_proto(reg).SerializeToString())
        for reg in source
    )


@pytest.mark.parametrize(
    "name",
    (
        "_sub_symb",
        "_sub_small",
        "_sub_large",
        "_sub_diff_size_regs",
        "_sub_symp_decomposition",
        "_sub_from_symb",
        "_sub_from_small",
        "_sub_from_large",
    ),
)
def test_all_actual_factories_wire_roundtrip_and_generic_native_resolution(
    name: str,
) -> None:
    from qualtran.bloqs.arithmetic import subtraction
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = getattr(subtraction, name).make()
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert (source == bloqs_from_proto(proto)[0]) == (
            name != "_sub_symp_decomposition" or depth != 0
        )
    library = deserialize_component_library(
        bloqs_to_proto(
            source, max_depth=1 if name == "_sub_symp_decomposition" else 0
        ).SerializeToString()
    )

    def factory(
        function_name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        if name == "_sub_symp_decomposition":
            return FunctionRecord(function_name, (), registers=registers)
        if "dtype" in arguments:
            dtype = arguments["dtype"]
            assert isinstance(dtype, RegisterType)
            declaration: SubtractionDeclaration | SubtractFromDeclaration = (
                SubtractFromDeclaration(dtype)
            )
        else:
            a, b = arguments["a_dtype"], arguments["b_dtype"]
            assert isinstance(a, RegisterType) and isinstance(b, RegisterType)
            declaration = SubtractionDeclaration(a, b)
        assert tuple(declaration.signature) == registers
        if any(isinstance(r.bitsize, sympy.Expr) for r in registers):
            with pytest.raises(ValueError):
                declaration.program()
            return FunctionRecord(function_name, (), registers=registers)
        graph = declaration.program()
        return _namespace_helpers(
            replace(
                graph,
                functions=(
                    replace(graph.functions[0], name=function_name),
                    *graph.functions[1:],
                ),
            ),
            function_name,
        )

    factories = {
        e.declaration.name: factory
        for e in library.entries
        if "CompositeBloq" not in e.declaration.name
    }
    resolved = resolve_component_program(library, 0, factories)
    assert resolved.functions[0].registers == library.entries[0].declaration.registers
    if name == "_sub_symp_decomposition":
        operations = resolved.functions[0].blocks[0].operations
        assert len(operations) == 8
        assert all(operation.opcode == "call" for operation in operations)
    elif name not in {"_sub_symb", "_sub_from_symb"}:
        direct = native_subtraction_declaration(source).program()
        left, right = (int(r.bitsize) for r in resolved.functions[0].registers or ())
        ports = (tuple(range(left)), tuple(range(left, left + right)))
        work = tuple(range(left + right, 3 * right - 1))
        assert (
            bind_register_program(resolved, ports, clean=work).circuit
            == bind_register_program(direct, ports, clean=work).circuit
        )


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 5)))
@pytest.mark.parametrize("reverse", (False, True))
def test_actual_sdk_stream_complex_spectator_all_histories_and_work(
    kind: str, left: int, right: int, reverse: bool
) -> None:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    dtype = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[kind]
    if reverse:
        left = right
    source = (
        SubtractFrom(dtype(left)) if reverse else Subtract(dtype(left), dtype(right))
    )
    logical = left + right
    signature = native_signature(source.signature)
    imported = import_cirq_register_gate(
        source.on(*cirq.LineQubit.range(logical)), signature=signature
    )
    assert imported is not None
    ports = (tuple(range(left)), tuple(range(left, logical)))
    scratch = right - 1 + (0 if reverse else right - left)
    work = tuple(range(logical + 1, logical + 1 + scratch))
    actual = bind_register_program(imported, ports, clean=work)
    direct = bind_register_program(
        native_subtraction_declaration(source).program(), ports, clean=work
    )
    assert actual.outputs == direct.outputs
    assert_source(actual.circuit.instructions, direct.circuit.instructions)
    assert Counter(g.name for g in actual.circuit.instructions) == Counter(
        g.name for g in direct.circuit.instructions
    )
    if right > 3:
        return
    amplitudes = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(amplitudes, 2 - 3 * amplitudes)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + scratch), dtype=torch.complex128)
    initial[:: 1 << scratch] = amplitudes
    expected = torch.zeros_like(initial)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected[:: 1 << scratch] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    circuit = DynamicCircuit(
        logical + 1 + scratch, inputs=initial, dtype=torch.complex128
    )
    circuit.append(FieldOperation(logical + 1 + scratch, actual.circuit.instructions))
    for history in product((0, 1), repeat=right - 1):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** ((right - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, 1 << scratch)[:, 1:]) == 0


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("cv", (0, 1))
def test_controlled_sdk_positive_specialization_and_negative_original_failure(
    reverse: bool, cv: int
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    source = (SubtractFrom if reverse else Subtract)(QUInt(3)).controlled(
        CtrlSpec(cvs=cv)
    )
    operation = source.on(*cirq.LineQubit.range(7))
    if cv == 0:
        unresolved = [
            leaf.gate
            for leaf in cirq.decompose(operation)
            if getattr(getattr(leaf.gate, "subbloq", None), "uncompute", False)
        ]
        assert unresolved
        with pytest.raises(RecursionError):
            cirq.unitary(unresolved[0])
        with pytest.raises(
            ValueError, match="Source leaf explicitly declares no unitary"
        ):
            import_cirq_register_gate(
                operation, signature=native_signature(source.signature)
            )
        return
    imported = import_cirq_register_gate(
        operation, signature=native_signature(source.signature)
    )
    assert imported is not None
    ports = ((0,), (1, 2, 3), (4, 5, 6))
    work = tuple(range(8, 13))
    actual = bind_register_program(imported, ports, clean=work)
    dtype = native_subtraction_declaration(source.subbloq).signature.get_left("a").dtype
    direct = bind_register_program(
        controlled_subtraction_program(dtype, subtract_from=reverse), ports, clean=work
    )
    assert_source(actual.circuit.instructions, direct.circuit.instructions)
    amplitudes = torch.arange(1, 257, dtype=torch.float64)
    amplitudes = torch.complex(amplitudes, 2 - 3 * amplitudes)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << 13, dtype=torch.complex128)
    initial[::32] = amplitudes
    expected = torch.zeros_like(initial)
    expected[::32] = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            128, 128
        )
        @ amplitudes.reshape(-1, 2)
    ).flatten()
    for history in product((0, 1), repeat=5):
        branch = forced_branch(initial, actual.circuit.instructions, history)
        torch.testing.assert_close(branch, expected / 2**2.5, rtol=0, atol=3e-12)
        assert torch.count_nonzero(branch.reshape(-1, 32)[:, 1:]) == 0


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("cv", (0, 1))
def test_controlled_foreign_library_resolves_to_specialized_native_chain(
    reverse: bool, cv: int
) -> None:
    from qualtran import CtrlSpec, QInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.compiler.formats.legacy_component_arguments import (
        ControlSpecification,
    )

    source = (SubtractFrom if reverse else Subtract)(QInt(2)).controlled(
        CtrlSpec(cvs=cv)
    )
    wire = bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert registers is not None
        if "subbloq" in arguments:
            child, spec = arguments["subbloq"], arguments["ctrl_spec"]
            assert isinstance(child, FunctionRecord) and child.registers is not None
            assert (
                isinstance(spec, ControlSpecification) and int(spec.clauses[0][1]) == cv
            )
            a, b = (r.dtype for r in child.registers)
            graph = controlled_subtraction_program(
                a, b, subtract_from=reverse, control_value=cv
            )
        else:
            a, b = (r.dtype for r in registers)
            declaration = (
                SubtractFromDeclaration(a) if reverse else SubtractionDeclaration(a, b)
            )
            graph = declaration.program()
        assert graph.functions[0].registers == registers
        return _namespace_helpers(
            replace(
                graph,
                functions=(
                    replace(graph.functions[0], name=name),
                    *graph.functions[1:],
                ),
            ),
            name,
        )

    resolved = resolve_component_program(
        library, 0, {entry.declaration.name: factory for entry in library.entries}
    )
    ports = ((0,), (1, 2), (3, 4))
    bound = bind_register_program(resolved, ports, clean=(5, 6, 7))
    dtype = native_subtraction_declaration(source.subbloq).signature.get_left("a").dtype
    direct = bind_register_program(
        controlled_subtraction_program(dtype, subtract_from=reverse, control_value=cv),
        ports,
        clean=(5, 6, 7),
    )
    assert bound.circuit == direct.circuit and bound.outputs == direct.outputs
