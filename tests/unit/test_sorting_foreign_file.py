"""Complete sorting libraries and SDK consumers retain original port ownership."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

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
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_import import (
    import_cirq_register_gate,
    import_cirq_register_optree,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_sorting_declaration import (
    Declaration,
    native_sorting_declaration,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "name",
    (
        "_comparator",
        "_comparator_symb",
        "_parallel_compare",
        "_bitonic_merge",
        "_bitonic_sort",
        "_bitonic_sort_symb",
    ),
)
def test_all_actual_factories_complete_wire_and_native_resolution(name: str) -> None:
    from qualtran.bloqs.arithmetic import sorting
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = getattr(sorting, name).make()
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert bloqs_from_proto(proto)[0] == source
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    )
    root = library.entries[0].declaration

    def factory(
        function_name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord | FunctionRecord:
        from flagquantum.compiler.registers.comparison.ordering_declaration import (
            ComparatorDeclaration,
            ParallelComparatorsDeclaration,
        )
        from flagquantum.compiler.registers.sorting.bitonic_declaration import (
            BitonicMergeDeclaration,
            BitonicSortDeclaration,
        )

        width = arguments["bitsize"]
        assert isinstance(width, (int, sympy.Expr))
        declaration: Declaration
        if type(source).__name__ == "Comparator":
            declaration = ComparatorDeclaration(width)
        elif type(source).__name__ == "ParallelComparators":
            count, offset = arguments["k"], arguments["offset"]
            assert isinstance(count, (int, sympy.Expr))
            assert isinstance(offset, (int, sympy.Expr))
            declaration = ParallelComparatorsDeclaration(count, offset, width)
        elif type(source).__name__ == "BitonicMerge":
            half_length = arguments["half_length"]
            assert isinstance(half_length, (int, sympy.Expr))
            declaration = BitonicMergeDeclaration(half_length, width)
        else:
            count = arguments["k"]
            assert isinstance(count, (int, sympy.Expr))
            declaration = BitonicSortDeclaration(count, width)
        assert tuple(declaration.signature) == registers
        if not isinstance(width, int):
            with pytest.raises(ValueError, match="concrete"):
                declaration.program()
            return FunctionRecord(function_name, (), registers=registers)
        graph = declaration.program()
        graph = replace(
            graph,
            functions=(
                replace(graph.functions[0], name=function_name),
                *graph.functions[1:],
            ),
        )
        return _namespace_helpers(graph, function_name)

    resolved = resolve_component_program(library, 0, {root.name: factory})
    assert resolved.functions[0].registers == root.registers
    if isinstance(source.bitsize, int):
        expected = native_sorting_declaration(source).program()
        assert len(resolved.functions) == len(expected.functions)


@pytest.mark.parametrize("kind", ("parallel", "merge", "sort"))
def test_symbolic_length_library_retains_assumption_loss_and_approved_declaration(
    kind: str,
) -> None:
    from qualtran.bloqs.arithmetic.sorting import (
        BitonicMerge,
        BitonicSort,
        ParallelComparators,
    )
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    length = sympy.Symbol("k", positive=True, integer=True)
    source = (
        ParallelComparators(length, 2, 3)
        if kind == "parallel"
        else (BitonicMerge if kind == "merge" else BitonicSort)(length, 3)
    )
    proto = bloqs_to_proto(source, max_depth=0)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert source != bloqs_from_proto(proto)[0]

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None and arguments
        return FunctionRecord(name, (), registers=registers)

    resolved = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    assert resolved.functions[0].registers == library.entries[0].declaration.registers
    assert not resolved.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(resolved, ())


@pytest.mark.parametrize("kind", ("compare", "parallel", "merge", "sort"))
@pytest.mark.parametrize("width", (1, 2))
def test_real_sdk_sided_ports_complete_complex_branches_and_clean_work(
    kind: str, width: int
) -> None:
    from qualtran.bloqs.arithmetic.sorting import (
        BitonicMerge,
        BitonicSort,
        Comparator,
        ParallelComparators,
    )
    from qualtran.serialization.registers import register_to_proto

    source = {
        "compare": Comparator(width),
        "parallel": ParallelComparators(3, 1, width),
        "merge": BitonicMerge(1, width),
        "sort": BitonicSort(2, width),
    }[kind]
    incoming = {}
    inputs: list[tuple[int, ...]] = []
    cursor = 0
    manager = cirq.SimpleQubitManager()
    for register in source.signature.lefts():
        count = int(register.total_bits())
        incoming[register.name] = np.array(manager.qalloc(count), dtype=object).reshape(
            (*register.shape, register.bitsize)
        )
        inputs.extend(
            tuple(range(i, i + width)) for i in range(cursor, cursor + count, width)
        )
        cursor += count
    signature = Signature(
        deserialize_register(register_to_proto(r).SerializeToString())
        for r in source.signature
    )
    imported: ProgramRecord | None
    if kind == "merge":
        # The original root SDK wrapper rejects different-name LEFT/RIGHT ports.
        # Its unchanged composite conversion retains the actual shared wires.
        tree, outgoing = source.decompose_bloq().to_cirq_circuit_and_quregs(
            qubit_manager=manager, **incoming
        )
        imported = import_cirq_register_optree(
            tree, signature=signature, in_quregs=incoming, out_quregs=outgoing
        )
    else:
        operation, outgoing = source.as_cirq_op(manager, **incoming)
        imported = import_cirq_register_gate(
            operation, signature=signature, in_quregs=incoming, out_quregs=outgoing
        )
    assert imported is not None
    assert imported.functions[0].registers == tuple(signature)
    bound = bind_register_program(
        imported,
        tuple(inputs),
        clean=tuple(range(cursor, cursor + 3 * width + 2)),
        allow_workspace_outputs=True,
    )
    total = bound.circuit.n_wires + 1
    raw = torch.arange(1, (1 << (cursor + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[:: 1 << (total - cursor)] = amplitudes[::2]
    initial[1 :: 1 << (total - cursor)] = amplitudes[1::2]
    outputs = tuple(w for register in bound.outputs for w in register)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << len(outputs), 1 << cursor
    )
    transformed = matrix @ amplitudes.reshape(-1, 2)
    expected = torch.zeros_like(initial)
    for value in range(1 << len(outputs)):
        physical = sum(
            ((value >> (len(outputs) - 1 - i)) & 1) << (total - 1 - wire)
            for i, wire in enumerate(outputs)
        )
        expected[physical] = transformed[value, 0]
        expected[physical + 1] = transformed[value, 1]
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert measurements == 2 * width - 1
    for history in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (measurements / 2), rtol=0, atol=3e-12
        )
        work = set(range(total - 1)) - set(outputs)
        dirty = torch.tensor(
            [
                any((basis >> (total - 1 - wire)) & 1 for wire in work)
                for basis in range(1 << total)
            ]
        )
        assert torch.linalg.vector_norm(actual[dirty]) < 3e-12


@pytest.mark.parametrize("width", (1, 2))
def test_original_merge_root_sdk_wrapper_retains_sided_port_arity_failure(
    width: int,
) -> None:
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge

    manager = cirq.SimpleQubitManager()
    incoming = {
        name: np.array(manager.qalloc(width), dtype=object).reshape(1, width)
        for name in ("xs", "ys")
    }
    with pytest.raises(ValueError, match="Wrong number of qubits"):
        BitonicMerge(1, width).as_cirq_op(manager, **incoming)
