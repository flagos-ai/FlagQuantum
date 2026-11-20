"""Frozen permutation serialization failures and executable SDK consumers."""

from collections import Counter
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
from flagquantum.compiler.registers.sorting.permutation_declaration import (
    PermutationCycleDeclaration,
    PermutationDeclaration,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from flagquantum.simulation.gate_matrix import gate_matrix
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_permutation_declaration import native_declaration

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "name",
    (
        "_permutation_cycle",
        "_permutation_cycle_symb_N",
        "_permutation_cycle_symb",
        "_permutation",
        "_permutation_symb",
        "_permutation_symb_with_cycles",
        "_sparse_permutation",
        "_sparse_permutation_with_symbolic_N",
    ),
)
def test_all_actual_factory_wire_success_or_original_serialization_failure(
    name: str,
) -> None:
    from qualtran.bloqs.arithmetic import permutation as source_module
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = getattr(source_module, name).make()
    for depth in (0, 1, 2):
        failure = {
            "_permutation_cycle_symb": "Cannot serialize Shaped",
            "_permutation_symb": "Cannot serialize Shaped",
            "_permutation_symb_with_cycles": "Object arrays cannot be saved",
            "_sparse_permutation": "inhomogeneous shape",
            "_sparse_permutation_with_symbolic_N": "inhomogeneous shape",
        }.get(name)
        if name == "_permutation_cycle_symb_N" and depth == 2:
            failure = "Cannot serialize HasLength"
        if failure is not None:
            with pytest.raises(ValueError, match=failure):
                bloqs_to_proto(source, max_depth=depth)
            continue
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        restored = bloqs_from_proto(proto)[0]
        assert (source == restored) == (name != "_permutation_cycle_symb_N")
        if name == "_permutation_cycle_symb_N":
            assert str(source.N) == str(restored.N)
            assert source.N.is_positive and source.N.is_integer
            assert restored.N.assumptions0 == {"commutative": True}


@pytest.mark.parametrize("composed", (False, True))
@pytest.mark.parametrize("symbolic", (False, True))
def test_library_argument_resolution_uses_general_owned_declaration(
    composed: bool, symbolic: bool
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle
    from qualtran.serialization.bloq import bloqs_to_proto

    size = sympy.Symbol("N") if symbolic else 4
    source = (
        Permutation(size, ((0, 1), (2, 3)), 2)
        if composed
        else PermutationCycle(size, (0, 1, 2), 2)
    )
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    )
    root = library.entries[0].declaration

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord | FunctionRecord:
        assert registers is not None
        bound, width = arguments["N"], arguments["bitsize"]
        assert isinstance(bound, (int, sympy.Expr)) and isinstance(width, int)
        data = arguments["cycles" if composed else "cycle"]
        assert isinstance(data, np.ndarray)
        declaration = (
            PermutationDeclaration(
                bound, tuple(tuple(int(v) for v in row) for row in data), width
            )
            if composed
            else PermutationCycleDeclaration(bound, tuple(int(v) for v in data), width)
        )
        assert tuple(declaration.signature) == registers
        if symbolic:
            with pytest.raises(ValueError, match="concrete"):
                declaration.program()
            return FunctionRecord(name, (), registers=registers)
        graph = declaration.program()
        graph = replace(
            graph,
            functions=(replace(graph.functions[0], name=name), *graph.functions[1:]),
        )
        return _namespace_helpers(graph, name)

    resolved = resolve_component_program(library, 0, {root.name: factory})
    assert resolved.functions[0].registers == root.registers
    if not symbolic:
        bound = bind_register_program(resolved, ((0, 1),), clean=(2, 3))
        expected = bind_register_program(
            native_declaration(source).program(), ((0, 1),), clean=(2, 3)
        )
        assert bound.circuit == expected.circuit and bound.outputs == expected.outputs


@pytest.mark.parametrize("kind", ("cycle", "composed", "identity"))
@pytest.mark.parametrize("width", (2, 3))
def test_sdk_real_bounded_signature_stream_amplitudes_and_clean_work(
    kind: str, width: int
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle
    from qualtran.serialization.registers import register_to_proto

    source = (
        PermutationCycle(1 << width, (0, 2), width)
        if kind == "cycle"
        else Permutation(
            1 << width, ((0, 1), (1, 2)) if kind == "composed" else (), width
        )
    )
    signature = Signature(
        deserialize_register(register_to_proto(r).SerializeToString())
        for r in source.signature
    )
    imported = import_cirq_register_gate(
        source.on(*reversed(cirq.LineQubit.range(width))), signature=signature
    )
    assert imported is not None
    work = tuple(range(width + 1, 2 * width + 1))
    actual = bind_register_program(imported, (tuple(range(width)),), clean=work)
    expected = bind_register_program(
        native_declaration(source).program(), (tuple(range(width)),), clean=work
    )
    assert actual.outputs == expected.outputs
    operation = source.on(*reversed(cirq.LineQubit.range(width)))
    source_leaves = cirq.decompose(
        operation,
        context=cirq.DecompositionContext(
            cirq.GreedyQubitManager("reference", maximize_reuse=True)
        ),
        keep=lambda op: (
            isinstance(op.gate, cirq.ControlledGate)
            and op.gate.sub_gate == cirq.X
            and op.gate.num_controls() <= 2
        )
        or isinstance(
            op.gate,
            (
                cirq.XPowGate,
                cirq.ZPowGate,
                cirq.CXPowGate,
                cirq.CCXPowGate,
                cirq.MatrixGate,
            ),
        ),
    )
    # Import normalizes one/two negative controls before SDK decomposition.
    # Independently prove that exact local basis change, then compare its stream.
    normalized = []
    for leaf in source_leaves:
        if isinstance(leaf.gate, cirq.ControlledGate):
            values = tuple(tuple(v) for v in leaf.gate.control_values)
            assert leaf.gate.sub_gate == cirq.X
            assert all(v in ((0,), (1,)) for v in values)
            flips = [
                cirq.X(q)
                for q, v in zip(leaf.qubits[:-1], values, strict=True)
                if v == (0,)
            ]
            positive = cirq.X.controlled(num_controls=len(values)).on(*leaf.qubits)
            sequence = [*flips, positive, *reversed(flips)]
            np.testing.assert_allclose(
                cirq.Circuit(sequence).unitary(qubit_order=leaf.qubits),
                cirq.unitary(leaf),
                atol=3e-15,
                rtol=0,
            )
            normalized.extend(sequence)
        else:
            normalized.append(leaf)
    source_leaves = normalized
    mapping = dict(zip(operation.qubits, range(width), strict=True))
    available = iter(work)
    assert len(actual.circuit.instructions) == len(source_leaves)
    for native, original in zip(
        actual.circuit.instructions, source_leaves, strict=True
    ):
        for qubit in original.qubits:
            if qubit not in mapping:
                mapping[qubit] = next(available)
        assert native.wires == tuple(mapping[q] for q in original.qubits)
        matrix = gate_matrix(native, bsz=1, device="cpu", dtype=torch.complex128)
        np.testing.assert_allclose(
            matrix.reshape(1 << len(native.wires), -1).numpy(),
            cirq.unitary(original.gate),
            atol=3e-15,
            rtol=0,
        )
    assert not any(g.name in {"measure", "reset"} for g in actual.circuit.instructions)
    # The original SDK uses coherent controlled-X leaves. The Bloq path uses
    # measured And compute/erase and retains its separate source gate evidence.
    direct_counts = Counter(g.name for g in expected.circuit.instructions)
    assert direct_counts["measure"] == direct_counts["reset"]
    if kind != "identity":
        assert direct_counts["measure"] > 0
    amplitudes = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(amplitudes, 2 - 3 * amplitudes)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (2 * width + 1), dtype=torch.complex128)
    initial[:: 1 << width] = amplitudes
    target = torch.zeros_like(initial)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << width, 1 << width
    )
    target[:: 1 << width] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    circuit = DynamicCircuit(2 * width + 1, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2 * width + 1, actual.circuit.instructions))
    measurements = sum(g.name == "measure" for g in actual.circuit.instructions)
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, target / 2 ** (measurements / 2), rtol=0, atol=3e-12
        )
        assert torch.linalg.vector_norm(branch.reshape(-1, 1 << width)[:, 1:]) < 3e-12
