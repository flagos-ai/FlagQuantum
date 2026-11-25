"""Complete k-XOR loader source, concrete QROM branches and declared oracles."""

from collections import Counter
from hashlib import sha1
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import sympy
import torch
from qualtran import CtrlSpec, DecomposeNotImplementedError
from qualtran.bloqs.optimization.k_xor_sat import load_kxor_instance as source
from qualtran.resource_counting import (
    QECGatesCost,
    SympySymbolAllocator,
    get_cost_value,
)

from flagquantum.algorithms.kxor_instance import Constraint, KXorInstance
from flagquantum.algorithms.kxor_loading import (
    LoadConstraintScopes,
    LoadUniqueScopeIndex,
    PRGAUniqueConstraintRHS,
    example_scope_loader,
    symbolic_scope_loader,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.gate_budget import GateCategories
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.core.ir import Instruction
from flagquantum.simulation.value_dimensions import SymbolicSequenceLength
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

_BLOB = "21a481e8a52dd2e853eccc71e34d6b32ce7ad1c4"

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_loader_file_is_frozen() -> None:
    assert source.__file__ is not None
    raw = Path(source.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == _BLOB


def test_original_example_factories_signatures_qrom_data_and_costs() -> None:
    for native, original in (
        (example_scope_loader(), source._load_scopes()),
        (symbolic_scope_loader(), source._load_scopes_symb()),
    ):
        assert [
            (r.name, r.bitsize, r.shape, r.side.name) for r in native.signature
        ] == [(r.name, r.bitsize, r.shape, r.side.name) for r in original.signature]
        assert native.scope_bitsize == original.scope_bitsize
        assert native.ancilla_bitsize == original.ancilla_bitsize == 0
        assert native.m_dtype.num_qubits == original.m_dtype.num_qubits
        table = native.qrom
        expected = original._qrom_bloq
        assert table.selection_bitsizes == expected.selection_bitsizes
        assert table.target_bitsizes == expected.target_bitsizes
        assert table.target_shapes == expected.target_shapes
        assert table.shape == expected.data_shape
        graph = native.source_graph(root="load", lookup="qrom")
        assert graph.leaf_counts == {"qrom": 1}
        assert original.build_call_graph(SympySymbolAllocator()) == {expected: 1}
        model = get_cost_value(original, QECGatesCost())
        actual: GateCategories
        if native.inst.is_symbolic():
            actual = table.resource_formula()
        else:
            assert table.datasets == tuple(
                tuple(int(x) for x in row.flat) for row in expected.data
            )
            actual = estimate_component_resources(table.resource_model()).gates
            for j, (scope, _) in enumerate(original.inst.batched_scopes):
                assert original.call_classically(j=j) == (
                    j,
                    original.inst.scope_as_int(scope),
                )
        for field in (
            "and_bloq",
            "clifford",
            "measurement",
            "t",
            "toffoli",
            "rotation",
            "cswap",
        ):
            assert sympy.simplify(getattr(actual, field) - getattr(model, field)) == 0


def test_complete_executable_loader_source_stream_and_all_complex_histories() -> None:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        Constraint as SourceConstraint,
    )
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        KXorInstance as SourceInstance,
    )

    clauses = (((0, 1), -1), ((0, 2), 1), ((1, 3), 1))
    inst = KXorInstance(4, 2, tuple(Constraint(s, b) for s, b in clauses))
    native = LoadConstraintScopes(inst)
    original = source.LoadConstraintScopes(
        SourceInstance(4, 2, tuple(SourceConstraint(s, b) for s, b in clauses))
    )
    j, output, work, bits = (0, 1), (2, 3, 4, 5), (6,), (51,)
    operation = native.make_on(j, output, workspace=work, classical_bits=bits)
    actual = Circuit(8).append(operation).to_ir().instructions
    expected = reference_arithmetic_gates(
        original,
        {"j": j},
        (*output, *work),
        {6: 51},
        cirq_style=True,
        expected_outputs={"j": j, "U": output},
    )
    assert_source(actual, expected)
    assert Counter(
        (g.name, tuple(sorted(g.params.items()))) for g in actual
    ) == Counter((g.name, tuple(sorted(g.params.items()))) for g in expected)
    generator = torch.Generator().manual_seed(921)
    logical = torch.randn(6, generator=generator, dtype=torch.complex128)
    logical /= torch.linalg.vector_norm(logical)
    before = torch.zeros(256, dtype=torch.complex128)
    loaded = torch.zeros_like(before)
    assert isinstance(inst.batched_scopes, tuple)
    for address in range(3):
        scope, _ = inst.batched_scopes[address]
        for spectator in range(2):
            value = logical[2 * address + spectator]
            before[(address << 6) | spectator] = value
            loaded[(address << 6) | (inst.scope_as_int(scope) << 2) | spectator] = value
    measures = sum(g.name == "measure" for g in actual)
    assert measures == 1
    for history in product((0, 1), repeat=measures):
        state = forced_branch(before, actual, history)
        torch.testing.assert_close(state, loaded / sqrt(2), atol=2e-14, rtol=0)
        torch.testing.assert_close(
            state, forced_branch(before, expected, history), atol=2e-14, rtol=0
        )
        # Work wire 6 and unused target-subsystem amplitudes are tested by the
        # full state comparison, including spectator coherence and absolute phase.
    for history in product((0, 1), repeat=2 * measures):
        state = forced_branch(before, (*actual, *actual), history)
        torch.testing.assert_close(state, before / 2, atol=3e-14, rtol=0)
    assert len(actual) == 27
    cost = get_cost_value(original, QECGatesCost())
    model = estimate_component_resources(native.qrom.resource_model()).gates
    assert model.and_bloq == cost.and_bloq == 1
    assert model.measurement == cost.measurement == 1
    for field in ("clifford", "t", "toffoli", "rotation", "cswap"):
        assert getattr(model, field) == getattr(cost, field)
    assert sum(g.name in ("t", "tdg") for g in actual) == 4
    assert native.qrom.workspace_size == 1
    assert Circuit(8).append(operation).to_ir().validate().instructions == actual


def test_original_example_all_valid_addresses_through_native_measured_lookup() -> None:
    native, original = example_scope_loader(), source._load_scopes()
    address_width, output_width = native.m_dtype.num_qubits, native.scope_bitsize
    assert isinstance(address_width, int) and isinstance(output_width, int)
    work_width = native.qrom.workspace_size
    address = tuple(range(address_width))
    output = tuple(range(address_width, address_width + output_width))
    work = tuple(
        range(address_width + output_width, address_width + output_width + work_width)
    )
    width = address_width + output_width + work_width + 1
    gates = (
        Circuit(width)
        .append(
            native.make_on(
                address,
                output,
                workspace=work,
                classical_bits=tuple(range(100, 100 + work_width)),
            )
        )
        .to_ir()
        .instructions
    )
    before, after = [], []
    for j, (scope, _) in enumerate(original.inst.batched_scopes):
        expected_j, expected_scope = original.call_classically(j=j)
        assert expected_j == j and expected_scope == original.inst.scope_as_int(scope)
        for target in (0, (1 << output_width) - 1):
            for spectator in (0, 1):
                incoming = (
                    (j << (width - address_width))
                    | (target << (work_width + 1))
                    | spectator
                )
                outgoing = incoming ^ (int(expected_scope) << (work_width + 1))
                before.append(tuple(map(int, f"{incoming:0{width}b}")))
                after.append(tuple(map(int, f"{outgoing:0{width}b}")))
    events = sum(g.name == "measure" for g in gates)
    assert events == 3
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
def test_complete_index_rhs_parameters_calls_and_control_routes(
    symbolic: bool, controlled: bool
) -> None:
    native_loader = symbolic_scope_loader() if symbolic else example_scope_loader()
    source_loader = source._load_scopes_symb() if symbolic else source._load_scopes()
    native_index, source_index = LoadUniqueScopeIndex(
        native_loader.inst
    ), source.LoadUniqueScopeIndex(source_loader.inst)
    assert [(r.name, r.bitsize, r.side.name) for r in native_index.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source_index.signature
    ]
    ((equality, count),) = source_index.build_call_graph(SympySymbolAllocator()).items()
    assert equality.bitsize == native_index.scope_bitsize
    assert str(equality.val) == "_c0"
    assert native_index.source_graph(root="index", equality="equals").leaf_counts == {
        "equals": count
    }
    index_cost = get_cost_value(source_index, QECGatesCost())
    native_index_cost = native_index.resource_gates()
    for field in (
        "and_bloq",
        "clifford",
        "measurement",
        "t",
        "toffoli",
        "rotation",
        "cswap",
    ):
        assert (
            sympy.simplify(
                getattr(native_index_cost, field) - getattr(index_cost, field)
            )
            == 0
        )
    width = sympy.Symbol("b", integer=True, positive=True) if symbolic else 8
    native = PRGAUniqueConstraintRHS(native_loader.inst, width, controlled)
    original = source.PRGAUniqueConstraintRHS(source_loader.inst, width, controlled)
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in original.signature
    ]
    assert native.n_ctrl == original.n_ctrl
    assert native.angle_dtype.num_qubits == original._angle_dtype.num_qubits
    assert (
        dict(native.angle_dtype.parameters)["fractional_width"]
        == original._angle_dtype.num_frac
    )
    data, signs = native.angle_data
    source_data, source_signs = original._qrom_angle_data
    if symbolic:
        assert isinstance(data, SymbolicSequenceLength) and isinstance(
            signs, SymbolicSequenceLength
        )
        assert data.length == source_data.n and signs.length == source_signs.n
    else:
        np.testing.assert_array_equal(data, source_data)
        np.testing.assert_array_equal(signs, source_signs)
    assert native.num_negative == original._num_negative
    table, expected_table = native.amplitude_qrom, original._amplitude_qrom
    assert table.target_bitsizes == expected_table.target_bitsizes
    assert table.num_controls == expected_table.num_controls
    if table.datasets is not None:
        assert table.datasets == tuple(
            tuple(int(x) for x in a.flat) for a in expected_table.data
        )
    rhs_cost = get_cost_value(original, QECGatesCost())
    native_rhs_cost = native.resource_gates()
    for field in (
        "and_bloq",
        "clifford",
        "measurement",
        "t",
        "toffoli",
        "rotation",
        "cswap",
    ):
        assert (
            sympy.simplify(getattr(native_rhs_cost, field) - getattr(rhs_cost, field))
            == 0
        )
    calls = original.build_call_graph(SympySymbolAllocator())
    roles = {}
    for child, multiplicity in calls.items():
        name = type(child).__name__
        if name == "QROM":
            role = "lookup"
            assert child == expected_table
        elif name == "Adjoint":
            role = "unload"
            assert child.subbloq == expected_table
        elif name == "Hadamard":
            role = "h"
        elif name == "RzViaPhaseGradient":
            role = "rz"
            assert child.angle_dtype == child.phasegrad_dtype == original._angle_dtype
        elif name == "LessThanConstant":
            role = "sign"
            assert child.bitsize == native.m_dtype.num_qubits
            assert child.less_than_val == native.num_negative
        else:
            assert name == "Controlled" and type(child.subbloq).__name__ == "SGate"
            role = "cs"
        roles[role] = multiplicity
    graph = native.source_graph(
        root="rhs",
        lookup="lookup",
        inverse_lookup="lookup",
        hadamard="h",
        rotation="rz",
        sign_compare="sign",
        controlled_s="cs",
    )
    assert (
        graph.leaf_counts == roles == {"lookup": 2, "h": 2, "rz": 1, "sign": 1, "cs": 1}
    )
    for values in ((0,), (1,), (0, 1)):
        actual = original.controlled(CtrlSpec(cvs=np.array(values)))
        route = native.source_control_route(values)
        if len(values) > 1 or (not controlled and values == (0,)):
            assert route == "generic" and type(actual).__name__ == "ControlledViaAnd"
        elif controlled:
            assert (
                route == "conjunction"
                and type(actual).__name__ == "_MultiControlledFromSinglyControlled"
            )
            assert actual.cvs == (*values, 1)
        else:
            assert route == "specialized" and actual.is_controlled
            assert native.with_positive_control().is_controlled
    with pytest.raises(DecomposeNotImplementedError):
        original.decompose_bloq()
    with pytest.raises(DecomposeNotImplementedError):
        source_index.decompose_bloq()


def test_loader_validation_and_symbolic_execution_boundary() -> None:
    with pytest.raises(ValueError, match="concrete"):
        symbolic_scope_loader().make_on((), (), workspace=(), classical_bits=())
    with pytest.raises(ValueError, match="widths"):
        example_scope_loader().make_on((), (), workspace=(), classical_bits=())


def test_original_full_example_emitted_stream_and_workspace() -> None:
    native, original = example_scope_loader(), source._load_scopes()
    j, output, work = tuple(range(3)), tuple(range(3, 15)), (15, 16)
    slots = (90, 91)
    operation = native.make_on(j, output, workspace=work, classical_bits=slots)
    expected = reference_arithmetic_gates(
        original,
        {"j": j},
        (*output, *work),
        dict(zip(work, slots, strict=True)),
        cirq_style=True,
        expected_outputs={"j": j, "U": output},
    )
    assert_source(operation.instructions, expected)
    assert len(operation.instructions) == len(expected) == 86
    assert sum(g.name == "measure" for g in expected) == 3
    assert native.qrom.workspace_size == 2


def test_scope_loader_uses_replaceable_qrom_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = example_scope_loader()
    called = []

    def alternate(
        table: QromTable,
        ports: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        assert table == loader.qrom
        called.append((ports, work, slots))
        return (Instruction("x", (ports[3],)),)

    monkeypatch.setattr(QromTable, "bind_packed", alternate)
    operation = loader.make_on(
        tuple(range(3)),
        tuple(range(3, 15)),
        workspace=(15, 16),
        classical_bits=(80, 81),
    )
    circuit = Circuit(17).append(operation)
    assert called == [(tuple(range(15)), (15, 16), (80, 81))]
    assert circuit.to_ir().instructions == (Instruction("x", (3,)),)
