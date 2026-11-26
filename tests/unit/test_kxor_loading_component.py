"""Actual k-XOR lookup calls transfer scope storage and retain bodyless controls."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.algorithms.kxor_instance import Constraint, KXorInstance
from flagquantum.algorithms.kxor_loading import (
    LoadConstraintScopes,
    LoadUniqueScopeIndex,
    PRGAUniqueConstraintRHS,
)
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.ecosystem._kxor_loading_component_library import kxor_loading_component
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def oracles(kind: str) -> tuple[Any, Any]:
    from qualtran.bloqs.optimization.k_xor_sat import load_kxor_instance as original
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        Constraint as OriginalConstraint,
    )
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        KXorInstance as OriginalInstance,
    )

    clauses = (((0, 1), -1), ((0, 2), 1), ((1, 3), 1))
    native = KXorInstance(4, 2, tuple(Constraint(s, b) for s, b in clauses))
    source = OriginalInstance(4, 2, tuple(OriginalConstraint(s, b) for s, b in clauses))
    if kind == "scopes":
        return LoadConstraintScopes(native), original.LoadConstraintScopes(source)
    if kind == "index":
        return LoadUniqueScopeIndex(native), original.LoadUniqueScopeIndex(source)
    return PRGAUniqueConstraintRHS(native, 3), original.PRGAUniqueConstraintRHS(
        source, 3
    )


def test_scope_allocate_qrom_partition_transfers_output_and_erases_only_private_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, source = oracles("scopes")
    graph = kxor_loading_component(oracle)
    assert not verify_program_record(graph)
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    with pytest.raises(ValueError, match="ownership transfer"):
        bind_register_program(graph, ((0, 1),), clean=tuple(range(2, 12)))
    bound = bind_register_program(
        graph, ((0, 1),), clean=tuple(range(2, 12)), allow_workspace_outputs=True
    )
    index, output = bound.outputs
    assert index == (0, 1) and len(output) == 4
    private = tuple(
        sorted({w for event in bound.events for w in event.wires} - set(output))
    )
    from flagquantum import Circuit
    from tests.unit.register_source_graph import assert_source_wire_streams
    from tests.unit.test_unary_component_calls import source_slot_names

    baseline = (
        Circuit(bound.circuit.n_wires)
        .append(
            oracle.make_on(
                index,
                output,
                workspace=private,
                classical_bits=tuple(range(len(private))),
            )
        )
        .to_ir()
        .instructions
    )
    assert_source_wire_streams(
        source_slot_names(bound.circuit.instructions, private),
        source_slot_names(baseline, private),
    )
    events = certify_erasure_histories(
        bound.circuit.instructions, n_wires=bound.circuit.n_wires, zero_wires=private
    )
    reference = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    weights = torch.tensor((1, 2j, -3, 0), dtype=torch.complex128)
    weights /= weights.norm()
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for value in range(4):
        initial[value << (bound.circuit.n_wires - 2)] = weights[value]
    logical = (*index, *output)
    for row, amplitude in enumerate(reference @ weights):
        address = sum(
            ((row >> (len(logical) - bit - 1)) & 1)
            << (bound.circuit.n_wires - wire - 1)
            for bit, wire in enumerate(logical)
        )
        expected[address] = amplitude
    import sys

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        executable = DynamicCircuit(
            bound.circuit.n_wires, inputs=initial, dtype=initial.dtype
        )
        executable.append(
            FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
        )
        result = run_dynamic(executable, shots=2, seed=21, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(2, -1),
            expected.expand(2, -1),
            atol=1e-11,
            rtol=0,
        )
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history)
            * 2 ** (events / 2),
            expected,
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("kind", ("scopes", "index", "rhs"))
@pytest.mark.parametrize("bits", (0, (0, 1)))
def test_actual_generic_control_wrappers_keep_positive_missing_body(
    kind: str, bits: Any
) -> None:
    from qualtran._infra.controlled import CtrlSpec

    oracle, source = oracles(kind)
    activation = np.array(bits)
    graph = kxor_loading_component(
        oracle, ControlSpecification(((quantum_bit_type(), activation),))
    )
    original = source.get_ctrl_system(CtrlSpec(cvs=activation))[0]
    assert not verify_program_record(graph)
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq())
    )
    assert any(not f.blocks for f in graph.functions)
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(graph, ())


@pytest.mark.parametrize("kind", ("scopes", "index", "rhs"))
def test_positive_declarations_preserve_original_missing_bodies(kind: str) -> None:
    from qualtran._infra.controlled import CtrlSpec

    from tests.unit.test_mean_estimation_component import _registers

    oracle, source = oracles(kind)
    graph = kxor_loading_component(
        oracle, ControlSpecification(((quantum_bit_type(), np.array(1)),))
    )
    original = source.get_ctrl_system(CtrlSpec())[0]
    assert graph.functions[0].registers == _registers(original.signature)
    assert not graph.functions[0].blocks
    assert not verify_program_record(graph)
    from qualtran._infra.bloq import DecomposeNotImplementedError, DecomposeTypeError

    with pytest.raises((DecomposeNotImplementedError, DecomposeTypeError)):
        original.decompose_bloq()


@pytest.mark.parametrize("bits", (0, 1, (0, 1)))
def test_existing_rhs_control_uses_original_shared_conjunction_calls(bits: Any) -> None:
    from dataclasses import replace

    import attrs
    from qualtran._infra.controlled import CtrlSpec

    oracle, source = oracles("rhs")
    oracle, source = replace(oracle, is_controlled=True), attrs.evolve(
        source, is_controlled=True
    )
    activation = np.array(bits)
    graph = kxor_loading_component(
        oracle, ControlSpecification(((quantum_bit_type(), activation),))
    )
    original = source.get_ctrl_system(CtrlSpec(cvs=activation))[0]
    assert not verify_program_record(graph)
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq())
    )
    assert any(not f.blocks for f in graph.functions)
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(graph, ())


def test_large_original_example_transfers_packed_output_with_identical_primitive_stream() -> (
    None
):
    from qualtran.bloqs.optimization.k_xor_sat.load_kxor_instance import _load_scopes

    from flagquantum import Circuit
    from flagquantum.algorithms.kxor_loading import example_scope_loader
    from tests.unit.register_source_graph import assert_source_wire_streams
    from tests.unit.test_unary_component_calls import source_slot_names

    oracle = example_scope_loader()
    graph = kxor_loading_component(oracle)
    source = _load_scopes.make()
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    bound = bind_register_program(
        graph, ((0, 1, 2),), clean=tuple(range(3, 19)), allow_workspace_outputs=True
    )
    index, output = bound.outputs
    assert output == tuple(range(3, 15))
    work = (15, 16)
    original_stream = (
        Circuit(17)
        .append(oracle.make_on(index, output, workspace=work, classical_bits=(0, 1)))
        .to_ir()
        .instructions
    )
    assert len(bound.circuit.instructions) == len(original_stream) == 86
    assert_source_wire_streams(
        source_slot_names(bound.circuit.instructions, work),
        source_slot_names(original_stream, work),
    )
