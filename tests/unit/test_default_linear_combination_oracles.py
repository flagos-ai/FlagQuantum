"""Close actual default oracle roles without inferring omitted source payloads."""

from importlib.metadata import version
from math import prod
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._default_linear_combination_oracles import (
    default_linear_combination_prepare_pair,
    default_linear_combination_select_pair,
)
from flagquantum.ecosystem._linear_combination_component_library import (
    capture_linear_combination_context,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def default_source() -> Any:
    import kickmix
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    return _linear_combination_block_encoding.make()


def bind(record: ProgramRecord) -> tuple[Any, int]:
    inputs = []
    offset = 0
    for register in record.functions[0].registers or ():
        for _ in range(prod(register.shape)):
            size = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + size)))
            offset += size
    provisional = bind_register_program(
        record, tuple(inputs), clean=tuple(range(offset, offset + 32))
    )
    used = {wire for event in provisional.events for wire in event.wires}
    work = max(used, default=offset - 1) + 1
    return (
        bind_register_program(record, tuple(inputs), clean=tuple(range(offset, work))),
        offset,
    )


@pytest.mark.parametrize("positive", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_default_select_full_calls_operator_and_all_histories(
    positive: bool, inverse: bool
) -> None:
    from qualtran import CtrlSpec

    source = default_source().select
    if positive:
        source = source.get_ctrl_system(CtrlSpec())[0]
    record = default_linear_combination_select_pair(source)[int(inverse)]
    if inverse:
        source = source.adjoint()
    assert not verify_program_record(record)
    assert all(f.blocks for f in record.functions)
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=8e-11,
        rtol=0,
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_original_default_prepare_recursive_calls(inverse: bool) -> None:
    source = default_source().prepare
    record = default_linear_combination_prepare_pair(source)[int(inverse)]
    compare_layers(
        record, source.adjoint() if inverse else source, atomic_leaves(record)
    )
    assert not verify_program_record(record)
    assert all(f.blocks for f in record.functions)
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(
            (source.adjoint() if inverse else source).tensor_contract(),
            dtype=torch.complex128,
        ),
        atol=8e-11,
        rtol=0,
    )


@pytest.mark.parametrize("positive", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_actual_default_constructor_context_closes_recursive_calls(
    positive: bool, inverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint

    source = default_source()
    if positive:
        source = source.get_ctrl_system(CtrlSpec())[0]
    context = capture_linear_combination_context(
        source,
        prepare_owner=default_linear_combination_prepare_pair,
        select_owner=default_linear_combination_select_pair,
    )
    record = context.program(adjoint=inverse)
    expected = Adjoint(source) if inverse else source
    assert not verify_program_record(record)
    assert all(f.blocks for f in record.functions)
    compare_layers(record, expected, atomic_leaves(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(expected.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


def test_captured_default_runs_without_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_linear_combination_factories import (
        linear_combination_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = default_source()
    context = capture_linear_combination_context(
        source,
        prepare_owner=default_linear_combination_prepare_pair,
        select_owner=default_linear_combination_select_pair,
    )
    library = _library(source, 0)
    expected, _ = bind(context.program())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(
            library, 0, linear_combination_factories({"main": context})
        )
        actual, _ = bind(record)
        circuit = Circuit(actual.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(actual.circuit.instructions, actual.circuit.n_wires)
        )
        assert circuit.to_ir().instructions == expected.circuit.instructions


def test_eight_bit_source_table_complex_branches_and_roundtrip() -> None:
    from attrs import evolve
    from qualtran._infra.gate_with_registers import get_named_qubits

    from tests.unit.measurement_certificate import certify_erasure_histories
    from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
    from tests.unit.test_sparse_chemistry_prepare import _branch

    cirq = pytest.importorskip("cirq")
    source = evolve(default_source(), lambd_bits=8, lambd=(0.125, -0.25, 0.375, -0.25))
    assert source.prepare.prepare.mu == 8
    context = capture_linear_combination_context(
        source,
        prepare_owner=default_linear_combination_prepare_pair,
        select_owner=default_linear_combination_select_pair,
    )
    forward, logical = bind(context.program())
    backward, _ = bind(context.program(adjoint=True))
    named = get_named_qubits(source.signature)
    visible = tuple(q for reg in source.signature for q in named[reg.name].flat)
    operations = list(
        cirq.decompose(
            source.on_registers(**named),
            keep=lambda op: cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
        )
    )
    initial = {0: 0.6 + 0j, 1 << (logical - 1): 0.8j}
    expected = _source_sparse_branch(operations, visible, initial, 0)
    work = forward.circuit.n_wires - logical
    for pattern in (0, 1):
        events = certify_erasure_histories(
            forward.circuit.instructions,
            n_wires=forward.circuit.n_wires,
            zero_wires=tuple(range(logical, forward.circuit.n_wires)),
            check_clean_output=False,
        )
        actual = _branch(
            forward.circuit.instructions,
            forward.circuit.n_wires,
            (pattern,) * events,
            {k << work: v for k, v in initial.items()},
            normalize_measurements=True,
        )
        assert all(key % (1 << work) == 0 for key in actual)
        decoded = {k >> work: v for k, v in actual.items()}
        assert all(
            abs(decoded.get(k, 0) - expected.get(k, 0)) < 2e-10
            for k in decoded.keys() | expected.keys()
        )
        inverse_work = backward.circuit.n_wires - logical
        inverse_events = certify_erasure_histories(
            backward.circuit.instructions,
            n_wires=backward.circuit.n_wires,
            zero_wires=tuple(range(logical, backward.circuit.n_wires)),
            check_clean_output=False,
        )
        restored = _branch(
            backward.circuit.instructions,
            backward.circuit.n_wires,
            (pattern,) * inverse_events,
            {k << inverse_work: v for k, v in decoded.items()},
            normalize_measurements=True,
        )
        target = {k << inverse_work: v for k, v in initial.items()}
        assert all(
            abs(restored.get(k, 0) - target.get(k, 0)) < 3e-10
            for k in restored.keys() | target.keys()
        )
