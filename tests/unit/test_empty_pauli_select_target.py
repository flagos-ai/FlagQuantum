"""Empty Pauli targets retain relative phase and linear storage ownership."""

from collections import Counter
from pickle import dumps, loads
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler.select_lowering import lower_select_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.physical_codec import (
    native_program_to_dict,
)
from flagquantum.ecosystem._pauli_select_component_library import (
    capture_pauli_select_pair,
)
from flagquantum.ecosystem.cirq._select_pauli_lcu import select_pauli_lcu_from_cirq
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.pauli_select_reference import (
    assert_all_branch_operators,
    compile_pauli_select,
)
from tests.unit.pauli_select_reference import source as source
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_multi_control_component import atomic_leaves

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
TABLES = (
    (1, ("", "-")),
    (2, ("", "-", "-")),
    (2, ("", "-", "", "-")),
    (1, ("", "")),
    (1, ("-", "-")),
)


@pytest.mark.parametrize("width,words", TABLES)
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_source_empty_target_graph_branches_ownership_and_roundtrip(
    source: Any, width: int, words: tuple[str, ...], control: int | None, inverse: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import Adjoint
    from qualtran.resource_counting import QubitCount, get_cost_value

    original = source.SelectPauliLCU(
        width,
        0,
        tuple(
            cirq.DensePauliString("", coefficient=-1 if word else 1) for word in words
        ),
        control,
    )
    selected = Adjoint(original) if inverse else original
    record = capture_pauli_select_pair(original)[int(inverse)]
    assert loads(dumps(record)) == record
    # That codec is deliberately limited to physical-context Programs.
    with pytest.raises(SerializationError, match="requires a physical context"):
        native_program_to_dict(record)
    terminals = atomic_leaves(record) | {
        f.name for f in record.functions if "pauli_leaf_" in f.name
    }
    compare_layers(record, selected, terminals)
    bound, logical = bind(record)
    offset = int(control is not None)
    expected_outputs: tuple[tuple[int, ...], ...] = (
        tuple(range(offset, offset + width)),
        (),
    )
    if control is not None:
        expected_outputs = ((0,), *expected_outputs)
    assert bound.outputs == expected_outputs
    assert logical == width + int(control is not None)
    assert bound.circuit.n_wires == get_cost_value(selected, QubitCount())
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    expected = expected.conj().T if inverse else expected
    assert_all_branch_operators(bound.circuit, logical, expected)
    assert (
        CircuitIR.from_json(bound.circuit.to_json()).instructions
        == bound.circuit.instructions
    )
    if not inverse:
        ir, _ = compile_pauli_select(original, words)
        assert Counter(g.name for g in ir.instructions) == Counter(
            g.name for g in bound.circuit.instructions
        )


@pytest.mark.parametrize("width,words", TABLES)
@pytest.mark.parametrize("control", (None, 0, 1))
def test_existing_logical_dense_unary_and_ladder_paths_keep_empty_target_phase(
    source: Any, width: int, words: tuple[str, ...], control: int | None
) -> None:
    cirq = pytest.importorskip("cirq")
    original = source.SelectPauliLCU(
        width,
        0,
        tuple(
            cirq.DensePauliString("", coefficient=-1 if word else 1) for word in words
        ),
        control,
    )
    logical = width + int(control is not None)
    operation = select_pauli_lcu_from_cirq(original)
    circuit = Circuit(logical).append(
        operation.make_on(
            selection=tuple(range(int(control is not None), logical)),
            target=(),
            control=() if control is None else (0,),
        )
    )
    ir = CircuitIR.from_json(circuit.to_ir().to_json())
    diagonal = torch.ones(1 << logical, dtype=torch.complex128)
    for index in range(1 << logical):
        address = index & ((1 << width) - 1)
        active = control is None or index >> width == control
        if active and address < len(words) and words[address]:
            diagonal[index] = -1
    expected = torch.diag(diagonal)
    torch.testing.assert_close(
        get_unitary(ir, dtype=torch.complex128), expected, atol=3e-11, rtol=0
    )
    for path in ("dense", "dynamic_unary_iteration", "unitary_ladder"):
        lowered = lower_select_program(ir, path)
        assert_all_branch_operators(lowered, logical, expected)


@pytest.mark.parametrize("control", (0, 1))
def test_existing_logical_constant_selection_controlled_phase(
    source: Any, control: int
) -> None:
    cirq = pytest.importorskip("cirq")
    original = source.SelectPauliLCU(
        0, 0, (cirq.DensePauliString("", coefficient=-1),), control
    )
    with pytest.raises(NotImplementedError, match="must not be empty"):
        original.decompose_bloq()
    operation = select_pauli_lcu_from_cirq(original)
    ir = (
        Circuit(1)
        .append(operation.make_on(selection=(), target=(), control=(0,)))
        .to_ir()
    )
    expected = torch.diag(
        torch.tensor(
            [-1 if bit == control else 1 for bit in (0, 1)], dtype=torch.complex128
        )
    )
    for path in ("dense", "dynamic_unary_iteration", "unitary_ladder"):
        assert_all_branch_operators(lower_select_program(ir, path), 1, expected)


@pytest.mark.parametrize("control", (None, False, True))
def test_boolean_empty_target_metadata_composes_with_approved_admission(
    source: Any, control: bool | None
) -> None:
    cirq = pytest.importorskip("cirq")
    original = source.SelectPauliLCU(
        True,
        False,
        (cirq.DensePauliString(""), cirq.DensePauliString("", coefficient=-1)),
        control,
    )
    compile_pauli_select(original, ("", "-"))
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    for record in capture_pauli_select_pair(original):
        bound, logical = bind(record)
        assert bound.outputs[-1] == ()
        assert_all_branch_operators(bound.circuit, logical, expected)
