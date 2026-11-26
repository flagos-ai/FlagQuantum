"""Own symbolic SELECT rows, then execute only explicitly materialized widths."""

import builtins
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pickle import dumps, loads
from types import ModuleType
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._pauli_select_component_library import (
    capture_pauli_select_pair,
)
from flagquantum.ecosystem.cirq._select_pauli_lcu import select_pauli_lcu_from_cirq
from flagquantum.errors import ValidationError
from tests.unit.pauli_select_reference import (
    assert_all_branch_operators,
    compile_pauli_select,
)
from tests.unit.pauli_select_reference import source as source
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("rows", (2, 3))
@pytest.mark.parametrize("control", (None, 0, 1))
def test_symbolic_owned_table_and_explicit_execution_without_source_callbacks(
    source: Any, rows: int, control: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    from attrs import evolve
    from qualtran import Adjoint, DecomposeTypeError
    from qualtran.resource_counting import QubitCount, get_cost_value

    symbol = sympy.Symbol("n", integer=True, positive=True)
    expression = symbol if rows == 2 else symbol + 1
    words = ("-X", "Y", "-Z")[:rows]
    original = source.SelectPauliLCU(
        expression,
        1,
        tuple(
            cirq.DensePauliString(
                word.removeprefix("-"), coefficient=-1 if word.startswith("-") else 1
            )
            for word in words
        ),
        control,
    )
    assert original.selection_registers[0].bitsize == expression
    with pytest.raises(DecomposeTypeError, match="parameterized"):
        original.decompose_bloq()
    operation = select_pauli_lcu_from_cirq(original)
    assert operation.selection_bitsize is expression
    assert operation.select_unitaries == words
    assert loads(dumps(operation)) == operation
    assert not isinstance(operation.selection_bitsize, int)
    assert next(iter(operation.selection_bitsize.free_symbols)).is_positive is True
    with pytest.raises(ValueError, match="requires a concrete selection width"):
        capture_pauli_select_pair(original)
    with pytest.raises(ValidationError, match="declared widths"):
        Circuit(5).append(operation.make_on(selection=(0, 1), target=(2,), control=()))

    imported: Callable[..., ModuleType] = builtins.__import__

    def native_only(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError(f"Native execution imported source SDK {name}")
        return imported(name, *args, **kwargs)

    minimum = (rows - 1).bit_length()
    for width in (minimum, minimum + 1):
        concrete = evolve(original, selection_bitsize=width)
        materialized = replace(operation, selection_bitsize=width)
        assert materialized == select_pauli_lcu_from_cirq(concrete)
        assert materialized.select_unitaries is operation.select_unitaries
        pair = capture_pauli_select_pair(concrete)
        expected = torch.as_tensor(concrete.tensor_contract(), dtype=torch.complex128)
        offset = int(control is not None)
        logical = offset + width + 1
        work = unary_iteration_workspace(width, control)
        controls = (0,) if offset else ()
        ports = {
            "selection": tuple(range(offset, offset + width)),
            "target": (offset + width,),
            "control": controls,
        }
        with monkeypatch.context() as isolated:
            isolated.setattr(builtins, "__import__", native_only)
            circuit = DynamicCircuit(logical + work)
            circuit.append(
                materialized.make_bounded_on(
                    **ports,
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(work)),
                )
            )
            logical_circuit = Circuit(logical).append(materialized.make_on(**ports))
            assert_all_branch_operators(circuit.to_ir(), logical, expected)
        assert (
            logical_circuit.to_ir().instructions[0].params["select_unitaries"] == words
        )
        source_ir, source_work = compile_pauli_select(concrete, words)
        assert source_work == work
        assert_source_dependencies(circuit.to_ir().instructions, source_ir.instructions)
        for inverse, record in enumerate(pair):
            selected = Adjoint(concrete) if inverse else concrete
            terminals = atomic_leaves(record) | {
                f.name for f in record.functions if "pauli_leaf_" in f.name
            }
            compare_layers(record, selected, terminals)
            bound, bound_logical = bind(record)
            assert bound_logical == logical
            assert bound.circuit.n_wires == get_cost_value(selected, QubitCount())
            assert Counter(g.name for g in bound.circuit.instructions) == Counter(
                g.name for g in source_ir.instructions
            )
            assert_all_branch_operators(
                bound.circuit, logical, expected.conj().T if inverse else expected
            )


def test_symbol_assumptions_and_capacity_refusals_follow_original(source: Any) -> None:
    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    from types import SimpleNamespace

    rows = (cirq.DensePauliString("X"), cirq.DensePauliString("Y"))
    unknown = sympy.Symbol("n", integer=True)
    with pytest.raises(TypeError, match="truth value"):
        source.SelectPauliLCU(unknown, 1, rows)
    with pytest.raises(TypeError, match="truth value"):
        select_pauli_lcu_from_cirq(
            SimpleNamespace(
                selection_bitsize=unknown,
                target_bitsize=1,
                select_unitaries=rows,
                control_val=None,
            )
        )
    negative = sympy.Symbol("n", integer=True, negative=True)
    with pytest.raises(ValueError, match="at-least"):
        source.SelectPauliLCU(negative, 1, rows)
    with pytest.raises(ValueError, match="nonnegative selection width"):
        select_pauli_lcu_from_cirq(
            SimpleNamespace(
                selection_bitsize=negative,
                target_bitsize=1,
                select_unitaries=rows,
                control_val=None,
            )
        )
