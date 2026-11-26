"""Keep complete source SDK metadata and stored-graph failures distinct."""

from typing import Any

import pytest
import torch

from flagquantum.ecosystem._pauli_select_component_library import (
    capture_pauli_select_pair,
)
from flagquantum.ecosystem.cirq._select_pauli_lcu import select_pauli_lcu_from_cirq
from tests.unit.pauli_select_reference import compile_pauli_select
from tests.unit.pauli_select_reference import source as source
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind", ("bool", "symbolic", "numpy_control", "float_control", "numpy_bool_control")
)
@pytest.mark.parametrize("control", (None, 0, 1))
def test_complete_sdk_metadata_calls_complex_branches_and_normal_append(
    source: Any, kind: str, control: int | None
) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    sympy = pytest.importorskip("sympy")
    width, target = (2, 1) if kind == "symbolic" else (1, 1)
    words = ("-X", "Y", "Z")[: 3 if width == 2 else 2]
    rows = tuple(
        cirq.DensePauliString(
            w.removeprefix("-"), coefficient=-1 if w.startswith("-") else 1
        )
        for w in words
    )
    selection_metadata: object = width
    target_metadata: object = target
    control_metadata: object = control
    if kind == "bool":
        selection_metadata, target_metadata = True, True
        control_metadata = None if control is None else bool(control)
    elif kind == "symbolic":
        selection_metadata, target_metadata = sympy.Integer(width), sympy.Integer(
            target
        )
        control_metadata = None if control is None else sympy.Integer(control)
    elif control is not None:
        control_metadata = {
            "numpy_control": np.int64,
            "float_control": float,
            "numpy_bool_control": np.bool_,
        }[kind](control)
    original = source.SelectPauliLCU(
        selection_metadata, target_metadata, rows, control_metadata
    )
    canonical = source.SelectPauliLCU(width, target, rows, control)
    converted = select_pauli_lcu_from_cirq(original)
    assert type(converted.selection_bitsize) is type(converted.target_bitsize) is int
    assert converted.control_val is None or type(converted.control_val) is int
    assert converted == select_pauli_lcu_from_cirq(canonical)
    pair = capture_pauli_select_pair(original)
    assert pair == capture_pauli_select_pair(canonical)
    if kind == "symbolic":
        from qualtran import DecomposeTypeError
        from qualtran.resource_counting import QECGatesCost, get_cost_value

        with pytest.raises(DecomposeTypeError, match="parameterized"):
            original.decompose_bloq()
        with pytest.raises(AttributeError, match="bit_length"):
            get_cost_value(original, QECGatesCost())
        ir, work = compile_pauli_select(canonical, words)
        offset = int(control is not None)
        logical = offset + width + target
        scratch = tuple(range(logical, logical + work))
        ports = {
            "selection": tuple(range(offset, offset + width)),
            "target": tuple(range(offset + width, logical)),
            **({"control": (0,)} if control is not None else {}),
        }
        gates = reference_arithmetic_gates(
            original,
            ports,
            scratch,
            dict(zip(scratch, range(work), strict=True)),
            cirq_style=True,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(ir.instructions, gates)
    else:
        assert (
            original.decompose_bloq().signature == canonical.decompose_bloq().signature
        )
        compile_pauli_select(original, words)
    expected = torch.as_tensor(canonical.tensor_contract(), dtype=torch.complex128)
    for record in pair:
        bound, logical = bind(record)
        assert_all_small_raw_histories(bound.circuit, logical, expected)


@pytest.mark.parametrize("kind", ("numpy", "float"))
def test_original_invalid_widths_are_not_silently_cast(source: Any, kind: str) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    width = np.int64(1) if kind == "numpy" else 1.0
    original = source.SelectPauliLCU(width, width, (cirq.DensePauliString("X"),) * 2, 1)
    with pytest.raises(ValueError, match="Bad bitsize"):
        original.decompose_bloq()
    with pytest.raises(ValueError, match="selection width"):
        select_pauli_lcu_from_cirq(original)
    with pytest.raises(ValueError, match="selection width"):
        capture_pauli_select_pair(original)
