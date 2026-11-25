"""Compare complete signed Pauli SELECT calls and every small erasure branch."""

from collections import Counter
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.lookup.pauli_select_component import (
    pauli_select_component_pair,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._pauli_select_component_library import (
    capture_pauli_select_pair,
)
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_pauli_select_file import source as source
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("source"),
]


def test_original_factory_capture_retains_complete_operator(source: Any) -> None:
    original = source._select_pauli_lcu.make()
    record = capture_pauli_select_pair(original)[0]
    bound, logical = bind(record)
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=3e-11, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)


@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize(
    "words,width", ((("-XY", "ZI", "II"), 2), (("X", "-I", "Y", "Z"), 2))
)
def test_full_calls_complex_operator_and_all_small_histories(
    control: int | None,
    inverse: bool,
    words: tuple[str, ...],
    width: int,
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import Adjoint
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU

    source = SelectPauliLCU(
        width,
        len(words[0].removeprefix("-")),
        tuple(
            cirq.DensePauliString(
                word.removeprefix("-"), coefficient=-1 if word.startswith("-") else 1
            )
            for word in words
        ),
        control,
    )
    pair = capture_pauli_select_pair(source)
    record = pair[int(inverse)]
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    terminals = atomic_leaves(record) | {
        f.name for f in record.functions if "pauli_leaf_" in f.name
    }
    compare_layers(record, source, terminals)
    bound, logical = bind(record)
    offset = int(control is not None)
    ports = {
        "selection": tuple(range(offset, offset + width)),
        "target": tuple(range(offset + width, logical)),
        **({"control": (0,)} if control is not None else {}),
    }
    workspace = tuple(range(logical, logical + width + offset - 1))
    primitives = reference_arithmetic_gates(
        source.subbloq if inverse else source,
        ports,
        workspace,
        dict(zip(workspace, range(len(workspace)), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in primitives
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=3e-11, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)


@pytest.mark.parametrize("control", (None, 0, 1))
def test_native_word_validation_and_source_singleton_boundary(
    control: int | None,
) -> None:
    if control is None:
        with pytest.raises(ValueError, match="supported controls"):
            pauli_select_component_pair(1, 1, ("X",), control_value=control)
    else:
        record = pauli_select_component_pair(1, 1, ("-Y",), control_value=control)[0]
        assert not verify_program_record(record)
