"""Retain complete available generic Majorana and sparse source call graphs."""

from dataclasses import replace
from importlib import import_module
from typing import Any, Literal

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import cirq

    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.registers.lookup.majorana_control_component import (
    controlled_majorana_program,
)
from flagquantum.compiler.registers.lookup.sparse_selection_control import (
    generic_sparse_select_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _source
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_generic_controlled_word_components import (
    frozen_generic_controls as frozen_generic_controls,
)
from tests.unit.test_legacy_majorana_factories import pinned_majorana as pinned_majorana
from tests.unit.test_momentum_source_declarations import frozen_source as frozen_source
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_cy_profile() -> Literal["cirq", "scalar"]:
    from qualtran.cirq_interop import cirq_gate_to_bloq

    ports = tuple(cirq_gate_to_bloq(cirq.Y.controlled()).signature)
    if tuple(port.name for port in ports) == ("q",):
        assert ports[0].shape == (2,)
        return "cirq"
    assert tuple(port.name for port in ports) == ("ctrl", "target")
    return "scalar"


def compare_available_graph(record: ProgramRecord, source: Any) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    assert verify_program_record(record) == ()
    leaves = atomic_leaves(record)
    pairs = compare_layers(record, source, leaves)
    functions = {function.name: function for function in record.functions}
    missing = [name for name in pairs if not functions[name].blocks]
    assert missing
    for name in missing:
        with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
            pairs[name].decompose_bloq()
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_used_workspace(record, allow_workspace_outputs=True)


@pytest.mark.parametrize("pauli", ("X", "Y"))
@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_generic_majorana_complete_available_calls(
    frozen_generic_controls: None,
    pauli: Literal["X", "Y"],
    controlled: bool,
    inverse: bool,
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    selections = tuple(
        Register(name, bounded_unsigned_register_type(1, 2)) for name in ("p", "spin")
    )
    control = Register("control", quantum_bit_type()) if controlled else None
    record = controlled_majorana_program(
        selections,
        pauli,
        control=control,
        adjoint=inverse,
        cy_profile=source_cy_profile(),
    )
    observer = _source(Signature([]))
    original = SelectedMajoranaFermion(
        selection_regs=tuple(observer.signature(Signature(selections))),
        control_regs=tuple(
            observer.signature(Signature((control,) if control else ()))
        ),
        target_gate=cirq.X if pauli == "X" else cirq.Y,
    )
    source = Controlled(original.adjoint() if inverse else original, CtrlSpec())
    compare_available_graph(record, source)


@pytest.mark.parametrize("size", (4, 6, 8))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_generic_sparse_five_call_root_and_majorana_dependencies(
    frozen_generic_controls: None,
    size: int,
    control: int | None,
    inverse: bool,
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse

    original = SelectSparse(size, control)
    source = Controlled(original.adjoint() if inverse else original, CtrlSpec())
    record = generic_sparse_select_program(
        size, control_value=control, adjoint=inverse, cy_profile=source_cy_profile()
    )
    compare_available_graph(record, source)


@pytest.mark.parametrize("inverse", (False, True))
def test_sparse_atomic_controlled_phase_full_operator_and_append(
    frozen_generic_controls: None,
    inverse: bool,
) -> None:
    import torch
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import SGate

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from tests.unit.test_default_linear_combination_oracles import bind
    from tests.unit.test_multiplexed_source_binding import logical_operator

    record = generic_sparse_select_program(4, adjoint=inverse)
    operations = record.functions[0].blocks[0].operations
    phase = operations[-1 if inverse else 0].callee
    function = next(f for f in record.functions if f.name == phase)
    atomic = ProgramRecord((replace(function, name="main"),))
    bound, logical = bind(atomic)
    assert logical == bound.circuit.n_wires == 2
    source = Controlled(SGate(is_adjoint=inverse), CtrlSpec())
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=1e-12, rtol=0
    )
    assert len(bound.circuit.instructions) == 1
    assert bound.circuit.instructions[0].name == "cphase"
    circuit = Circuit(2).append(_BoundBlockOperation(bound.circuit.instructions, 2))
    assert circuit.to_ir().instructions == bound.circuit.instructions


@pytest.mark.parametrize("size,peak", ((4, 16), (6, 23), (8, 25)))
@pytest.mark.parametrize("inverse", (False, True))
def test_sparse_generic_logical_peak_and_original_missing_gate_model(
    frozen_generic_controls: None,
    size: int,
    peak: int,
    inverse: bool,
) -> None:
    from qualtran import Controlled, CtrlSpec, DecomposeTypeError
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler._component_trace.program_logical_width import (
        query_program_logical_width,
    )

    original = SelectSparse(size)
    source = Controlled(original.adjoint() if inverse else original, CtrlSpec())
    record = generic_sparse_select_program(
        size, adjoint=inverse, cy_profile=source_cy_profile()
    )
    assert (
        query_program_logical_width(record)
        == get_cost_value(source, QubitCount())
        == peak
    )
    # A logical peak is available even though the source cannot expand its
    # Controlled(And) cost. Neither quantity is a physical primitive inventory.
    with pytest.raises(DecomposeTypeError, match="And"):
        get_cost_value(source, QECGatesCost())
