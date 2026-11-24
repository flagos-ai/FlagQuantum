"""Preserve original Phase calls and exact scalar-carrier complex action."""

from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.compiler.registers.layout.packed_block_component import (
    unitary_block_component_program,
)
from flagquantum.compiler.registers.phase.phased_block_component import (
    phased_block_component_pair,
)
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def scalar_pair(exponent: float) -> tuple[ProgramRecord, ProgramRecord]:
    factory = scalar_phase_factories()[
        "qualtran.bloqs.basic_gates.global_phase.GlobalPhase"
    ]
    pair = tuple(
        factory("scalar", (), {"exponent": sign * exponent, "eps": 0})
        for sign in (1, -1)
    )
    assert all(isinstance(record, ProgramRecord) for record in pair)
    assert isinstance(pair[0], ProgramRecord) and isinstance(pair[1], ProgramRecord)
    return pair[0], pair[1]


def assert_phased_source(record: ProgramRecord, source: Any, inverse: bool) -> None:
    from qualtran._infra.adjoint import Adjoint

    if inverse:
        source = Adjoint(source)
    root = record.functions[0]
    # The scalar is atomic in the original source. Its native owned carrier is
    # an execution recipe, not a claimed upstream primitive decomposition.
    functions = {function.name: function for function in record.functions}
    scalars = {
        operation.callee
        for operation in root.blocks[0].operations
        if operation.callee is not None and functions[operation.callee].registers == ()
    }
    assert len(scalars) == 1
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record) | scalars)
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("gate", ("H", "T"))
@pytest.mark.parametrize("exponent", (0.0, 0.25, -0.5, 1.0))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_phase_root_calls_full_complex_operator_and_zero_carrier(
    gate: str, exponent: float, inverse: bool
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary

    suffix = "hadamard.Hadamard" if gate == "H" else "t_gate.TGate"
    factory = fixed_gate_factories()["qualtran.bloqs.basic_gates." + suffix]
    forward = factory("leaf", None, {})
    backward = factory("leaf_inverse", None, {"is_adjoint": 1} if gate == "T" else {})
    assert isinstance(forward, FunctionRecord) and isinstance(backward, FunctionRecord)
    child, child_inverse = ProgramRecord((forward,)), ProgramRecord((backward,))
    pair = tuple(
        unitary_block_component_program(child, inverse=child_inverse, adjoint=d)
        for d in (False, True)
    )
    native = phased_block_component_pair(pair[0], pair[1], *scalar_pair(exponent))[
        int(inverse)
    ]
    source = Phase(Unitary(Hadamard() if gate == "H" else TGate()), phi=exponent, eps=0)
    assert_phased_source(native, source, inverse)


@pytest.mark.parametrize("inverse", (False, True))
def test_scalar_carrier_with_original_measured_lcu_and_all_histories(
    inverse: bool,
) -> None:
    from qualtran.bloqs.block_encoding.phase import Phase

    from flagquantum.ecosystem._default_linear_combination_oracles import (
        default_linear_combination_prepare_pair,
        default_linear_combination_select_pair,
    )
    from flagquantum.ecosystem._linear_combination_component_library import (
        capture_linear_combination_context,
    )
    from tests.unit.test_default_linear_combination_oracles import default_source

    child = default_source()
    context = capture_linear_combination_context(
        child,
        prepare_owner=default_linear_combination_prepare_pair,
        select_owner=default_linear_combination_select_pair,
    )
    native = phased_block_component_pair(
        context.program(), context.program(adjoint=True), *scalar_pair(0.125)
    )[int(inverse)]
    assert_phased_source(native, Phase(child, phi=0.125, eps=0), inverse)
