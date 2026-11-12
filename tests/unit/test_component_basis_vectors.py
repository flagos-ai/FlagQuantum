"""Integer state/effect functions use existing IR lifetime and loader owners."""

from collections import Counter
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_all_small_states_effects_and_owned_roundtrip(width: int) -> None:
    from qualtran.bloqs.basic_gates import IntState, OneState
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype = RegisterType("uint", width)
    for value in range(1 << width):
        state = integer_basis_function("state", dtype, value)
        effect = integer_basis_function("effect", dtype, value, state=False)
        assert verify_program_record(ProgramRecord((state, effect))) == ()
        bound = bind_register_program(
            ProgramRecord((state,)),
            (),
            entry="state",
            clean=tuple(range(width)),
            allow_workspace_outputs=True,
        )
        original = IntState(value, width)
        leaves = [
            b.bloq
            for b, _, _ in original.decompose_bloq().iter_bloqsoqs()
            if hasattr(b.bloq, "bit")
        ]
        active = [i for i, leaf in enumerate(leaves) if isinstance(leaf, OneState)]
        assert [g.wires for g in bound.circuit.instructions] == [
            (bound.outputs[0][i],) for i in active
        ]
        assert Counter(g.name for g in bound.circuit.instructions) == (
            {"x": value.bit_count()} if value else {}
        )
        expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
        initial = torch.zeros(1 << width, dtype=torch.complex128)
        initial[0] = 1
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, ()), expected
        )
        assert get_cost_value(original, QECGatesCost()).clifford == 1
        builder = ComponentProgramBuilder(Signature(()), (state, effect))
        word = builder.call_dict("state", {})["val"]
        builder.call_dict("effect", {"val": word})
        closed = bind_register_program(
            builder.finalize({}), (), clean=tuple(range(width))
        )
        torch.testing.assert_close(
            forced_branch(initial, closed.circuit.instructions, ()), initial
        )
        assert closed.outputs == ()


@pytest.mark.parametrize("width,value", [(0, 0), (2, -1), (2, 4), (2, 0.5), (2, True)])
def test_native_loader_refusals(width: int, value: Any) -> None:
    with pytest.raises(ValueError):
        integer_basis_function("state", RegisterType("uint", width), value)


def test_incremental_definition_reuses_constructor_conflict_rules() -> None:
    leaf = integer_basis_function("ket", RegisterType("bit", 1), 1)
    builder = ComponentProgramBuilder(Signature(()), name="root")
    builder.define(leaf)
    for duplicate in [leaf, integer_basis_function("root", RegisterType("bit", 1), 0)]:
        with pytest.raises(ValueError, match="conflicting"):
            builder.define(duplicate)
    invalid: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        integer_basis_function("ket", RegisterType("bit", 1), 0, state=invalid)
