"""Verify the original controlled Phase root and its complete complex action."""

import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.phase.controlled_block_phase_component import (
    controlled_block_phase_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_indexed_selection_component import positive_gate_pairs
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def controlled_h_block() -> ProgramRecord:
    """Supply an exact opaque one-bit controlled block at the child boundary."""
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("ctrl", quantum_bit_type()),
                Register("system", opaque_register_type(1)),
            )
        )
    )
    system = builder.inputs["system"]
    assert isinstance(system, RegisterBinding)
    bit = builder.split(system)[0]
    child = positive_gate_pairs()[1][0]
    result = builder.call_program(
        child,
        {"ctrl": builder.inputs["ctrl"], "target": bit},
        entry=child.functions[0].name,
    )
    target = result["target"]
    assert isinstance(target, RegisterBinding)
    return builder.finalize(
        {
            "ctrl": result["ctrl"],
            "system": builder.join((target,), opaque_register_type(1)),
        }
    )


@pytest.mark.parametrize("exponent", (0.0, 0.25, -0.5, 1.0))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_controlled_phase_root_and_full_operator(
    exponent: float, inverse: bool
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary

    source = Controlled(Phase(Unitary(Hadamard()), phi=exponent, eps=0), CtrlSpec())
    child = controlled_h_block()
    record = controlled_block_phase_program(
        child, exponent, inverse=child, adjoint=inverse
    )
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    # This proof owns the Phase root; the explicitly supplied child is a boundary.
    terminals = {f.name for f in record.functions[1:]}
    compare_layers(record, source, terminals)
    bound = bind_register_program(record, ((0,), (1,)), clean=(2, 3))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=6e-11,
        rtol=0,
    )
