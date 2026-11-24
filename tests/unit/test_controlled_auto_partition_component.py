"""Keep source data partitions outside explicit positive child control."""

import pytest
import torch

from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.controlled_auto_partition_component import (
    controlled_auto_partition_pair,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_indexed_selection_component import positive_gate_pairs
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("padding", (0, 1, 2))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_controlled_partition_calls_and_complete_operator(
    padding: int, inverse: bool
) -> None:
    from qualtran import Controlled, CtrlSpec, QAny
    from qualtran import Register as SourceRegister
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.bookkeeping.auto_partition import (
        AutoPartition as SourcePartition,
    )
    from qualtran.bloqs.bookkeeping.auto_partition import (
        Unused,
    )

    source = Controlled(
        SourcePartition(
            Hadamard(),
            (
                (
                    SourceRegister("system", QAny(1 + padding)),
                    ("q", *((Unused(padding),) if padding else ())),
                ),
            ),
        ),
        CtrlSpec(),
    )
    pair = positive_gate_pairs()[1]
    native = controlled_auto_partition_pair(
        *pair,
        (Register("system", opaque_register_type(1 + padding)),),
        AutoPartition(
            (("system", (("q", 1), *(((None, padding),) if padding else ()))),)
        ),
        data_registers=(Register("q", quantum_bit_type()),),
    )[int(inverse)]
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(native)
    assert all(f.blocks for f in native.functions)
    compare_layers(native, source, atomic_leaves(native))
    logical = 2 + padding
    bound = bind_register_program(
        native, ((0,), tuple(range(1, logical))), clean=(logical,)
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=6e-11,
        rtol=0,
    )
