"""Exercise LCU child ancilla/resource partitions with independent oracle owners."""

from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.registers.bits.on_each import on_each_word_program
from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.packed_oracle_component import (
    packed_oracle_component_program,
)
from flagquantum.compiler.registers.preparation.linear_combination_component import (
    linear_combination_component_program,
    linear_combination_signature,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_block_encoding_factories import _oracles
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("ancilla,resource", ((1, 0), (0, 1), (1, 1)))
@pytest.mark.parametrize("inverse", (False, True))
def test_packed_child_widths_calls_and_complete_complex_operator(
    ancilla: int, resource: int, inverse: bool
) -> None:
    from attrs import frozen
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    width = 1 + ancilla + resource

    class Select:
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("address", QBit()),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("data", QAny(width)),)

        def build_composite_bloq(
            self, bb: Any, address: Any, data: Any
        ) -> dict[str, Any]:
            return {"address": address, "data": bb.add(OnEach(width, XGate()), q=data)}

    prepare, _, prepare_pair, _, _ = _oracles()
    select_type = frozen(
        type("WideSelect", (Select, SelectOracle), {"__module__": __name__})
    )
    source = LinearCombination(
        (
            Unitary(XGate()),
            Unitary(XGate(), ancilla_bitsize=ancilla, resource_bitsize=resource),
        ),
        (0.5, 0.5),
        1,
        prepare=BlackBoxPrepare(prepare),
        select=BlackBoxSelect(select_type()),
    )
    bit, word = quantum_bit_type(), opaque_register_type(width)
    leaf = fixed_gate_factories()["qualtran.bloqs.basic_gates.x_basis.XGate"](
        "x", None, {}
    )
    assert isinstance(leaf, FunctionRecord)
    repeated = on_each_word_program(width, leaf)
    builder = ComponentProgramBuilder(
        Signature((Register("address", bit), Register("data", word)))
    )
    out = builder.call_program(repeated, {"q": builder.inputs["data"]})
    select = builder.finalize({"address": builder.inputs["address"], "data": out["q"]})
    leaf_program = ProgramRecord((leaf,))
    unrepeated = adjoint_component_calls(repeated, ((leaf_program, leaf_program),))
    unselect = adjoint_component_calls(select, ((repeated, unrepeated),))

    def pack(
        child: Any,
        pair: tuple[ProgramRecord, ProgramRecord],
        groups: tuple[tuple[str, tuple[Register, ...]], ...],
    ) -> tuple[ProgramRecord, ProgramRecord]:
        signature = Signature(_library(child, 0).entries[0].declaration.registers or ())
        forward, reverse = (
            packed_oracle_component_program(
                signature, groups, pair[0], inverse=pair[1], adjoint=direction
            )
            for direction in (False, True)
        )
        return forward, reverse

    native_prepare = pack(
        source.prepare,
        prepare_pair,
        (
            ("selection", (Register("address", bit),)),
            ("junk", (Register("junk", bit),)),
        ),
    )
    native_select = pack(
        source.select,
        (select, unselect),
        (
            ("selection", (Register("address", bit),)),
            ("system", (Register("data", word),)),
        ),
    )
    record = linear_combination_component_program(
        linear_combination_signature(
            1, (0, ancilla), (0, resource), prepare_selection=1, prepare_junk=1
        ),
        constituent_ancilla=ancilla,
        constituent_resource=resource,
        prepare_selection=1,
        prepare_junk=1,
        prepare_pair=native_prepare,
        select_pair=native_select,
        adjoint=inverse,
    )
    if inverse:
        source = Adjoint(source)
    compare_layers(record, source, atomic_leaves(record))
    logical = 3 + ancilla + resource
    bound = bind_register_program(
        record, ((0,), tuple(range(1, 2 + ancilla)), tuple(range(2 + ancilla, logical)))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )
