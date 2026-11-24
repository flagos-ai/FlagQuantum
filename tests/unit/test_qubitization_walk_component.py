"""Compare native walk roots with literal source calls and complex operators."""

from importlib.metadata import version
from math import pi
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.preparation.block_encoding_component import (
    lcu_block_component_program,
    select_block_component_program,
)
from flagquantum.compiler.registers.preparation.identity_preparation_component import (
    identity_preparation_program,
)
from flagquantum.compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from flagquantum.compiler.registers.preparation.qubitization_walk_component import (
    qubitization_walk_component_pair,
)
from flagquantum.compiler.registers.primitives.global_phase_component import (
    scalar_phase_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from tests.unit.test_chebyshev_component import chebyshev_terminals
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_legacy_block_encoding_factories import _oracles
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_qubitization_walk_file import (
    complete_frozen_sources as complete_frozen_sources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def reference_runtime() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")


def walk_fixture(
    lcu: bool,
    *,
    controlled: bool = False,
) -> tuple[Any, tuple[ProgramRecord, ProgramRecord], ProgramRecord]:
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding,
        SelectBlockEncoding,
    )
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    prepare, select, preparation, selection, positive_selection = _oracles()
    block = (LCUBlockEncoding if lcu else SelectBlockEncoding)(select, prepare)
    source = QubitizationWalkOperator(block)
    positive_block = block.get_ctrl_system(CtrlSpec())[0] if controlled else block
    block_signature = Signature(_signature(positive_block.signature))

    def add_positive(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        result = builder.call_program(
            positive_selection[0], {"control": control, **values}
        )
        return result["control"], (result["address"], result["data"])

    if lcu:
        blocks = tuple(
            lcu_block_component_program(
                block_signature,
                *preparation,
                selection[0],
                controlled_select=add_positive if controlled else None,
                adjoint=inverse,
                inverse_pairs=(selection, positive_selection),
            )
            for inverse in (False, True)
        )
    elif not controlled:
        blocks = tuple(
            select_block_component_program(
                block_signature, selection[0], adjoint=inverse, inverse=selection[1]
            )
            for inverse in (False, True)
        )
    else:
        builder = ComponentProgramBuilder(block_signature)
        values = dict(builder.inputs)
        result = builder.call_program(
            positive_selection[0], {"control": values.pop("ctrl"), **values}
        )
        result["ctrl"] = result.pop("control")
        forward = builder.finalize(result)
        blocks = forward, adjoint_component_calls(forward, (positive_selection,))
    # LCU's signal state is the identity, although source.walk.prepare still
    # exposes its original PREPARE. Reflection must bind the actual signal.
    reflection_preparation = (
        identity_preparation_program(_signature(block.signal_state.selection_registers))
        if lcu
        else None
    )
    reflection_pair = (
        (reflection_preparation, reflection_preparation)
        if reflection_preparation is not None
        else preparation
    )
    reflection = prepared_reflection_program(
        *reflection_pair,
        ("address",),
        control_value=1 if controlled else None,
        global_phase=-1,
        scalar_phase=None if controlled else scalar_phase_program(pi),
    )
    if controlled:
        source = source.controlled()
    return source, (blocks[0], blocks[1]), reflection


@pytest.mark.parametrize("lcu", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
def test_source_order_named_storage_complex_operator_and_clean_work(
    lcu: bool,
    inverse: bool,
    controlled: bool,
) -> None:
    from qualtran._infra.adjoint import Adjoint

    source, blocks, reflection = walk_fixture(lcu, controlled=controlled)
    pair = qubitization_walk_component_pair(
        Signature(_signature(source.signature)),
        blocks,
        (reflection, reflection),
        controlled=controlled,
    )
    record = pair[int(inverse)]
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-11,
        rtol=0,
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_zero_port_walk_retains_complex_scalar_and_private_carrier(
    inverse: bool,
) -> None:
    import attrs
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity

    class ScalarSelect:
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return ()

        def build_composite_bloq(self, bb: Any) -> dict[str, Any]:
            bb.add(GlobalPhase(exponent=0.37))
            return {}

    source = QubitizationWalkOperator(
        SelectBlockEncoding(
            attrs.frozen(
                type("ScalarSelectOracle", (ScalarSelect, SelectOracle), {})
            )(),
            PrepareIdentity(()),
        )
    )
    scalars = scalar_phase_program(0.37 * pi), scalar_phase_program(-0.37 * pi)
    builder = ComponentProgramBuilder(Signature(()))
    builder.call_program(scalars[0], {})
    select = builder.finalize({})
    selection = select, adjoint_component_calls(select, (scalars,))
    blocks = tuple(
        select_block_component_program(
            Signature(()), selection[0], adjoint=d, inverse=selection[1]
        )
        for d in (False, True)
    )
    preparation = identity_preparation_program(())
    reflection_scalar = scalar_phase_program(pi)
    reflection = prepared_reflection_program(
        preparation, preparation, (), global_phase=-1, scalar_phase=reflection_scalar
    )
    record = qubitization_walk_component_pair(
        Signature(()), (blocks[0], blocks[1]), (reflection, reflection)
    )[int(inverse)]
    if inverse:
        source = Adjoint(source)
    # Do not classify the empty-signature walk, block or SELECT wrapper as an
    # atomic scalar. Only the literal accepted GlobalPhase carrier is a leaf.
    terminals = atomic_leaves(record)
    for function in record.functions[1:]:
        if function.registers == () and any(
            matches_component_definition(record, function.name, scalar)
            for scalar in (*scalars, reflection_scalar)
        ):
            terminals.add(function.name)
    compare_layers(record, source, terminals)
    bound, logical = bind(record)
    assert logical == 0 and bound.circuit.n_wires > 0
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(1, 1),
        atol=3e-11,
        rtol=0,
    )
