"""Locked SELECT-only block encoding composition and branch parity."""

from dataclasses import replace

import pytest

from flagquantum.algorithms.primitives.alias_preparation import alias_black_box_prepare
from flagquantum.algorithms.primitives.alias_sampling import alias_sampling_table
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler.registers.lookup.multiplexed_swap import (
    multiplexed_cswap_ir,
    source_multiplexed_cswap_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import (
    assert_controlled_exchange_branches,
    assert_source_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_select_block_source_ports_and_measured_branches(
    packed: bool, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import Adjoint, BQUInt, Register
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        SelectBlockEncoding as SourceEncoding,
    )
    from qualtran.bloqs.bookkeeping import Partition
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    source_select = MultiplexedCSwap(Register("selection", BQUInt(2, 3)), 1)
    source_prepare = StatePreparationAliasSampling.from_probabilities(
        [1, 2, 3], precision=0.1
    )
    forward = SourceEncoding(
        BlackBoxSelect(source_select) if packed else source_select,
        BlackBoxPrepare(source_prepare) if packed else source_prepare,
    )
    source = forward.adjoint() if adjoint else forward
    signal = alias_black_box_prepare(alias_sampling_table((1, 2, 3), precision=0.1))
    target_layout = (("system", 4),) if packed else (("targets", 3), ("output", 1))
    selection, target, workspace, slots = (0, 1), (2, 3, 4, 5), (6,), (0,)
    ports = {"selection": selection}
    if packed:
        ports["system"] = target
    else:
        ports.update(targets=target[:-1], output=target[-1:])
    calls = 0

    def select(bound, work, bits, inverse):
        nonlocal calls
        calls += 1
        assert (
            bound == ports
            and work == workspace
            and bits == slots
            and inverse == adjoint
        )
        rows = bound["system"][:-1] if packed else bound["targets"]
        output = bound["system"][-1:] if packed else bound["output"]
        return multiplexed_cswap_ir(
            (),
            (bound["selection"],),
            (3,),
            tuple((wire,) for wire in rows),
            output,
            work,
            bits,
            adjoint=inverse,
        )

    encoding = SelectBlockEncoding((("selection", 2),), target_layout, select, signal)
    op = encoding.make_on(
        selection=selection,
        target=target,
        workspace=workspace,
        classical_bits=slots,
        adjoint=adjoint,
    )
    circuit = DynamicCircuit(7).append(op)
    assert calls == 1
    actual = circuit.to_ir().instructions
    assert encoding.signal_state is signal
    assert [reg.name for reg in source.signature] == list(ports)
    assert (
        encoding.ancilla_bitsize,
        encoding.resource_bitsize,
        encoding.system_bitsize,
    ) == (forward.ancilla_bitsize, forward.resource_bitsize, forward.system_bitsize)
    assert encoding.alpha == forward.alpha and encoding.epsilon == forward.epsilon

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Partition):
            arrays = {
                reg.name: np.array(
                    [cirq.LineQubit(wire) for wire in inputs[reg.name]]
                ).reshape((*reg.shape, int(reg.bitsize)))
                for reg in bloq.signature.lefts()
            }
            operation, returned = bloq.as_cirq_op(qubit_manager=None, **arrays)
            assert operation is None
            return (), {
                name: tuple(wire.x for wire in word.flat)
                for name, word in returned.items()
            }
        child = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if child != source_select:
            return None
        assert isinstance(bloq, Adjoint) == adjoint
        assert inputs == {
            "selection": selection,
            "targets": target[:-1],
            "output": target[-1:],
        }
        return (
            reference_arithmetic_gates(
                bloq,
                inputs,
                workspace,
                dict(zip(workspace, slots, strict=True)),
                cirq_style=not adjoint,
                preserve_components=True,
            ),
            inputs,
        )

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source_dependencies(actual, expected)
    inverse = multiplexed_cswap_ir(
        (),
        (selection,),
        (3,),
        tuple((wire,) for wire in target[:-1]),
        target[-1:],
        workspace,
        slots,
        adjoint=not adjoint,
    )
    assert_controlled_exchange_branches(actual, expected, inverse, 6, workspace)
    child = source_multiplexed_cswap_cost((3,), (2,), 0, 1)
    assert_source_model(encoding.resource_model(child), source)


def test_select_block_bindings_validate_before_emission() -> None:
    signal = alias_black_box_prepare(alias_sampling_table((1, 1), precision=0.25))

    def escaped(*_):
        return (Instruction("x", (3,)),)

    encoding = SelectBlockEncoding(
        (("selection", 1),), (("target", 1),), escaped, signal
    )
    with pytest.raises(ValueError, match="outside"):
        encoding.emit((0,), (1,), (), ())
    with pytest.raises(ValueError, match="distinct"):
        encoding.emit((0,), (0,), (), ())
    with pytest.raises(ValueError, match="distinct"):
        replace(encoding, target_layout=(("selection", 1),)).emit((0,), (1,), (), ())
    with pytest.raises(ValueError, match="L1"):
        _ = replace(encoding, signal_state=replace(signal, normalization=None)).alpha
