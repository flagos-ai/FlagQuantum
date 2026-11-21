"""Import complete blocked lookup graphs through owned Partition dependencies."""

import builtins
from collections import Counter
from typing import Any

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_blocked_lookup_restores_owned_and_borrowed_banks(
    dirty: bool, adjoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = SelectSwapQROM.build_from_data(
        np.array([0, 1, 2, 3]),
        target_bitsizes=(2,),
        log_block_sizes=(1,),
        use_dirty_ancilla=dirty,
    )
    graph = source.adjoint() if adjoint else source
    program = import_register_graph(graph)
    # The original inverse allocates clean words even for a dirty forward bank.
    borrowed = tuple(range(4, 8)) if dirty and not adjoint else ()
    clean = tuple(range(8 if borrowed else 4, 10))
    bound = bind_register_program(
        program, ((1, 0), (3, 2)), clean=clean, borrowed=borrowed
    )
    assert bound.outputs == ((1, 0), (3, 2))
    assert Counter(event.kind for event in bound.events) == {"allocate": 2, "free": 2}
    assert all(
        event.dirty == bool(borrowed)
        for event in bound.events
        if event.kind == "allocate"
    )
    counts = Counter(op.name for op in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == (32 if dirty else 16)
    assert counts["measure"] == 0
    model = get_cost_value(graph, QECGatesCost())
    assert model.total_t_count() == counts["t"] + counts["tdg"]
    physical_width = bound.circuit.n_wires

    def pack(address: int, target: int, bank: int, spectator: int) -> int:
        values = {
            1: address >> 1,
            0: address & 1,
            3: target >> 1,
            2: target & 1,
            9: spectator,
        }
        values.update({4 + i: bank >> (3 - i) & 1 for i in range(4)})
        return sum(
            value << (physical_width - wire - 1) for wire, value in values.items()
        )

    before, expected = {}, {}
    for address in range(4):
        for target in range(4):
            for bank in range(16 if borrowed else 1):
                spectator = (address + target + bank) % 2
                amplitude = complex(1 + address + bank, target - bank - 1)
                before[pack(address, target, bank, spectator)] = amplitude
                expected[pack(address, target ^ address, bank, spectator)] = amplitude
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before = {key: value / norm for key, value in before.items()}
    expected = {key: value / norm for key, value in expected.items()}
    source_import = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".")[0] not in {"qualtran", "cirq", "kickmix"}
        return source_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", native_import)
        ir = (
            DynamicCircuit(physical_width)
            .append(FieldOperation(physical_width, bound.circuit.instructions))
            .to_ir()
        )
        ir = CircuitIR.from_json(ir.to_json())
        actual = execute_sparse_register(ir, before, seed=0)
    assert actual == pytest.approx(expected, abs=3e-12)


@pytest.mark.parametrize(
    "shape,block,error",
    [
        ((4,), (2,), "Finalizing requires"),
        ((2, 2), (0, 1), "register dtypes are not consistent"),
        ((2, 2), (1, 1), "Finalizing requires"),
    ],
)
def test_original_missing_selector_boundaries(
    shape: tuple[int, ...], block: tuple[int, ...], error: str
) -> None:
    import numpy as np
    from qualtran import BloqError
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    source = SelectSwapQROM.build_from_data(
        np.arange(4).reshape(shape), target_bitsizes=(2,), log_block_sizes=block
    )
    with pytest.raises(BloqError, match=error):
        source.decompose_bloq()
    with pytest.raises(BloqError, match=error):
        import_register_graph(source)


@pytest.mark.parametrize("reverse", (False, True))
def test_partition_only_keeps_shaped_views_without_a_cirq_operation(
    reverse: bool,
) -> None:
    import numpy as np
    from qualtran import QAny, Register
    from qualtran.bloqs.bookkeeping import Partition

    source = Partition(
        6,
        (Register("words", QAny(2), shape=(2,)), Register("tail", QAny(2))),
        partition=not reverse,
    )
    wires = tuple(reversed(range(6)))
    inputs = (wires[:2], wires[2:4], wires[4:]) if reverse else (wires,)
    incoming = (
        {"words": np.array(wires[:4]).reshape(2, 2), "tail": np.array(wires[4:])}
        if reverse
        else {"x": np.array(wires)}
    )
    operation, outgoing = source.as_cirq_op(None, **incoming)
    assert operation is None
    expected = tuple(
        tuple(int(wire) for wire in word)
        for value in outgoing.values()
        for word in value.reshape(-1, value.shape[-1])
    )
    bound = bind_register_program(
        import_register_graph(source.as_composite_bloq()), inputs
    )
    assert bound.outputs == expected
    assert not bound.circuit.instructions and not bound.events


@pytest.mark.parametrize("kind", ("split", "join"))
@pytest.mark.parametrize("adjoint", (False, True))
def test_two_word_partition_family_uses_the_same_view_owner(
    kind: str, adjoint: bool
) -> None:
    import numpy as np
    from qualtran.bloqs.bookkeeping import Join2, Split2

    source = Split2(2, 4) if kind == "split" else Join2(2, 4)
    if adjoint:
        source = source.adjoint()
    wires = tuple(reversed(range(6)))
    incoming = (
        {"x": np.array(wires)}
        if source.partition
        else {"y1": np.array(wires[:2]), "y2": np.array(wires[2:])}
    )
    operation, outgoing = source.as_cirq_op(None, **incoming)
    assert operation is None
    inputs = (wires,) if source.partition else (wires[:2], wires[2:])
    expected = tuple(tuple(int(wire) for wire in word) for word in outgoing.values())
    bound = bind_register_program(
        import_register_graph(source.as_composite_bloq()), inputs
    )
    assert bound.outputs == expected
    assert not bound.circuit.instructions and not bound.events
