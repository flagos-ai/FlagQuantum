"""Check the source serial, parallel and projected independent fixtures."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from tests.unit.atomic_register_fixtures import AtomicRegisterFixture
from tests.unit.composition_fixtures import Mode, composition_fixture
from tests.unit.register_source_graph import SourcePorts, SourceResult
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def supply(
    function: FunctionRecord, inputs: Any, outputs: Any
) -> tuple[TensorNetworkNode, ...]:
    assert function.registers is not None
    signature = Signature(function.registers)
    if function.name.startswith("atom"):
        tensor = AtomicRegisterFixture("x").tensor()
    elif not inputs or not outputs:
        tensor = torch.tensor([1, 0], dtype=torch.complex128)
    else:
        tensor = torch.eye(8, dtype=torch.complex128).reshape((2,) * 6)
        # Each side is a three-bit word or three single-bit words.
        tensor = tensor.reshape(
            tuple(
                1 << int(reg.bitsize)
                for reg in (*signature.rights(), *signature.lefts())
                for _ in reg.all_idxs()
            )
        )
    return (bind_register_tensor(tensor, signature, inputs, outputs),)


@pytest.mark.parametrize("mode", ["serial", "parallel", "independent"])
def test_composition_port_actions_and_projection(mode: Mode) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.atom import TestAtom
    from qualtran.bloqs.for_testing.with_decomposition import (
        TestIndependentParallelCombo,
        TestParallelCombo,
        TestSerialCombo,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = {
        "serial": TestSerialCombo,
        "parallel": TestParallelCombo,
        "independent": TestIndependentParallelCombo,
    }[mode]()
    width = {"serial": 1, "parallel": 3, "independent": 0}[mode]
    native = composition_fixture(mode)
    main = next(function for function in native.functions if function.name == "main")
    assert main.registers is not None
    assert [(r.name, r.bitsize) for r in main.registers] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    order = tuple(position for position, _, _ in iter_component_calls(native))
    actual = contract_component_tensors(
        native, order, supply, torch.empty((), dtype=torch.complex128)
    )
    expected = torch.as_tensor(source.tensor_contract().copy(), dtype=actual.dtype)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, atol=0, rtol=0)

    ports = tuple(reversed(range(width)))
    work = () if width else (0,)
    bound = bind_register_program(native, (ports,) if width else (), clean=work)

    def atom(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if not isinstance(bloq, TestAtom):
            return None
        assert all(inputs[name] == wires for name, wires in desired.items())
        return (Instruction("x", inputs["q"]),), inputs

    gates, lifetime = source_workspace_gates(
        source, {"reg": ports} if width else {}, work, (), custom_leaf=atom
    )
    assert bound.outputs == ((ports,) if width else ())
    assert_source(bound.circuit.instructions, gates)
    assert [(e.kind, e.wires, e.dirty) for e in bound.events] == list(lifetime)
    assert len(gates) == 3 and all(g.name == "x" for g in gates)
    assert get_cost_value(source, QECGatesCost()).t == 300
    assert (
        get_cost_value(source, QubitCount()) == bound.circuit.n_wires == max(width, 1)
    )
    if mode == "serial":
        original = sorted(
            source.decompose_bloq().bloq_instances, key=lambda node: node.i
        )
        calls = [op.callee for op in main.blocks[0].operations]
        assert (
            calls == [node.bloq.tag for node in original] == ["atom0", "atom1", "atom2"]
        )
    if width:
        amplitudes = torch.arange(1, 1 + (1 << width), dtype=torch.float64)
        ket = torch.complex(amplitudes, 3 - amplitudes)
        ket /= ket.norm()
        state = Circuit(width, inputs=ket, dtype=ket.dtype).append(
            FieldOperation(width, bound.circuit.instructions)
        )
        axes = tuple(reversed(range(width)))
        expected_state = (
            (expected @ ket.reshape((2,) * width).permute(axes).flatten())
            .reshape((2,) * width)
            .permute(axes)
            .flatten()
        )
        torch.testing.assert_close(
            state.state().flatten(), expected_state, atol=1e-14, rtol=0
        )
    else:
        assert actual.item() == 0
        raw = Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, bound.circuit.instructions)
        )
        torch.testing.assert_close(
            raw.state().flatten(),
            torch.tensor([0, 1], dtype=actual.dtype),
            atol=0,
            rtol=0,
        )
