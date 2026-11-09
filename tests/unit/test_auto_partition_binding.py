"""Check AutoPartition binding and bystanders against the pinned source graph."""

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    PartitionedCall,
)
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_unused_ports_preserve_order_and_full_complex_action() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping.auto_partition import (
        AutoPartition as SourceAutoPartition,
    )
    from qualtran.bloqs.bookkeeping.auto_partition import Unused
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    layout = AutoPartition(
        (
            ("packed", ((None, 1), ("target", 1), ("ctrl", 1))),
            ("idle", ((None, 1),)),
        )
    )
    source = SourceAutoPartition(
        CNOT(),
        (
            (Register("packed", QAny(3)), (Unused(1), "target", "ctrl")),
            (Register("idle", QAny(1)), (Unused(1),)),
        ),
    )
    physical = {"idle": (2,), "packed": (3, 1, 0)}
    ports, unused = layout.bind(physical)
    assert ports == {"target": (1,), "ctrl": (0,)}
    assert layout.restore(ports, unused) == physical
    assert layout.restore({"target": (0,), "ctrl": (1,)}, unused)["packed"] == (3, 0, 1)
    assert layout.logical_width(2) == get_cost_value(source, QubitCount()) == 4
    assert query_qec_gate_cost(
        PartitionedCall(CliffordCall("cx", 2), layout.groups)
    ) == QECGateCost(clifford=1)
    assert get_cost_value(source, QECGatesCost()).asdict() == {"clifford": 1}

    rows = torch.arange(1, 17, dtype=torch.float64)
    state = torch.complex(rows, rows.remainder(7) - 4)
    state /= state.norm()
    circuit = fq.Circuit(4, inputs=state, dtype=state.dtype)
    circuit.cx(ports["ctrl"][0], ports["target"][0])
    order = (3, 1, 0, 2)
    before = state.reshape((2,) * 4).permute(order).reshape(-1)
    after = circuit.state().reshape((2,) * 4).permute(order).reshape(-1)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=state.dtype).reshape(
        16, 16
    )
    torch.testing.assert_close(after, matrix @ before, rtol=0, atol=2e-12)
    assert tuple(gate.name for gate in circuit.to_ir().instructions) == ("cx",)


@pytest.mark.ftqc_reference
def test_left_only_preserves_child_output_order_and_consumption() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QAny, QBit, Register, Side, Signature
    from qualtran.bloqs.basic_gates import CNOT, ZeroEffect, ZeroState
    from qualtran.bloqs.bookkeeping.auto_partition import (
        AutoPartition as SourceAutoPartition,
    )

    signature = Signature(
        [
            Register("aux", QBit(), side=Side.LEFT),
            Register("ctrl", QBit()),
            Register("target", QBit()),
        ]
    )
    builder, inputs = BloqBuilder.from_signature(signature)
    ctrl, target = builder.add(CNOT(), ctrl=inputs["ctrl"], target=inputs["target"])
    builder.add(ZeroEffect(), q=inputs["aux"])
    source = SourceAutoPartition(
        builder.finalize(ctrl=ctrl, target=target),
        ((Register("packed", QAny(3)), ("target", "aux", "ctrl")),),
        left_only=True,
    )
    layout = AutoPartition(
        (("packed", (("target", 1), ("aux", 1), ("ctrl", 1))),),
        child_outputs=("ctrl", "target"),
        left_only=True,
    )
    ports, unused = layout.bind({"packed": (2, 1, 0)})
    outputs = layout.restore({"target": ports["target"], "ctrl": ports["ctrl"]}, unused)
    assert (
        tuple(outputs)
        == tuple(reg.name for reg in source.signature.rights())
        == ("ctrl", "target")
    )
    rows = torch.arange(1, 9, dtype=torch.float64)
    state = torch.complex(rows, rows.remainder(3) - 1)
    state /= state.norm()
    circuit = fq.Circuit(3, inputs=state, dtype=state.dtype)
    circuit.cx(ports["ctrl"][0], ports["target"][0])
    plan = build_tensor_network(circuit, dtype=state.dtype)
    actual = tensor_network_basis_effect(plan, {ports["aux"][0]: ("z", 0)})[0]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=state.dtype).reshape(4, 8)
    expected = matrix @ state.reshape(2, 2, 2).permute(2, 1, 0).reshape(-1)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    with pytest.raises(ValueError, match="signature"):
        SourceAutoPartition(ZeroState(), (), left_only=True)
    with pytest.raises(ValueError, match="input layout"):
        AutoPartition((), child_outputs=("q",), left_only=True)


@pytest.mark.ftqc_reference
def test_symbolic_partition_matches_upstream_width_without_binding() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny, QUInt, Register
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.bloqs.bookkeeping.auto_partition import (
        AutoPartition as SourceAutoPartition,
    )
    from qualtran.bloqs.bookkeeping.auto_partition import Unused
    from qualtran.resource_counting import QubitCount, get_cost_value
    from sympy import Symbol

    n = Symbol("n", integer=True, positive=True)
    layout = AutoPartition((("packed", (("q", n), (None, 2))),))
    source = SourceAutoPartition(
        Identity(n), ((Register("packed", QAny(n + 2)), ("q", Unused(2))),)
    )
    assert layout.logical_width(n) == get_cost_value(source, QubitCount()) == n + 2
    with pytest.raises(ValueError, match="concrete widths"):
        layout.bind({"packed": ()})

    adder = Add(QUInt(2), QUInt(2))
    with_work = AutoPartition((("packed", (("a", 2), (None, 2), ("b", 2))),))
    source_with_work = SourceAutoPartition(
        adder, ((Register("packed", QAny(6)), ("a", Unused(2), "b")),)
    )
    assert (
        with_work.logical_width(get_cost_value(adder, QubitCount()))
        == get_cost_value(source_with_work, QubitCount())
        == 7
    )


def test_auto_partition_rejects_invalid_bindings() -> None:
    groups = (("packed", (("q", 1), (None, 1))),)
    with pytest.raises(ValueError, match="unused"):
        AutoPartition(groups, left_only=True)
    layout = AutoPartition(groups)
    with pytest.raises(ValueError, match="declared packed"):
        layout.bind({})
    with pytest.raises(ValueError, match="distinct"):
        layout.bind({"packed": (1, 1)})
    _, unused = layout.bind({"packed": (1, 2)})
    with pytest.raises(ValueError, match="original unused"):
        layout.restore({"q": (1,)}, {})
    with pytest.raises(ValueError, match="distinct"):
        layout.restore({"q": (2,)}, unused)
