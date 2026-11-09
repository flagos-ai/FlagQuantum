"""Compare ordered source products, success flags and both source cost routes."""

from dataclasses import asdict
from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit
CASES = (
    (("t",), (0,)),
    (("t", "h"), (0, 0)),
    (("t", "h"), (1, 1)),
    (("h", "t"), (2, 1)),
    (("t", "h"), (1, 2)),
    (("t", "h"), (1, 3)),
    (("t", "h"), (3, 0)),
    (("t", "h"), (0, 3)),
    (("t", "h", "x"), (2, 0, 1)),
    (("x", "h", "t"), (1, 2, 3)),
)


def _blocks(kinds, ancs, resource):
    blocks = []
    for kind, anc in zip(kinds, ancs, strict=True):

        def emit(ports, work, bits, inverse, kind=kind):
            return (
                Instruction("tdg" if kind == "t" and inverse else kind, ports["q"]),
            )

        blocks.append(
            UnitaryBlockEncoding(
                (("q", 1),),
                emit,
                alpha=0.5,
                ancilla_bitsize=anc,
                resource_bitsize=resource,
                epsilon=0.01,
            )
        )
    return tuple(blocks)


def _source(kinds, ancs, resource):
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate
    from qualtran.bloqs.block_encoding.product import Product
    from qualtran.bloqs.block_encoding.unitary import Unitary

    return Product(
        tuple(
            Unitary(
                {"t": TGate, "h": Hadamard, "x": XGate}[kind](),
                alpha=0.5,
                ancilla_bitsize=anc,
                resource_bitsize=resource,
                epsilon=0.01,
            )
            for kind, anc in zip(kinds, ancs, strict=True)
        )
    )


def _bind(blocks, adjoint):
    root = ProductBlockEncoding(blocks)
    data = root.system_bitsize + root.ancilla_bitsize + root.resource_bitsize
    need = max((max(0, b.ancilla_bitsize - 1) for b in blocks[1:]), default=0)
    calls = []

    def factory(index):
        def bind(ports, work, slots, inverse):
            calls.append((index, inverse, dict(ports)))
            block = blocks[index]
            return block.make_on(
                system=ports["system"],
                ancilla=ports["ancilla"],
                resource=ports["resource"],
                workspace=work,
                classical_bits=slots,
                adjoint=inverse,
            )

        return bind

    kwargs = dict(
        system=tuple(range(root.system_bitsize)),
        ancilla=tuple(
            range(root.system_bitsize, root.system_bitsize + root.ancilla_bitsize)
        ),
        resource=tuple(range(root.system_bitsize + root.ancilla_bitsize, data)),
        workspaces=((),) * len(blocks),
        classical_bits=((),) * len(blocks),
        flag_workspace=tuple(range(data, data + need)),
        flag_classical_bits=tuple(range(40, 40 + need)),
        adjoint=adjoint,
    )
    op = root.make_on(tuple(factory(i) for i in range(len(blocks))), **kwargs)
    return root, op, kwargs, calls


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kinds,ancs", CASES)
@pytest.mark.parametrize("resource,adjoint", tuple(product((0, 1), (False, True))))
def test_original_gate_order_and_abstract_vs_decomposed_costs(
    kinds, ancs, resource, adjoint
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    blocks = _blocks(kinds, ancs, resource)
    root, op, kw, calls = _bind(blocks, adjoint)
    source = _source(kinds, ancs, resource)
    for attr in (
        "system_bitsize",
        "ancilla_bitsize",
        "resource_bitsize",
        "alpha",
        "epsilon",
    ):
        assert getattr(root, attr) == getattr(source, attr)
    assert root.signal_state.selection_width == source.signal_state.selection_bitsize
    original = source.adjoint() if adjoint else source
    order = (
        tuple(range(len(blocks))) if adjoint else tuple(reversed(range(len(blocks))))
    )
    assert tuple(i for i, _, _ in calls) == order
    assert all(inverse == adjoint for _, inverse, _ in calls)
    graph = original.decompose_bloq()
    ports = {name: kw[name] for name in ("system", "ancilla", "resource") if kw[name]}
    work = kw["flag_workspace"]
    slots = kw["flag_classical_bits"]
    expected = reference_arithmetic_gates(
        graph.flatten(),
        ports,
        work,
        dict(zip(work, slots, strict=True)),
        cirq_style=True,
        preserve_components=True,
    )
    assert_source_dependencies(op.instructions, expected)
    costs = tuple(
        block.resource_model(
            ComponentCost(
                kind, 1, GateCost(t=1) if kind == "t" else GateCost(clifford=1)
            )
        )
        for kind, block in zip(kinds, blocks, strict=True)
    )
    for legacy, route in product((False, True), ("call_graph", "decomposition")):
        model = root.resource_model(
            costs, adjoint=adjoint, legacy_shims=legacy, flag_model=route
        )
        native = estimate_ftqc_resources(model).logical
        target = original if route == "call_graph" else graph
        source_gates = get_cost_value(target, QECGatesCost(legacy_shims=legacy))
        assert asdict(native.gates) == {
            name: getattr(source_gates, name) for name in asdict(native.gates)
        }
        assert native.peak_logical_qubits == get_cost_value(original, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("ancs", [(0, 0), (1, 1), (1, 2), (1, 3)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_complex_source_matrix_and_every_flag_measurement(ancs, adjoint):
    blocks = _blocks(("t", "h"), ancs, 0)
    root, op, kw, _ = _bind(blocks, adjoint)
    source = _source(("t", "h"), ancs, 0)
    if adjoint:
        source = source.adjoint()
    data = root.system_bitsize + root.ancilla_bitsize
    work = op.required_qubits - data
    amplitudes = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << data)], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << work] = amplitudes
    expected[:: 1 << work] = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            1 << data, 1 << data
        )
        @ amplitudes
    )
    events = sum(g.name == "measure" for g in op.instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            expected * 2 ** (-events / 2),
            atol=5e-12,
            rtol=0,
        )


def _signal_source(kinds, ancs):
    import numpy as np
    from attrs import frozen
    from qualtran import Signature
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate
    from qualtran.bloqs.block_encoding.block_encoding_base import BlockEncoding
    from qualtran.bloqs.block_encoding.product import Product
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    @frozen
    class SignalBlock(BlockEncoding):
        kind: str
        ancilla_bitsize: int
        system_bitsize: int = 1
        resource_bitsize: int = 0
        alpha: float = 1.0
        epsilon: float = 0.0

        @property
        def signature(self):
            return Signature.build(system=1, ancilla=self.ancilla_bitsize)

        @property
        def signal_state(self):
            return BlackBoxPrepare(
                PrepareIdentity.from_bitsizes([self.ancilla_bitsize])
            )

        def build_composite_bloq(self, bb, system, ancilla):
            sys = bb.split(system)
            anc = bb.split(ancilla)
            anc[0] = bb.add(Hadamard(), q=anc[0])
            anc[0], sys[0] = bb.add_t(CNOT(), ctrl=anc[0], target=sys[0])
            sys[0] = bb.add(TGate() if self.kind == "t" else Hadamard(), q=sys[0])
            return {"system": bb.join(np.array(sys)), "ancilla": bb.join(np.array(anc))}

    return Product(
        tuple(SignalBlock(kind, anc) for kind, anc in zip(kinds, ancs, strict=True))
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("ancs", [(1, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_nontrivial_failure_flags_full_source_branches_and_projected_product(
    ancs, adjoint
):
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.product_block_encoding import SignalBlock

    blocks = (SignalBlock("t", ancs[0]), SignalBlock("h", ancs[1]))
    root, op, kw, _ = _bind(blocks, adjoint)
    source = _signal_source(("t", "h"), ancs)
    if adjoint:
        source = source.adjoint()
    graph = source.decompose_bloq()
    work = kw["flag_workspace"]
    slots = kw["flag_classical_bits"]
    original = reference_arithmetic_gates(
        graph.flatten(),
        {"system": kw["system"], "ancilla": kw["ancilla"]},
        work,
        dict(zip(work, slots, strict=True)),
        cirq_style=True,
        preserve_components=True,
    )
    assert_source_dependencies(op.instructions, original)
    data = root.system_bitsize + root.ancilla_bitsize
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << data, 1 << data
    )
    selected = torch.tensor([0, 1 << (data - 1)])
    projected = matrix[selected[:, None], selected]
    ideal = (
        torch.as_tensor(
            TGate().tensor_contract() @ Hadamard().tensor_contract(),
            dtype=torch.complex128,
        )
        / 2
    )
    if adjoint:
        ideal = ideal.mH
    torch.testing.assert_close(projected, ideal, atol=4e-13, rtol=0)
    amps = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << data)], dtype=torch.complex128
    )
    amps /= amps.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << len(work)] = amps
    expected[:: 1 << len(work)] = matrix @ amps
    events = sum(g.name == "measure" for g in op.instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            expected * 2 ** (-events / 2),
            atol=5e-12,
            rtol=0,
        )
    costs = tuple(block.resource_model() for block in blocks)
    for legacy, route in product((False, True), ("call_graph", "decomposition")):
        result = estimate_ftqc_resources(
            root.resource_model(
                costs, adjoint=adjoint, legacy_shims=legacy, flag_model=route
            )
        ).logical
        original_cost = get_cost_value(
            source if route == "call_graph" else graph,
            QECGatesCost(legacy_shims=legacy),
        )
        assert asdict(result.gates) == {
            name: getattr(original_cost, name) for name in asdict(result.gates)
        }
        assert result.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_measured_child_and_success_flag_share_clean_work_lifetimes(adjoint):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.block_encoding.product import Product
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
        source_add_cost,
    )
    from tests.unit.test_unitary_block_encoding import _fixture

    blocks = (_fixture("add2", 2, 0)[0], _fixture("add2", 3, 0)[0])
    root = ProductBlockEncoding(blocks)

    def factory(block):
        def bind(ports, work, slots, inverse):
            return block.make_on(
                system=ports["system"],
                ancilla=ports["ancilla"],
                resource=ports["resource"],
                workspace=work,
                classical_bits=slots,
                adjoint=inverse,
            )

        return bind

    op = root.make_on(
        tuple(factory(b) for b in blocks),
        system=(0, 1, 2, 3),
        ancilla=(4, 5, 6, 7),
        workspaces=((8,), (8,)),
        classical_bits=((40,), (40,)),
        flag_workspace=(8, 9),
        flag_classical_bits=(40, 41),
        adjoint=adjoint,
    )
    source = Product(
        tuple(
            Unitary(Add(QUInt(2)), alpha=0.5, epsilon=0.01, ancilla_bitsize=a)
            for a in (2, 3)
        )
    )
    if adjoint:
        source = source.adjoint()
    graph = source.decompose_bloq()
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt import MultiControlX

    from tests.unit.register_source_graph import (
        source_partition_leaf,
        walk_source_graph,
    )

    def leaf(bloq, inputs, desired):
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        child = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(child, (Unitary, MultiControlX)):
            work = (8,) if isinstance(child, Unitary) else (8, 9)
            gate_source = bloq.decompose_bloq().flatten()
            gates = reference_arithmetic_gates(
                gate_source,
                inputs,
                work,
                {8: 40, 9: 41},
                cirq_style=True,
                preserve_components=True,
            )
            return gates, inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        return None

    ports = {"system": (0, 1, 2, 3), "ancilla": (4, 5, 6, 7)}
    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source_dependencies(op.instructions, expected)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        256, 256
    )
    amps = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(256)], dtype=torch.complex128
    )
    amps /= amps.norm()
    state = torch.zeros(1024, dtype=torch.complex128)
    reference = torch.zeros_like(state)
    state[::4] = amps
    reference[::4] = matrix @ amps
    events = sum(g.name == "measure" for g in op.instructions)
    assert events == 4
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, op.instructions, history),
            reference / 4,
            atol=5e-12,
            rtol=0,
        )
    for legacy, route in product((False, True), ("call_graph", "decomposition")):
        costs = tuple(
            b.resource_model(source_add_cost(2, 2, legacy_shims=legacy)) for b in blocks
        )
        result = estimate_ftqc_resources(
            root.resource_model(
                costs, adjoint=adjoint, legacy_shims=legacy, flag_model=route
            )
        ).logical
        source_gates = get_cost_value(
            source if route == "call_graph" else graph,
            QECGatesCost(legacy_shims=legacy),
        )
        assert asdict(result.gates) == {
            name: getattr(source_gates, name) for name in asdict(result.gates)
        }
        assert result.peak_logical_qubits == get_cost_value(source, QubitCount()) == 10


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("resources", [(1, 2), (2, 1), (1, 3)])
def test_unequal_nonzero_source_resource_partitions_remain_unsupported(resources):
    from dataclasses import replace

    from qualtran import BloqError
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.product import Product
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Product(
        (
            Unitary(TGate(), resource_bitsize=resources[0]),
            Unitary(Hadamard(), resource_bitsize=resources[1]),
        )
    )
    with pytest.raises(BloqError):
        source.decompose_bloq()
    with pytest.raises(ValueError):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(RuntimeError):
        get_cost_value(source, QubitCount())
    blocks = tuple(
        replace(b, resource_bitsize=r)
        for b, r in zip(_blocks(("t", "h"), (0, 0), 0), resources, strict=True)
    )
    root = ProductBlockEncoding(blocks)
    assert root.resource_bitsize == source.resource_bitsize == max(resources)

    def refuse(*args):
        pytest.fail("Unsupported resource layout reached child")

    with pytest.raises(ValueError, match="unequal nonzero"):
        root.make_on(
            (refuse, refuse),
            system=(0,),
            ancilla=(1,),
            resource=tuple(range(2, 2 + max(resources))),
            workspaces=((), ()),
            classical_bits=((), ()),
        )
    with pytest.raises(ValueError, match="unequal nonzero"):
        root.resource_model(
            (ComponentCost("a", 1 + resources[0]), ComponentCost("b", 1 + resources[1]))
        )


@pytest.mark.parametrize(
    "change",
    [
        {"system": ()},
        {"ancilla": ()},
        {"flag_workspace": (3,)},
        {"flag_classical_bits": (0,)},
        {"workspaces": ((0,), ())},
        {"classical_bits": ((0, 0), ())},
        {"adjoint": 1},
    ],
)
def test_invalid_layout_is_rejected_before_child_factory(change):
    root = ProductBlockEncoding(_blocks(("t", "h"), (1, 1), 0))

    def refuse(*args):
        pytest.fail("Invalid product reached child")

    kwargs = dict(
        system=(0,), ancilla=(1, 2), workspaces=((), ()), classical_bits=((), ())
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        root.make_on((refuse, refuse), **kwargs)


def test_constructor_and_source_model_contracts_reject_invalid_inputs():
    from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
    from flagquantum.algorithms.primitives.select_block_encoding import (
        SelectBlockEncoding,
    )

    with pytest.raises(ValueError):
        ProductBlockEncoding(())
    wrong = UnitaryBlockEncoding((("q", 2),), lambda *args: ())
    (one,) = _blocks(("t",), (0,), 0)
    with pytest.raises(ValueError, match="same system"):
        ProductBlockEncoding((one, wrong))
    signal = BlackBoxPrepare((("s", 1),), (), lambda *args: (), 1)
    block = SelectBlockEncoding((("s", 1),), (("t", 1),), lambda *args: (), signal)
    with pytest.raises(ValueError, match="identity"):
        ProductBlockEncoding((block,))
    root = ProductBlockEncoding((one,))
    with pytest.raises(ValueError):
        root.resource_model(())
    with pytest.raises(ValueError):
        root.resource_model((ComponentCost("T", 1),), flag_model="invalid")
    with pytest.raises(ValueError):
        root.resource_model((ComponentCost("T", 1),), legacy_shims=1)
    with pytest.raises(ValueError):
        root.resource_model((ComponentCost("narrow", 0),))


def test_runtime_product_projected_block_and_exact_inverse(capsys):
    from examples.algorithms.product_block_encoding import main

    main()
    output = capsys.readouterr().out
    assert "Model T equivalents / emitted T instructions 9 9" in output
    assert "Abstract / decomposition model Clifford counts 9 21" in output
    assert "Model logical peak / reserved wires 7 7" in output
