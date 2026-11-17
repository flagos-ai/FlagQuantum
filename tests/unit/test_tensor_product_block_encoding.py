"""Source packed partitions, exact child streams and tensor-product branches."""

from dataclasses import replace
from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
    TensorProductBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_indexed_component_cost_profiles import _assert_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unitary_block_encoding import _fixture

pytestmark = pytest.mark.unit
CASES = (
    ("t",),
    ("t", "cx"),
    ("cx", "t"),
    ("t", "add2"),
    ("add2", "t"),
    ("add2", "add2"),
    ("add2", "add3"),
    ("t", "cx", "add2"),
)


@pytest.mark.ftqc_reference
def test_original_overrides_preserve_full_complex_auxiliary_ports():
    from qualtran.bloqs.block_encoding.tensor_product import (
        _tensor_product_block_encoding_properties,
    )

    from flagquantum import Circuit
    from flagquantum.core.ir import Instruction
    from flagquantum.simulation.unitary import get_unitary

    source = _tensor_product_block_encoding_properties()
    t, *_ = _fixture("t", 2, 1)
    cx, *_ = _fixture("cx", 1, 1)
    blocks = (t, replace(cx, epsilon=0.1))
    native, operation, binding, _ = _binding(blocks, False, True)
    assert (
        tuple(map(len, (binding["system"], binding["ancilla"], binding["resource"])))
        == (
            3,
            3,
            2,
        )
        == tuple(register.total_bits() for register in source.signature)
    )
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    circuit = Circuit(8, dtype=torch.complex128).append(operation)
    assert circuit.to_ir().instructions == (
        Instruction("t", (0,)),
        Instruction("cx", (1, 2)),
    )
    matrix = get_unitary(circuit)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(matrix, expected, atol=2e-12, rtol=0)
    selected = torch.arange(8) * 32
    projector = torch.eye(32, dtype=torch.complex128)
    torch.testing.assert_close(
        matrix,
        torch.kron(expected[selected][:, selected], projector),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.ftqc_reference
def test_original_nested_x_x_cnot_cirq_export():
    import cirq
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.block_encoding.tensor_product import TensorProduct
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives.unitary_block_encoding import (
        UnitaryBlockEncoding,
    )
    from flagquantum.core.ir import Instruction
    from flagquantum.ecosystem.cirq import to_cirq

    def x(ports, work, bits, inverse):
        return (Instruction("x", ports["q"]),)

    def bind(block):
        def apply(ports, work, bits, inverse):
            return block.make_on(system=ports["system"], adjoint=inverse)

        return apply

    bit_flip = UnitaryBlockEncoding((("q", 1),), x)
    pair = TensorProductBlockEncoding((bit_flip, bit_flip))
    cnot, *_ = _fixture("cx", 0, 0)
    root = TensorProductBlockEncoding((pair, cnot))

    def bind_pair(ports, work, bits, inverse):
        return pair.make_on(
            (bind(bit_flip), bind(bit_flip)),
            system=ports["system"],
            workspaces=((), ()),
            classical_bits=((), ()),
            adjoint=inverse,
        )

    operation = root.make_on(
        (bind_pair, bind(cnot)),
        system=(0, 1, 2, 3),
        workspaces=((), ()),
        classical_bits=((), ()),
    )
    actual = to_cirq(Circuit(4).append(operation).to_ir())
    original = TensorProduct(
        (TensorProduct((Unitary(XGate()), Unitary(XGate()))), Unitary(CNOT()))
    )
    qubits = cirq.LineQubit.range(4)
    expected = cirq.Circuit(
        cirq.decompose_once(
            original.on_registers(system=qubits),
            context=cirq.DecompositionContext(
                cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
            ),
        )
    )
    gates = (cirq.X(qubits[0]), cirq.X(qubits[1]), cirq.CNOT(*qubits[2:]))
    lowered = cirq.decompose(expected, keep=lambda op: op.gate in (cirq.X, cirq.CNOT))
    assert tuple(actual.all_operations()) == tuple(lowered) == gates
    matrix = torch.as_tensor(actual.unitary(qubit_order=qubits))
    torch.testing.assert_close(
        matrix, torch.as_tensor(expected.unitary(qubit_order=qubits))
    )
    assert matrix[:, 0b0111].abs().argmax().item() == 0b1010


def _blocks(kinds, mixed):
    blocks = []
    for i, kind in enumerate(kinds):
        block, *_ = _fixture(kind, int(mixed and i % 2 == 0), int(mixed and i % 2 == 1))
        blocks.append(block)
    return tuple(blocks)


def _binding(blocks, adjoint, reuse):
    data = sum(
        b.system_bitsize + b.ancilla_bitsize + b.resource_bitsize for b in blocks
    )
    offset = 0
    workspaces = []
    sizes = [
        (
            max(width for _, width in b.system_layout) - 1
            if any(name == "a" for name, _ in b.system_layout)
            else 0
        )
        for b in blocks
    ]
    for size in sizes:
        workspaces.append(
            tuple(
                range(
                    data + (0 if reuse else offset),
                    data + (0 if reuse else offset) + size,
                )
            )
        )
        offset += size
    bits = tuple(tuple(w + 40 for w in work) for work in workspaces)
    calls = []

    def binder(i):
        def bind(ports, work, slots, inverse):
            calls.append((i, inverse, dict(ports)))
            return blocks[i].make_on(
                system=ports["system"],
                ancilla=ports["ancilla"],
                resource=ports["resource"],
                workspace=work,
                classical_bits=slots,
                adjoint=inverse,
            )

        return bind

    nsys = sum(b.system_bitsize for b in blocks)
    nanc = sum(b.ancilla_bitsize for b in blocks)
    kwargs = dict(
        system=tuple(range(nsys)),
        ancilla=tuple(range(nsys, nsys + nanc)),
        resource=tuple(range(nsys + nanc, data)),
        workspaces=tuple(workspaces),
        classical_bits=bits,
        adjoint=adjoint,
    )
    root = TensorProductBlockEncoding(blocks)
    operation = root.make_on(tuple(binder(i) for i in range(len(blocks))), **kwargs)
    return root, operation, kwargs, calls


def _source(kinds, mixed):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, TGate
    from qualtran.bloqs.block_encoding.tensor_product import TensorProduct
    from qualtran.bloqs.block_encoding.unitary import Unitary

    children = []
    for i, kind in enumerate(kinds):
        child = (
            Add(QUInt(int(kind[-1])))
            if kind.startswith("add")
            else TGate()
            if kind == "t"
            else CNOT()
        )
        children.append(
            Unitary(
                child,
                alpha=0.5,
                epsilon=0.01,
                ancilla_bitsize=int(mixed and i % 2 == 0),
                resource_bitsize=int(mixed and i % 2 == 1),
            )
        )
    return TensorProduct(tuple(children))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kinds", CASES)
@pytest.mark.parametrize("mixed,adjoint,reuse", tuple(product((False, True), repeat=3)))
def test_original_child_streams_models_and_source_order(kinds, mixed, adjoint, reuse):
    blocks = _blocks(kinds, mixed)
    root, op, binding, calls = _binding(blocks, adjoint, reuse)
    source = _source(kinds, mixed)
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
        tuple(reversed(range(len(blocks)))) if adjoint else tuple(range(len(blocks)))
    )
    assert tuple(i for i, _, _ in calls) == order
    assert all(reverse == adjoint for _, reverse, _ in calls)
    ports = {
        name: binding[name]
        for name in ("system", "ancilla", "resource")
        if binding[name]
    }
    works = binding["workspaces"]
    allocations = tuple(dict.fromkeys(w for i in order for w in works[i]))
    expected = reference_arithmetic_gates(
        original.decompose_bloq().flatten(),
        ports,
        allocations,
        {w: w + 40 for w in allocations},
        cirq_style=True,
        reuse_allocations=reuse,
    )
    assert_source_dependencies(op.instructions, expected)
    for legacy in (False, True):
        models = []
        for kind, block in zip(kinds, blocks, strict=True):
            child = (
                source_add_cost(int(kind[-1]), int(kind[-1]), legacy_shims=legacy)
                if kind.startswith("add")
                else ComponentCost(
                    kind,
                    block.system_bitsize,
                    GateCost(t=1) if kind == "t" else GateCost(clifford=1),
                )
            )
            models.append(block.resource_model(child))
        _assert_model(
            root.resource_model(tuple(models), adjoint=adjoint), original, legacy
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "kinds,mixed",
    [(("t", "cx"), True), (("add2", "t"), True), (("add2", "add2"), False)],
)
@pytest.mark.parametrize("adjoint,reuse", tuple(product((False, True), repeat=2)))
def test_complete_source_tensor_on_entangled_ports_and_every_branch(
    kinds, mixed, adjoint, reuse
):
    blocks = _blocks(kinds, mixed)
    root, op, binding, _ = _binding(blocks, adjoint, reuse)
    source = _source(kinds, mixed)
    if adjoint:
        source = source.adjoint()
    data = root.system_bitsize + root.ancilla_bitsize + root.resource_bitsize
    workcount = op.required_qubits - data
    amplitudes = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << data)], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << workcount] = amplitudes
    expected[:: 1 << workcount] = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            1 << data, 1 << data
        )
        @ amplitudes
    )
    count = sum(g.name == "measure" for g in op.instructions)
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            expected * 2 ** (-count / 2),
            atol=5e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_noncontiguous_packed_registers_preserve_original_ports(adjoint):
    blocks = _blocks(("t", "cx"), True)
    root = TensorProductBlockEncoding(blocks)

    def first(ports, work, bits, inverse):
        return blocks[0].make_on(
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            adjoint=inverse,
        )

    def second(ports, work, bits, inverse):
        return blocks[1].make_on(
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            adjoint=inverse,
        )

    op = root.make_on(
        (first, second),
        system=(6, 1, 5),
        ancilla=(0,),
        resource=(8,),
        workspaces=((), ()),
        classical_bits=((), ()),
        adjoint=adjoint,
    )
    source = _source(("t", "cx"), True)
    if adjoint:
        source = source.adjoint()
    expected = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        {"system": (6, 1, 5), "ancilla": (0,), "resource": (8,)},
        (),
        {},
        cirq_style=True,
    )
    assert_source_dependencies(op.instructions, expected)


@pytest.mark.ftqc_reference
def test_symbolic_normalization_error_and_nested_source_metadata():
    from dataclasses import replace

    from qualtran.bloqs.basic_gates import CNOT, TGate
    from qualtran.bloqs.block_encoding.tensor_product import TensorProduct
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from sympy import Symbol, simplify

    a, b, e, f = [Symbol(name, positive=True) for name in ("a", "b", "e", "f")]
    first, second = _blocks(("t", "cx"), False)
    first = replace(first, alpha=a, epsilon=e)
    second = replace(second, alpha=b, epsilon=f)
    native = TensorProductBlockEncoding((first, second))
    source = TensorProduct(
        (Unitary(TGate(), alpha=a, epsilon=e), Unitary(CNOT(), alpha=b, epsilon=f))
    )
    assert simplify(native.alpha - source.alpha) == 0
    assert simplify(native.epsilon - source.epsilon) == 0
    nested = TensorProductBlockEncoding((native, first))
    original = TensorProduct((source, Unitary(TGate(), alpha=a, epsilon=e)))
    assert simplify(nested.alpha - original.alpha) == 0
    assert simplify(nested.epsilon - original.epsilon) == 0
    assert nested.system_bitsize == original.system_bitsize


@pytest.mark.ftqc_reference
def test_source_and_native_refuse_nonidentity_signal_state():
    from qualtran import BQUInt, Register
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        SelectBlockEncoding as SourceSelectBlock,
    )
    from qualtran.bloqs.block_encoding.tensor_product import TensorProduct
    from qualtran.bloqs.state_preparation.black_box_prepare import (
        BlackBoxPrepare as SourcePrepare,
    )
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
    from flagquantum.algorithms.primitives.select_block_encoding import (
        SelectBlockEncoding,
    )

    prepare = BlackBoxPrepare((("s", 1),), (("junk", 1),), lambda *args: (), 1)
    block = SelectBlockEncoding((("s", 1),), (("t", 1),), lambda *args: (), prepare)
    with pytest.raises(NotImplementedError):
        source_block = SourceSelectBlock(
            MultiplexedCSwap(Register("selection", BQUInt(1, 2)), 1),
            SourcePrepare(
                StatePreparationAliasSampling.from_probabilities(
                    [0.25, 0.75], precision=1
                )
            ),
        )
        _ = TensorProduct((source_block,)).signal_state
    with pytest.raises(NotImplementedError):
        _ = TensorProductBlockEncoding((block,)).signal_state


def test_capture_and_model_do_not_enter_numerical_execution(monkeypatch):
    from flagquantum.circuit import Circuit
    from flagquantum.qec.resources.estimation import estimate_ftqc_resources

    def refuse(*args, **kwargs):
        pytest.fail("IR composition entered numerical execution")

    monkeypatch.setattr(Circuit, "state", refuse)
    monkeypatch.setattr(Circuit, "run", refuse)
    blocks = _blocks(("add2", "t"), False)
    root, op, _, _ = _binding(blocks, False, True)
    assert op.instructions
    costs = (
        blocks[0].resource_model(source_add_cost(2, 2)),
        blocks[1].resource_model(ComponentCost("T", 1, GateCost(t=1))),
    )
    report = estimate_ftqc_resources(root.resource_model(costs))
    assert report.logical.peak_logical_qubits == 6
    assert report.logical.gates.total_t_count() == 5
    assert report.emitted is None


@pytest.mark.parametrize(
    "change",
    [
        {"system": (0,)},
        {"ancilla": (0,)},
        {"system": (0, 0)},
        {"workspaces": ((0,), ())},
        {"workspaces": ((),)},
        {"classical_bits": ((1, 1), ())},
        {"adjoint": 1},
    ],
)
def test_invalid_tensor_layout_refused_before_child(change):
    blocks = _blocks(("t", "t"), False)
    root = TensorProductBlockEncoding(blocks)

    def refuse(*args):
        pytest.fail("Invalid tensor layout reached child factory")

    kwargs = dict(system=(0, 1), workspaces=((), ()), classical_bits=((), ()))
    kwargs.update(change)
    with pytest.raises(ValueError):
        root.make_on((refuse, refuse), **kwargs)


def test_child_cannot_access_another_partition():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.core.ir import Instruction

    blocks = _blocks(("t", "t"), False)
    root = TensorProductBlockEncoding(blocks)

    def escaped(*args):
        return FieldOperation(2, (Instruction("x", (1,)),))

    with pytest.raises(ValueError, match="another child's"):
        root.make_on(
            (escaped, escaped),
            system=(0, 1),
            workspaces=((), ()),
            classical_bits=((), ()),
        )


def test_empty_product_and_invalid_child_models_refused():
    with pytest.raises(ValueError, match="at least one"):
        TensorProductBlockEncoding(())
    (block,) = _blocks(("cx",), False)
    root = TensorProductBlockEncoding([block])
    assert isinstance(root.block_encodings, tuple)
    with pytest.raises(ValueError, match="one child"):
        root.resource_model(())
    with pytest.raises(ValueError, match="through ports"):
        root.resource_model((ComponentCost("narrow", 1),))


def test_tensor_append_runtime_round_trip_example(capsys):
    from examples.algorithms.tensor_product_block_encoding import main

    main()
    output = capsys.readouterr().out
    assert "Model T equivalents 5" in output
    assert "Emitted T/T-dagger instructions 5" in output
    assert "Model logical peak / reserved wires 6 6" in output


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_nested_tensor_keeps_child_work_and_original_grouping(adjoint):
    inner_blocks = _blocks(("add2", "t"), False)
    (last,) = _blocks(("cx",), False)
    inner = TensorProductBlockEncoding(inner_blocks)
    outer = TensorProductBlockEncoding((inner, last))

    def plain(block):
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

    def nested(ports, work, slots, inverse):
        return inner.make_on(
            (plain(inner_blocks[0]), plain(inner_blocks[1])),
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            workspaces=(work, ()),
            classical_bits=(slots, ()),
            adjoint=inverse,
        )

    op = outer.make_on(
        (nested, plain(last)),
        system=tuple(range(7)),
        workspaces=((7,), ()),
        classical_bits=((0,), ()),
        adjoint=adjoint,
    )
    from qualtran.bloqs.block_encoding.tensor_product import TensorProduct

    original = TensorProduct(
        (_source(("add2", "t"), False), _source(("cx",), False).block_encodings[0])
    )
    assert outer.alpha == original.alpha and outer.epsilon == original.epsilon
    source = original.adjoint() if adjoint else original
    gates = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        {"system": tuple(range(7))},
        (7,),
        {7: 0},
        cirq_style=True,
    )
    assert_source_dependencies(op.instructions, gates)
    amplitudes = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(128)], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    state = torch.zeros(256, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    state[::2] = amplitudes
    expected[::2] = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            128, 128
        )
        @ amplitudes
    )
    for history in ((0,), (1,)):
        torch.testing.assert_close(
            forced_branch(state, op.instructions, history),
            expected / 2**0.5,
            atol=5e-12,
            rtol=0,
        )
    inner_cost = inner.resource_model(
        (
            inner_blocks[0].resource_model(source_add_cost(2, 2)),
            inner_blocks[1].resource_model(ComponentCost("T", 1, GateCost(t=1))),
        ),
        adjoint=adjoint,
    )
    last_cost = last.resource_model(ComponentCost("CX", 2, GateCost(clifford=1)))
    _assert_model(
        outer.resource_model((inner_cost, last_cost), adjoint=adjoint), source, False
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("case", ("ancillas", "system", "resources"))
def test_symbolic_tensor_signature_and_fallback(case):
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import Hadamard, Identity, TGate
    from qualtran.bloqs.block_encoding import TensorProduct, Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol, simplify

    from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
        source_symbolic_tensor_product_resource_formula,
        source_tensor_product_block_signature,
    )
    from flagquantum.algorithms.primitives.unitary_block_encoding import (
        source_unitary_block_resource_formula,
    )
    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    a, b, r, s = (Symbol(name) for name in "abrs")
    cases = {
        "ancillas": (((1, a, 0), (1, b, 0)), TGate(), 1),
        "system": (((s, 2, 0), (1, 0, r)), Identity(s), 0),
        "resources": (((1, 0, 2), (1, 0, r)), TGate(), 1),
    }
    widths, first, first_t = cases[case]
    source = TensorProduct.of(
        *(
            Unitary(gate, ancilla_bitsize=ancilla, resource_bitsize=resource)
            for gate, (_, ancilla, resource) in zip(
                (first, Hadamard()), widths, strict=True
            )
        )
    )
    signature = source_tensor_product_block_signature(widths)
    assert [(reg.name, reg.bitsize) for reg in signature] == [
        (reg.name, reg.bitsize) for reg in source.signature
    ]
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        source.decompose_bloq()

    child_models = tuple(
        source_unitary_block_resource_formula(
            SymbolicResourceFormula(0, cliffords, 0, system, t=t),
            ancilla=ancilla,
            resource=resource,
        )
        for (cliffords, t), (system, ancilla, resource) in zip(
            ((0, first_t), (1, 0)), widths, strict=True
        )
    )
    model = source_symbolic_tensor_product_resource_formula(widths, child_models)
    for legacy in (False, True):
        expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for name in GateCost.__dataclass_fields__:
            assert simplify(getattr(model, name) - getattr(expected, name)) == 0
    assert simplify(model.logical_qubits - get_cost_value(source, QubitCount())) == 0
