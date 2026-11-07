"""Verify source scalar atoms independently from their native carrier expansion."""

from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_indexed_component_cost_profiles import _assert_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unitary_block_encoding import _fixture

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kind", ["t", "cx", "add2"])
@pytest.mark.parametrize("phi", [0.0, 0.25, -0.5, 1.0, 1 / 3])
@pytest.mark.parametrize("adjoint,auxiliary", tuple(product((False, True), repeat=2)))
def test_scalar_source_tensors_child_gates_complete_branches_and_models(
    kind, phi, adjoint, auxiliary
):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, GlobalPhase, TGate
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary

    encoding, system, a, r, work, bits = _fixture(kind, int(auxiliary), int(auxiliary))
    child = Add(QUInt(2)) if kind == "add2" else TGate() if kind == "t" else CNOT()
    source_block = Unitary(
        child,
        alpha=encoding.alpha,
        ancilla_bitsize=len(a),
        resource_bitsize=len(r),
        epsilon=encoding.epsilon,
    )
    source = Phase(source_block, phi=phi, eps=0.03)
    native = PhaseBlockEncoding(encoding, phi, 0.03)
    for name in (
        "system_bitsize",
        "ancilla_bitsize",
        "resource_bitsize",
        "alpha",
        "epsilon",
    ):
        assert getattr(native, name) == getattr(source, name)
    assert native.signal_state == encoding.signal_state
    root = source.adjoint() if adjoint else source
    source_graph = root.decompose_bloq()
    scalar = [
        instance.bloq
        for instance in source_graph.bloq_instances
        if isinstance(instance.bloq, GlobalPhase)
    ]
    assert len(scalar) == 1
    assert scalar[0].exponent == (-phi if adjoint else phi)
    data = len(system) + len(a) + len(r)
    total = data + len(work)
    anchor = (r or system)[-1]
    bound = encoding.make_on(
        system=system,
        ancilla=a,
        resource=r,
        workspace=work,
        classical_bits=bits,
        adjoint=adjoint,
    )
    circuit = DynamicCircuit(total).append(
        native.make_on(bound, anchor=anchor, adjoint=adjoint)
    )
    gates = circuit.to_ir().instructions
    body = gates[:-4] if phi and adjoint else gates[4:] if phi else gates
    ports = {"system": system}
    if a:
        ports["ancilla"] = a
    if r:
        ports["resource"] = r
    original_child = source_block.adjoint() if adjoint else source_block
    original_gates = reference_arithmetic_gates(
        original_child.decompose_bloq().flatten(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
    )
    assert_source_dependencies(body, original_gates)
    matrix = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128).reshape(
        1 << data, 1 << data
    )
    amplitudes = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << data)], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << len(work)] = amplitudes
    expected[:: 1 << len(work)] = matrix @ amplitudes
    for history in product((0, 1), repeat=len(work)):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected * 2 ** (-len(work) / 2),
            atol=4e-12,
            rtol=0,
        )
    for legacy in (False, True):
        child_cost = (
            source_add_cost(2, 2, legacy_shims=legacy)
            if kind == "add2"
            else ComponentCost(
                kind,
                len(system),
                GateCost(t=1) if kind == "t" else GateCost(clifford=1),
            )
        )
        model = native.resource_model(encoding.resource_model(child_cost))
        _assert_model(model, root, legacy)
        report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
        assert report.emitted is not None
        assert model.calls[0].component.name == "GlobalPhase"
        assert model.calls[0].component.gates == GateCost()


@pytest.mark.parametrize("phi", [float("nan"), float("inf"), True])
def test_nonfinite_or_nonconcrete_phase_refused_before_child(phi):
    encoding, *_ = _fixture("t", 0, 0)

    class Refuse:
        def append_to(self, circuit):
            pytest.fail("Invalid phase reached child")

    with pytest.raises(ValueError):
        PhaseBlockEncoding(encoding, phi, 0).make_on(Refuse(), anchor=0)


def test_symbolic_phase_models_without_execution():
    from sympy import Symbol

    encoding, *_ = _fixture("t", 0, 0)
    phase = PhaseBlockEncoding(
        encoding, Symbol("phi", real=True), Symbol("eps", positive=True)
    )
    assert phase.epsilon == 0.01 + Symbol("eps", positive=True)
    assert (
        estimate_ftqc_resources(
            phase.resource_model(ComponentCost("T", 1, GateCost(t=1)))
        ).logical.gates.t
        == 1
    )
    with pytest.raises(ValueError, match="concrete"):
        phase.make_on(encoding.make_on(system=(0,)), anchor=0)


@pytest.mark.parametrize("anchor", [-1, True, 3])
def test_invalid_anchor_preserves_circuit(anchor):
    encoding, *_ = _fixture("t", 0, 0)
    circuit = DynamicCircuit(1)
    before = circuit.to_ir()
    with pytest.raises(ValueError):
        circuit.append(
            PhaseBlockEncoding(encoding, 0.25, 0).make_on(
                encoding.make_on(system=(0,)), anchor=anchor
            )
        )
    assert circuit.to_ir() == before


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("lcu,adjoint", tuple(product((False, True), repeat=2)))
def test_existing_select_and_lcu_consumers_need_no_binding_changes(lcu, adjoint):
    from qualtran import BQUInt, Register
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding as SourceLCU,
    )
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        SelectBlockEncoding as SourceSelectBlock,
    )
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    from flagquantum.algorithms.primitives.black_box_prepare import (
        BlackBoxPrepare,
        _identity_prepare,
    )
    from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
    from flagquantum.algorithms.primitives.select_block_encoding import (
        SelectBlockEncoding,
    )
    from flagquantum.compiler.registers.multiplexed_swap import (
        multiplexed_cswap_ir,
        source_multiplexed_cswap_cost,
    )

    class NormalizedIdentity(PrepareIdentity):
        @property
        def l1_norm_of_coeffs(self):
            return 1.0

    source_select = MultiplexedCSwap(Register("selection", BQUInt(2, 3)), 1)
    source_prepare = NormalizedIdentity(source_select.selection_registers)
    source_block = (SourceLCU if lcu else SourceSelectBlock)(
        source_select, source_prepare
    )
    source = Phase(source_block, phi=0.25, eps=0.001)
    if adjoint:
        source = source.adjoint()
    signal = BlackBoxPrepare((("selection", 2),), (), _identity_prepare, 1.0)

    def select(ports, workspace, bits, inverse):
        return multiplexed_cswap_ir(
            (),
            (ports["selection"],),
            (3,),
            tuple((wire,) for wire in ports["target"][:3]),
            ports["target"][3:],
            workspace,
            bits,
            adjoint=inverse,
        )

    selection = SelectBlockEncoding(
        (("selection", 2),), (("target", 4),), select, signal
    )
    block = LCUBlockEncoding(selection) if lcu else selection
    phase = PhaseBlockEncoding(block, 0.25, 0.001)
    assert phase.signal_state == block.signal_state
    assert phase.alpha == 1.0
    if lcu:
        bound = block.make_on(
            selection=(0, 1),
            junk=(),
            target=(2, 3, 4, 5),
            select_workspace=(6,),
            select_classical_bits=(0,),
            adjoint=adjoint,
        )
        cost = block.resource_model(
            source_multiplexed_cswap_cost((3,), (2,), 0, 1),
            ComponentCost("identity", 2),
            ComponentCost("identity", 2),
            adjoint=adjoint,
        )
    else:
        bound = block.make_on(
            selection=(0, 1),
            target=(2, 3, 4, 5),
            workspace=(6,),
            classical_bits=(0,),
            adjoint=adjoint,
        )
        cost = block.resource_model(source_multiplexed_cswap_cost((3,), (2,), 0, 1))
    circuit = DynamicCircuit(7).append(phase.make_on(bound, anchor=5, adjoint=adjoint))
    state = torch.zeros(128, dtype=torch.complex128)
    amplitudes = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(64)], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    state[::2] = amplitudes
    expected = torch.zeros_like(state)
    from dataclasses import asdict

    from qualtran import BloqError
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    # Original Phase assumes packed system/ancilla/resource names, while
    # these existing source children expose selection/target names instead.
    with pytest.raises(BloqError, match="Soquet named"):
        source.tensor_contract()
    with pytest.raises(RuntimeError, match="Soquet named"):
        get_cost_value(source, QubitCount())
    body_source = source_block.adjoint() if adjoint else source_block
    coefficient = complex(GlobalPhase(exponent=-0.25 if adjoint else 0.25).coefficient)
    expected[::2] = coefficient * (
        torch.as_tensor(body_source.tensor_contract(), dtype=torch.complex128).reshape(
            64, 64
        )
        @ amplitudes
    )
    for history in ((0,), (1,)):
        torch.testing.assert_close(
            forced_branch(state, circuit.to_ir().instructions, history),
            expected / 2**0.5,
            atol=4e-12,
            rtol=0,
        )
    gates = estimate_ftqc_resources(phase.resource_model(cost)).logical.gates
    original = get_cost_value(source, QECGatesCost())
    assert asdict(gates) == {key: getattr(original, key) for key in asdict(gates)}


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_nested_phase_wrappers_preserve_each_scalar_and_precision(adjoint):
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary

    encoding, system, a, r, work, bits = _fixture("t", 1, 1)
    inner = PhaseBlockEncoding(encoding, 0.125, 0.001)
    outer = PhaseBlockEncoding(inner, -0.25, 0.002)
    source = Phase(
        Phase(
            Unitary(
                TGate(), ancilla_bitsize=1, resource_bitsize=1, alpha=0.5, epsilon=0.01
            ),
            phi=0.125,
            eps=0.001,
        ),
        phi=-0.25,
        eps=0.002,
    )
    assert outer.epsilon == source.epsilon
    bound = encoding.make_on(system=system, ancilla=a, resource=r, adjoint=adjoint)
    circuit = DynamicCircuit(3).append(
        outer.make_on(
            inner.make_on(bound, anchor=2, adjoint=adjoint), anchor=1, adjoint=adjoint
        )
    )
    state = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(8)], dtype=torch.complex128
    )
    original = source.adjoint() if adjoint else source
    expected = (
        torch.as_tensor(original.tensor_contract(), dtype=torch.complex128).reshape(
            8, 8
        )
        @ state
    )
    torch.testing.assert_close(
        forced_branch(state, circuit.to_ir().instructions, ()),
        expected,
        atol=4e-12,
        rtol=0,
    )
    _assert_model(
        outer.resource_model(
            inner.resource_model(
                encoding.resource_model(ComponentCost("T", 1, GateCost(t=1)))
            )
        ),
        original,
        False,
    )


def test_phase_append_and_runtime_round_trip_example(capsys):
    from examples.algorithms.phase_block_encoding import main

    main()
    output = capsys.readouterr().out
    assert "Source-model T equivalents 4" in output
    assert "Emitted T/T-dagger instructions 4" in output
    assert "Emitted unsynthesized phase instructions 2" in output
    assert "Model logical peak / reserved wires 7 7" in output


def test_direction_and_undersized_cost_rejected():
    encoding, *_ = _fixture("t", 1, 1)
    phase = PhaseBlockEncoding(encoding, 0.25, 0.01)
    bound = encoding.make_on(system=(0,), ancilla=(1,), resource=(2,))
    with pytest.raises(ValueError, match="boolean"):
        phase.make_on(bound, anchor=0, adjoint=1)
    with pytest.raises(ValueError, match="quantum ports"):
        phase.resource_model(ComponentCost("narrow", 2))
