"""Source unitary wrapper gates, metadata, idle ports and exact branch amplitudes."""

from collections.abc import Mapping
from itertools import product
from math import pi

import pytest
import torch

from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_indexed_component_cost_profiles import _assert_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit


def _fixture(kind: str, ancilla: int, resource: int):
    if kind.startswith("add"):
        n = int(kind[-1])
        layout = (("a", n), ("b", n))
        work_count = n - 1
    else:
        n = 1
        layout = (("q", 1),) if kind == "t" else (("ctrl", 1), ("target", 1))
        work_count = 0

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        if kind.startswith("add"):
            return (add_adjoint_ir if adjoint else add_ir)(
                ports["a"], ports["b"], work, bits
            )
        if kind == "t":
            return (Instruction("tdg" if adjoint else "t", ports["q"]),)
        return (Instruction("cx", (*ports["ctrl"], *ports["target"])),)

    encoding = UnitaryBlockEncoding(
        layout,
        emit,
        alpha=0.5,
        ancilla_bitsize=ancilla,
        resource_bitsize=resource,
        epsilon=0.01,
    )
    system = tuple(range(encoding.system_bitsize))
    a = tuple(range(len(system), len(system) + ancilla))
    r = tuple(range(len(system) + ancilla, len(system) + ancilla + resource))
    work = tuple(
        range(
            len(system) + ancilla + resource,
            len(system) + ancilla + resource + work_count,
        )
    )
    return encoding, system, a, r, work, tuple(range(40, 40 + work_count))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kind", ["t", "cx", "add2", "add3"])
@pytest.mark.parametrize("ancilla,resource", [(0, 0), (1, 2), (3, 1)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_packed_child_graph_metadata_and_profiles(
    kind, ancilla, resource, adjoint
):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    encoding, system, a, r, work, bits = _fixture(kind, ancilla, resource)
    if kind.startswith("add"):
        child = Add(QUInt(int(kind[-1])))
    else:
        child = TGate() if kind == "t" else CNOT()
    source = Unitary(
        child,
        alpha=0.5,
        ancilla_bitsize=ancilla,
        resource_bitsize=resource,
        epsilon=0.01,
    )
    assert (
        encoding.system_bitsize,
        encoding.ancilla_bitsize,
        encoding.resource_bitsize,
        encoding.alpha,
        encoding.epsilon,
    ) == (
        source.system_bitsize,
        source.ancilla_bitsize,
        source.resource_bitsize,
        source.alpha,
        source.epsilon,
    )
    assert (
        encoding.signal_state.selection_width == source.signal_state.selection_bitsize
    )
    assert encoding.signal_state.junk_width == source.signal_state.junk_bitsize
    assert encoding.signal_state.emit(a, (), (), ()) == ()
    assert encoding.signal_state.normalization is None
    with pytest.raises(NotImplementedError):
        _ = source.signal_state.l1_norm_of_coeffs
    bound = encoding.make_on(
        system=system,
        ancilla=a,
        resource=r,
        workspace=work,
        classical_bits=bits,
        adjoint=adjoint,
    )
    gates = bound.to_ir()
    ports = {"system": system}
    if a:
        ports["ancilla"] = a
    if r:
        ports["resource"] = r
    original = source.adjoint() if adjoint else source
    expected = reference_arithmetic_gates(
        original.decompose_bloq().flatten(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
    )
    assert_source_dependencies(gates, expected)
    for legacy in (False, True):
        if kind.startswith("add"):
            cost = source_add_cost(int(kind[-1]), int(kind[-1]), legacy_shims=legacy)
        else:
            cost = ComponentCost(
                kind,
                len(system),
                GateCost(t=1) if kind == "t" else GateCost(clifford=1),
            )
        _assert_model(encoding.resource_model(cost), original, legacy)


@pytest.mark.parametrize("kind", ["t", "cx", "add2", "add3"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_complex_state_and_every_measurement_branch(kind, adjoint):
    from cmath import exp

    encoding, system, a, r, work, bits = _fixture(kind, 1, 1)
    bound = encoding.make_on(
        system=system,
        ancilla=a,
        resource=r,
        workspace=work,
        classical_bits=bits,
        adjoint=adjoint,
    )
    data = len(system) + len(a) + len(r)
    total = data + len(work)
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for value in range(1 << data):
        amp = complex(value + 1, 3 - value) / (1 << data)
        initial[value << len(work)] = amp
        sys = value >> 2
        output = sys
        factor = 1
        if kind == "t":
            factor = exp((-1 if adjoint else 1) * 1j * pi / 4) if sys else 1
        elif kind == "cx":
            output = sys ^ int(bool(sys & 2))
        else:
            n = int(kind[-1])
            left = sys >> n
            right = sys & ((1 << n) - 1)
            output = (left << n) | ((right + (-1 if adjoint else 1) * left) % (1 << n))
        expected[(((output << 2) | (value & 3)) << len(work))] = amp * factor
    for history in product((0, 1), repeat=len(work)):
        torch.testing.assert_close(
            forced_branch(initial, bound.to_ir(), history),
            expected * 2 ** (-len(work) / 2),
            atol=4e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"system": (0,)},
        {"ancilla": ()},
        {"resource": (0,)},
        {"workspace": (0,)},
        {"classical_bits": (0, 0)},
        {"adjoint": 1},
    ],
)
def test_binding_rejected_before_child(change):
    def refuse(*args):
        pytest.fail("Invalid block binding reached child")

    encoding = UnitaryBlockEncoding(
        (("a", 1), ("b", 1)), refuse, ancilla_bitsize=1, resource_bitsize=1
    )
    args = dict(system=(0, 1), ancilla=(2,), resource=(3,))
    args.update(change)
    with pytest.raises(ValueError):
        encoding.make_on(**args)


@pytest.mark.parametrize(
    "layout", [(("q", -1),), (("q", 1.5),), (("q", True),), (("q", 1), ("q", 1))]
)
def test_layout_uses_shared_concrete_contract(layout):
    with pytest.raises(ValueError):
        UnitaryBlockEncoding(layout, lambda *args: ())


def test_child_cannot_touch_idle_auxiliary_port():
    encoding = UnitaryBlockEncoding(
        (("q", 1),), lambda *args: (Instruction("x", (1,)),), ancilla_bitsize=1
    )
    with pytest.raises(ValueError, match="outside"):
        encoding.make_on(system=(0,), ancilla=(1,)).to_ir()


def test_runtime_append_and_model_example(capsys):
    from examples.algorithms.unitary_block_encoding import main

    main()
    output = capsys.readouterr().out
    assert "Model T equivalents 4" in output
    assert "Emitted T/T-dagger instructions 4" in output
    assert "Model logical peak 7" in output


def test_out_of_range_binding_refused_before_append():
    from flagquantum.dynamic import DynamicCircuit

    encoding, system, a, r, work, bits = _fixture("add2", 1, 1)
    bound = encoding.make_on(
        system=system, ancilla=a, resource=r, workspace=work, classical_bits=bits
    )
    circuit = DynamicCircuit(6)
    before = circuit.to_ir()
    with pytest.raises(ValueError, match="exceeds"):
        circuit.append(bound)
    assert circuit.to_ir() == before


@pytest.mark.parametrize("ancilla,resource", [(-1, 0), (True, 0), (0, 0.5)])
def test_auxiliary_dimensions_refused(ancilla, resource):
    with pytest.raises(ValueError, match="nonnegative"):
        UnitaryBlockEncoding(
            (("q", 1),),
            lambda *args: (),
            ancilla_bitsize=ancilla,
            resource_bitsize=resource,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_noncontiguous_packed_ports_match_original(adjoint):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.block_encoding.unitary import Unitary

    encoding, *_ = _fixture("add2", 1, 1)
    bound = encoding.make_on(
        system=(6, 1, 8, 3),
        ancilla=(0,),
        resource=(5,),
        workspace=(9,),
        classical_bits=(17,),
        adjoint=adjoint,
    )
    source = Unitary(Add(QUInt(2)), ancilla_bitsize=1, resource_bitsize=1)
    if adjoint:
        source = source.adjoint()
    expected = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        {"system": (6, 1, 8, 3), "ancilla": (0,), "resource": (5,)},
        (9,),
        {9: 17},
        cirq_style=True,
    )
    assert_source_dependencies(bound.to_ir(), expected)


def test_dimension_only_resource_query_never_invokes_child():
    from flagquantum.qec.resources.estimation import estimate_ftqc_resources

    def refuse(*args):
        pytest.fail("Resource query entered child emission")

    encoding = UnitaryBlockEncoding(
        (("a", 1024), ("b", 1024)), refuse, ancilla_bitsize=2, resource_bitsize=3
    )
    report = estimate_ftqc_resources(
        encoding.resource_model(source_add_cost(1024, 1024))
    )
    assert report.logical.peak_logical_qubits == 3076
    assert report.logical.gates.total_t_count() == 4092
    assert report.emitted is None


def test_child_model_includes_all_system_ports():
    encoding, *_ = _fixture("add2", 0, 0)
    with pytest.raises(ValueError, match="every system"):
        encoding.resource_model(ComponentCost("narrow", 3))
