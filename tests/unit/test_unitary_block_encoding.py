"""Source unitary wrapper gates, metadata, idle ports and exact branch amplitudes."""

from collections.abc import Mapping
from dataclasses import replace
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


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("overrides", [False, True])
def test_original_t_wrapper_examples_exact_matrix_and_ports(overrides):
    from qualtran.bloqs.block_encoding.unitary import (
        _unitary_block_encoding,
        _unitary_block_encoding_properties,
    )

    from flagquantum import Circuit
    from flagquantum.simulation.unitary import get_unitary

    if overrides:
        source = _unitary_block_encoding_properties()
        ancilla, resource = (1, 2), (3,)
        native, *_ = _fixture("t", 2, 1)
    else:
        source = _unitary_block_encoding()
        ancilla, resource = (), ()
        native, *_ = _fixture("t", 0, 0)
        native = replace(native, alpha=1, epsilon=0)
    width = 1 + len(ancilla) + len(resource)
    assert tuple((reg.name, reg.bitsize) for reg in native.signature) == tuple(
        (reg.name, reg.bitsize) for reg in source.signature
    )
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    model = native.resource_model(ComponentCost("T", 1, GateCost(t=1)))
    assert source.build_call_graph(None) == {source.U: 1}
    assert len(model.calls) == model.calls[0].repetitions == 1
    for legacy in (False, True):
        _assert_model(model, source, legacy)
    binding = dict(system=(0,), ancilla=ancilla, resource=resource)
    assert native.signal_state.emit(ancilla, (), (), ()) == ()
    circuit = Circuit(width, dtype=torch.complex128)
    circuit.append(native.make_on(**binding))
    assert circuit.to_ir().instructions == (Instruction("t", (0,)),)
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )
    circuit.append(native.make_on(**binding, adjoint=True))
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.eye(1 << width, dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.ftqc_reference
def test_original_rejects_preparation_as_a_unitary_child():
    from qualtran.bloqs.basic_gates import IntState
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register, RegisterSide
    from flagquantum.core.ir.program.signatures import Signature

    source = IntState(55, bitsize=8)
    with pytest.raises(ValueError, match="THRU"):
        Unitary(source)
    (register,) = source.signature
    native = Signature(
        (Register(register.name, opaque_register_type(8), side=RegisterSide.RIGHT),)
    )
    phase, *_ = _fixture("t", 0, 0)
    with pytest.raises(ValueError, match="THRU"):
        UnitaryBlockEncoding.from_signature(native, phase.unitary)
    bound = UnitaryBlockEncoding.from_signature(Signature.build(q=1), phase.unitary)
    assert bound.make_on(system=(3,)).to_ir() == (Instruction("t", (3,)),)


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


@pytest.mark.ftqc_reference
def test_original_shaped_ports_and_symbolic_declarations() -> None:
    from dataclasses import dataclass

    import numpy as np
    from qualtran import Bloq, BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard, Identity, XGate
    from qualtran.bloqs.block_encoding import Unitary
    from qualtran.resource_counting import QubitCount, get_cost_value
    from sympy import Symbol

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives.unitary_block_encoding import (
        source_unitary_block_resource_formula,
        source_unitary_block_signature,
    )
    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        SymbolicResourceFormula,
    )
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.unitary import get_unitary

    @dataclass(frozen=True)
    class ShapedX(Bloq):
        shape: tuple

        @property
        def signature(self) -> SourceSignature:
            return SourceSignature([SourceRegister("q", QBit(), shape=self.shape)])

        def build_composite_bloq(
            self, bb: BloqBuilder, q: np.ndarray
        ) -> dict[str, np.ndarray]:
            return {
                "q": np.array(
                    [bb.add(XGate(), q=wire) for wire in q.flat], dtype=object
                ).reshape(q.shape)
            }

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("x", ports["q"]),)

    for shape in ((1,), (1, 1), (2,), (Symbol("k", positive=True, integer=True),)):
        typed = Signature((Register("q", quantum_bit_type(), shape=shape),))
        locked = Unitary(ShapedX(shape))
        declared = source_unitary_block_signature(typed)
        assert [(r.name, r.bitsize) for r in declared] == [
            (r.name, r.bitsize) for r in locked.signature
        ]
        if shape in ((1,), (1, 1)):
            native = UnitaryBlockEncoding.from_signature(typed, emit)
            actual = Circuit(1, dtype=torch.complex128).append(
                native.make_on(system=(0,))
            )
            assert actual.to_ir().instructions == (Instruction("x", (0,)),)
            torch.testing.assert_close(
                get_unitary(actual),
                torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128),
            )
        else:
            with pytest.raises(ValueError, match="Total bitsize"):
                locked.tensor_contract()
            with pytest.raises(
                ValueError, match="source packed width|concrete quantum"
            ):
                UnitaryBlockEncoding.from_signature(typed, emit)

    a, r = (Symbol(name, positive=True, integer=True) for name in ("a", "r"))
    locked = Unitary(Hadamard(), ancilla_bitsize=a, resource_bitsize=r)
    signature = source_unitary_block_signature(
        Signature.build(q=1), ancilla=a, resource=r
    )
    assert [(reg.name, reg.bitsize) for reg in signature] == [
        (reg.name, reg.bitsize) for reg in locked.signature
    ]
    formula = source_unitary_block_resource_formula(
        SymbolicResourceFormula(0, 1, 0, 1), ancilla=a, resource=r
    )
    assert formula.logical_qubits == get_cost_value(locked, QubitCount())
    assert formula.clifford == 1
    n = Symbol("n", positive=True, integer=True)
    variable_child = Signature.build(q=n)
    variable = source_unitary_block_signature(variable_child, ancilla=a, resource=r)
    locked_variable = Unitary(Identity(n), ancilla_bitsize=a, resource_bitsize=r)
    assert [(reg.name, reg.bitsize) for reg in variable] == [
        (reg.name, reg.bitsize) for reg in locked_variable.signature
    ]
    with pytest.raises(ValueError, match="concrete quantum"):
        UnitaryBlockEncoding.from_signature(variable_child, emit)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("pattern", [(1,), (0,), (1, 0), (1, 0, 1)])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("aux", [False, True])
def test_generic_unitary_control_preserves_complex_branches(
    pattern: tuple[int, ...], adjoint: bool, aux: bool
) -> None:
    import numpy as np
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding import Unitary

    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.dynamic import DynamicCircuit

    child, *_ = _fixture("t", int(aux), int(aux))
    original = Unitary(TGate(), ancilla_bitsize=int(aux), resource_bitsize=int(aux))
    spec = CtrlSpec(cvs=pattern[0] if len(pattern) == 1 else np.array(pattern))
    root = original.controlled(ctrl_spec=spec)
    if adjoint:
        root = root.adjoint()
    target = len(pattern)
    ancilla, resource = ((target + 1,), (target + 2,)) if aux else ((), ())
    logical = target + 1 + 2 * int(aux)
    work = tuple(range(logical, logical + target - 1))
    slots = tuple(range(3, 3 + len(work)))
    width = logical + len(work)
    registers = (ControlRegister(1, pattern),)
    predicate = work[-1] if work else 0

    def bind(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        child_work: tuple[int, ...],
        child_slots: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        assert (control, ports["q"], child_work, child_slots, inverse) == (
            predicate,
            (target,),
            (),
            (),
            adjoint,
        )
        return (
            Instruction(
                "cphase",
                (control, target),
                params={"theta": (-1 if inverse else 1) * pi / 4},
            ),
        )

    # The concrete fixture's child register name is q.
    operation = child.make_controlled_on(
        bind,
        controls=(tuple(range(target)),),
        control_registers=registers,
        system=(target,),
        ancilla=ancilla,
        resource=resource,
        control_workspace=work,
        control_classical_bits=slots,
        adjoint=adjoint,
    )
    gates = DynamicCircuit(width).append(operation).to_ir().instructions
    assert tuple(g.wires for g in gates if g.name == "cphase") == ((predicate, target),)
    assert tuple(
        g.metadata["classical_bit"] for g in gates if g.name == "measure"
    ) == tuple(reversed(slots))
    values = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(values + 1, values % 7 - 3)
    state /= state.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << len(work)] = state
    source = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
    expected[:: 1 << len(work)] = source @ state
    for history in product((0, 1), repeat=len(work)):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected * 2 ** (-len(work) / 2),
            rtol=0,
            atol=3e-12,
        )
    for legacy in (False, True):
        _assert_model(
            child.controlled_resource_model(
                ComponentCost("ControlledT", 2, GateCost(rotation=1)),
                registers,
                legacy_shims=legacy,
            ),
            root,
            legacy,
        )
