"""Audit source PrepareOracle through existing named and packed consumers."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path

import pytest
from sympy import Symbol

from examples.algorithms.hubbard_lcu_catalog import HubbardLCUExample
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.value_math import SymbolicInt, SymbolicReal

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_prepare_documentation_has_no_hidden_factory_or_execution_body() -> None:
    from qualtran.bloqs.state_preparation import prepare_base

    declaration = prepare_base._PREPARE_ORACLE_DOC
    assert declaration.bloq_cls is prepare_base.PrepareOracle
    assert tuple(declaration.examples) == ()
    assert declaration.import_line == (
        "from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle"
    )
    assert PrepareOracle.__abstractmethods__ == frozenset({"selection_registers"})
    with pytest.raises(TypeError, match="selection_registers"):
        PrepareOracle()


@pytest.mark.ftqc_reference
def test_prepare_oracle_contract_and_existing_hubbard_binding() -> None:
    from qualtran.bloqs.state_preparation import prepare_base
    from qualtran.bloqs.state_preparation.black_box_prepare import _black_box_prepare
    from qualtran.bloqs.state_preparation.black_box_prepare_test import (
        TestPrepareOracle,
    )

    raw = Path(prepare_base.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "6870d7752aeb7da0c251ad1abb3b90519f2d0556"
    )
    with pytest.raises(TypeError, match="selection_registers"):
        prepare_base.PrepareOracle()

    source = _black_box_prepare.make().prepare
    native = HubbardLCUExample(2)
    selected, junk = source.selection_registers, source.junk_registers
    assert tuple(source.signature) == (*selected, *junk)
    assert (
        tuple((r.name, r.total_bits()) for r in selected)
        == native.prepare_oracle.selection_layout
    )
    assert (
        tuple((r.name, r.total_bits()) for r in junk)
        == native.prepare_oracle.junk_layout
    )
    assert native.prepare_oracle.normalization == source.l1_norm_of_coeffs == 32

    dummy = TestPrepareOracle()
    assert tuple(dummy.signature) == (*dummy.selection_registers, *dummy.junk_registers)
    with pytest.raises(NotImplementedError):
        _ = dummy.l1_norm_of_coeffs


@pytest.mark.ftqc_reference
def test_prepare_oracle_default_junk_and_missing_normalization() -> None:
    from qualtran import QAny, Register
    from qualtran.bloqs.state_preparation.prepare_base import (
        PrepareOracle as SourcePrepare,
    )

    class SelectionOnly(SourcePrepare):  # type: ignore[misc]
        @property
        def selection_registers(self) -> tuple[object, ...]:
            return (Register("data", QAny(2)),)

    source = SelectionOnly()
    typed = _SelectionOnly()
    assert PrepareOracle.__abstractmethods__ == SourcePrepare.__abstractmethods__
    assert tuple(reg.name for reg in typed.signature) == ("i", "j")
    assert typed.junk_registers == source.junk_registers == ()
    with pytest.raises(NotImplementedError):
        _ = typed.l1_norm_of_coeffs
    assert tuple(source.signature) == source.selection_registers
    with pytest.raises(NotImplementedError):
        _ = source.l1_norm_of_coeffs
    native = BlackBoxPrepare((("data", 2),), (), lambda *_: ())
    assert native.junk_layout == ()
    assert native.normalization is None
    assert native.selection_width == source.signature.n_qubits() == 2


class _SelectionOnly(PrepareOracle):
    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("i", quantum_bit_type()), Register("j", quantum_bit_type()))


def test_prepare_signature_keeps_symbolic_and_output_descriptors() -> None:
    n = Symbol("n", integer=True, positive=True)
    junk = Register(
        "work", opaque_register_type(n), shape=(2,), side=RegisterSide.RIGHT
    )

    class _DeclaredJunk(_SelectionOnly):
        @property
        def junk_registers(self) -> tuple[Register, ...]:
            return (junk,)

    oracle = _DeclaredJunk()
    assert tuple(oracle.signature) == (*oracle.selection_registers, junk)
    assert oracle.signature is oracle.signature
    assert oracle.signature.get_right("work") is junk
    assert tuple(oracle.signature.lefts()) == oracle.selection_registers
    assert junk.total_bits() == 2 * n


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_hubbard_uses_the_ordered_prepare_signature(dimension: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as SourceHubbard,
    )

    native = PrepareHubbard(dimension, t=1.5, u=2.0)
    reference = SourceHubbard(dimension, dimension, t=1.5, u=2.0)
    assert native.l1_norm_of_coeffs == native.alpha == reference.l1_norm_of_coeffs
    assert [
        (reg.name, reg.bitsize, reg.side.value, dict(reg.dtype.parameters).get("bound"))
        for reg in native.signature
    ] == [
        (
            reg.name,
            reg.bitsize,
            reg.side.value,
            getattr(reg.dtype, "iteration_length", None),
        )
        for reg in reference.signature
    ]
    assert (
        tuple((reg.name, reg.total_bits()) for reg in native.signature) == native.layout
    )


def test_packed_black_box_exposes_its_distinct_binding() -> None:
    oracle = _SelectionOnly()
    names = tuple(reg.name for reg in oracle.selection_registers)
    bound: list[tuple[int, ...]] = []

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[()]:
        bound.extend(ports[name] for name in names)
        return ()

    box = BlackBoxPrepare(tuple((name, 1) for name in names), (), emit)
    assert tuple(reg.name for reg in box.signature) == ("selection",)
    assert box.normalization is None
    assert box.emit((1, 0), (), (), ()) == ()
    assert bound == [(1,), (0,)]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_symbolic_prepare_resolves_into_actual_external_execution(
    width: int, adjoint: bool
) -> None:
    from typing import Any

    import torch
    from attrs import evolve, frozen
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import Hadamard, OnEach, Rz
    from qualtran.bloqs.state_preparation.prepare_base import (
        PrepareOracle as SourcePrepare,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import sympify

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives.black_box_prepare import (
        PackedPrepareDeclaration,
    )
    from flagquantum.core.ir import CircuitIR
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
    from flagquantum.simulation.unitary import get_unitary

    @frozen
    class ExternalPrepare(SourcePrepare):
        bitsize: Any
        angle: Any
        norm: Any

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("data", QAny(self.bitsize)),)

        @property
        def l1_norm_of_coeffs(self) -> Any:
            return self.norm

        def build_composite_bloq(self, bb: Any, data: Any) -> dict[str, Any]:
            data = bb.add(OnEach(self.bitsize, Hadamard()), q=data)
            bits = bb.split(data)
            bits[0] = bb.add(Rz(self.angle), q=bits[0])
            return {"data": bb.join(bits)}

    @dataclass(frozen=True)
    class Declaration(PrepareOracle):
        bitsize: SymbolicInt
        norm: SymbolicReal

        @property
        def selection_registers(self) -> tuple[Register, ...]:
            return (Register("data", opaque_register_type(self.bitsize)),)

        @property
        def l1_norm_of_coeffs(self) -> SymbolicReal:
            return self.norm

    n, theta = Symbol("n", integer=True, positive=True), Symbol("theta", real=True)
    source_declaration = ExternalPrepare(n, theta, 2**n)
    native_declaration = Declaration(n, 2**n)
    with pytest.raises(ValueError, match="concrete width for data"):
        PackedPrepareDeclaration(native_declaration).bind(lambda *_: ())
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(source_declaration)

    # Resolve immutable constructor parameters, not the reference algorithm.
    # This is the same explicit binding used by other symbolic native modules.
    values = {n: width, theta: 0.37 * width}
    source = evolve(
        source_declaration,
        bitsize=int(source_declaration.bitsize.subs(values)),
        angle=float(source_declaration.angle.subs(values)),
        norm=source_declaration.norm.subs(values),
    )
    declaration = replace(
        native_declaration,
        bitsize=int(sympify(native_declaration.bitsize).subs(values)),
        norm=sympify(native_declaration.norm).subs(values),
    )
    emitter = import_register_emitter(source)
    native = PackedPrepareDeclaration(declaration).bind(emitter)
    assert native.normalization == source.l1_norm_of_coeffs == 2**width
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    ir = (
        Circuit(width)
        .append(native.make_on(selection=tuple(range(width)), junk=(), adjoint=adjoint))
        .to_ir()
    )
    original = source.adjoint() if adjoint else source
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    restored = CircuitIR.from_json(ir.to_json())
    torch.testing.assert_close(
        get_unitary(restored, dtype=torch.complex128), expected, atol=5e-12, rtol=0
    )
    assert len(ir.instructions) == width + 1
    assert sum(g.name == "h" for g in ir.instructions) == width
    assert sum(g.name == "rz" for g in ir.instructions) == 1
    assert sum(g.name == "measure" for g in ir.instructions) == 0
    model = get_cost_value(original, QECGatesCost())
    assert model.clifford == width and model.rotation == 1 and model.measurement == 0
    assert get_cost_value(original, QubitCount()) == width

    # Exercise the actual source AutoPartition wrapper after the same explicit
    # symbolic resolution, then restore only owned IR at the SDK boundary.
    from qualtran.bloqs.state_preparation.black_box_prepare import (
        BlackBoxPrepare as SourcePackedPrepare,
    )

    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(SourcePackedPrepare(source_declaration))
    packed_source = SourcePackedPrepare(source)
    packed_emitter = import_register_emitter(packed_source)
    assert packed_emitter.layout == (("selection", width),)
    restored_emitter = replace(
        packed_emitter,
        snapshots=tuple(
            CircuitIR.from_json(snapshot.to_json())
            for snapshot in packed_emitter.snapshots
        ),
    )
    packed_native = BlackBoxPrepare(
        (("selection", width),), (), restored_emitter, packed_source.l1_norm_of_coeffs
    )
    packed_ir = (
        Circuit(width)
        .append(
            packed_native.make_on(
                selection=tuple(range(width)), junk=(), adjoint=adjoint
            )
        )
        .to_ir()
    )
    packed_original = packed_source.adjoint() if adjoint else packed_source
    torch.testing.assert_close(
        get_unitary(packed_ir, dtype=torch.complex128),
        torch.as_tensor(packed_original.tensor_contract(), dtype=torch.complex128),
        atol=5e-12,
        rtol=0,
    )
    assert packed_ir.instructions == ir.instructions
    assert packed_native.normalization == source.l1_norm_of_coeffs
