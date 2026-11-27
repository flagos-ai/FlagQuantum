"""Packed preparation resolves raw storage size without changing descriptors."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import attrs
    import cirq
    import numpy as np
    import sympy
    import torch
    from qualtran._infra.data_types import QAny, QBit, QDType, QFxp
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran.bloqs.state_preparation.black_box_prepare import (
        BlackBoxPrepare as SourcePacked,
    )
    from qualtran.bloqs.state_preparation.prepare_base import (
        PrepareOracle as SourceOracle,
    )
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import PackedPrepareDeclaration
from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@attrs.frozen
class NamedSourceIdentity(SourceOracle):
    """Explicit identity child supplied through the unchanged source interface."""

    dtype: QDType[int]
    shape: tuple[int | sympy.Expr, ...] = ()

    @property
    def selection_registers(self) -> tuple[SourceRegister, ...]:
        return (
            SourceRegister("x", self.dtype, self.shape),
            SourceRegister("flag", QBit()),
        )

    @property
    def junk_registers(self) -> tuple[SourceRegister, ...]:
        return (SourceRegister("a", QAny(2)),)

    @property
    def l1_norm_of_coeffs(self) -> int:
        return 2

    def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
        return soqs


@dataclass(frozen=True)
class NamedIdentity(PrepareOracle):
    dtype: RegisterType
    shape: tuple[int | sympy.Expr, ...] = ()

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (
            Register("x", self.dtype, self.shape),
            Register("flag", quantum_bit_type()),
        )

    @property
    def junk_registers(self) -> tuple[Register, ...]:
        return (Register("a", opaque_register_type(2)),)

    @property
    def l1_norm_of_coeffs(self) -> int:
        return 2


@pytest.mark.parametrize(
    "width,shape",
    ((1, ()), (sympy.Integer(1), ()), (sympy.Integer(3), ()), (sympy.Integer(2), (2,))),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_exact_size_sdk_binding_preserves_types_named_order_and_complex_spectator(
    width: int | sympy.Expr, shape: tuple[int, ...], adjoint: bool
) -> None:
    source = SourcePacked(NamedSourceIdentity(QAny(width), shape))
    native_dtype = opaque_register_type(width)
    declaration = PackedPrepareDeclaration(NamedIdentity(native_dtype, shape))
    assert declaration.selection_width == source.selection_bitsize
    assert declaration.junk_width == source.junk_bitsize == 2
    calls: list[tuple[dict[str, tuple[int, ...]], bool]] = []

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        assert workspace == classical_bits == ()
        calls.append((dict(ports), inverse))
        return ()

    bound = declaration.bind(emit)
    assert declaration.prepare.selection_registers[0].dtype is native_dtype
    assert bound.normalization == source.l1_norm_of_coeffs == 2
    selection_width = int(source.selection_bitsize)
    selection = tuple(reversed(range(selection_width)))
    junk = (selection_width, selection_width + 1)
    total = selection_width + 3
    operation, output = source.as_cirq_op(
        cirq.SimpleQubitManager(),
        selection=np.array([cirq.LineQubit(w) for w in selection]),
        junk=np.array([cirq.LineQubit(w) for w in junk]),
    )
    sdk = cirq.Circuit(cirq.decompose(operation))
    assert tuple(q.x for q in output["selection"].flat) == selection
    assert tuple(q.x for q in output["junk"].flat) == junk
    expected = torch.tensor(
        sdk.unitary(qubit_order=cirq.LineQubit.range(total)), dtype=torch.complex128
    )
    values = torch.arange(1, (1 << total) + 1, dtype=torch.float64)
    initial = torch.complex(values, 1 - 3 * values)
    initial /= initial.norm()
    circuit = Circuit(total, inputs=initial, dtype=torch.complex128).append(
        bound.make_on(selection=selection, junk=junk, adjoint=adjoint)
    )
    torch.testing.assert_close(
        circuit.state().flatten(), expected @ initial, atol=2e-12, rtol=0
    )
    assert calls == [
        ({"x": selection[:-1], "flag": selection[-1:], "a": junk}, adjoint)
    ]
    assert circuit.to_ir().instructions == ()
    assert CircuitIR.from_json(circuit.to_ir().to_json()) == circuit.to_ir()


def test_symbolic_interpretation_and_unresolved_array_boundaries() -> None:
    parameter = sympy.Symbol("f", integer=True)
    declaration = PackedPrepareDeclaration(
        NamedIdentity(fixed_point_register_type(3, parameter))
    )
    bound = declaration.bind(lambda *args: ())
    assert bound.selection_layout == (("x", 3), ("flag", 1))
    for dtype, shape in (
        (opaque_register_type(sympy.Symbol("n")), ()),
        (opaque_register_type(2), (sympy.Integer(2),)),
    ):
        with pytest.raises(ValueError, match="concrete width"):
            PackedPrepareDeclaration(NamedIdentity(dtype, shape)).bind(lambda *args: ())
    source = SourcePacked(NamedSourceIdentity(QFxp(3, parameter)))
    operation, _ = source.as_cirq_op(
        cirq.SimpleQubitManager(),
        selection=np.array(cirq.LineQubit.range(4)),
        junk=np.array(cirq.LineQubit.range(4, 6)),
    )
    assert list(cirq.Circuit(cirq.decompose(operation)).all_operations()) == []
