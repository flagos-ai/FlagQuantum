"""Audit the frozen block export families and their native declaration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import isclass

import pytest

from flagquantum.algorithms.primitives._block_metadata import (
    _BlockDeclaration,
    _BlockMetadata,
    _packed_block_signature,
)
from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
    ChebyshevBlockEncoding,
)
from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
)
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.algorithms.primitives.scaled_chebyshev_block_encoding import (
    ScaledChebyshevBlockEncoding,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
    SparseMatrixBlockEncoding,
)
from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
    TensorProductBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.value_math import SymbolicInt, SymbolicReal

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_locked_block_package_exports_have_native_owners() -> None:
    from qualtran.bloqs import block_encoding as locked

    native = {
        "BlockEncoding": _BlockMetadata,
        "ChebyshevPolynomial": ChebyshevBlockEncoding,
        "ScaledChebyshevPolynomial": ScaledChebyshevBlockEncoding,
        "LCUBlockEncoding": LCUBlockEncoding,
        "SelectBlockEncoding": SelectBlockEncoding,
        "LinearCombination": LinearCombinationBlockEncoding,
        "Phase": PhaseBlockEncoding,
        "Product": ProductBlockEncoding,
        "SparseMatrix": SparseMatrixBlockEncoding,
        "SparseMatrixHermitian": HermitianSparseMatrixBlockEncoding,
        "TensorProduct": TensorProductBlockEncoding,
        "Unitary": UnitaryBlockEncoding,
    }
    exports = {
        name
        for name, member in vars(locked).items()
        if isclass(member) and issubclass(member, locked.BlockEncoding)
    }
    assert exports == native.keys()
    assert len(set(native.values())) == len(exports) == 12
    required = (
        locked.BlockEncoding.__abstractmethods__ & vars(locked.BlockEncoding).keys()
    )
    assert required == {
        name
        for name, member in vars(_BlockDeclaration).items()
        if isinstance(member, property)
    }
    for owner in native.values():
        assert owner.__module__.startswith("flagquantum.algorithms.primitives.")
        fields = set(dir(owner)) | vars(owner).get("__annotations__", {}).keys()
        assert not (missing := required - fields), (owner.__name__, missing)


@dataclass(frozen=True)
class _SymbolicSignal(PrepareOracle):
    bitsize: SymbolicInt

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("selection", opaque_register_type(self.bitsize)),)


def test_symbolic_block_declaration_retains_all_six_source_properties() -> None:
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.bloqs.block_encoding import Unitary
    from sympy import Symbol

    system, ancilla, resource = (
        Symbol(name, positive=True, integer=True) for name in ("s", "a", "r")
    )
    normalization, error = (
        Symbol(name, positive=True) for name in ("alpha", "epsilon")
    )

    @dataclass(frozen=True)
    class Declaration:
        system_bitsize: SymbolicInt = system
        ancilla_bitsize: SymbolicInt = ancilla
        resource_bitsize: SymbolicInt = resource
        alpha: SymbolicReal = normalization
        epsilon: SymbolicReal = error
        signal_state: PrepareOracle = _SymbolicSignal(ancilla)

    native: _BlockDeclaration[SymbolicInt, PrepareOracle] = Declaration()
    source = Unitary(
        Identity(system),
        alpha=normalization,
        epsilon=error,
        ancilla_bitsize=ancilla,
        resource_bitsize=resource,
    )
    actual = _packed_block_signature(
        native.system_bitsize, native.ancilla_bitsize, native.resource_bitsize
    )
    assert [(reg.name, reg.bitsize) for reg in actual] == [
        (reg.name, reg.bitsize) for reg in source.signature
    ]
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    assert (
        native.signal_state.signature.n_qubits()
        == source.signal_state.signature.n_qubits()
    )
    assert native.signal_state.signature.n_qubits() == ancilla
    assert len(_packed_block_signature(system, 0, resource)) == 2
