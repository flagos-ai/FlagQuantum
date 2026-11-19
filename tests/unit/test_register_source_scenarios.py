# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source behavior audited against the frozen Qualtran v0.7.0 registers_test.py.
"""Run complete frozen register test bodies with native metadata owners."""

from hashlib import sha1
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    classical_bit_type,
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.core.value_parameters import is_symbolic_value
from flagquantum.ecosystem.cirq._register_export import _named_qubits, _register_shape

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")


def native_bounded(bitsize: SymbolicInt, iteration_length: SymbolicInt) -> RegisterType:
    """Translate the source test's keyword spelling at its external boundary."""
    return bounded_unsigned_register_type(bitsize, iteration_length)


def native_named_qubits(registers: Signature) -> dict[str, Any]:
    cirq = pytest.importorskip("cirq")
    return {
        reg.name: _named_qubits(cirq, reg, _register_shape(reg)) for reg in registers
    }


@pytest.fixture(scope="module")
def original() -> ModuleType:
    from qualtran._infra import registers_test

    assert isinstance(registers_test, ModuleType)
    return registers_test


@pytest.mark.parametrize(
    "name",
    [
        "test_register",
        "test_classical_register",
        "test_multidim_register",
        "test_selection_registers_consistent",
        "test_registers_getitem_raises",
        "test_signature",
        "test_get_named_qubits",
        "test_signature_symbolic",
        "test_partial_classical_signature_n_qubits",
        "test_signature_build",
        "test_and_regs",
        "test_agg_split",
        "test_get_named_qubits_multidim",
        "test_duplicate_names",
        "test_dtypes_converter",
        "test_is_symbolic",
    ],
)
def test_original_register_scenario_uses_native_dependencies(
    original: ModuleType, name: str
) -> None:
    namespace = dict(vars(original))
    namespace.update(
        Register=Register,
        Side=RegisterSide,
        Signature=Signature,
        QAny=opaque_register_type,
        QBit=quantum_bit_type,
        QInt=signed_register_type,
        QUInt=unsigned_register_type,
        CBit=classical_bit_type,
        BQUInt=native_bounded,
        get_named_qubits=native_named_qubits,
        is_symbolic=is_symbolic_value,
    )
    source_function = getattr(original, name)
    assert isinstance(source_function, FunctionType)
    # Preserve every assertion while isolating the test's imported owner names.
    native_function = FunctionType(source_function.__code__, namespace)
    native_function()


@pytest.mark.parametrize(
    "x_width,x_bound,y_width,y_bound", [(4, 10, 5, 19), (4, 16, 5, 32)]
)
def test_original_selection_indexing_uses_native_bound_parameters(
    original: ModuleType, x_width: int, x_bound: int, y_width: int, y_bound: int
) -> None:
    original.test_selection_registers_indexing(x_width, x_bound, y_width, y_bound)
    dtypes = (
        bounded_unsigned_register_type(x_width, x_bound),
        bounded_unsigned_register_type(y_width, y_bound),
    )
    registers = [
        Register(name, dtype) for name, dtype in zip(("x", "y"), dtypes, strict=True)
    ]
    bounds = tuple(int(dict(reg.dtype.parameters)["bound"]) for reg in registers)
    assert bounds == (x_bound, y_bound)
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            assert np.ravel_multi_index((x, y), bounds) == x * y_bound + y
            assert np.array_equal(np.unravel_index(x * y_bound + y, bounds), (x, y))
    assert np.prod(bounds) == x_bound * y_bound


def test_complete_register_source_and_tests_keep_frozen_identity() -> None:
    root = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/_infra"
    for name, blob in (
        ("registers.py", "cfc9191ff398687cd0f86ea3a38ee33a58c25cee"),
        ("registers_test.py", "7da20da8624fee7203e9d4a7354091d47f0a3e1c"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
