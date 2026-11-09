"""The same register metadata scenarios consume source-backed and native records.

The source-backed backend is a test-only contract stand-in. It does not enter
Core and does not pretend that component calls or graph execution are ported.
"""

import builtins
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


def type_pair(kind, width):
    from qualtran import BQUInt, CBit, QAny, QBit, QInt, QUInt

    if kind == "bit":
        return RegisterType("bit", 1), QBit()
    if kind == "classical":
        return RegisterType("bit", 0, 1), CBit()
    if kind == "bounded":
        return RegisterType("uint", width, parameters=(("bound", 5),)), BQUInt(width, 5)
    factory = {"opaque": QAny, "uint": QUInt, "signed": QInt}[kind]
    return RegisterType(kind, width), factory(width)


@pytest.mark.parametrize(
    "kind,width,shape,side",
    tuple(
        (*dtype, shape, side)
        for dtype, shape, side in product(
            (
                ("opaque", -1),
                ("opaque", 0),
                ("opaque", 3),
                ("opaque", N),
                ("uint", 0),
                ("uint", 3),
                ("uint", N),
                ("bit", 1),
                ("classical", 1),
                ("signed", 3),
                ("bounded", 3),
            ),
            ((), 2, True, (2, 3), (0, 2), (-1, 2), (N, 3)),
            ("LEFT", "RIGHT", "THRU"),
        )
    ),
)
def test_register_methods_match_unchanged_source(kind, width, shape, side):
    from qualtran import Register as SourceRegister
    from qualtran import Side

    native_type, source_type = type_pair(kind, width)
    native = Register("word", native_type, shape, RegisterSide[side])
    source = SourceRegister("word", source_type, shape=shape, side=Side[side])
    assert native.dtype is native_type
    assert native.shape_symbolic == source.shape_symbolic
    assert native.is_symbolic() == source.is_symbolic()
    for name in ("bitsize", "total_bits", "total_qubits", "total_cbits"):
        actual, expected = getattr(native, name), getattr(source, name)
        if callable(actual):
            actual, expected = actual(), expected()
        assert actual == expected
        assert type(actual) is type(expected)
    if any(isinstance(size, sympy.Basic) for size in source.shape_symbolic):
        for register in (source, native):
            with pytest.raises(ValueError, match="Cannot get real-valued shape"):
                _ = register.shape
            with pytest.raises(ValueError, match="Cannot get real-valued shape"):
                list(register.all_idxs())
    else:
        assert native.shape == source.shape
        assert list(native.all_idxs()) == list(source.all_idxs())
    actual, expected = native.adjoint(), source.adjoint()
    assert actual.side.name == expected.side.name
    assert actual.dtype is native.dtype
    assert actual.shape_symbolic == expected.shape_symbolic
    assert actual.adjoint() == native
    assert (actual is native) == (expected is source) == (side == "THRU")


@pytest.mark.parametrize("shape", [np.int64(2), sympy.Integer(2), 1.5, None])
def test_shape_converter_failure_precedes_dtype_validation(shape):
    from qualtran import Register as SourceRegister

    errors = []
    for factory in (SourceRegister, Register):
        with pytest.raises(TypeError) as caught:
            factory("word", object(), shape=shape)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]


def test_shape_generator_is_consumed_before_nominal_type_rejection():
    from qualtran import Register as SourceRegister

    for factory, expected_type in (
        (SourceRegister, "QCDType"),
        (Register, "RegisterType"),
    ):
        seen = []

        def dimensions(seen):
            for dimension in (2, 3):
                seen.append(dimension)
                yield dimension

        with pytest.raises(ValueError, match=f"dtype must be a {expected_type}"):
            factory("word", object(), shape=dimensions(seen))
        assert seen == [2, 3]


@pytest.mark.parametrize("raw", [None, [], [1], 1, np.bool_(True)])
def test_scalar_dtype_checker_result_is_not_coerced(raw):
    from qualtran import QAny
    from qualtran import Register as SourceRegister

    class SourceType(QAny):
        def is_symbolic(self):
            return raw

    class NativeType(RegisterType):
        def is_symbolic(self):
            return raw

    source = SourceRegister("word", SourceType(3))
    native = Register("word", NativeType("opaque", 3))
    assert native.is_symbolic() is source.is_symbolic() is raw
    assert Register("word", native.dtype, (2,)).is_symbolic() == bool(raw)


def test_native_records_are_immutable_and_preserve_encoding_identity():
    quantum = RegisterType("uint", 3)
    assert quantum != RegisterType("opaque", 3)
    assert quantum != RegisterType("uint", 3, parameters=(("bound", 5),))
    assert quantum != RegisterType("uint", 0, 3)
    word = Register("word", quantum, (2, 3))
    assert word == Register("word", quantum, (2, 3))
    assert hash(word) == hash(Register("word", quantum, (2, 3)))
    with pytest.raises(FrozenInstanceError):
        word.name = "changed"
    with pytest.raises(FrozenInstanceError):
        quantum.quantum_bits = 4


def test_concrete_metadata_does_not_require_optional_sympy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sympy":
            raise ModuleNotFoundError("No module named 'sympy'", name="sympy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    register = Register("word", RegisterType("uint", 3), shape=(2, 3))
    assert register.is_symbolic() is False
    assert register.total_bits() == register.total_qubits() == 18
    assert register.total_cbits() == 0
    assert list(register.all_idxs()) == list(product(range(2), range(3)))


@pytest.mark.parametrize("backend", ["source_contract", "native"])
@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
def test_two_consumer_signatures_replace_source_contract_metadata(backend, consumer):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.data_loading.qrom import QROM

    if consumer == "add":
        source = Add(QUInt(2), QUInt(3))
        native = (
            Register("a", RegisterType("uint", 2)),
            Register("b", RegisterType("uint", 3)),
        )
        expected = [("a", 2, (), "THRU"), ("b", 3, (), "THRU")]
    else:
        source = QROM.build_from_data(np.array([1, 2, 0, 3]), target_bitsizes=(2,))
        native = (
            Register("selection", RegisterType("uint", 2, parameters=(("bound", 4),))),
            Register("target0_", RegisterType("opaque", 2)),
        )
        expected = [("selection", 2, (), "THRU"), ("target0_", 2, (), "THRU")]
    registers = tuple(source.signature) if backend == "source_contract" else native
    assert [
        (reg.name, reg.bitsize, reg.shape, reg.side.name) for reg in registers
    ] == expected
    assert sum(reg.total_qubits() for reg in registers) == (
        5 if consumer == "add" else 4
    )
    assert all(reg.total_cbits() == 0 for reg in registers)
    assert all(reg.adjoint() is reg for reg in registers)


@pytest.mark.parametrize("side", [RegisterSide(0), "THRU", None])
def test_unknown_side_adjoint_is_rejected_without_constructor_normalization(side):
    register = Register("word", RegisterType("opaque", 1), side=side)
    assert register.side is side
    with pytest.raises(ValueError, match="Unknown side"):
        register.adjoint()
