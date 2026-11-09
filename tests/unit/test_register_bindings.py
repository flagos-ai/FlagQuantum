"""Replace pinned binding construction and traversal with native responsibilities."""

import re
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.model import ValueRef
from flagquantum.core.ir.program.registers import Register, RegisterType

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


class SourceContract:
    """Test-only source-backed stand-in for the internal binding contract."""

    @staticmethod
    def register(register):
        from qualtran import CBit, QAny, QBit, QInt, QUInt, Side
        from qualtran import Register as SourceRegister

        dtype = register.dtype
        if dtype.num_cbits:
            source_dtype = CBit()
        elif dtype.encoding == "bit":
            source_dtype = QBit()
        else:
            source_dtype = {"opaque": QAny, "uint": QUInt, "signed": QInt}[
                dtype.encoding
            ](dtype.num_qubits)
        return SourceRegister(
            register.name,
            source_dtype,
            shape=register.shape_symbolic,
            side=Side[register.side.name],
        )

    @classmethod
    def binding(cls, register, index=()):
        from qualtran import LeftDangle, Soquet

        return Soquet(LeftDangle, cls.register(register), index)

    @staticmethod
    def index(binding):
        return binding.idx

    @staticmethod
    def name(binding):
        return binding.reg.name

    @classmethod
    def process(cls, registers, bindings, visit):
        from qualtran._infra.composite_bloq import _process_soquets

        return _process_soquets(
            map(cls.register, registers), bindings, "consumer", visit
        )


class NativeContract:
    @staticmethod
    def binding(register, index=()):
        from flagquantum.core.ir.program.register_bindings import RegisterBinding

        return RegisterBinding(ValueRef("entry", 0, "qubit"), register, index)

    @staticmethod
    def index(binding):
        return binding.index

    @staticmethod
    def name(binding):
        return binding.register.name

    @staticmethod
    def process(registers, bindings, visit):
        from flagquantum.compiler.registers.component_bindings import (
            process_register_bindings,
        )

        return process_register_bindings(registers, bindings, "consumer", visit)


@pytest.fixture(params=[SourceContract, NativeContract], ids=["source", "native"])
def backend(request):
    return request.param


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2), (N, 2)])
@pytest.mark.parametrize("index", [(), 0, 1, -1, (0, 0), (-1, 1), (2, 0)])
def test_index_conversion_and_validation(backend, shape, index):
    register = Register("word", RegisterType("uint", N), shape)
    try:
        expected = SourceContract.binding(register, index)
    except ValueError as error:
        with pytest.raises(ValueError, match=re.escape(str(error).split(" for ")[0])):
            backend.binding(register, index)
    else:
        actual = backend.binding(register, index)
        assert backend.index(actual) == expected.idx
        assert backend.name(actual) == "word"
        assert hash(actual) == hash(backend.binding(register, index))


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
@pytest.mark.parametrize("width", [0, 1, 3, N])
def test_ordered_scalar_array_and_zero_width_binding(backend, shape, width):
    register = Register("words", RegisterType("uint", width), shape)
    values = np.empty(shape, dtype=object)
    for index in register.all_idxs():
        values[index] = backend.binding(register, index)
    seen = []
    result = backend.process(
        [register],
        {"words": values},
        lambda value, port, index: seen.append(
            (backend.index(value), port.name, index)
        ),
    )
    assert result is None
    assert seen == [(index, "words", index) for index in register.all_idxs()]


@pytest.mark.parametrize(
    "case", ["missing", "extra", "short", "long", "ragged", "wrong"]
)
def test_failures_preserve_partial_callback_order(backend, case):
    register = Register("word", RegisterType("uint", 3), (2,))
    a, b = (backend.binding(register, (i,)) for i in range(2))
    bindings = {"word": [a, b]}
    expected_seen, error = [0, 1], None
    if case == "missing":
        bindings = {}
        expected_seen, error = [], ValueError
    elif case == "extra":
        bindings["extra"] = a
        error = ValueError
    elif case == "short":
        bindings["word"] = [a]
        expected_seen, error = [0], IndexError
    elif case == "long":
        bindings["word"] = [a, b, a]
    elif case == "ragged":
        bindings["word"] = [[a], [a, b]]
        expected_seen, error = [], ValueError
    else:
        bindings["word"] = [a, object()]
        expected_seen, error = [0], AssertionError
    seen = []

    def visit(value, port, index):
        seen.append(index[0])

    if error is None:
        backend.process([register], bindings, visit)
    else:
        with pytest.raises(error):
            backend.process([register], bindings, visit)
    assert seen == expected_seen


@pytest.mark.parametrize(
    "source_kind,target_kind", tuple(product(["uint", "opaque", "signed"], repeat=2))
)
@pytest.mark.parametrize("widths", [(1, 1), (3, 3), (2, 3)])
def test_callback_precedes_loose_type_consistency(
    backend, source_kind, target_kind, widths
):
    from qualtran._infra.data_types import check_dtypes_consistent

    origin = Register("origin", RegisterType(source_kind, widths[0]))
    target = Register("target", RegisterType(target_kind, widths[1]))
    binding = backend.binding(origin)
    expected = check_dtypes_consistent(
        SourceContract.register(origin).dtype, SourceContract.register(target).dtype
    )
    seen = []

    def visit(value, port, index):
        seen.append((value, port.name, index))

    if expected:
        backend.process([target], {"target": binding}, visit)
    else:
        with pytest.raises(ValueError, match="register dtypes are not consistent"):
            backend.process([target], {"target": binding}, visit)
    assert seen == [(binding, "target", ())]


def test_callback_failure_precedes_dtype_and_surplus_names(backend):
    origin = Register("origin", RegisterType("uint", 2))
    target = Register("target", RegisterType("uint", 3))
    binding = backend.binding(origin)
    seen = []

    def visit(value, port, index):
        seen.append(value)
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        backend.process([target], {"target": binding, "extra": object()}, visit)
    assert seen == [binding]


@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
def test_real_builder_initial_bindings_match_native_port_trace(backend, consumer):
    from qualtran import BloqBuilder, QUInt
    from qualtran._infra.composite_bloq import _process_soquets
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.data_loading.qrom import QROM

    operation = (
        Add(QUInt(2), QUInt(3))
        if consumer == "add"
        else QROM.build_from_data(np.array([1, 2, 0, 3]), target_bitsizes=(2,))
    )
    _, initial = BloqBuilder.from_signature(operation.signature)
    expected = []
    _process_soquets(
        operation.signature.lefts(),
        initial,
        "consumer",
        lambda value, port, index: expected.append((value.reg.name, port.name, index)),
    )
    registers = [
        Register(port.name, RegisterType("opaque", port.bitsize), port.shape)
        for port in operation.signature.lefts()
    ]
    bindings = {port.name: backend.binding(port) for port in reversed(registers)}
    actual = []
    backend.process(
        registers,
        bindings,
        lambda value, port, index: actual.append(
            (backend.name(value), port.name, index)
        ),
    )
    assert actual == expected


def test_native_binding_preserves_ssa_identity_without_mutation():
    from flagquantum.core.ir.program.register_bindings import RegisterBinding

    register = Register("word", RegisterType("uint", 3), (2,))
    value = ValueRef("entry", 7, "qubit")
    binding = RegisterBinding(value, register, 0)
    assert binding.value is value
    assert binding.register is register
    assert binding.index == (0,)
    assert binding != RegisterBinding(ValueRef("entry", 8, "qubit"), register, 0)
    with pytest.raises(FrozenInstanceError):
        binding.index = (1,)
