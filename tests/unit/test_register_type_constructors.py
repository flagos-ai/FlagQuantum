"""Replace source-backed metadata constructors without changing consumers."""

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OMITTED = object()
N = sympy.Symbol("n", integer=True, positive=True)


def source_factory(kind, width=OMITTED, bound=OMITTED):
    from qualtran import (
        BCUInt,
        BQUInt,
        CBit,
        CInt,
        CIntOnesComp,
        CUInt,
        QAny,
        QBit,
        QInt,
        QIntOnesComp,
        QUInt,
    )

    if kind in ("bit", "classical_bit"):
        return {"bit": QBit, "classical_bit": CBit}[kind]()
    if kind in ("bounded", "classical_bounded"):
        factory = BQUInt if kind == "bounded" else BCUInt
        return factory(width) if bound is OMITTED else factory(width, bound)
    return {
        "opaque": QAny,
        "uint": QUInt,
        "signed": QInt,
        "ones_complement": QIntOnesComp,
        "classical_uint": CUInt,
        "classical_signed": CInt,
        "classical_ones_complement": CIntOnesComp,
    }[kind](width)


def source_contract(kind, width=OMITTED, bound=OMITTED):
    dtype = source_factory(kind, width, bound)
    classical = kind.startswith("classical_")
    encoding = kind.removeprefix("classical_")
    encoding = "uint" if encoding == "bounded" else encoding
    declared_width = 1 if encoding == "bit" else dtype.bitsize
    parameters = (("storage", "classical"),) if classical and encoding != "bit" else ()
    if kind == "bounded":
        parameters += (("bound", dtype.iteration_length),)
    elif kind == "classical_bounded":
        parameters += (("bound", dtype.bound),)
    return RegisterType(
        encoding,
        0 if classical else declared_width,
        declared_width if classical else 0,
        parameters=parameters,
    )


def native_factory(kind, width=OMITTED, bound=OMITTED):
    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
        classical_bit_type,
        classical_bounded_unsigned_register_type,
        classical_ones_complement_register_type,
        classical_signed_register_type,
        classical_unsigned_register_type,
        ones_complement_register_type,
        opaque_register_type,
        quantum_bit_type,
        signed_register_type,
        unsigned_register_type,
    )

    if kind in ("bit", "classical_bit"):
        return {"bit": quantum_bit_type, "classical_bit": classical_bit_type}[kind]()
    if kind in ("bounded", "classical_bounded"):
        factory = (
            bounded_unsigned_register_type
            if kind == "bounded"
            else classical_bounded_unsigned_register_type
        )
        return factory(width) if bound is OMITTED else factory(width, bound)
    return {
        "opaque": opaque_register_type,
        "uint": unsigned_register_type,
        "signed": signed_register_type,
        "ones_complement": ones_complement_register_type,
        "classical_uint": classical_unsigned_register_type,
        "classical_signed": classical_signed_register_type,
        "classical_ones_complement": classical_ones_complement_register_type,
    }[kind](width)


@pytest.fixture(params=["source_contract", "native"])
def construct_type(request):
    return source_contract if request.param == "source_contract" else native_factory


def outcome(factory, kind, width, bound=OMITTED):
    try:
        value = factory(kind, width, bound)
    except (TypeError, ValueError) as error:
        message = (
            str(error)
            .replace("QAny", "opaque register")
            .replace("BQUInt", "Bounded unsigned register")
            .replace("BCUInt", "Bounded classical unsigned register")
        )
        return type(error), message
    return (
        value.encoding,
        value.quantum_bits,
        type(value.quantum_bits),
        value.classical_bits,
        type(value.classical_bits),
        tuple(
            (name, parameter, type(parameter)) for name, parameter in value.parameters
        ),
    )


@pytest.mark.parametrize("kind", ["opaque", "uint", "bounded"])
@pytest.mark.parametrize(
    "width",
    [-2, -1, 0, 1, 2, True, False, np.int64(2), sympy.Integer(2), N, 1.5, None, "3"],
)
def test_width_construction_matches_original_outcomes(construct_type, kind, width):
    assert outcome(construct_type, kind, width) == outcome(source_contract, kind, width)


@pytest.mark.parametrize("width", [-1, 0, 1, 3, np.int64(3), sympy.Integer(3), N])
@pytest.mark.parametrize(
    "bound", [-1, 0, 1, 3, 8, 9, True, None, 1.5, sympy.Integer(8), N]
)
def test_explicit_bounded_construction_matches_original(construct_type, width, bound):
    assert outcome(construct_type, "bounded", width, bound) == outcome(
        source_contract, "bounded", width, bound
    )


@pytest.mark.parametrize(
    "kind,width", [("bit", 1), ("opaque", 0), ("opaque", -1), ("uint", 0), ("uint", 3)]
)
def test_constructor_identity_and_signature_consumer(construct_type, kind, width):
    dtype = construct_type(kind, width)
    assert dtype == RegisterType(kind, width)
    assert type(dtype.quantum_bits) is int
    register = Register("word", dtype, shape=(2, 3))
    signature = Signature((register,))
    assert signature.get_left("word").dtype is dtype
    assert signature.get_right("word").dtype is dtype
    assert signature.n_qubits() == 6 * width
    assert list(register.all_idxs()) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_default_and_explicit_bound_are_distinct(construct_type):
    dtype = construct_type("bounded", 3)
    assert dtype == RegisterType("uint", 3, parameters=(("bound", 8),))
    with pytest.raises(TypeError):
        construct_type("bounded", 3, None)
    assert construct_type("bounded", N).parameters == (("bound", 2**N),)
    # The source accepts an explicit None bound when width is symbolic.
    assert construct_type("bounded", N, None).parameters == (("bound", None),)


def test_default_bound_evaluates_before_symbolic_guard(construct_type):
    events = []

    class Width:
        def __rpow__(self, base):
            events.append(("power", base))
            return 7

        def is_symbolic(self):
            events.append("symbolic")
            return True

    width = Width()
    dtype = construct_type("bounded", width)
    assert dtype.quantum_bits is width
    assert dtype.parameters == (("bound", 7),)
    assert events == [("power", 2), "symbolic"]


def test_bounded_guard_short_circuits_later_checker(construct_type):
    class UntouchedBound:
        def is_symbolic(self):
            raise AssertionError(
                "Symbolic width should short-circuit the bound checker"
            )

    bound = UntouchedBound()
    dtype = construct_type("bounded", N, bound)
    assert dtype.parameters[0][1] is bound


@pytest.mark.parametrize("consumer", ["signature", "xor_qrom"])
def test_actual_consumers_accept_factory_replacement(
    construct_type, consumer, monkeypatch
):
    from flagquantum.compiler.registers.lookup import qrom_table
    from flagquantum.core.ir.program import signatures

    events = []

    def bit():
        events.append(("bit",))
        return construct_type("bit")

    def opaque(width):
        events.append(("opaque", width))
        return construct_type("opaque", width)

    def bounded(width, bound):
        events.append(("bounded", width, bound))
        return construct_type("bounded", width, bound)

    if consumer == "signature":
        monkeypatch.setattr(signatures, "quantum_bit_type", bit)
        monkeypatch.setattr(signatures, "opaque_register_type", opaque)
        result = Signature.build(bit=1, word=3, empty=0)
        assert [reg.dtype for reg in result] == [
            RegisterType("bit", 1),
            RegisterType("opaque", 3),
        ]
        assert events == [("bit",), ("opaque", 3)]
    else:
        baseline = qrom_table.QromTable.from_data([0, 1, 2, 3], num_controls=1)
        work = tuple(range(5, 5 + baseline.workspace_size))
        bits = tuple(range(baseline.workspace_size))
        expected = baseline.bind_packed(tuple(range(5)), work, bits)
        monkeypatch.setattr(qrom_table, "opaque_register_type", opaque)
        monkeypatch.setattr(qrom_table, "bounded_unsigned_register_type", bounded)
        table = qrom_table.QromTable.from_data([0, 1, 2, 3], num_controls=1)
        assert table.signature == baseline.signature
        assert table.bind_packed(tuple(range(5)), work, bits) == expected
        assert events == [("opaque", 1), ("bounded", 2, 4), ("opaque", 2)]


@pytest.mark.parametrize("kind", ["bit", "opaque", "uint", "bounded"])
def test_native_concrete_factories_work_without_optional_sympy(kind, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sympy":
            raise ModuleNotFoundError("No module named 'sympy'", name="sympy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    dtype = native_factory(kind, 3)
    assert dtype.quantum_bits == (1 if kind == "bit" else 3)


INTEGER_KINDS = (
    "signed",
    "ones_complement",
    "classical_bit",
    "classical_uint",
    "classical_signed",
    "classical_ones_complement",
    "classical_bounded",
)


@pytest.fixture(params=["source_contract", "native"])
def construct_integer_type(request):
    return source_contract if request.param == "source_contract" else native_factory


@pytest.mark.parametrize("kind", INTEGER_KINDS)
@pytest.mark.parametrize(
    "width",
    [-2, -1, 0, 1, 2, True, False, np.int64(1), sympy.Integer(1), N, 1.5, None, "3"],
)
def test_integer_constructor_outcomes(construct_integer_type, kind, width):
    assert outcome(construct_integer_type, kind, width) == outcome(
        source_contract, kind, width
    )


@pytest.mark.parametrize("width", [-1, 0, 1, 3, np.int64(3), sympy.Integer(3), N])
@pytest.mark.parametrize(
    "bound", [-1, 0, 1, 3, 8, 9, True, None, 1.5, sympy.Integer(8), N]
)
def test_classical_bounded_constructor_outcomes(construct_integer_type, width, bound):
    assert outcome(
        construct_integer_type, "classical_bounded", width, bound
    ) == outcome(source_contract, "classical_bounded", width, bound)


@pytest.mark.parametrize("kind", ["uint", "signed", "ones_complement", "bounded"])
def test_zero_width_preserves_quantum_classical_identity(construct_integer_type, kind):
    quantum = source_contract(kind, 0)
    classical = construct_integer_type("classical_" + kind, 0)
    assert quantum.num_bits == classical.num_bits == 0
    assert quantum != classical
    assert classical.parameters[0] == ("storage", "classical")
    assert classical == source_contract("classical_" + kind, 0)
    word = Register("empty", classical)
    assert list(word.all_idxs()) == [()]
    assert list(Signature.build_from_dtypes(empty=classical)) == []


@pytest.mark.parametrize("kind", INTEGER_KINDS)
def test_integer_factories_feed_signature(construct_integer_type, kind):
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    dtype = construct_integer_type(kind, 3)
    native = Signature([Register("word", dtype, shape=(2, 3))])
    source = SourceSignature(
        [SourceRegister("word", source_factory(kind, 3), shape=(2, 3))]
    )
    for method in ("n_bits", "n_qubits", "n_cbits"):
        assert getattr(native, method)() == getattr(source, method)()
    assert native.get_left("word").dtype is dtype
    assert native.get_right("word").dtype is dtype
    assert len(Signature.build_from_dtypes(word=dtype)) == len(
        SourceSignature.build_from_dtypes(word=source_factory(kind, 3))
    )


@pytest.mark.parametrize("kind", ["ones_complement", "classical_ones_complement"])
def test_ones_complement_rejects_only_python_int_one(construct_integer_type, kind):
    for width in (1, True):
        with pytest.raises(ValueError, match="bitsize must be > 1"):
            construct_integer_type(kind, width)
    for width in (np.int64(1), sympy.Integer(1), 1.0):
        dtype = construct_integer_type(kind, width)
        assert (
            dtype.classical_bits
            if kind.startswith("classical_")
            else dtype.quantum_bits
        ) is width


def test_classical_bound_default_precedes_symbolic_guard(construct_integer_type):
    events = []

    class Width:
        def __rpow__(self, base):
            events.append(("power", base))
            return 7

        def is_symbolic(self):
            events.append("symbolic")
            return True

    width = Width()
    dtype = construct_integer_type("classical_bounded", width)
    assert dtype.classical_bits is width
    assert dtype.parameters == (("storage", "classical"), ("bound", 7))
    assert events == [("power", 2), "symbolic"]


@pytest.mark.parametrize("kind", INTEGER_KINDS)
def test_constructed_integer_words_reuse_existing_codecs(construct_integer_type, kind):
    from flagquantum.simulation.bit_encoding import bit_from_bits, bit_to_bits
    from flagquantum.simulation.bounded_integer_encoding import (
        bounded_integer_from_bits,
        bounded_integer_to_bits,
    )
    from flagquantum.simulation.integer_encoding import (
        integer_from_bits,
        integer_to_bits,
    )
    from flagquantum.simulation.ones_complement_encoding import (
        ones_complement_from_bits,
        ones_complement_to_bits,
    )

    dtype = construct_integer_type(kind, 3)
    width = (
        dtype.classical_bits if kind.startswith("classical_") else dtype.quantum_bits
    )
    source = source_factory(kind, 3)
    for value in source.get_classical_domain():
        expected = source.to_bits(value)
        if dtype.encoding == "bit":
            bits = bit_to_bits(value)
            decoded = bit_from_bits(bits)
        elif dtype.encoding == "ones_complement":
            bits = ones_complement_to_bits(value, width)
            decoded = ones_complement_from_bits(bits, width)
        elif "bound" in dict(dtype.parameters):
            bound = dict(dtype.parameters)["bound"]
            bits = bounded_integer_to_bits(value, width, bound)
            decoded = bounded_integer_from_bits(bits, width, bound)
        else:
            signed = dtype.encoding == "signed"
            bits = integer_to_bits(value, width, signed=signed)
            decoded = integer_from_bits(bits, width, signed=signed)
        assert bits == expected
        assert decoded == source.from_bits(expected) == value
        assert type(decoded) is type(source.from_bits(expected))


def test_ones_complement_consumers_share_core_guard_identity():
    from flagquantum.core.ir.program import register_types
    from flagquantum.simulation import (
        classical_signed_truncation,
        ones_complement_encoding,
    )

    assert (
        ones_complement_encoding.validate_ones_complement_width
        is register_types.validate_ones_complement_width
    )
    assert (
        classical_signed_truncation.validate_ones_complement_width
        is register_types.validate_ones_complement_width
    )


@pytest.mark.parametrize("kind", INTEGER_KINDS)
def test_integer_factories_without_optional_sympy(kind, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sympy":
            raise ModuleNotFoundError("No module named 'sympy'", name="sympy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    dtype = native_factory(kind, 3)
    assert dtype.num_bits == (1 if kind == "classical_bit" else 3)
