"""Signature metadata and real consumers match unchanged pinned source calls."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_metadata import type_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def register_record(register):
    from qualtran import BQUInt, CBit, QAny, QBit, QInt, QUInt

    dtype = register.dtype
    if isinstance(dtype, RegisterType):
        encoding, parameters = dtype.encoding, dtype.parameters
    elif isinstance(dtype, BQUInt):
        encoding, parameters = "uint", (("bound", dtype.iteration_length),)
    else:
        encoding = {
            QAny: "opaque",
            QBit: "bit",
            CBit: "bit",
            QInt: "signed",
            QUInt: "uint",
        }[type(dtype)]
        parameters = ()
    return (
        register.name,
        encoding,
        dtype.num_qubits,
        dtype.num_cbits,
        parameters,
        register.shape_symbolic,
        register.side.name,
    )


def signature_pair(specs):
    from qualtran import Register as SourceRegister
    from qualtran import Side
    from qualtran import Signature as SourceSignature

    native, source = [], []
    for name, kind, width, shape, side in specs:
        native_type, source_type = type_pair(kind, width)
        native.append(Register(name, native_type, shape, RegisterSide[side]))
        source.append(SourceRegister(name, source_type, shape=shape, side=Side[side]))
    return Signature(native), SourceSignature(source)


def compare_signature(native, source):
    def records(registers):
        return [register_record(register) for register in registers]

    assert records(native) == records(source)
    assert records(native.lefts()) == records(source.lefts())
    assert records(native.rights()) == records(source.rights())
    assert [(name, records(group)) for name, group in native.groups()] == [
        (name, records(group)) for name, group in source.groups()
    ]
    assert native.thru_registers_only is source.thru_registers_only
    assert len(native) == len(source)
    assert (
        len(set((native, Signature(native))))
        == len(set((source, type(source)(source))))
        == 1
    )
    for method in ("n_bits", "n_qubits", "n_cbits"):
        actual, expected = getattr(native, method)(), getattr(source, method)()
        assert actual == expected
        assert type(actual) is type(expected)
    for register in native.lefts():
        assert native.get_left(register.name) is register
    for register in native.rights():
        assert native.get_right(register.name) is register


@pytest.mark.parametrize(
    "shape,side,kind,width",
    [
        (shape, side, kind, width)
        for shape, side, (kind, width) in product(
            ((), (0, 2), (2, 3), (sympy.Symbol("n", positive=True),)),
            ("LEFT", "RIGHT", "THRU"),
            (("opaque", 0), ("opaque", -1), ("uint", 3), ("classical", 1)),
        )
    ],
)
def test_direction_shape_zero_width_and_adjoint(shape, side, kind, width):
    native, source = signature_pair([("word", kind, width, shape, side)])
    compare_signature(native, source)
    compare_signature(native.adjoint(), source.adjoint())
    assert native.adjoint().adjoint() == native


def test_same_name_two_sides_grouping_and_independent_width_peaks():
    native, source = signature_pair(
        [
            ("word", "uint", 3, (), "LEFT"),
            ("word", "classical", 1, (7,), "RIGHT"),
        ]
    )
    compare_signature(native, source)
    compare_signature(native.adjoint(), source.adjoint())
    assert native.n_bits() == 7
    assert native.n_qubits() + native.n_cbits() == 10
    assert [name for name, _ in native.groups()] == ["word"]


@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_duplicate_namespace_error_and_generator_consumption(side):
    from qualtran import QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    for factory, register, dtype, direction in (
        (Signature, Register, RegisterType("bit", 1), RegisterSide[side]),
        (SourceSignature, SourceRegister, QBit(), Side[side]),
    ):
        seen = []

        def rows(seen, register, dtype, direction):
            for name in ("word", "word", "later"):
                seen.append(name)
                yield register(name, dtype, side=direction)

        with pytest.raises(
            ValueError, match="Register word is specified more than once per side"
        ) as error:
            factory(rows(seen, register, dtype, direction))
        assert error.value.__cause__ is None
        assert seen == ["word", "word", "later"]


@pytest.mark.parametrize(
    "width",
    [
        None,
        False,
        True,
        -1,
        0,
        1,
        2,
        0.0,
        1.0,
        2.0,
        np.int64(1),
        np.int64(2),
        sympy.Integer(0),
        sympy.Integer(2),
        sympy.Symbol("n"),
        [1],
        np.array([1, 2]),
    ],
)
def test_width_builder_filter_one_bit_and_opaque_validation_order(width):
    from qualtran import Signature as SourceSignature

    outcomes = []
    for factory in (SourceSignature, Signature):
        try:
            signature = factory.build(before=0, word=width, after=2)
            outcomes.append(("value", [register_record(reg) for reg in signature]))
        except Exception as error:
            outcomes.append(
                ("error", type(error), str(error).replace("QAny", "opaque register"))
            )
    assert outcomes[0] == outcomes[1]


def test_dtype_builder_filters_quantum_width_and_explicit_classical_ports_remain():
    from qualtran import CBit, QAny, QBit
    from qualtran import Signature as SourceSignature

    native = Signature.build_from_dtypes(
        classical=RegisterType("bit", 0, 1),
        empty=RegisterType("opaque", 0),
        bit=RegisterType("bit", 1),
    )
    source = SourceSignature.build_from_dtypes(
        classical=CBit(), empty=QAny(0), bit=QBit()
    )
    compare_signature(native, source)
    assert [reg.name for reg in native] == ["bit"]
    assert Signature([Register("c", RegisterType("bit", 0, 1))]).n_cbits() == 1


@pytest.mark.parametrize(
    "dtype", [object(), SimpleNamespace(num_qubits=0), SimpleNamespace(num_qubits=2)]
)
def test_dtype_filter_runs_before_nominal_register_validation(dtype):
    from qualtran import Signature as SourceSignature

    results = []
    for factory in (SourceSignature, Signature):
        try:
            results.append(("value", len(factory.build_from_dtypes(word=dtype))))
        except Exception as error:
            results.append(
                ("error", type(error), str(error).replace("QCDType", "RegisterType"))
            )
    assert results[0] == results[1]


def test_subclass_convenience_builder_retains_lazy_constructor_consumption():
    from qualtran import Signature as SourceSignature

    for base in (SourceSignature, Signature):

        class DeferredSignature(base):
            def __init__(self, registers):
                self.registers = registers

        signature = DeferredSignature.build(valid=1, invalid=2.5)
        assert next(signature.registers).name == "valid"
        with pytest.raises(ValueError, match="Bad bitsize"):
            next(signature.registers)


@pytest.mark.parametrize(
    "key", [0, -1, True, np.int64(1), slice(None, None, -1), 7, 1.5, "word"]
)
def test_index_slice_and_bad_lookup_match_source(key):
    native, source = signature_pair(
        [("a", "uint", 2, (), "THRU"), ("b", "bit", 1, (), "THRU")]
    )
    outcomes = []
    for signature in (source, native):
        try:
            value = signature[key]
            outcomes.append(
                (
                    "value",
                    (
                        [register_record(reg) for reg in value]
                        if isinstance(value, tuple)
                        else register_record(value)
                    ),
                )
            )
        except Exception as error:
            outcomes.append(("error", type(error), str(error)))
        assert signature.get_left("a") in signature
        with pytest.raises(KeyError) as error:
            signature.get_right("missing")
        assert error.value.args == ("missing",)
    assert outcomes[0] == outcomes[1]


def test_empty_and_foreign_equality_follow_source_behavior():
    from qualtran import Signature as SourceSignature

    compare_signature(Signature(()), SourceSignature(()))
    for factory in (SourceSignature, Signature):
        with pytest.raises(AttributeError, match="_registers"):
            factory(()).__eq__(None)


@pytest.mark.parametrize(
    "case,controls",
    [
        (case, control)
        for case, control in product(
            (
                "scalar",
                "array",
                "leading_singleton",
                "middle_singleton",
                "only_singletons",
                "rank_zero",
                "zero_width",
                "symbolic",
                "symbolic_one",
                "symbolic_zero_width",
            ),
            (0, 2),
        )
    ],
)
def test_qrom_uses_shared_typed_signature_before_existing_wire_partition(
    case, controls
):
    from qualtran.bloqs.data_loading.qrom import QROM

    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from tests.unit.test_qrom_table_layout import shape_case

    shape, widths, targets = shape_case(case)
    native = QromTable.from_shape(
        shape, widths, target_shapes=targets, num_controls=controls
    )
    source = QROM.build_from_bitsize(
        shape, widths, target_shapes=targets, num_controls=controls
    )
    compare_signature(native.signature, source.signature)
    assert native.signature is native.signature
    assert native.register_layout == tuple(
        (reg.name, reg.total_bits()) for reg in native.signature
    )


def test_add_symbolic_cost_consumer_reads_replaceable_signature(monkeypatch):
    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        source_add_resource_formula,
    )

    n = sympy.Symbol("n", integer=True, positive=True)
    before = source_add_resource_formula(n, n)
    monkeypatch.setattr(Signature, "n_qubits", lambda _: 17)
    after = source_add_resource_formula(n, n)
    assert after.logical_qubits == 17
    assert before.and_bloq == after.and_bloq
    assert before.clifford == after.clifford


def test_qrom_partition_consumer_accepts_replacement_signature(monkeypatch):
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable

    replacement = Signature([Register("bound", RegisterType("opaque", 3), (2,))])
    monkeypatch.setattr(QromTable, "signature", property(lambda _: replacement))
    assert QromTable.from_shape(4, 2).register_layout == (("bound", 6),)
