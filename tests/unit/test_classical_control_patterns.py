"""Control grouping follows original conversion before numerical evaluation."""

from collections.abc import Sequence

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_control_activation import (
    classical_controls_active,
)
from flagquantum.simulation.classical_control_patterns import normalize_control_patterns
from flagquantum.simulation.classical_controlled_values import (
    controlled_classical_values,
)
from flagquantum.simulation.value_dimensions import (
    SymbolicArrayShape,
    SymbolicSequenceLength,
)
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -2,
        2**100,
        True,
        np.int8(-1),
        np.uint64(2**63),
        np.bool_(True),
        None,
        1.5,
        sympy.Integer(1),
        "10",
        [],
        (),
        [0, 1, 1, 0],
        (True, False),
        [np.bool_(True), np.bool_(False)],
        [[0, 1], [1, 0]],
        [[0], [1, 0]],
        [0, [1, 0]],
        [[[], []]],
        range(3),
        np.array(1),
        np.array([0, 1]),
        np.array([[0, 1], [1, 0]]),
        np.zeros((0, 2), dtype=np.uint8),
        np.arange(8).reshape(2, 4)[:, ::2],
        np.ma.array([0, 1], mask=[False, True]),
    ],
)
def test_conversion_grouping_dtype_identity_and_failures_match_original(value):
    from qualtran._infra.controlled import _cvs_convert

    try:
        expected = _cvs_convert(value)
    except Exception as error:
        with pytest.raises(type(error)):
            normalize_control_patterns(value)
        return
    actual = normalize_control_patterns(value)
    assert type(actual) is type(expected) is tuple
    assert len(actual) == len(expected)
    for native, source in zip(actual, expected, strict=True):
        same_value(native, source)
    if isinstance(value, np.ndarray):
        assert actual[0] is expected[0] is value
    elif isinstance(value, Sequence):
        for item, native, source in zip(value, actual, expected, strict=False):
            if isinstance(item, np.ndarray):
                assert item is native is source


@pytest.mark.parametrize("shape", [(), (2, 2), (sympy.Symbol("n"),)])
@pytest.mark.parametrize("nested", [False, True])
def test_unknown_array_identity_is_preserved_without_object_array_conversion(
    shape, nested
):
    from qualtran._infra.controlled import _cvs_convert
    from qualtran.symbolics import Shaped

    source, native = Shaped(shape), SymbolicArrayShape(shape)
    svalue = [source, np.array([1])] if nested else source
    nvalue = [native, np.array([1])] if nested else native
    expected, actual = _cvs_convert(svalue), normalize_control_patterns(nvalue)
    assert len(expected) == len(actual)
    assert expected[0] is source and actual[0] is native
    assert actual[0].shape is expected[0].shape is shape
    if nested:
        same_value(actual[1], expected[1])
    with pytest.raises(ValueError, match="symbolic"):
        classical_controls_active(tuple(0 for _ in actual), actual)


def test_other_symbolic_objects_are_not_mistaken_for_array_descriptors():
    from qualtran._infra.controlled import _cvs_convert
    from qualtran.symbolics import HasLength

    for source, native in [(HasLength(3), SymbolicSequenceLength(3))]:
        with pytest.raises(TypeError):
            _cvs_convert(source)
        with pytest.raises(TypeError):
            normalize_control_patterns(native)

    class OpaqueSymbolic:
        def is_symbolic(self):
            return True

    source, native = _cvs_convert([OpaqueSymbolic()]), normalize_control_patterns(
        [OpaqueSymbolic()]
    )
    assert source[0].shape == native[0].shape == ()
    assert type(source[0].item()) is type(native[0].item()) is OpaqueSymbolic


@pytest.mark.parametrize("items", [[], [0, 1], [[0, 1], 0, 1], [0, [1], [0]]])
def test_original_single_pass_iterable_consumption_is_retained(items):
    from qualtran._infra.controlled import _cvs_convert

    source_input, native_input = iter(items), iter(items)
    expected = _cvs_convert(source_input)
    actual = normalize_control_patterns(native_input)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        if e.shape == () and e.item() is source_input:
            assert a.shape == () and a.item() is native_input
        else:
            same_value(a, e)
    assert list(source_input) == list(native_input) == []


@pytest.mark.parametrize(
    "pattern,values",
    [
        (0, (0,)),
        ([0, 1], (np.array([0, 1]),)),
        ([[0, 1], [1, 0]], (np.array([0, 1]), np.array([1, 0]))),
        (np.array([[0, 1], [1, 0]]), (np.array([[0, 1], [1, 0]]),)),
    ],
)
@pytest.mark.parametrize("active", [False, True])
def test_normalized_groups_feed_existing_controlled_callback(pattern, values, active):
    from qualtran import CtrlSpec, QBit
    from qualtran._infra.controlled import Controlled, _cvs_convert
    from qualtran.bloqs.basic_gates import XGate

    patterns = normalize_control_patterns(pattern)
    source = Controlled(
        XGate(), CtrlSpec(qdtypes=(QBit(),) * len(_cvs_convert(pattern)), cvs=pattern)
    )
    actual_values = values if active else (np.asarray(values[0]) ^ 1,) + values[1:]
    named = dict(zip(source.ctrl_reg_names, actual_values, strict=True), q=0)
    expected = source.on_classical_vals(**named)
    actual = controlled_classical_values(
        named, source.ctrl_reg_names, ("q",), patterns, XGate().on_classical_vals
    )
    assert actual.keys() == expected.keys()
    for name in expected:
        same_value(actual[name], expected[name])


def test_append_consumer_uses_replaceable_converter_before_existing_emitter(
    monkeypatch,
):
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate

    import examples.algorithms.numerical_value_dimensions as example
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    seen = []

    def convert(value):
        seen.append(value)
        return (np.ones((2, 2), dtype=int),)

    monkeypatch.setattr(example, "normalize_control_patterns", convert)
    circuit = DynamicCircuit(9)
    circuit.append(example.ShapedControlledX())
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], [[0, 1], [1, 0]])
    source = XGate().controlled(CtrlSpec(cvs=np.ones((2, 2), dtype=int)))
    original = reference_arithmetic_gates(
        source,
        {source.ctrl_reg_names[0]: (0, 1, 2, 3), "q": (4,)},
        (5, 6, 7),
        {5: 0, 6: 1, 7: 2},
    )
    assert_source(tuple(circuit.to_ir().instructions), original)
