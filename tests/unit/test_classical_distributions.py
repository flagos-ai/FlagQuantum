"""Random and fixed numerical outcomes preserve the original handler rules."""

from types import SimpleNamespace

import numpy as np
import pytest

from flagquantum.simulation.classical_distributions import (
    ClassicalDistribution,
    draw_classical_value,
)
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "choices,probabilities",
    [
        (2, None),
        (5, [0, 0.1, 0.2, 0.3, 0.4]),
        ([3, 7, 9], None),
        (np.array([1.5, 4.5]), [0.2, 0.8]),
        (np.array([1j, -1j]), None),
        (["zero", "one"], None),
        (np.arange(6).reshape(3, 2), [0, 0, 1]),
        (0, None),
        (-1, None),
        ([], None),
        (2, [1]),
        (2, [0.3, 0.3]),
        (2, [-1, 2]),
        (2, [float("nan"), 1]),
        (np.array(1), None),
    ],
)
@pytest.mark.parametrize("seed", [0, 197])
def test_rng_values_types_sequence_errors_and_state_match_original(
    choices, probabilities, seed
):
    from qualtran.simulation.classical_sim import (
        ClassicalValDistribution,
        _RandomClassicalValHandler,
    )

    original_rng = np.random.default_rng(seed)
    native_rng = np.random.default_rng(seed)
    source = ClassicalValDistribution(choices, probabilities)
    native = ClassicalDistribution(choices, probabilities)
    assert source.a is native.choices is choices
    assert source.p is native.probabilities is probabilities
    original = _RandomClassicalValHandler(original_rng)
    for _ in range(16):
        try:
            expected = original.get(SimpleNamespace(i=5), source)
        except Exception as error:
            with pytest.raises(type(error)):
                draw_classical_value(native, instance_id=5, rng=native_rng)
            break
        actual = draw_classical_value(native, instance_id=5, rng=native_rng)
        same_value(actual, expected)
    assert original_rng.bit_generator.state == native_rng.bit_generator.state


@pytest.mark.parametrize("fixed", [0, 1, -1, 1j, None, np.array([1, 0]), object()])
def test_fixed_value_identity_ignores_invalid_distribution_and_missing_id(fixed):
    from qualtran.simulation.classical_sim import (
        ClassicalValDistribution,
        _FixedClassicalValHandler,
    )

    mapping = {8: fixed}
    source = _FixedClassicalValHandler(mapping)
    invalid_source = ClassicalValDistribution(-1, [4, -3])
    invalid_native = ClassicalDistribution(-1, [4, -3])
    assert source.get(SimpleNamespace(i=8), invalid_source) is fixed
    assert (
        draw_classical_value(invalid_native, instance_id=8, fixed_values=mapping)
        is fixed
    )
    with pytest.raises(KeyError):
        source.get(SimpleNamespace(i=9), invalid_source)
    with pytest.raises(KeyError):
        draw_classical_value(invalid_native, instance_id=9, fixed_values=mapping)
    mapping[8] = object()
    assert source.get(SimpleNamespace(i=8), invalid_source) is mapping[8]
    assert (
        draw_classical_value(invalid_native, instance_id=8, fixed_values=mapping)
        is mapping[8]
    )


def test_rng_precedence_matches_original_phased_simulator_factory():
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    class ForbiddenFixed(dict):
        def __getitem__(self, key):
            pytest.fail("The RNG must be selected ahead of the fixed mapping")

    mapping = ForbiddenFixed()
    source_rng, native_rng = np.random.default_rng(19), np.random.default_rng(19)
    for _ in range(16):
        expected, phase = do_phased_classical_simulation(
            MeasureX(), {"q": 0}, rng=source_rng, fixed_random_vals=mapping
        )
        actual = draw_classical_value(
            ClassicalDistribution(2),
            instance_id=0,
            rng=native_rng,
            fixed_values=mapping,
        )
        same_value(actual, expected["c"])
        assert phase == 1


def test_no_handler_rejects_and_empty_fixed_mapping_does_not_fall_back():
    from qualtran.simulation.classical_sim import (
        ClassicalValDistribution,
        _BannedClassicalValHandler,
        _FixedClassicalValHandler,
    )

    with pytest.raises(ValueError, match="non-deterministic"):
        _BannedClassicalValHandler().get(
            SimpleNamespace(i=0), ClassicalValDistribution(2)
        )
    with pytest.raises(ValueError, match="non-deterministic"):
        draw_classical_value(ClassicalDistribution(2), instance_id=0)
    with pytest.raises(KeyError):
        _FixedClassicalValHandler({}).get(
            SimpleNamespace(i=0), ClassicalValDistribution(2)
        )
    with pytest.raises(KeyError):
        draw_classical_value(ClassicalDistribution(2), instance_id=0, fixed_values={})


def test_distribution_is_immutable_and_unhashable_like_original():
    from qualtran.simulation.classical_sim import ClassicalValDistribution

    for value in (ClassicalValDistribution(2), ClassicalDistribution(2)):
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises(AttributeError):
            value.p = None
