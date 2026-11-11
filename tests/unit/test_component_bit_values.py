"""Bit graph ports retain source equality validation instead of integer ranges."""

import warnings
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import sympy
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_measurement import measure_x_basis_phase
from flagquantum.simulation.component_classical_state import ComponentClassicalState

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def identity_record(
    shape: tuple[int, ...], classical: bool = False, *, integer: bool = False
) -> ProgramRecord:
    dtype = (
        unsigned_register_type(1)
        if integer
        else classical_bit_type() if classical else quantum_bit_type()
    )
    signature = Signature([Register("x", dtype, shape)])
    leaf_builder = ComponentProgramBuilder(signature, name="identity")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(signature, (leaf,))
    return builder.finalize(builder.call_dict("identity", builder.inputs))


def source_identity(
    shape: tuple[int, ...],
    classical: bool = False,
    *,
    output: object = NotImplemented,
    phase: complex | None = None,
) -> Any:
    from qualtran import Bloq, CBit, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    factory: Any = type(
        "BitIdentity",
        (Bloq,),
        {
            "signature": property(
                lambda self: SourceSignature(
                    [SourceRegister("x", CBit() if classical else QBit(), shape=shape)]
                )
            ),
            "on_classical_vals": lambda self, **values: (
                values if output is NotImplemented else {"x": output}
            ),
            "basis_state_phase": lambda self, **values: phase,
        },
    )
    return factory().as_composite_bloq()


SCALARS: list[object] = [
    0,
    1,
    False,
    True,
    np.int8(1),
    np.bool_(True),
    0.0,
    1.0,
    np.float32(1),
    0j,
    1 + 0j,
    Fraction(1),
    Decimal(0),
    np.array(1),
    np.array([1]),
    sympy.Integer(1),
    sympy.Symbol("v"),
    0.5,
    float("nan"),
    2,
    -1,
    "1",
    None,
    [1],
]


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("value", SCALARS)
def test_bit_scalar_admission_identity_and_failures_match_original(
    classical: bool, value: object
) -> None:
    from qualtran.simulation.classical_sim import ClassicalSimState

    source = source_identity((), classical)
    record = identity_record((), classical)
    try:
        expected = ClassicalSimState.from_cbloq(source, {"x": value}).simulate()["x"]
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            ComponentClassicalState(
                record, (0,), {"x": value}, lambda f, v: v
            ).simulate()
        return
    actual = ComponentClassicalState(
        record, (0,), {"x": value}, lambda f, v: v
    ).simulate()["x"]
    assert actual is expected is value
    assert type(actual) is type(expected)


ARRAYS: list[np.ndarray[Any, Any]] = [
    np.array([0, 1]),
    np.array([0.0, 1.0]),
    np.array([False, True]),
    np.array([0j, 1 + 0j]),
    np.array([Fraction(0), Fraction(1)], dtype=object),
    np.array([0.5]),
    np.array([1.5]),
    np.array([float("nan")]),
    np.array([-1]),
    np.array([2]),
    np.array(["0", "1"]),
    np.array([[0.0, 1.0], [1.0, 0.0]]).T,
    np.array([], dtype=float),
]


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("values", ARRAYS)
def test_bit_array_equality_precedes_uint8_collection(
    classical: bool, values: np.ndarray[Any, Any]
) -> None:
    from qualtran.simulation.classical_sim import ClassicalSimState

    shape = values.shape
    original = values.copy()
    record = identity_record(shape, classical)
    source = source_identity(shape, classical)
    with warnings.catch_warnings(record=True) as source_warnings:
        warnings.simplefilter("always")
        try:
            expected = ClassicalSimState.from_cbloq(source, {"x": values}).simulate()[
                "x"
            ]
        except (ValueError, TypeError, KeyError) as error:
            if isinstance(error, KeyError):
                assert values.size == 0
                result = ComponentClassicalState(
                    record, (0,), {"x": values}, lambda f, v: v
                ).simulate()["x"]
                assert (
                    isinstance(result, np.ndarray)
                    and result.dtype == np.uint8
                    and result.shape == shape
                )
            else:
                with warnings.catch_warnings(record=True) as invalid_warnings:
                    warnings.simplefilter("always")
                    with pytest.raises(type(error)):
                        ComponentClassicalState(
                            record, (0,), {"x": values}, lambda f, v: v
                        ).simulate()
                assert not invalid_warnings
            np.testing.assert_array_equal(values, original)
            return
    with warnings.catch_warnings(record=True) as native_warnings:
        warnings.simplefilter("always")
        actual = ComponentClassicalState(
            record, (0,), {"x": values}, lambda f, v: v
        ).simulate()["x"]
    assert isinstance(actual, np.ndarray) and actual.dtype == np.uint8
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)
    assert [warning.category for warning in native_warnings] == [
        warning.category for warning in source_warnings
    ]


@pytest.mark.parametrize("value", [0.0, 1.0, np.float32(1), 1 + 0j])
def test_uint_width_one_retains_integer_policy(value: object) -> None:
    from qualtran import QUInt

    with pytest.raises(ValueError):
        QUInt(1).assert_valid_classical_val(value)
    with pytest.raises(ValueError, match="integer"):
        ComponentClassicalState(
            identity_record((), integer=True), (0,), {"x": value}, lambda f, v: v
        )
    state = ComponentClassicalState(
        identity_record(()), (0,), {"x": value}, lambda f, v: v
    )
    assert state.simulate()["x"] is value


@pytest.mark.parametrize("output,contribution", [(1.0, 2), (np.float32(1), 1), (0j, 2)])
@pytest.mark.parametrize("phased", [False, True])
def test_bit_output_is_assigned_before_phase_rejection(
    output: object, contribution: complex, phased: bool
) -> None:
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        PhasedClassicalSimState,
    )

    record = identity_record(())
    state = ComponentClassicalState(
        record,
        (0,),
        {"x": 0},
        lambda f, v: {"x": output},
        lambda f, v: contribution,
        phased=phased,
    )
    source_type = PhasedClassicalSimState if phased else ClassicalSimState
    reference = source_type.from_cbloq(
        source_identity((), output=output, phase=contribution), {"x": 0}
    )
    if phased and contribution == 1:
        assert state.simulate()["x"] is reference.simulate()["x"] is output
        return
    with pytest.raises(ValueError):
        reference.simulate()
    with pytest.raises(ValueError):
        state.step()
    result = record.functions[0].blocks[0].operations[0].results[0]
    assert state.assignments[result] is output
    assert state.phase == 1
    assert state.last_position == 0
    assert any(value is output for value in reference.soq_assign.values())


@pytest.mark.parametrize("representation", ["float", "numpy_float", "complex"])
@pytest.mark.parametrize("input_bit,outcome", list(product((0, 1), repeat=2)))
def test_measurement_float_equivalent_fixed_values_keep_phase_and_complex_branches(
    representation: str, input_bit: int, outcome: int
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_classical_state import (
        measurement_record,
        measurement_values,
    )

    value: object = (
        float(outcome)
        if representation == "float"
        else (
            np.float32(outcome) if representation == "numpy_float" else complex(outcome)
        )
    )
    record = measurement_record()
    state = ComponentClassicalState(
        record,
        (0,),
        {"q": float(input_bit)},
        measurement_values,
        lambda f, v: measure_x_basis_phase(v["q"]),
        phased=True,
        instance_ids=(73,),
        fixed_values={73: value},
    )
    reference = PhasedClassicalSimState.from_cbloq(
        MeasureX().as_composite_bloq(),
        {"q": float(input_bit)},
        fixed_random_vals={0: value},
    )
    actual, expected = state.simulate(), reference.simulate()
    assert actual["c"] is expected["c"] is value
    assert state.phase == reference.phase == (-1 if input_bit == outcome == 1 else 1)
    bound = bind_register_program(record, ((0,),), clean=(1,))
    initial = torch.zeros(4, dtype=torch.complex128)
    initial[2 * input_bit] = 1
    branch = forced_branch(initial, bound.circuit.instructions, (outcome,))
    target = torch.zeros_like(initial)
    target[2 * outcome] = torch.tensor(state.phase / 2**0.5, dtype=target.dtype)
    torch.testing.assert_close(branch, target, atol=2e-14, rtol=0)
    assert torch.count_nonzero(branch[1::2]) == 0
    assert Counter(gate.name for gate in bound.circuit.instructions) == {
        "h": 1,
        "measure": 1,
    }
    model = get_cost_value(MeasureX(), QECGatesCost())
    assert model.measurement == 1 and model.clifford == 0


def test_bit_dispatch_reuses_shared_traversal_and_existing_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.component_classical_state as owner
    from flagquantum.simulation.bit_encoding import (
        validate_bit_array,
        validate_bit_value,
    )
    from flagquantum.simulation.classical_registers import (
        bind_classical_register_values,
    )

    bind = Mock(wraps=bind_classical_register_values)
    monkeypatch.setattr(owner, "bind_classical_register_values", bind)
    monkeypatch.setattr(
        owner,
        "bind_integer_register_values",
        lambda *args, **kwargs: pytest.fail("integer validator"),
    )
    value = np.array([0.0, 1.0])
    result = owner.ComponentClassicalState(
        identity_record((2,)), (0,), {"x": value}, lambda f, v: v
    ).simulate()
    np.testing.assert_array_equal(result["x"], [0, 1])
    assert bind.call_count == 2
    for invocation in bind.call_args_list:
        assert invocation.args[2] is validate_bit_value
        assert invocation.args[3] is validate_bit_array
