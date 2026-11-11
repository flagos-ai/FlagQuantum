"""Opaque graph values follow QAny validation and NumPy collection semantics."""

import warnings
from typing import Any

import numpy as np
import pytest
import sympy
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_distributions import ClassicalDistribution
from flagquantum.simulation.classical_registers import bind_opaque_register_values
from flagquantum.simulation.component_classical_state import ComponentClassicalState

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def identity_record(width: int, shape: tuple[int, ...] = ()) -> ProgramRecord:
    signature = Signature([Register("x", opaque_register_type(width), shape)])
    leaf_builder = ComponentProgramBuilder(signature, name="identity")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(signature, (leaf,))
    return builder.finalize(builder.call_dict("identity", builder.inputs))


def source_identity(width: int, shape: tuple[int, ...] = ()) -> Any:
    from qualtran import Bloq, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    factory: Any = type(
        "OpaqueIdentity",
        (Bloq,),
        {
            "signature": SourceSignature([SourceRegister("x", QAny(width), shape)]),
            "on_classical_vals": lambda self, **values: values,
            "build_composite_bloq": lambda self, bb, **values: values,
        },
    )
    return factory().as_composite_bloq()


SCALARS: list[object] = [
    0,
    -1,
    257,
    0.5,
    float("nan"),
    1j,
    "opaque",
    b"bytes",
    None,
    object(),
    [1, 2],
    {"key": "value"},
    np.float32(0.5),
    sympy.Symbol("x"),
]


@pytest.mark.parametrize("width", [0, 1, 5, 64, 65])
@pytest.mark.parametrize("value", SCALARS)
def test_unrestricted_scalar_identity_matches_source(width: int, value: object) -> None:
    from qualtran.simulation.classical_sim import ClassicalSimState

    record = identity_record(width)
    inputs = {"x": value}
    native = ComponentClassicalState(record, (0,), inputs, lambda f, v: v)
    source = ClassicalSimState.from_cbloq(source_identity(width), inputs)
    assert native.simulate()["x"] is source.simulate()["x"] is value
    assert all(assigned is value for assigned in native.assignments.values())
    assert inputs["x"] is value


@pytest.mark.parametrize("width", [1, 9, 65])
@pytest.mark.parametrize("shape", [(2,), (1, 2)])
@pytest.mark.parametrize(
    "items",
    [[0.5, 2.5], [-1, 257], [1j, -1j], ["3", "7"], [None, object()], [float("nan"), 0]],
)
def test_array_binding_then_collection_matches_source_errors_and_warnings(
    width: int, shape: tuple[int, ...], items: list[object]
) -> None:
    from qualtran.simulation.classical_sim import ClassicalSimState

    array = np.array(items).reshape(shape)
    original = array.copy()
    inputs = {"x": array}
    native = ComponentClassicalState(
        identity_record(width, shape), (0,), inputs, lambda f, v: v
    )
    source = ClassicalSimState.from_cbloq(source_identity(width, shape), inputs)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        try:
            expected = source.simulate()["x"]
        except (ValueError, TypeError, OverflowError) as error:
            expected_error: type[Exception] | None = type(error)
        else:
            expected_error = None
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        if expected_error is not None:
            with pytest.raises(expected_error):
                native.simulate()
        else:
            actual = native.simulate()["x"]
            assert isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
            assert actual.dtype == expected.dtype
            for left, right in zip(actual.flat, expected.flat, strict=True):
                np.testing.assert_equal(left, right)
    assert [w.category for w in actual_warnings] == [
        w.category for w in expected_warnings
    ]
    np.testing.assert_equal(array, original)


@pytest.mark.parametrize("value", [0.5, -1, None, object()])
@pytest.mark.parametrize("at_input", [False, True])
def test_fixed_opaque_distribution_retains_resolved_object(
    value: object, at_input: bool
) -> None:
    from qualtran.simulation.classical_sim import (
        ClassicalValDistribution,
        PhasedClassicalSimState,
    )

    source = source_identity(3)
    source_state = None
    if at_input:
        # The source fixed handler needs binst.i, absent on LeftDangle.
        # Native input identity -1 is an explicit representation difference.
        with pytest.raises(AttributeError, match="has no attribute 'i'"):
            PhasedClassicalSimState.from_cbloq(
                source,
                {"x": ClassicalValDistribution(-1)},
                fixed_random_vals={-1: value},
            )
    else:
        # Override only the declared numerical action at the source Bloq boundary.
        # The source state, binding, distribution and collection remain real.
        leaf = next(i.bloq for i, _, _ in source.iter_bloqsoqs())
        cls = type(leaf)
        output_type: Any = type(
            "OpaqueOutput",
            (cls,),
            {
                "on_classical_vals": lambda self, **v: {
                    "x": ClassicalValDistribution(-1)
                }
            },
        )
        source_state = PhasedClassicalSimState.from_cbloq(
            output_type().as_composite_bloq(), {"x": 0}, fixed_random_vals={0: value}
        )
    native = ComponentClassicalState(
        identity_record(3),
        (0,),
        {"x": ClassicalDistribution(-1) if at_input else 0},
        lambda f, v: v if at_input else {"x": ClassicalDistribution(-1)},
        phased=True,
        fixed_values={-1 if at_input else 0: value},
    )
    assert native.simulate()["x"] is value
    assert native.phase == 1
    if source_state is not None:
        assert source_state.simulate()["x"] is value
        assert source_state.phase == native.phase


@pytest.mark.parametrize("seed", [0, 19, 201])
def test_rng_opaque_values_and_generator_state_match_source(seed: int) -> None:
    from qualtran.simulation.classical_sim import (
        ClassicalValDistribution,
        PhasedClassicalSimState,
    )

    rng, source_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    for _ in range(16):
        native = ComponentClassicalState(
            identity_record(2),
            (0,),
            {"x": ClassicalDistribution([-2.5, 0.5])},
            lambda f, v: v,
            phased=True,
            rng=rng,
            fixed_values={},
        )
        source = PhasedClassicalSimState.from_cbloq(
            source_identity(2),
            {"x": ClassicalValDistribution([-2.5, 0.5])},
            rng=source_rng,
            fixed_random_vals={},
        )
        assert native.simulate() == source.simulate()
        assert rng.bit_generator.state == source_rng.bit_generator.state


def test_shared_binding_guards_do_not_disappear_with_opaque_validation() -> None:
    with pytest.raises(ValueError, match="requires register"):
        bind_opaque_register_values("x", {})
    with pytest.raises(ValueError, match="shape differs"):
        bind_opaque_register_values("x", {"x": [1, 2]}, shape=(3,))
    with pytest.raises(ValueError, match="non-deterministic"):
        bind_opaque_register_values("x", {"x": ClassicalDistribution(2)})
    with pytest.raises(KeyError):
        ComponentClassicalState(
            identity_record(3),
            (0,),
            {"x": ClassicalDistribution(2)},
            lambda f, v: v,
            fixed_values={},
        )


@pytest.mark.parametrize("width", [1, 2, 3])
def test_physical_identity_full_matrix_append_and_clean_spectator(width: int) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program

    record = identity_record(width)
    matrix = np.zeros((2**width, 2**width), dtype=complex)
    for value in range(2**width):
        result = ComponentClassicalState(
            record, (0,), {"x": value}, lambda f, v: v
        ).simulate()
        assert isinstance(result["x"], int)
        matrix[result["x"], value] = 1
    np.testing.assert_equal(matrix, source_identity(width).tensor_contract())
    bound = bind_register_program(record, (tuple(range(width)),), clean=(width,))
    assert bound.circuit.instructions == ()
    active = torch.arange(1, 2**width + 1, dtype=torch.float64).to(torch.complex128)
    active += 1j * active.flip(0)
    active /= torch.linalg.vector_norm(active)
    initial = torch.zeros(2 ** (width + 1), dtype=torch.complex128)
    initial[::2] = active
    actual = (
        Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, initial, rtol=0, atol=1e-12)
    assert torch.count_nonzero(actual[1::2]) == 0
