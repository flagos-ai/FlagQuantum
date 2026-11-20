"""Typed one-bit fixed-point controls reuse full measured-component witnesses."""

from itertools import product
from typing import Any

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip(
        "NumPy is needed for fixed-point control fixtures", allow_module_level=True
    )

from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from tests.unit import test_component_structured_control as structured
from tests.unit import test_field_control_component as field
from tests.unit.test_register_bindings import SourceContract
from tests.unit.test_register_control_predicate import ControlCase

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES: tuple[ControlCase, ...] = tuple(
    ((f"QFxp{fraction}", 1, shape, (bit,), "unsigned"),)
    for fraction, shape, bit in product((0, 1), ((), (1,)), (0, 1))
)


def controls(case: ControlCase, names: tuple[str, ...]) -> tuple[Register, ...]:
    return tuple(
        Register(name, fixed_point_register_type(1, int(kind[-1])), shape)
        for name, (kind, _, shape, _, _) in zip(names, case, strict=True)
    )


def source_dtype(register: Register) -> Any:
    from qualtran._infra.data_types import QFxp

    return QFxp(1, dict(register.dtype.parameters)["fractional_width"], False)


def fixture(
    case: ControlCase,
) -> tuple[Any, tuple[ControlRegister, ...], tuple[tuple[int, ...], ...]]:
    from qualtran._infra.controlled import CtrlSpec

    port = controls(case, ("ctrl",))[0]
    dtype = source_dtype(port)
    bit = case[0][3][0]
    value = dtype.from_bits([bit])
    from flagquantum.simulation.fixed_point_encoding import fixed_point_real_value

    fraction = int(case[0][0][-1])
    real_value = bit / (2**fraction)
    assert fixed_point_real_value(value, 1, fraction) == real_value
    assert value == bit
    assert dtype.to_fixed_width_int(real_value, require_exact=True) == bit
    spec = CtrlSpec(dtype, np.full(port.shape, value))
    assert spec.get_single_ctrl_val() == bit
    return spec, (ControlRegister(1, (bit,)),), ((0,),)


@pytest.fixture(autouse=True)
def fixed_point_reference_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapt test metadata while executing the unchanged source implementation."""
    from qualtran._infra.registers import Register as ForeignRegister
    from qualtran._infra.registers import Side

    original: Any = SourceContract.register

    def source_register(register: Register) -> Any:
        if register.dtype.encoding == "fixed_point":
            assert register.side.name is not None
            return ForeignRegister(
                register.name,
                source_dtype(register),
                register.shape,
                Side[register.side.name],
            )
        return original(register)

    monkeypatch.setattr(SourceContract, "register", staticmethod(source_register))
    for module in (field, structured):
        monkeypatch.setattr(
            module, "controls" if module is field else "native_controls", controls
        )
        monkeypatch.setattr(
            module, "fixture" if module is field else "control_fixture", fixture
        )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("expanded", [False, True])
def test_calls_branches_feedback_resources_and_zero_work(
    case: ControlCase, expanded: bool
) -> None:
    field.test_complete_source_wrapper_calls_branches_and_resources(case, expanded)


@pytest.mark.parametrize("case", CASES)
def test_stored_append_replaces_previous_specialization(case: ControlCase) -> None:
    structured.test_existing_append_protocol_and_specialization_replacement(case)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("expanded", [False, True])
def test_exact_adjoint_restores_all_history_pairs(
    case: ControlCase, expanded: bool
) -> None:
    structured.test_exact_adjoint_body_restores_all_forward_inverse_histories(
        case, expanded
    )


@pytest.mark.parametrize("signed", [False, True])
def test_original_wider_predicate_refusal(signed: bool) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QFxp
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    with pytest.raises(NotImplementedError, match="only supports integer types"):
        CtrlSpecAnd(CtrlSpec(QFxp(2, 1, signed), 0))
    with pytest.raises(ValueError):
        from examples.algorithms.controlled_programs import controlled_x_program

        controlled_x_program(
            (Register("ctrl", fixed_point_register_type(2, 1, signed)),),
            (ControlRegister(2, (0,)),),
        )


def test_original_signed_bit_constructor_refusal() -> None:
    from qualtran._infra.data_types import QFxp

    with pytest.raises(ValueError, match="num_qubits must be > 1"):
        QFxp(1, 1, True)
    with pytest.raises(ValueError, match="num_qubits must be > 1"):
        fixed_point_register_type(1, 1, True)


def test_original_symbolic_fraction_activation_refusal() -> None:
    import sympy
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QFxp
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    fraction = sympy.Symbol("f", integer=True)
    spec = CtrlSpec(QFxp(1, fraction), 1)
    with pytest.raises(ValueError, match="symbolic"):
        spec.get_single_ctrl_val()
    with pytest.raises(ValueError, match="symbolic"):
        ControlledViaAnd(XGate(), spec).decompose_bloq()
