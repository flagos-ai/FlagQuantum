"""Typed packed specialization preserves the original wrapped callback contract."""

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.component_control_selection import (
    select_component_control_system,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    montgomery_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase
from tests.unit.test_component_control_selection import (
    assert_gate_parameters,
    fixtures,
    native_fallback,
)
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_hierarchy_adjoint import source_adjoint
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_packed_specialized_control import packed_fixture, source_gates
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


# Data words are three-bit raw storage. Every control remains a separate QBit.
PACKED_PROFILES = (
    "opaque",
    "uint",
    "bounded",
    "montgomery",
    "fixed-2",
    "fixed0",
    "fixed3",
    "field",
    "polynomial",
    "signed-2",
    "signed1",
    "signed2",
    "field5",
    "polynomial5",
)


def packed_dtype(profile: str) -> tuple[RegisterType, Any]:
    """Pair independently constructed native and frozen-source descriptors."""
    from qualtran._infra.data_types import (
        QGF,
        BQUInt,
        QAny,
        QFxp,
        QGFPoly,
        QMontgomeryUInt,
        QUInt,
    )

    if profile in {"montgomery_symbolic", "bounded_symbolic", "fixed_symbolic"}:
        from sympy import Symbol

        parameter = Symbol("p", positive=True, integer=True)
        if profile == "montgomery_symbolic":
            return montgomery_register_type(3, parameter), QMontgomeryUInt(3, parameter)
        if profile == "bounded_symbolic":
            return bounded_unsigned_register_type(3, parameter), BQUInt(3, parameter)
        return fixed_point_register_type(3, parameter), QFxp(3, parameter)
    if profile.startswith("empty_signed"):
        fraction = int(profile.removeprefix("empty_signed"))
        return fixed_point_register_type(0, fraction, signed=True), QFxp(
            0, fraction, signed=True
        )

    if profile == "opaque":
        return RegisterType("opaque", 3), QAny(3)
    if profile == "uint":
        return RegisterType("uint", 3), QUInt(3)
    if profile == "bounded":
        return bounded_unsigned_register_type(3, 6), BQUInt(3, 6)
    if profile == "montgomery":
        return montgomery_register_type(3, 5), QMontgomeryUInt(3, 5)
    if profile == "field":
        return FiniteFieldEncoding(2, 3).register_type(), QGF(2, 3)
    if profile == "polynomial":
        field = FiniteFieldEncoding(2, 1)
        return FiniteFieldPolynomialEncoding(2, field).register_type(), QGFPoly(
            2, QGF(2, 1)
        )
    if profile in {"field3", "field9"}:
        characteristic, degree = (3, 1) if profile == "field3" else (3, 2)
        return FiniteFieldEncoding(characteristic, degree).register_type(), QGF(
            characteristic, degree
        )
    if profile.endswith("_linear"):
        characteristic, degree = {
            "polynomial3_linear": (3, 1),
            "polynomial5_linear": (5, 1),
            "polynomial9_linear": (3, 2),
        }[profile]
        field = FiniteFieldEncoding(characteristic, degree)
        return FiniteFieldPolynomialEncoding(1, field).register_type(), QGFPoly(
            1, QGF(characteristic, degree)
        )
    if profile == "field5":
        return FiniteFieldEncoding(5, 1).register_type(), QGF(5, 1)
    if profile == "polynomial5":
        field = FiniteFieldEncoding(5, 1)
        return FiniteFieldPolynomialEncoding(0, field).register_type(), QGFPoly(
            0, QGF(5, 1)
        )
    if profile.startswith("empty_fixed"):
        fraction = int(profile.removeprefix("empty_fixed"))
        return fixed_point_register_type(0, fraction), QFxp(0, fraction)
    if profile.startswith("signed"):
        fraction = int(profile.removeprefix("signed"))
        return fixed_point_register_type(3, fraction, signed=True), QFxp(
            3, fraction, signed=True
        )
    fraction = int(profile.removeprefix("fixed"))
    return fixed_point_register_type(3, fraction), QFxp(3, fraction)


@pytest.mark.parametrize(
    "profile",
    (*PACKED_PROFILES, "montgomery_symbolic", "bounded_symbolic", "fixed_symbolic"),
)
@pytest.mark.parametrize(
    "layers,value,inverse", tuple(product((1, 2), (0, 1), (False, True)))
)
def test_complete_typed_packed_return_protocol_and_exact_direction(
    profile: str, layers: int, value: int, inverse: bool
) -> None:
    from attrs import evolve
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as OriginalRegister
    from qualtran._infra.registers import Signature as OriginalSignature
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    dtype, original_dtype = packed_dtype(profile)
    module, wrapper, child = packed_fixture(1, True, False, activation=value)
    if inverse:
        module = adjoint_component_hierarchy(
            module,
            {
                "ch": "ch",
                "cp": "cp_inverse",
                "__fq_predicate_0_x": "__fq_predicate_0_x",
            },
        )
        child = source_adjoint(child.decompose_bloq())
        wrapper = evolve(wrapper, bloq=child)
    ports = (Register("control", quantum_bit_type()), Register("data", dtype))
    # Rebuild the existing opaque wrapper through the owned typed entry point.
    layout = AutoPartition(
        (("control", (("ctrl", 1),)), ("data", (("q", 1), (None, 1), ("spectator", 1))))
    )
    module = auto_partition_component_program(
        module, layout, name="typed0", packed_ports=ports
    )
    wrapper = evolve(
        wrapper,
        partitions=(
            (OriginalRegister("control", QBit()), wrapper.partitions[0][1]),
            (OriginalRegister("data", original_dtype), wrapper.partitions[1][1]),
        ),
    )
    if layers == 2:
        layout = AutoPartition(
            (("control", (("control", 1),)), ("data", (("data", 3),)))
        )
        module = auto_partition_component_program(
            module, layout, entry="typed0", name="typed1", packed_ports=ports
        )
        from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original

        wrapper = Original(
            wrapper,
            (
                (OriginalRegister("control", QBit()), ("control",)),
                (OriginalRegister("data", original_dtype), ("data",)),
            ),
        )
    entry = f"typed{layers-1}"
    forward, original_body, _, _ = fixtures("spectator", True)
    controls = (Register("added", quantum_bit_type()),)
    selected, adder = select_component_control_system(
        controls,
        (ControlRegister(1, (value,)),),
        lambda activation: (module, "control"),
        native_fallback(forward, {"h": add_h, "p": add_phase}),
        entry=entry,
    )
    returned, callback = get_ctrl_system_1bit_cv(
        original_body,
        CtrlSpec(cvs=value),
        current_ctrl_bit=None,
        get_ctrl_bloq_and_ctrl_reg_name=lambda activation: (wrapper, "control"),
    )
    assert returned == child
    assert selected.functions[0] is module.functions[0]
    assert next(f for f in module.functions if f.name == entry).registers == ports
    signature = Signature((controls[0], ports[1]))
    builder = ComponentProgramBuilder(signature, selected.functions, name="caller")
    original_builder, inputs = BloqBuilder.from_signature(
        OriginalSignature(
            (OriginalRegister("added", QBit()), wrapper.signature.get_left("data"))
        )
    )
    values = {"data": builder.inputs["data"]}
    source_values = {"data": inputs["data"]}
    added, data = adder(builder, (builder.inputs["added"],), values)
    original_added, original_data = callback(
        original_builder, (inputs["added"],), source_values
    )
    caller = builder.finalize({"added": added[0], "data": data[0]})
    original = original_builder.finalize(
        added=tuple(original_added)[0], data=tuple(original_data)[0]
    )
    assert isinstance(data[0], RegisterBinding)
    assert data[0].register.dtype == dtype
    assert original.signature.get_right("data").dtype == original_dtype
    assert tuple(values) == tuple(source_values) == ("data", "control")
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(SourceCalls, original)
    )
    bound = bind_register_program(caller, ((0,), (1, 2, 3)), clean=(4,), entry="caller")
    assert bound.outputs == ((0,), (1, 2, 3))
    gates = bound.circuit.instructions
    bits = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in gates
        if gate.name == "measure"
    }
    expected_gates = source_gates(
        wrapper, child, {"control": (0,), "data": (1, 2, 3)}, (4,), bits
    )
    assert_gate_parameters(gates, expected_gates)
    count = Counter(gate.name for gate in gates)
    model = get_cost_value(wrapper, QECGatesCost())
    assert count["measure"] == count["reset"] == model.measurement == 1
    assert get_cost_value(wrapper, QubitCount()) == bound.circuit.n_wires == 5
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    appended = DynamicCircuit(6).append(FieldOperation(6, gates)).to_ir().instructions
    assert appended == gates
    amplitudes = torch.tensor(
        [complex(i + 1, 13 - i) for i in range(32)], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    target = torch.zeros_like(initial)
    for control, q, unused, spare, spectator in product((0, 1), repeat=5):
        label = (control << 4) | (q << 3) | (unused << 2) | (spare << 1) | spectator
        address = (label >> 1 << 2) | spectator
        initial[address] = amplitudes[label]
        for out in (0, 1):
            factor = complex(out == q)
            if control == value:
                phase_bit = q if inverse else out
                factor = (
                    (-1 if q and out else 1)
                    * np.exp(((-1 if inverse else 1) * 1j * np.pi / 4) * phase_bit)
                    / 2**0.5
                )
            destination = (
                (control << 5) | (out << 4) | (unused << 3) | (spare << 2) | spectator
            )
            target[destination] += amplitudes[label] * factor
    for outcome in (0, 1):
        actual = forced_branch(initial, appended, (outcome,))
        torch.testing.assert_close(actual, target / 2**0.5, atol=2e-12, rtol=0)
        assert actual.reshape(-1, 4)[:, 2:].abs().max().item() < 2e-12
