"""Literal SU2 controls retain source calls and the relative scalar phase."""

import sys
from collections import Counter
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from math import pi
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import cirq as cirq
    import numpy as np

    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_su2_factories import controlled_su2_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_su2_source() -> None:
    import kickmix
    from qualtran.bloqs.basic_gates import su2_rotation

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for filename, digest in (
        ("su2_rotation.py", "91283ba1822f1cc3c1b1cedc37a4dfa74ec21085"),
        ("su2_rotation_test.py", "61b4d09dbc41dac7590d4655a6b5ecb6c188228f"),
    ):
        raw = (Path(su2_rotation.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("polarity", (0, 1))
@pytest.mark.parametrize(
    "angles", ((0.3, 0.4, 0.2, 0.1), (0.0, 0.0, 0.0, 0.0), (pi / 2, -0.2, 0.8, 7.2))
)
@pytest.mark.parametrize("depth", (0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_controlled_su2_calls_complex_operator_and_sdk_free_append(
    polarity: int, angles: tuple[float, float, float, float], depth: int, inverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate

    original = SU2RotationGate(*angles)
    source = (original.adjoint() if inverse else original).controlled(
        ctrl_spec=CtrlSpec(cvs=polarity)
    )
    library = _library(source, depth)
    factories = controlled_su2_factories()
    record = resolve_component_program(library, 0, factories)
    assert verify_program_record(record) == ()
    pairs = compare_layers(record, source, atomic_leaves(record))
    assert any(type(child).__name__ == "CRz" for child in pairs.values())
    bound = bind_register_program(record, ((2,), (0,)), borrowed=(1,))
    assert bound.outputs == ((2,), (0,)) and bound.events == ()
    expected = cirq.Circuit(
        source.on(cirq.LineQubit(2), cirq.LineQubit(0)), cirq.I(cirq.LineQubit(1))
    ).unitary(qubit_order=cirq.LineQubit.range(3))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3),
        torch.as_tensor(expected, dtype=torch.complex128),
        atol=4e-14,
        rtol=0,
    )
    counts = Counter(instruction.name for instruction in bound.circuit.instructions)
    assert counts == {
        "cx": 4,
        "crx": 1,
        "phase": 5,
        **({"x": 2} if polarity == 0 else {}),
    }
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        restored = resolve_component_program(library, 0, controlled_su2_factories())
        assert restored == record
        initial = torch.tensor([1, 2j, 3, -1j, 2, 1j, -2, 3j], dtype=torch.complex128)
        initial = initial / torch.linalg.vector_norm(initial)
        circuit = Circuit(3, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(3, bound.circuit.instructions))
        assert circuit.to_ir().instructions == bound.circuit.instructions
        assert CircuitIR.from_json(circuit.to_ir().to_json()) == circuit.to_ir()
        torch.testing.assert_close(
            circuit.state().flatten(),
            torch.as_tensor(expected, dtype=torch.complex128) @ initial,
            atol=4e-14,
            rtol=0,
        )


@pytest.mark.parametrize("polarity", (0, 1))
def test_symbolic_controlled_su2_keeps_owned_parameters_until_binding(
    polarity: int,
) -> None:
    import sympy
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate

    theta, phi, shift = sympy.symbols("theta phi shift", real=True)
    source = SU2RotationGate(sympy.sin(theta), phi, 0.2, shift).controlled(
        ctrl_spec=CtrlSpec(cvs=polarity)
    )
    record = resolve_component_program(
        _library(source, 0), 0, controlled_su2_factories()
    )
    bound = bind_register_program(record, ((0,), (1,)))
    initial = torch.tensor([1, 2j, 3, -1j], dtype=torch.complex128) / 15**0.5
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    values = {"theta": 0.3, "phi": 0.4, "shift": -0.2}
    numeric = SU2RotationGate(float(sympy.sin(0.3)), 0.4, 0.2, -0.2).controlled(
        ctrl_spec=CtrlSpec(cvs=polarity)
    )
    torch.testing.assert_close(
        circuit.bind_parameters(values).state().flatten(),
        torch.as_tensor(cirq.unitary(numeric), dtype=torch.complex128) @ initial,
        atol=4e-14,
        rtol=0,
    )


def test_unknown_child_and_invalid_control_are_not_inferred_from_ports() -> None:
    from flagquantum.compiler.formats.legacy_component_arguments import (
        ControlSpecification,
    )
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.errors import SerializationError

    factory = controlled_su2_factories()["qualtran._infra.controlled.Controlled"]
    child = FunctionRecord("unknown", (), registers=tuple(Signature.build(q=1)))
    spec = ControlSpecification(((quantum_bit_type(), np.array(1)),))
    with pytest.raises(SerializationError, match="explicit SU2 control recipe"):
        factory("main", None, {"subbloq": child, "ctrl_spec": spec})
    with pytest.raises(SerializationError, match="one positive scalar bit"):
        factory("main", None, {"subbloq": child, "ctrl_spec": ControlSpecification(())})


@pytest.mark.parametrize("polarity", (0, 1))
def test_generic_nonclifford_inventory_and_independent_source_model(
    polarity: int,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler._component_trace.program_logical_width import (
        query_program_logical_width,
    )

    source = SU2RotationGate(0.3, 0.4, 0.2, 0.1).controlled(
        ctrl_spec=CtrlSpec(cvs=polarity)
    )
    record = resolve_component_program(
        _library(source, 0), 0, controlled_su2_factories()
    )
    # The model is an independent source category account. No work is allocated
    # and the controlled-Rx atom has no deeper literal source primitive stream.
    assert get_cost_value(source, QECGatesCost()).asdict() == {
        "clifford": 4 + 2 * (polarity == 0),
        "rotation": 6,
    }
    assert (
        query_program_logical_width(record) == get_cost_value(source, QubitCount()) == 2
    )
