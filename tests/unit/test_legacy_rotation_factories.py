"""Execute every locked rotation constructor through native library resolution."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any, Literal

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledZPowerCall,
    ControlledZRotationCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
    controlled_z_power_program,
)
from flagquantum.compiler.registers.primitives.pauli_rotation import (
    PauliAxis,
    pauli_power_ir,
    pauli_rotation_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicReal
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_fixed_gate_factories import _library
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KINDS = ("Rx", "Ry", "Rz", "CRz", "XPowGate", "YPowGate", "ZPowGate", "CZPowGate")
AXES: dict[str, PauliAxis] = {
    "Rx": "x",
    "Ry": "y",
    "Rz": "z",
    "XPowGate": "x",
    "YPowGate": "y",
    "ZPowGate": "z",
}


@pytest.mark.parametrize(
    "example", ("_rx", "_ry", "_rz", "_crz", "_x_pow", "_y_pow", "_z_pow", "_cz_pow")
)
def test_all_original_factories_and_serialized_depths(example: str) -> None:
    rotation = import_module("qualtran.bloqs.basic_gates.rotation")
    codec = import_module("qualtran.serialization.bloq")
    original = getattr(rotation, example).make()
    previous = None
    for depth in (0, 1, 2):
        library = deserialize_component_library(
            codec.bloqs_to_proto(original, max_depth=depth).SerializeToString()
        )
        program = resolve_component_program(library, 0, rotation_factories())
        assert verify_program_record(program) == ()
        root_arguments = {
            argument.name: argument.value
            for argument in library.entries[0].declaration.arguments
        }
        assert root_arguments["eps"] == original.eps
        ports = tuple((wire,) for wire in range(original.signature.n_qubits()))
        bound = bind_register_program(
            program, ports, clean=(2,) if example == "_cz_pow" else ()
        )
        if previous is not None:
            assert previous == bound
        previous = bound


def _source(
    kind: str, value: SymbolicReal, shift: float = 0.0, eps: SymbolicReal = 1e-9
) -> Any:
    constructor = getattr(import_module("qualtran.bloqs.basic_gates.rotation"), kind)
    arguments = {"exponent" if kind.endswith("PowGate") else "angle": value, "eps": eps}
    if kind in ("XPowGate", "YPowGate"):
        arguments["global_shift"] = shift
    return constructor(**arguments)


def _action(
    matrix: torch.Tensor, state: torch.Tensor, wires: tuple[int, ...], width: int
) -> torch.Tensor:
    order = (*wires, *(wire for wire in range(width) if wire not in wires))
    reverse = tuple(order.index(wire) for wire in range(width))
    return (
        (
            matrix
            @ state.reshape((2,) * width).permute(order).reshape(matrix.shape[0], -1)
        )
        .reshape((2,) * width)
        .permute(reverse)
        .flatten()
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", (0.0, 0.123, -0.7, 0.5, 1.0))
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_complete_constructor_gate_stream_amplitudes_all_branches_and_clean_work(
    kind: str, value: float, dtype: torch.dtype
) -> None:
    shift = -0.31 if kind in ("XPowGate", "YPowGate") else 0.0
    source = _source(kind, value, shift)
    library = _library(source)
    data = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(data)) == data
    program = resolve_component_program(library, 0, rotation_factories())
    assert verify_program_record(program) == ()
    controlled = kind in ("CRz", "CZPowGate")
    measured = kind == "CZPowGate"
    width = 4 if controlled else 3
    wires = (2, 0) if controlled else (1,)
    bound = bind_register_program(
        program, tuple((wire,) for wire in wires), clean=(3,) if measured else ()
    )
    assert bound.outputs == tuple((wire,) for wire in wires)
    if measured:
        reference = bind_register_program(
            controlled_z_power_program(value), ((2,), (0,)), clean=(3,)
        )
        assert bound == reference
    elif kind == "CRz":
        assert (
            bound.circuit.instructions
            == bind_register_program(
                controlled_rz_program(value), ((2,), (0,))
            ).circuit.instructions
        )
    else:
        expected_gates = (
            pauli_power_ir(AXES[kind], 1, value, global_shift=shift)
            if kind.endswith("PowGate")
            else pauli_rotation_ir(AXES[kind], 1, value)
        )
        assert bound.circuit.instructions == expected_gates
    if controlled:
        ports: dict[str, tuple[int, ...]] = (
            {"q": wires} if measured else {"ctrl": (2,), "q": (0,)}
        )
        original_gates, phase_calls = source_gradient_gates(
            source, ports, ports, work=3 if measured else None
        )
        assert_source(bound.circuit.instructions, original_gates)
        assert [leaf.eps for leaf in phase_calls] == (
            [1e-9] if measured else [5e-10, 5e-10]
        )
    values = torch.arange(1 << (width - 1), dtype=torch.float64)
    amplitudes = torch.complex(values + 1, 2 - 3 * values).to(dtype)
    initial = torch.zeros(1 << width, dtype=dtype)
    initial[::2] = amplitudes / amplitudes.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=dtype)
    expected = _action(matrix, initial, wires, width)
    tolerance = 6e-7 if dtype == torch.complex64 else 3e-12
    histories = ((0,), (1,)) if measured else ((),)
    for history in histories:
        branch = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            branch,
            expected.to(branch.dtype) / (2**0.5 if measured else 1),
            rtol=0,
            atol=tolerance,
        )
        assert torch.count_nonzero(branch[1::2]) == 0
    if measured:
        from flagquantum.dynamic import DynamicCircuit
        from flagquantum.runtime.dynamic.execution import run_dynamic

        circuit = DynamicCircuit(width, inputs=initial, dtype=dtype)
        circuit.append(FieldOperation(width, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=8, seed=31, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(8, -1),
            expected.expand(8, -1),
            rtol=0,
            atol=tolerance,
        )
    else:
        circuit = Circuit(width, inputs=initial, dtype=dtype)
        circuit.append(FieldOperation(width, bound.circuit.instructions))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=tolerance
        )
        if not controlled:
            logical = bind_register_program(program, ((0,),))
            operator = Circuit(1, dtype=torch.complex128).append(
                FieldOperation(1, logical.circuit.instructions)
            )
            torch.testing.assert_close(
                get_unitary(operator),
                torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
                rtol=0,
                atol=3e-12,
            )
    inverse_source = _source(kind, -value, shift)
    inverse = bind_register_program(
        resolve_component_program(_library(inverse_source), 0, rotation_factories()),
        tuple((wire,) for wire in wires),
        clean=(3,) if measured else (),
    )
    for first in histories:
        branch = forced_branch(initial, bound.circuit.instructions, first)
        for second in histories:
            restored = forced_branch(branch, inverse.circuit.instructions, second)
            torch.testing.assert_close(
                restored,
                initial.to(restored.dtype) / (2 if measured else 1),
                rtol=0,
                atol=tolerance,
            )
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["measure"] == counts["reset"] == int(measured)
    if measured:
        assert counts == {
            "h": 3,
            "cx": 6,
            "t": 2,
            "tdg": 2,
            "s": 1,
            "phase": 1,
            "measure": 1,
            "cz": 1,
            "reset": 1,
        }
    elif controlled:
        assert counts == {"phase": 2, "cx": 2}
    else:
        assert not bound.events
    key: ResourceCall
    if kind == "CZPowGate":
        key = ControlledZPowerCall(value, 1e-9)
    elif kind == "CRz":
        key = ControlledZRotationCall(value, 1e-9)
    else:
        opcodes: dict[str, Literal["rx", "ry", "rz", "xpow", "ypow", "zpow"]] = {
            "Rx": "rx",
            "Ry": "ry",
            "Rz": "rz",
            "XPowGate": "xpow",
            "YPowGate": "ypow",
            "ZPowGate": "zpow",
        }
        key = RotationCall(opcodes[kind], value, 1e-9)
    resource = import_module("qualtran.resource_counting")
    assert (
        query_qec_gate_cost(key).asdict()
        == resource.get_cost_value(source, resource.QECGatesCost()).asdict()
    )
    assert resource.get_cost_value(source, resource.QubitCount()) == (
        3 if measured else 2 if controlled else 1
    )


@pytest.mark.parametrize("kind", KINDS)
def test_symbolic_source_factory_binding_and_nonlinear_angles(kind: str) -> None:
    theta = sympy.Symbol("theta")
    expression = sympy.sin(theta) + theta**2 / 3
    source = _source(kind, expression)
    program = resolve_component_program(_library(source), 0, rotation_factories())
    controlled = kind in ("CRz", "CZPowGate")
    measured = kind == "CZPowGate"
    wires = (1, 0) if controlled else (0,)
    bound = bind_register_program(
        program, tuple((wire,) for wire in wires), clean=(2,) if measured else ()
    )
    ir = CircuitIR.from_json(bound.circuit.to_json())
    numeric = 0.37
    native_angle = float(expression.subs(theta, numeric))
    original = _source(kind, native_angle)
    if measured:
        # Resolve owned instruction parameters before forcing both physical histories.
        gates = tuple(
            Instruction(
                g.name,
                g.wires,
                params={
                    key: (
                        item.bind({"theta": numeric}) if hasattr(item, "bind") else item
                    )
                    for key, item in g.params.items()
                },
                metadata=g.metadata,
            )
            for g in ir.instructions
        )
        state = torch.zeros(8, dtype=torch.complex128)
        state[::2] = torch.tensor([1, 2j, -3, 4j], dtype=state.dtype)
        state /= state.norm()
        expected = _action(
            torch.as_tensor(original.tensor_contract(), dtype=state.dtype),
            state,
            wires,
            3,
        )
        for history in ((0,), (1,)):
            torch.testing.assert_close(
                forced_branch(state, gates, history),
                expected / 2**0.5,
                rtol=0,
                atol=3e-12,
            )
    else:
        size = 2 if controlled else 1
        circuit = Circuit(size, dtype=torch.complex128).append(
            FieldOperation(size, ir.instructions)
        )
        assert circuit.parameter_names == ("theta",)
        actual = get_unitary(circuit.bind_parameters({"theta": numeric}))
        expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
        if controlled:
            expected = expected.reshape(2, 2, 2, 2).permute(1, 0, 3, 2).reshape(4, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_constructor_admission_defaults_and_symbolic_precision(kind: str) -> None:
    factory = rotation_factories()[f"qualtran.bloqs.basic_gates.rotation.{kind}"]
    arguments: dict[str, ResolvedArgument] = {
        "exponent" if kind.endswith("PowGate") else "angle": 0.123,
        "eps": sympy.Symbol("eps"),
    }
    assert isinstance(
        factory("native", None, arguments), (FunctionRecord, ProgramRecord)
    )
    with pytest.raises(SerializationError, match="Unexpected constructor"):
        factory("bad", None, {**arguments, "extra": 1})
    with pytest.raises(SerializationError, match="Invalid register signature"):
        factory("bad", tuple(Signature.build(other=1)), arguments)
    for invalid in (True, "0.3", complex(1, 1)):
        with pytest.raises(SerializationError, match="real scalar"):
            factory(
                "bad",
                None,
                {"exponent" if kind.endswith("PowGate") else "angle": invalid},
            )
    for invalid in (float("inf"), float("nan"), 10**400):
        with pytest.raises(ValueError, match="finite"):
            factory(
                "bad",
                None,
                {"exponent" if kind.endswith("PowGate") else "angle": invalid},
            )
    if kind.endswith("PowGate"):
        assert factory("default", None, {})
    else:
        with pytest.raises(SerializationError, match="angle"):
            factory("missing", None, {})
    if kind in ("XPowGate", "YPowGate"):
        assert factory("constant_shift", None, {"global_shift": sympy.Rational(-1, 2)})
        with pytest.raises(SerializationError, match="concrete real scalar"):
            factory("bad", None, {"global_shift": sympy.Symbol("s")})
    fresh = rotation_factories()
    fresh.clear()
    assert len(rotation_factories()) == 8


def test_multiple_controlled_libraries_keep_independent_helper_names_and_feedback() -> (
    None
):
    q = import_module("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    codec = import_module("qualtran.serialization.bloq")
    builder, inputs = q.BloqBuilder.from_signature(q.Signature.build(a=1, b=1))
    a, b = inputs["a"], inputs["b"]
    for angle in (0.123, -0.37):
        a, b = builder.add(gates.CRz(angle), ctrl=a, q=b)
        a, b = builder.add(gates.CZPowGate(angle), q=[a, b])
    composite = builder.finalize(a=a, b=b)
    previous = None
    for depth in (1, 2, 3):
        library = deserialize_component_library(
            codec.bloqs_to_proto(composite, max_depth=depth).SerializeToString()
        )
        program = resolve_component_program(
            library, 0, {**rotation_factories(), **fixed_gate_factories()}
        )
        assert verify_program_record(program) == ()
        bound = bind_register_program(program, ((1,), (0,)), clean=(2,))
        assert len([g for g in bound.circuit.instructions if g.name == "measure"]) == 2
        if previous is not None:
            assert bound == previous
        previous = bound
    assert previous is not None
    state = torch.zeros(8, dtype=torch.complex128)
    state[::2] = torch.tensor([1 + 2j, -3j, 4 + 1j, -2], dtype=state.dtype)
    state /= state.norm()
    expected = _action(
        torch.as_tensor(composite.tensor_contract(), dtype=state.dtype),
        state,
        (1, 0),
        3,
    )
    for history in product((0, 1), repeat=2):
        branch = forced_branch(state, previous.circuit.instructions, history)
        torch.testing.assert_close(branch, expected / 2, rtol=0, atol=3e-12)
        assert torch.count_nonzero(branch[1::2]) == 0
