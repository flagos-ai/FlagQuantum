"""Ad hoc construction and inferred outputs against the frozen source builder."""

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_register_bindings import SourceContract
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


def original_builder() -> Any:
    from qualtran import BloqBuilder

    return BloqBuilder()


def original_register(register: Register) -> Any:
    adapter: Any = SourceContract
    return adapter.register(register)


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_explicit_dynamic_ports_match_source_identity_and_shape(
    width: int, shape: tuple[int, ...]
) -> None:
    register = Register("input", RegisterType("uint", width), shape)
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    source = original_builder()
    value = native.add_register(register)
    expected_value = source.add_register(original_register(register))
    output = native.finalize({"input": value})
    expected = source.finalize(input=expected_value)
    assert _native_trace(output) == _source_trace(SourceCalls, expected) == []
    assert np.asarray(value).shape == np.asarray(expected_value).shape == shape
    assert output.functions[0].registers == (register,)
    assert verify_program_record(output) == ()


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_inferred_output_retains_dtype_shape_and_mapped_value(
    shape: tuple[int, ...],
) -> None:
    register = Register("input", RegisterType("opaque", 3), shape, RegisterSide.LEFT)
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    source = original_builder()
    value = native.add_register(register)
    expected_value = source.add_register(original_register(register))
    output = native.finalize({"renamed": value})
    expected = source.finalize(renamed=expected_value)
    ports = output.functions[0].registers or ()
    assert [(r.name, r.shape, r.side.name) for r in ports] == [
        (r.name, r.shape, r.side.name) for r in expected.signature
    ]
    assert ports[1] == Register("renamed", register.dtype, shape, RegisterSide.RIGHT)
    assert (
        output.functions[0].blocks[0].arguments
        == output.functions[0].blocks[0].return_values
    )


def test_late_input_and_inferred_output_execute_exact_crossed_cnot() -> None:
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from examples.algorithms.component_construction_tutorial import (
        BoundComponentExample,
    )

    gate = bit_gate_function(
        "CX",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    native = ComponentProgramBuilder(
        Signature([]), (gate,), allow_register_addition=True
    )
    source = original_builder()
    a = native.add_register("a", 1)
    sa = source.add_register("a", 1)
    b = native.add_register("b", RegisterType("bit", 1))
    sb = source.add_register("b", 1)
    a, b = native.call_tuple("CX", {"ctrl": a, "target": b})
    sa, sb = source.add_t(CNOT(), ctrl=sa, target=sb)
    # This argument receives an SSA identity after the first call's results.
    spectator = native.add_register(
        Register("spectator", RegisterType("bit", 1), side=RegisterSide.LEFT)
    )
    sspectator = source.add_register(
        original_register(
            Register("spectator", RegisterType("bit", 1), side=RegisterSide.LEFT)
        )
    )
    b, a = native.call_tuple("CX", {"ctrl": b, "target": a})
    sb, sa = source.add_t(CNOT(), ctrl=sb, target=sa)
    result = native.finalize({"b": b, "a": a, "other": spectator})
    expected = source.finalize(b=sb, a=sa, other=sspectator)
    assert _native_trace(result) == _source_trace(SourceCalls, expected)
    bound = bind_register_program(result, ((0,), (1,), (2,)))
    assert bound.events == ()
    assert bound.circuit.instructions == (
        Instruction("cx", (0, 1)),
        Instruction("cx", (1, 0)),
    )
    state = torch.complex(
        torch.arange(8, dtype=torch.float64) + 1,
        torch.arange(8, dtype=torch.float64).flip(0) + 3,
    )
    state /= state.norm()
    reference = torch.as_tensor(expected.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        get_unitary(bound.circuit, dtype=torch.complex128) @ state,
        reference @ state,
        rtol=0,
        atol=0,
    )
    circuit = fq.Circuit(3)
    circuit.append(BoundComponentExample(result, ((0,), (1,), (2,))))
    assert circuit.to_ir().instructions == bound.circuit.instructions
    assert get_cost_value(expected, QECGatesCost()).clifford == 2


@pytest.mark.parametrize("kind", ["width", "dtype", "explicit"])
def test_register_argument_forms(kind: str) -> None:
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    if kind == "width":
        value = native.add_register("q", 3)
    elif kind == "dtype":
        value = native.add_register("q", RegisterType("uint", 3))
    else:
        value = native.add_register(Register("q", RegisterType("signed", 3)))
    assert isinstance(value, RegisterBinding)
    assert value.register.bitsize == 3
    assert verify_program_record(native.finalize({"q": value})) == ()


def test_explicit_output_only_and_owned_input_release() -> None:
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    source = original_builder()
    register = Register("out", RegisterType("opaque", 2), side=RegisterSide.RIGHT)
    assert (
        native.add_register(register)
        is source.add_register(original_register(register))
        is None
    )
    output = native.allocate(2)
    sout = source.allocate(2)
    result = native.finalize({"out": output})
    expected = source.finalize(out=sout)
    assert _native_trace(result) == _source_trace(SourceCalls, expected)
    bound = bind_register_program(
        result, (), clean=(0, 1), allow_workspace_outputs=True
    )
    assert bound.outputs == ((0, 1),)
    assert bound.circuit.instructions == ()
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (0, 1))
    ]


def test_left_only_input_can_be_released_without_inferred_output() -> None:
    register = Register("input", RegisterType("opaque", 3), side=RegisterSide.LEFT)
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    source = original_builder()
    value = native.add_register(register)
    assert isinstance(value, RegisterBinding)
    native.free(value)
    source.free(source.add_register(original_register(register)))
    result = native.finalize({})
    expected = source.finalize()
    assert _native_trace(result) == _source_trace(SourceCalls, expected)
    bound = bind_register_program(
        result, ((0, 1, 2),), input_ownership=((0, False), (1, False), (2, False))
    )
    assert bound.outputs == () and bound.circuit.instructions == ()
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("free", (0, 1, 2))
    ]


def test_fixed_signature_refuses_dynamic_ports_and_preserves_state() -> None:
    native = ComponentProgramBuilder(Signature([]))
    with pytest.raises(ValueError, match="fixed"):
        native.add_register("q", 1)
    assert native._arguments == () and native._registers == ()
    invalid: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        ComponentProgramBuilder(Signature([]), allow_register_addition=invalid)


def test_inferred_output_rejects_nonbinding_array_before_consumption() -> None:
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    value = native.allocate(1)
    with pytest.raises(ValueError, match="non-binding"):
        native.finalize({"bad": np.array([3])})
    assert native._registers == ()
    assert native._available == {value}
    assert verify_program_record(native.finalize({"out": value})) == ()


@pytest.mark.parametrize(
    "failure", ["missing", "extra_dtype", "invalid", "duplicate", "empty_inferred"]
)
def test_failures_preserve_source_timing(failure: str) -> None:
    native = ComponentProgramBuilder(Signature([]), allow_register_addition=True)
    source = original_builder()
    register = Register("q", RegisterType("opaque", 2))
    if failure == "missing":
        with pytest.raises(ValueError):
            native.add_register("q")
        with pytest.raises(ValueError):
            source.add_register("q")
    elif failure == "extra_dtype":
        with pytest.raises(ValueError):
            native.add_register(register, 2)
        with pytest.raises(ValueError):
            source.add_register_from_dtype(original_register(register), object())
    elif failure == "invalid":
        with pytest.raises(ValueError):
            native.add_register("q", -2)
        # Source permits negative-width metadata. Native finite execution
        # rejects it before allocating logical handles.
        assert source.add_register("q", -2).reg.bitsize == -2
    elif failure == "duplicate":
        native.add_register("q", 1)
        source.add_register("q", 1)
        value = native.add_register("q", 2)
        svalue = source.add_register("q", 2)
        with pytest.raises(ValueError):
            native.finalize({"q": value})
        with pytest.raises(ValueError):
            source.finalize(q=svalue)
    else:
        value = native.allocate(2)
        svalue = source.allocate(2)
        with pytest.raises(IndexError):
            native.finalize({"created": value, "empty": np.empty((0,), dtype=object)})
        with pytest.raises(IndexError):
            source.finalize(created=svalue, empty=np.empty((0,), dtype=object))
        assert (
            [r.name for r in native._registers]
            == [r.name for r in source._regs]
            == ["created"]
        )
        assert len(native._available) == len(source._available) == 1
