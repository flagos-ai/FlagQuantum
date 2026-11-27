"""Complete guiding-state child Programs retain helpers and measured ownership."""

import sys
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.kikuchi_guiding_state import (
    GuidingState,
    ProbabilisticUncompute,
    example_guiding_state,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.bit_view_call_checks import assert_source_bit_view_calls
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_mean_estimation_component import _registers
from tests.unit.test_unary_component_calls import source_slot_names

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def x_leaf() -> FunctionRecord:
    return bit_gate_function(
        "flag_x", tuple(Signature.build(q=1)), CircuitIR(1, (Instruction("x", (0,)),))
    )


def context(
    length: int,
) -> tuple[GuidingState, Any, tuple[FunctionRecord | ProgramRecord, ...]]:
    import attrs
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        _guiding_state,
    )
    from qualtran.resource_counting._call_graph import SympySymbolAllocator

    base = example_guiding_state()
    native = GuidingState(base.inst, length)
    source = attrs.evolve(_guiding_state.make(), ell=length)
    allocator: Any = SympySymbolAllocator
    roles = {
        type(child).__name__: child for child in source.build_call_graph(allocator())
    }
    children: tuple[FunctionRecord | ProgramRecord, ...] = (
        *(
            FunctionRecord(name, (), registers=_registers(roles[key].signature))
            for name, key in (
                ("scope", "SimpleGuidingState"),
                ("sort", "SortInPlace"),
                ("duplicates", "HasDuplicates"),
            )
        ),
        ProbabilisticUncompute(int(native.sort_ancilla_bitsize)).program(),
        multi_control_x_program((0, 0)),
        x_leaf(),
    )
    return native, source, children


@pytest.mark.parametrize("length", (2, 4))
def test_complete_program_context_preserves_source_root_bit_operands_and_missing_prefix(
    length: int,
) -> None:
    native, source, children = context(length)
    graph = native.program(*children)
    assert not verify_program_record(graph)
    from tests.unit.test_component_graph import _native_trace, _source_trace
    from tests.unit.test_projectile_nu_library import _named_inputs
    from tests.unit.test_register_component_calls import SourceCalls

    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    assert len(graph.functions) > 30
    missing = {f.registers for f in graph.functions if not f.blocks}
    expected = {
        child.registers for child in children[:3] if isinstance(child, FunctionRecord)
    }
    assert missing == expected
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(graph, ())


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_actual_probabilistic_program_primitive_stream_preserves_existing_source_recipe(
    width: int,
) -> None:
    from flagquantum import Circuit

    oracle = ProbabilisticUncompute(width)
    logical = width + 1
    clean = tuple(range(logical, logical + width - 1))
    bound = bind_register_program(
        oracle.program(), (tuple(range(width)), (width,)), clean=clean
    )
    baseline = (
        Circuit(logical + len(clean))
        .append(
            oracle.bind(tuple(range(width)), width, clean, tuple(range(len(clean))))
        )
        .to_ir()
        .instructions
    )
    assert_source_wire_streams(
        source_slot_names(bound.circuit.instructions, clean),
        source_slot_names(baseline, clean),
    )
    assert bound.outputs == (tuple(range(width)), (width,))


def suffix(width: int) -> tuple[ProgramRecord, Any]:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QAny, QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        ProbabilisticUncompute as SourceUncompute,
    )

    signature = Signature(
        (
            Register("q", opaque_register_type(width)),
            Register("flags", quantum_bit_type(), (3,)),
        )
    )
    builder = ComponentProgramBuilder(signature)
    flags = np.asarray(builder.inputs["flags"]).copy()
    result = builder.call_program(
        ProbabilisticUncompute(width).program(),
        {"q": builder.inputs["q"], "flag": flags[1]},
    )
    flags[1] = result["flag"]
    marked = builder.call_program(
        multi_control_x_program((0, 0)), {"controls": flags[:2], "target": flags[2]}
    )
    flags[:2] = marked["controls"]
    flags[2] = builder.call_program(
        ProgramRecord((x_leaf(),)), {"q": marked["target"]}, entry="flag_x"
    )["q"]
    native = builder.finalize({"q": result["q"], "flags": flags})
    original, inputs = BloqBuilder.from_signature(
        SourceSignature(
            (
                SourceRegister("q", QAny(width)),
                SourceRegister("flags", QBit(), shape=(3,)),
            )
        )
    )
    soq_flags = np.asarray(inputs["flags"], dtype=object).copy()
    word, soq_flags[1] = original.add(
        SourceUncompute(width), q=inputs["q"], flag=soq_flags[1]
    )
    soq_flags[:2], soq_flags[2] = original.add(
        MultiControlX(cvs=(0, 0)), controls=soq_flags[:2], target=soq_flags[2]
    )
    soq_flags[2] = original.add(XGate(), q=soq_flags[2])
    return native, original.finalize(q=word, flags=soq_flags)


@pytest.mark.parametrize("width", (1, 2, 3))
def test_complete_known_suffix_complex_columns_all_branches_and_only_private_work_zero(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, source = suffix(width)
    assert_source_bit_view_calls(source, native)
    logical = width + 3
    clean = tuple(range(logical, logical + max(width - 1, 1)))
    bound = bind_register_program(
        native, (tuple(range(width)), (width,), (width + 1,), (width + 2,)), clean=clean
    )
    events = certify_erasure_histories(
        bound.circuit.instructions, n_wires=bound.circuit.n_wires, zero_wires=clean
    )
    reference = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    stride = 1 << len(clean)
    for column in range(1 << logical):
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        initial[column * stride] = 1
        expected = torch.zeros_like(initial)
        expected[::stride] = reference[:, column]
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(initial, bound.circuit.instructions, history)
                * 2 ** (events / 2),
                expected,
                atol=1e-11,
                rtol=0,
            )
    weights = torch.arange(1, (1 << logical) + 1).to(torch.complex128)
    weights[1::2] *= 1j
    weights /= weights.norm()
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[::stride] = weights
    expected = torch.zeros_like(initial)
    expected[::stride] = reference @ weights
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        executable = DynamicCircuit(
            bound.circuit.n_wires, inputs=initial, dtype=initial.dtype
        )
        executable.append(
            FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
        )
        result = run_dynamic(executable, shots=3, seed=23, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(3, -1),
            expected.expand(3, -1),
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("width", (1, 2, 4))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_original_uncompute_constructor_wire_roundtrip_and_sdk_free_resolution(
    width: int, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        ProbabilisticUncompute as SourceUncompute,
    )

    from flagquantum.compiler.formats.legacy_component_library import (
        deserialize_component_library,
        serialize_component_library,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._guiding_state_component_library import (
        guiding_state_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = SourceUncompute(width)
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        graph = resolve_component_program(library, 0, guiding_state_library_factories())
    assert not verify_program_record(graph)
    clean = tuple(range(width + 1, 2 * width))
    bound = bind_register_program(graph, (tuple(range(width)), (width,)), clean=clean)
    direct = bind_register_program(
        ProbabilisticUncompute(width).program(),
        (tuple(range(width)), (width,)),
        clean=clean,
    )
    assert bound == direct


@pytest.mark.parametrize("symbolic", (False, True))
def test_source_zero_and_symbolic_constructor_declarations_do_not_invent_execution(
    symbolic: bool,
) -> None:
    import sympy
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        ProbabilisticUncompute as SourceUncompute,
    )

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._guiding_state_component_library import (
        guiding_state_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    width = sympy.Symbol("n", positive=True, integer=True) if symbolic else 0
    source = SourceUncompute(width)
    graph = resolve_component_program(
        _library(source, 0), 0, guiding_state_library_factories()
    )
    assert graph.functions[0].registers == _registers(source.signature)
    assert not graph.functions[0].blocks
    assert not verify_program_record(graph)
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(graph, ())


def test_missing_child_helpers_refuse_without_mutating_supplied_modules() -> None:
    native, _, children = context(2)
    measured = children[3]
    assert isinstance(measured, ProgramRecord) and len(measured.functions) > 1
    original = repr(measured)
    incomplete = ProgramRecord((measured.functions[0],))
    with pytest.raises(ValueError, match="invalid|unknown|callee"):
        native.program(*(*children[:3], incomplete, *children[4:]))
    assert repr(measured) == original
    graph = native.program(*children)
    assert not verify_program_record(graph)
    assert repr(measured) == original
