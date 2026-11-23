"""Frozen conjunction libraries retain exact calls, phases and erase branches."""

from collections import Counter
from dataclasses import replace
from importlib import import_module
from itertools import product
from math import sqrt
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentLibrary,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_conjunction_factories import (
    conjunction_factories,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
PATTERNS = tuple(values for n in (2, 3, 4) for values in product((0, 1), repeat=n))


def _library(source: Any) -> ComponentLibrary:
    codec = import_module("qualtran.serialization.bloq")
    return deserialize_component_library(
        codec.bloqs_to_proto(source, max_depth=1).SerializeToString()
    )


def _source(values: tuple[int, ...]) -> Any:
    source = import_module("qualtran.bloqs.mcmt.and_bloq")
    return source.And(*values) if len(values) == 2 else source.MultiAnd(values)


def _roundtrip(values: tuple[int, ...]) -> Any:
    source = import_module("qualtran")
    forward = _source(values)
    builder, inputs = source.BloqBuilder.from_signature(
        source.Signature(forward.signature.lefts())
    )
    return builder.finalize(
        **builder.add_d(forward.adjoint(), **builder.add_d(forward, **inputs))
    )


@pytest.mark.parametrize("values", PATTERNS)
def test_all_polarities_source_gates_complex_branches_clean_work_and_models(
    values: tuple[int, ...],
) -> None:
    resource = import_module("qualtran.resource_counting")
    original = _roundtrip(values)
    library = _library(original)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    program = resolve_component_program(library, 0, conjunction_factories())
    assert verify_program_record(program) == ()
    n = len(values)
    controls, work = tuple(range(n)), tuple(range(n, 2 * n - 1))
    bound = bind_register_program(program, tuple((q,) for q in controls), clean=work)
    assert bound.outputs == tuple((q,) for q in controls)
    gates = (
        Circuit(2 * n)
        .append(FieldOperation(2 * n, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    expected = reference_arithmetic_gates(
        original,
        {"ctrl": controls},
        work,
        dict(zip(reversed(work), range(n - 1), strict=True)),
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = resource.get_cost_value(original, resource.QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
    assert counts["measure"] == model.measurement == n - 1
    assert counts["reset"] == n - 1
    assert len(gates) == 17 * (n - 1) + 4 * values.count(0)
    for erase in (False, True):
        reference = _source(values)
        if erase:
            reference = reference.adjoint()
        assert (
            query_qec_gate_cost(ConjunctionCall(values, erase)).asdict()
            == resource.get_cost_value(reference, resource.QECGatesCost()).asdict()
        )
    # The last wire is an entangled spectator. All produced outputs begin clean.
    amplitudes = torch.arange(1, (1 << (n + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(amplitudes, amplitudes.flip(0))
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (2 * n), dtype=torch.complex128)
    indices = ((torch.arange(amplitudes.numel()) >> 1) << n) | (
        torch.arange(amplitudes.numel()) & 1
    )
    initial[indices] = amplitudes
    for history in product((0, 1), repeat=n - 1):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, initial / sqrt(1 << (n - 1)), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(1 << n, 1 << (n - 1), 2)[:, 1:]) == 0


@pytest.mark.parametrize("values", ((0, 1), (0, 1, 0), (1, 0, 1, 0, 1, 0)))
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_normal_append_forward_full_amplitudes_and_trajectory_roundtrip(
    values: tuple[int, ...], dtype: torch.dtype
) -> None:
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    n = len(values)
    factories = conjunction_factories()
    forward = resolve_component_program(_library(_source(values)), 0, factories)
    bound = bind_register_program(
        forward,
        tuple((q,) for q in range(n)),
        clean=tuple(range(n, 2 * n - 1)),
        allow_workspace_outputs=True,
    )
    assert len(bound.outputs) == 2 * n - 1
    coeff = torch.arange(1, (1 << n) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coeff, 3 - coeff).to(dtype)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (2 * n - 1), dtype=dtype)
    initial[:: 1 << (n - 1)] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(1 << n):
        prefix = 1
        output = 0
        for position, value in enumerate(values):
            prefix &= ((word >> (n - 1 - position)) & 1) == value
            if position:
                output = (output << 1) | prefix
        expected[(word << (n - 1)) | output] = amplitudes[word]
    circuit = Circuit(2 * n - 1, inputs=initial, dtype=dtype).append(
        FieldOperation(2 * n - 1, bound.circuit.instructions)
    )
    tolerance = 2e-6 if dtype == torch.complex64 else 4e-12
    torch.testing.assert_close(
        circuit.state().flatten(), expected, atol=tolerance, rtol=0
    )
    program = resolve_component_program(_library(_roundtrip(values)), 0, factories)
    both = bind_register_program(
        program, tuple((q,) for q in range(n)), clean=tuple(range(n, 2 * n - 1))
    )
    dynamic = DynamicCircuit(2 * n - 1, inputs=initial, dtype=dtype)
    dynamic.append(FieldOperation(2 * n - 1, both.circuit.instructions))
    result = run_dynamic(dynamic, shots=64, seed=13, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1),
        initial.expand(64, -1),
        atol=tolerance,
        rtol=0,
    )


def test_original_examples_depths_and_decomposed_rotation_consumer() -> None:
    source = import_module("qualtran.bloqs.mcmt.and_bloq")
    codec = import_module("qualtran.serialization.bloq")
    factories = {
        **fixed_gate_factories(),
        **rotation_factories(),
        **conjunction_factories(),
    }
    for original in (
        source._and_bloq.make(),
        source._multi_and.make(),
        source._multi_and.make().adjoint(),
    ):
        for depth in (0, 1, 2):
            library = deserialize_component_library(
                codec.bloqs_to_proto(original, max_depth=depth).SerializeToString()
            )
            assert (
                verify_program_record(resolve_component_program(library, 0, factories))
                == ()
            )
    rotation = import_module("qualtran.bloqs.basic_gates.rotation").CZPowGate(0.37)
    native = []
    for original in (rotation, rotation.decompose_bloq()):
        program = resolve_component_program(_library(original), 0, factories)
        native.append(
            bind_register_program(
                program, ((1,), (0,)), clean=(2,)
            ).circuit.instructions
        )
    assert_source(native[0], native[1])


def test_upstream_symbolic_serialization_failures_and_native_declaration() -> None:
    source = import_module("qualtran.bloqs.mcmt.and_bloq")
    symbolics = import_module("qualtran.symbolics")
    symbol = sympy.Symbol("cv")
    for original, message in (
        (source.MultiAnd((symbol, 1, 0)), "Object arrays"),
        (
            source.MultiAnd(symbolics.HasLength(sympy.Symbol("n"))),
            "Cannot serialize HasLength",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _library(original)
    for erase in (False, True):
        program = resolve_component_program(
            _library(source.And(symbol, 1, erase)), 0, conjunction_factories()
        )
        assert not program.functions[0].blocks
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(program, ((0,), (1,)))


def test_constructor_validation_exact_context_and_double_adjoint() -> None:
    factories = conjunction_factories()
    and_factory = factories["qualtran.bloqs.mcmt.and_bloq.And"]
    multi = factories["qualtran.bloqs.mcmt.and_bloq.MultiAnd"]
    adjoint = factories["qualtran._infra.adjoint.Adjoint"]
    for args in (
        {"extra": 1},
        {"cv1": 2},
        {"cv2": 0.5},
        {"uncompute": 2},
        {"uncompute": 0.5},
    ):
        with pytest.raises(SerializationError):
            and_factory("a", None, args)
    assert isinstance(and_factory("a", None, {"cv1": sympy.Integer(0)}), ProgramRecord)
    for controls in ((), (0, 1), 3, np.zeros((2, 2)), (0, 2, 1)):
        with pytest.raises(SerializationError):
            multi("a", None, {"cvs": controls})
    with pytest.raises(SerializationError, match="only cvs"):
        multi("a", None, {})
    with pytest.raises(SerializationError, match="signature"):
        and_factory("a", (), {})
    for arguments in ({}, {"subbloq": FunctionRecord("unknown", ())}, {"subbloq": 1}):
        with pytest.raises(SerializationError):
            adjoint("a", None, arguments)
    program = multi("forward", None, {"cvs": (0, 1, 0)})
    assert isinstance(program, ProgramRecord)
    erased = adjoint("inverse", None, {"subbloq": program.functions[0]})
    assert isinstance(erased, ProgramRecord)
    restored = adjoint("forward", None, {"subbloq": erased.functions[0]})
    assert restored == program
    # A foreign function with the same name cannot borrow stored constructor context.
    with pytest.raises(SerializationError, match="context"):
        adjoint("a", None, {"subbloq": replace(program.functions[0], blocks=())})
    symbolic = multi("symbolic", None, {"cvs": (sympy.Symbol("x"), 1, 0)})
    assert isinstance(symbolic, ProgramRecord) and not symbolic.functions[0].blocks
    with pytest.raises(SerializationError, match="Ambiguous"):
        multi("symbolic", None, {"cvs": (sympy.Symbol("y"), 1, 0)})


def test_catalog_context_isolated_by_complete_function_and_reusable() -> None:
    factories = conjunction_factories()
    for values in ((0, 1, 0), (True, False, True), (0, 1, 0), (0, 0, 1, 1)):
        program = resolve_component_program(_library(_roundtrip(values)), 0, factories)
        n = len(values)
        bound = bind_register_program(
            program, tuple((q,) for q in range(n)), clean=tuple(range(n, 2 * n - 1))
        )
        assert (
            sum(gate.name == "measure" for gate in bound.circuit.instructions) == n - 1
        )
    # Source Adjoint(And()) explicitly requests a decomposition of an atomic
    # source, while And.adjoint() returns the supported specialized constructor.
    atomic_wrapper = import_module("qualtran._infra.adjoint").Adjoint(_source((1, 1)))
    with pytest.raises(SerializationError, match="context"):
        resolve_component_program(_library(atomic_wrapper), 0, factories)


@pytest.mark.parametrize("value", (0.0, 1.0, np.float64(0), np.float32(1)))
@pytest.mark.parametrize("uncompute", (False, True))
def test_numeric_source_controls_normalize_only_at_constructor_boundary(
    value: float, uncompute: bool
) -> None:
    factory = conjunction_factories()["qualtran.bloqs.mcmt.and_bloq.And"]
    assert factory(
        "and", None, {"cv1": value, "cv2": value, "uncompute": uncompute}
    ) == factory(
        "and", None, {"cv1": int(value), "cv2": int(value), "uncompute": uncompute}
    )


@pytest.mark.parametrize("value", (0.5, float("nan"), float("inf"), "1"))
def test_nonbinary_numeric_source_controls_refuse(value: float | str) -> None:
    factory = conjunction_factories()["qualtran.bloqs.mcmt.and_bloq.And"]
    with pytest.raises(SerializationError):
        factory("and", None, {"cv1": value})
