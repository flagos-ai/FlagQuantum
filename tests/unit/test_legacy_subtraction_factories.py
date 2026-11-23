"""Complete subtraction and fanout constructor closures retain source execution."""

from collections import Counter
from importlib import import_module
from itertools import product

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fanout_factories import fanout_factories
from flagquantum.compiler.formats.legacy_subtraction_factories import (
    subtraction_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import (
    _factories as arithmetic_factories,
)
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _factories() -> dict[str, ComponentFactory]:
    return {**arithmetic_factories(), **subtraction_factories(), **fanout_factories()}


@pytest.mark.parametrize("kind", ("Subtract", "SubtractFrom"))
@pytest.mark.parametrize("dtype", ("QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 3), (4, 6)))
def test_complete_subtraction_source_calls_operands_feedback_and_resources(
    kind: str, dtype: str, left: int, right: int
) -> None:
    source = import_module("qualtran")
    module = import_module("qualtran.bloqs.arithmetic.subtraction")
    original = (
        getattr(module, kind)(getattr(source, dtype)(right))
        if kind == "SubtractFrom"
        else module.Subtract(
            getattr(source, dtype)(left), getattr(source, dtype)(right)
        )
    )
    a_width = right if kind == "SubtractFrom" else left
    a = tuple(reversed(range(a_width)))
    b = tuple(reversed(range(a_width, a_width + right)))
    logical = a_width + right
    count = right - 1 + (right - left if kind == "Subtract" else 0)
    work = tuple(range(logical, logical + count))
    native = []
    for op in (original, original.decompose_bloq()):
        program = resolve_component_program(_library(op), 0, _factories())
        assert verify_program_record(program) == ()
        bound = bind_register_program(program, (a, b), clean=work)
        assert bound.outputs == (a, b)
        native.append(bound.circuit.instructions)
    slots = {
        g.wires[0]: int(g.metadata["classical_bit"])
        for g in native[0]
        if g.name == "measure"
    }
    expected = reference_arithmetic_gates(original, {"a": a, "b": b}, work, slots)
    assert_source(native[0], expected)
    assert_source(native[1], expected)
    cost_owner = import_module("qualtran.resource_counting")
    model = cost_owner.get_cost_value(original, cost_owner.QECGatesCost())
    counts = Counter(g.name for g in native[0])
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (right - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == right - 1


@pytest.mark.parametrize("kind", ("Subtract", "SubtractFrom"))
@pytest.mark.parametrize("dtype", ("QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 3), (2, 3)))
def test_full_complex_source_all_histories_clean_prefix_and_inverse(
    kind: str, dtype: str, left: int, right: int
) -> None:
    source = import_module("qualtran")
    module = import_module("qualtran.bloqs.arithmetic.subtraction")
    original = (
        getattr(module, kind)(getattr(source, dtype)(right))
        if kind == "SubtractFrom"
        else module.Subtract(
            getattr(source, dtype)(left), getattr(source, dtype)(right)
        )
    )
    a_width = right if kind == "SubtractFrom" else left
    logical = a_width + right
    scratch = right - 1 + (right - left if kind == "Subtract" else 0)
    ports = (tuple(range(a_width)), tuple(range(a_width, logical)))
    clean = tuple(range(logical + 1, logical + 1 + scratch))
    program = resolve_component_program(_library(original), 0, _factories())
    forward = bind_register_program(program, ports, clean=clean)
    inverse = (
        original
        if kind == "Subtract"
        else import_module("qualtran.bloqs.arithmetic.addition").Add(
            getattr(source, dtype)(right)
        )
    )
    backward = bind_register_program(
        resolve_component_program(_library(inverse), 0, _factories()),
        ports,
        clean=clean,
    )
    coeff = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coeff, 7 - 3 * coeff)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + scratch), dtype=torch.complex128)
    initial[:: 1 << scratch] = amplitudes
    matrix = torch.as_tensor(
        original.tensor_contract(), dtype=torch.complex128
    ).reshape(1 << logical, 1 << logical)
    expected = torch.zeros_like(initial)
    expected[:: 1 << scratch] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for history in product((0, 1), repeat=right - 1):
        actual = forced_branch(initial, forward.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** ((right - 1) / 2), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << scratch)[:, 1:]) == 0
        for second in product((0, 1), repeat=right - 1):
            recovered = forced_branch(actual, backward.circuit.instructions, second)
            torch.testing.assert_close(
                recovered, initial / 2 ** (right - 1), atol=3e-12, rtol=0
            )
    for precision in (torch.complex64, torch.complex128):
        circuit = DynamicCircuit(
            logical + 1 + scratch, inputs=initial.to(precision), dtype=precision
        )
        circuit.append(FieldOperation(circuit.n_qubits, forward.circuit.instructions))
        result = run_dynamic(circuit, shots=4, seed=37, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(4, -1),
            expected.to(precision).expand(4, -1),
            atol=2e-6 if precision == torch.complex64 else 3e-12,
            rtol=0,
        )


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
def test_fanout_full_coherent_dirty_targets_source_tree_involution_and_model(
    width: int,
) -> None:
    source = import_module("qualtran.bloqs.mcmt.multi_target_cnot").MultiTargetCNOT(
        width
    )
    targets = tuple(reversed(range(1, width + 1)))
    program = resolve_component_program(_library(source), 0, _factories())
    bound = bind_register_program(program, ((0,), targets), borrowed=(width + 1,))
    assert not bound.events and bound.outputs == ((0,), targets)
    expected = reference_arithmetic_gates(
        source, {"control": (0,), "targets": targets}, (), {}
    )
    assert_source(bound.circuit.instructions, expected)
    assert len(bound.circuit.instructions) == 2 * width - 1
    resource = import_module("qualtran.resource_counting")
    cost = resource.get_cost_value(source, resource.QECGatesCost())
    assert cost.clifford == 1 and cost.total_t_count() == cost.measurement == 0
    raw = torch.arange(1, (1 << (width + 2)) + 1, dtype=torch.float64)
    state = torch.complex(raw, 9 - raw)
    state /= state.norm()
    indexes = torch.arange(state.numel())
    flipped = torch.where(
        (indexes >> (width + 1)) != 0, indexes ^ (((1 << width) - 1) << 1), indexes
    )
    for precision in (torch.complex64, torch.complex128):
        circuit = Circuit(width + 2, inputs=state.to(precision), dtype=precision)
        operation = FieldOperation(width + 2, bound.circuit.instructions)
        circuit.append(operation)
        tolerance = 5e-7 if precision == torch.complex64 else 3e-12
        torch.testing.assert_close(
            circuit.state().flatten(),
            state[flipped].to(precision),
            atol=tolerance,
            rtol=0,
        )
        circuit.append(operation)
        torch.testing.assert_close(
            circuit.state().flatten(), state.to(precision), atol=tolerance, rtol=0
        )


@pytest.mark.parametrize(
    "example",
    (
        "_sub_symb",
        "_sub_small",
        "_sub_large",
        "_sub_diff_size_regs",
        "_sub_symp_decomposition",
        "_sub_from_symb",
        "_sub_from_small",
        "_sub_from_large",
    ),
)
def test_every_original_subtraction_example_with_retained_symbolic_boundaries(
    example: str,
) -> None:
    original = getattr(
        import_module("qualtran.bloqs.arithmetic.subtraction"), example
    ).make()
    for depth in (0, 1, 2):
        library = _library(original, depth)
        if example == "_sub_symp_decomposition" and depth == 0:
            with pytest.raises(SerializationError, match="Missing connection"):
                resolve_component_program(library, 0, _factories())
            continue
        program = resolve_component_program(library, 0, _factories())
        assert verify_program_record(program) == ()
        if "symb" in example or "symp" in example:
            with pytest.raises(ValueError, match="component bodies"):
                bind_register_program(program, ())


def test_constructor_validation_and_symbolic_fanout_declaration() -> None:
    mapping = subtraction_factories()
    for identifier, factory in mapping.items():
        from_target = identifier.endswith("SubtractFrom")
        key = "dtype" if from_target else "a_dtype"
        for args in ({}, {key: 1}, {key: unsigned_register_type(2), "bad": 1}):
            with pytest.raises(SerializationError):
                factory("main", None, args)
        with pytest.raises(SerializationError, match="signature"):
            factory("main", (), {key: unsigned_register_type(2)})
        with pytest.raises(ValueError):
            factory("main", None, {key: opaque_register_type(2)})
        assert isinstance(
            factory("main", None, {key: unsigned_register_type(2)}), ProgramRecord
        )
    fanout = next(iter(fanout_factories().values()))
    for fanout_args in ({}, {"bitsize": True}, {"bitsize": 3, "bad": 1}):
        with pytest.raises(SerializationError):
            fanout("main", None, fanout_args)
    with pytest.raises(SerializationError, match="signature"):
        fanout("main", (), {"bitsize": 2})
    for width in (0, -1):
        with pytest.raises(ValueError):
            fanout("main", None, {"bitsize": width})
    for width in (sympy.Symbol("n"), sympy.Integer(3)):
        declared = fanout("main", None, {"bitsize": width})
        assert isinstance(declared, FunctionRecord) and not declared.blocks


@pytest.mark.parametrize("example", ("_c_multi_not", "_c_multi_not_symb"))
def test_complete_fanout_examples_all_serialized_depths(example: str) -> None:
    original = getattr(
        import_module("qualtran.bloqs.mcmt.multi_target_cnot"), example
    ).make()
    for depth in (0, 1, 2):
        program = resolve_component_program(_library(original, depth), 0, _factories())
        assert verify_program_record(program) == ()
        if "symb" in example:
            assert not program.functions[0].blocks
            with pytest.raises(ValueError, match="component bodies"):
                bind_register_program(program, ())
