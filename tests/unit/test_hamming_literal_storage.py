"""Concrete symbolic Hamming SDK inputs keep unused logical junk ownership."""

from collections import Counter
from importlib import import_module
from itertools import product

from tests._reference_dependencies import unavailable_reference

# Typed-register owners require NumPy before their modules can be collected.
try:
    import_module("numpy")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_hamming_weight_factories import (
    hamming_weight_factories,
)
from flagquantum.compiler.registers.bits.hamming_weight_component import (
    source_hamming_weight_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5, 7, 8))
def test_literal_sdk_operands_unused_outputs_and_models(width: int) -> None:
    try:
        from qualtran._infra.bloq import DecomposeTypeError
        from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
        from qualtran.resource_counting._bloq_counts import QECGatesCost
        from qualtran.resource_counting._costing import get_cost_value
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    literal = sympy.Integer(width)
    source = HammingWeightCompute(literal)
    with pytest.raises(DecomposeTypeError):
        source.decompose_bloq()
    program = source_hamming_weight_program(literal)
    assert verify_program_record(program) == ()
    factory = next(iter(hamming_weight_factories().values()))
    resolved = factory("main", None, {"bitsize": literal})
    assert isinstance(resolved, ProgramRecord)
    assert resolved.functions[0].registers == program.functions[0].registers
    total = width + width - 1 + width.bit_length()
    bound = bind_register_program(
        program,
        (tuple(reversed(range(width))),),
        clean=tuple(reversed(range(width, total))),
        allow_workspace_outputs=True,
    )
    words = dict(zip(("x", "junk", "out"), bound.outputs, strict=True))
    expected = reference_arithmetic_gates(source, words, (), {}, cirq_style=True)
    assert_source(bound.circuit.instructions, expected)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    counts = Counter(g.name for g in bound.circuit.instructions)
    used = width - width.bit_count()
    assert counts["t"] + counts["tdg"] == 4 * used
    # The symbolic source model counts allocated junk, including untouched bits.
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == 4 * (width - 1)
    touched = {wire for gate in bound.circuit.instructions for wire in gate.wires}
    assert len(set(words["junk"]) - touched) == width.bit_count() - 1


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_literal_exact_inverse_append_every_complex_history_and_zero_work(
    width: int,
) -> None:
    literal = sympy.Integer(width)
    factories = hamming_weight_factories()
    forward = factories[
        "qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute"
    ]("compute", None, {"bitsize": literal})
    assert isinstance(forward, ProgramRecord)
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "erase", None, {"subbloq": forward.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord)
    assert verify_program_record(inverse) == ()
    total = width + width - 1 + width.bit_length() + 1
    clean = tuple(range(width, total - 1))
    first = bind_register_program(
        forward,
        (tuple(range(width)),),
        entry="compute",
        clean=clean,
        allow_workspace_outputs=True,
    )
    second = bind_register_program(
        inverse,
        first.outputs,
        entry="erase",
        input_ownership=tuple((wire, False) for wire in clean),
    )
    gates = (*first.circuit.instructions, *second.circuit.instructions)
    circuit = DynamicCircuit(total).append(FieldOperation(total, gates))
    assert circuit.to_ir().instructions == gates
    raw = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial.reshape(1 << width, -1)[:, :2] = amplitudes.reshape(1 << width, 2)
    measures = sum(gate.name == "measure" for gate in gates)
    assert measures == width - width.bit_count()
    for history in product((0, 1), repeat=measures):
        actual = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            actual, initial / 2 ** (measures / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(actual.reshape(1 << width, -1)[:, 2:]) == 0


@pytest.mark.parametrize("width", (sympy.Symbol("n"), sympy.Rational(3, 2), 0, -1))
def test_nonphysical_literal_inputs_refuse(width: sympy.Expr | int) -> None:
    with pytest.raises(ValueError):
        source_hamming_weight_program(width)
