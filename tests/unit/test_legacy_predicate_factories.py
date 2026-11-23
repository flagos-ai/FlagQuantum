"""Grouped predicate constructors preserve source feedback and exact owned erasure."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_partition_factories import partition_factories
from flagquantum.compiler.formats.legacy_predicate_factories import predicate_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import (
    CASES,
    ControlCase,
    control_fixture,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolve_predicate(source: Any, depth: int = 0) -> ProgramRecord:
    factories = {**_factories(), **partition_factories()}
    factories.update(
        predicate_factories(
            adjoint_factory=factories["qualtran._infra.adjoint.Adjoint"]
        )
    )
    program = resolve_component_program(_library(source, depth), 0, factories)
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("case", CASES)
def test_complete_grouped_constructor_source_stream_ownership_complex_branches_and_model(
    case: ControlCase,
) -> None:
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.control.control_predicate_resource_model import (
        control_predicate_resource_formula,
    )

    spec, patterns, groups = control_fixture(case)
    source = CtrlSpecAnd(spec)
    n = int(source.n_ctrl_qubits)
    inputs = tuple(
        group[i : i + pattern.width]
        for group, pattern in zip(groups, patterns, strict=True)
        for i in range(0, len(group), pattern.width)
    )
    work = tuple(range(n, 2 * n - 1))
    forward = bind_register_program(
        resolve_predicate(source), inputs, clean=work, allow_workspace_outputs=True
    )
    back = bind_register_program(
        resolve_predicate(source.adjoint()),
        forward.outputs,
        input_ownership=tuple((w, False) for w in work),
    )
    assert back.outputs == inputs
    controls = {
        register.name: group
        for register, group in zip(source.control_registers, groups, strict=True)
    }
    outputs = {**controls, "target": forward.outputs[-1]}
    if n > 2:
        outputs["junk"] = forward.outputs[-2]
    for original, bound, before, after, uncompute in (
        (source, forward, controls, outputs, False),
        (source.adjoint(), back, outputs, controls, True),
    ):
        assert_source(
            bound.circuit.instructions,
            reference_arithmetic_gates(
                original,
                before,
                work,
                MeasurementSlots(bound.circuit.instructions),
                expected_outputs=after,
            ),
        )
        gate_counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(original, QECGatesCost())
        native = control_predicate_resource_formula(
            tuple(int(bit) for bit in source._flat_cvs), uncompute=uncompute
        )
        for key in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert getattr(native, key) == getattr(model, key)
        assert gate_counts["t"] + gate_counts["tdg"] == model.total_t_count()
        assert gate_counts["measure"] == gate_counts["reset"] == model.measurement
    assert get_cost_value(source, QubitCount()) == 2 * n - 1
    coefficients = torch.arange(1, (1 << (n + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coefficients, 7 - 3 * coefficients)
    amplitudes /= amplitudes.norm()
    # A spectator occupies the least significant wire after the owned prefix.
    initial = torch.zeros(1 << (2 * n), dtype=torch.complex128)
    initial[:: 1 << n] = amplitudes[::2]
    initial[1 :: 1 << n] = amplitudes[1::2]
    expected = torch.zeros_like(initial)
    for value in range(1 << n):
        prefix = 0
        matched = True
        for index, bit in enumerate(source._flat_cvs):
            matched = matched and (value >> (n - index - 1)) & 1 == bit
            if index:
                prefix = (prefix << 1) | matched
        for spectator in (0, 1):
            expected[(value << n) | (prefix << 1) | spectator] = amplitudes[
                2 * value + spectator
            ]
    prepared = forced_branch(initial, forward.circuit.instructions, ())
    torch.testing.assert_close(prepared, expected, rtol=0, atol=3e-12)
    for history in product((0, 1), repeat=n - 1):
        actual = forced_branch(prepared, back.circuit.instructions, history)
        torch.testing.assert_close(
            actual, initial / 2 ** ((n - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << n)[:, 2:]) == 0
    if n <= 4:
        matrix = torch.as_tensor(
            source.tensor_contract(), dtype=torch.complex128
        ).reshape(1 << (2 * n - 1), 1 << n)
        torch.testing.assert_close(
            expected, (matrix @ amplitudes.reshape(-1, 2)).flatten(), rtol=0, atol=3e-12
        )
    for precision in (torch.complex64, torch.complex128):
        circuit = DynamicCircuit(2 * n, inputs=initial.to(precision), dtype=precision)
        circuit.append(
            FieldOperation(
                circuit.n_qubits,
                (*forward.circuit.instructions, *back.circuit.instructions),
            )
        )
        result = run_dynamic(circuit, shots=2, seed=51, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(2, -1),
            initial.to(precision).expand(2, -1),
            rtol=0,
            atol=2e-6 if precision == torch.complex64 else 3e-12,
        )
    # The original codec fails at the first NumPy-valued And argument.
    failing_depth = 1 if n == 2 else 2
    with pytest.raises(ValueError, match="Cannot serialize .*numpy.uint8"):
        _library(source, failing_depth)
    if n > 2:
        named = resolve_predicate(source, 1)
        assert (
            bind_register_program(
                named, inputs, clean=work, allow_workspace_outputs=True
            ).circuit.instructions
            == forward.circuit.instructions
        )


@pytest.mark.parametrize(
    "example",
    ("_ctrl_on_int", "_ctrl_on_bits", "_ctrl_on_nd_bits", "_ctrl_on_multiple_values"),
)
def test_all_concrete_original_factory_signatures(example: str) -> None:
    source = getattr(import_module("qualtran.bloqs.mcmt.ctrl_spec_and"), example).make()
    for original in (source, source.adjoint()):
        assert (
            resolve_predicate(original).functions[0].registers
            == _library(original, 0).entries[0].declaration.registers
        )


def test_symbolic_original_codec_failure_and_logical_widths() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    n = sympy.Symbol("n", integer=True, positive=True)
    source = CtrlSpecAnd(CtrlSpec(QUInt(n), cvs=1))
    for original in (source, source.adjoint()):
        assert not resolve_predicate(original, 0).functions[0].blocks
    module = import_module("qualtran.bloqs.mcmt.ctrl_spec_and")
    for example in (
        "_ctrl_on_symbolic_cv",
        "_ctrl_on_symbolic_cv_multi",
        "_ctrl_on_symbolic_n_ctrls",
    ):
        with pytest.raises(ValueError, match="cannot serialize Shaped"):
            _library(getattr(module, example).make(), 0)


def test_predicate_constructor_validation_exact_nested_adjoints_and_fallback() -> None:
    np = pytest.importorskip("numpy")
    existing = _factories()
    factories = predicate_factories(
        adjoint_factory=existing["qualtran._infra.adjoint.Adjoint"]
    )
    factory = factories["qualtran.bloqs.mcmt.ctrl_spec_and.CtrlSpecAnd"]
    spec = ControlSpecification(((quantum_bit_type(), np.array([0, 1, 1])),))
    result = factory("custom", None, {"ctrl_spec": spec})
    assert isinstance(result, ProgramRecord)
    assert (
        factory("custom", result.functions[0].registers, {"ctrl_spec": spec}) == result
    )
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "back", None, {"subbloq": result.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord)
    double = factories["qualtran._infra.adjoint.Adjoint"](
        "restored", None, {"subbloq": inverse.functions[0]}
    )
    assert isinstance(double, ProgramRecord)
    assert double.functions[0].registers == result.functions[0].registers
    bad: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"ctrl_spec": 1},
        {"ctrl_spec": ControlSpecification(())},
        {
            "ctrl_spec": ControlSpecification(
                (
                    (
                        RegisterType(
                            "fixed_point", 2, parameters=(("fractional_width", 1),)
                        ),
                        np.array(0),
                    ),
                )
            )
        },
    )
    for args in bad:
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"ctrl_spec": spec})
    with pytest.raises(ValueError, match="at least two"):
        factory(
            "bad",
            None,
            {"ctrl_spec": ControlSpecification(((quantum_bit_type(), np.array(1)),))},
        )
    with pytest.raises(SerializationError, match="context"):
        factories["qualtran._infra.adjoint.Adjoint"](
            "bad", None, {"subbloq": FunctionRecord("unknown", ())}
        )
    # Preserve the preceding MultiAnd recipe resolver in this same catalogue.
    and_root = existing["qualtran.bloqs.mcmt.and_bloq.MultiAnd"](
        "and", None, {"cvs": (0, 1, 1)}
    )
    assert isinstance(and_root, ProgramRecord)
    delegated = factories["qualtran._infra.adjoint.Adjoint"](
        "and_inverse", None, {"subbloq": and_root.functions[0]}
    )
    assert isinstance(delegated, ProgramRecord)
