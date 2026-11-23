"""Multi-control constructor dependencies preserve literal polarities and phases."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_multi_control_factories import (
    multi_control_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _resolve(source: Any, depth: int = 1) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, depth), 0, {**_factories(), **multi_control_factories()}
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("axis", ("X", "Z"))
@pytest.mark.parametrize(
    "values", [values for n in range(5) for values in product((0, 1), repeat=n)]
)
def test_multi_control_named_and_direct_source_dependencies(
    axis: str, values: tuple[int, ...]
) -> None:
    source = getattr(import_module("qualtran.bloqs.mcmt"), "MultiControl" + axis)(
        values
    )
    width = len(values)
    inputs = (*((i,) for i in range(width)), (width,))
    clean = tuple(range(width + 1, 2 * width))
    for original in (source, source.decompose_bloq()):
        bound = bind_register_program(_resolve(original, 1), inputs, clean=clean)
        ports: dict[str, tuple[int, ...]] = (
            {"controls": tuple(range(width)), "target": (width,)}
            if width
            else {"target": (0,)}
        )
        assert_source(
            bound.circuit.instructions,
            reference_arithmetic_gates(
                source,
                ports,
                clean,
                MeasurementSlots(bound.circuit.instructions),
                keep_pauli_leaves=True,
            ),
        )
        from qualtran.resource_counting import QECGatesCost, get_cost_value

        actual = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(source, QECGatesCost())
        assert (
            actual["t"] + actual["tdg"]
            == model.total_t_count()
            == 4 * max(width - 1, 0)
        )
        assert actual["measure"] == model.measurement == max(width - 1, 0)
        if width > 3:
            continue
        logical = width + 1
        coeff = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
        amplitudes = torch.complex(coeff, 3 - 7 * coeff)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(1 << (logical + len(clean)), dtype=torch.complex128)
        initial[:: 1 << len(clean)] = amplitudes
        matrix = torch.as_tensor(
            source.tensor_contract(), dtype=torch.complex128
        ).reshape(1 << logical, 1 << logical)
        expected = torch.zeros_like(initial)
        expected[:: 1 << len(clean)] = matrix @ amplitudes
        for history in product((0, 1), repeat=len(clean)):
            torch.testing.assert_close(
                forced_branch(initial, bound.circuit.instructions, history),
                expected / 2 ** (len(clean) / 2),
                rtol=0,
                atol=3e-12,
            )


@pytest.mark.parametrize("axis", ("X", "Z"))
@pytest.mark.parametrize("values", (0, 1, [0], [1], [0, 1, 0]))
def test_scoped_controlled_via_and_scalar_and_array_shapes(
    axis: str, values: int | list[int]
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    child = getattr(import_module("qualtran.bloqs.basic_gates"), axis + "Gate")()
    source = ControlledViaAnd(child, CtrlSpec(cvs=values))
    program = _resolve(source, 0)
    count = int(source.ctrl_spec.num_qubits)
    bound = bind_register_program(
        program,
        (*((i,) for i in range(count)), (count,)),
        clean=tuple(range(count + 1, 2 * count)),
    )
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source,
            {"ctrl": tuple(range(count)), "q": (count,)},
            tuple(range(count + 1, 2 * count)),
            MeasurementSlots(bound.circuit.instructions),
            keep_pauli_leaves=True,
        ),
    )


def test_multi_control_invalid_inputs_and_scoped_child_boundary() -> None:
    np = pytest.importorskip("numpy")
    factories = multi_control_factories()
    factory = factories["qualtran.bloqs.mcmt.multi_control_pauli.MultiControlX"]
    invalid: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"cvs": 1},
        {"cvs": np.zeros((2, 2), dtype=int)},
        {"cvs": (2,)},
        {"cvs": (sympy.Symbol("activation", integer=True),)},
    )
    for args in invalid:
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"cvs": (0, 1)})
    controlled = factories["qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"]
    leaves = fixed_gate_factories()
    x = leaves["qualtran.bloqs.basic_gates.x_basis.XGate"]("leaf", None, {})
    h = leaves["qualtran.bloqs.basic_gates.hadamard.Hadamard"]("leaf", None, {})
    bit = quantum_bit_type()
    assert isinstance(x, FunctionRecord) and isinstance(h, FunctionRecord)
    bad_controlled: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"subbloq": 1, "ctrl_spec": ControlSpecification(())},
        {"subbloq": h, "ctrl_spec": ControlSpecification(())},
        {"subbloq": x, "ctrl_spec": 1},
        {
            "subbloq": x,
            "ctrl_spec": ControlSpecification(
                ((unsigned_register_type(2), np.array(1)),)
            ),
        },
        {
            "subbloq": x,
            "ctrl_spec": ControlSpecification(((bit, np.array([], dtype=int)),)),
        },
    )
    for args in bad_controlled:
        with pytest.raises(SerializationError):
            controlled("bad", None, args)
    assert isinstance(x, FunctionRecord)
