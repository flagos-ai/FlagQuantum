"""Stored ArcTan lookup replaces the declaration only through an explicit factory."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.lookup.arctan_lookup_component import (
    arctan_lookup_program,
)
from flagquantum.compiler.registers.lookup.arctangent_declaration import (
    ArctangentDeclaration,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def pinned_arctangent() -> None:
    from qualtran.bloqs.arithmetic.trigonometric import arctan

    raw = Path(arctan.__file__).read_bytes()
    assert version("qualtran") == "0.7.0"
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "746411d9f44fa95c5209f5575419a99749937e99"
    )


@pytest.mark.parametrize("source_width,target_width", ((1, 1), (2, 3), (3, 8), (4, 6)))
def test_stored_lookup_preserves_source_qrom_operands_feedback_and_resources(
    source_width: int, target_width: int
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.trigonometric import ArcTan
    from qualtran.bloqs.data_loading import QROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = ArcTan(source_width, target_width)
    logical = source_width + 1 + target_width
    ports = (
        tuple(range(source_width)),
        (source_width,),
        tuple(range(source_width + 1, logical)),
    )
    work = tuple(range(logical, logical + source_width - 1))
    program = arctan_lookup_program(source_width, target_width)
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == tuple(
        ArctangentDeclaration(source_width, target_width).signature
    )
    bound = bind_register_program(program, ports, clean=work)
    assert bound.outputs == ports
    rows = tuple(original.apply(x, 0, 0) for x in range(1 << source_width))
    qrom = QROM(
        (np.array([r[1] for r in rows]), np.array([r[2] for r in rows])),
        (source_width,),
        (1, target_width),
        ((), ()),
    )
    # Each source measurement event binds to its native SSA result, even when
    # the flat emitter reuses a classical slot for successive erasures.
    expected = reference_arithmetic_gates(
        qrom,
        dict(zip(("selection", "target0_", "target1_"), ports, strict=True)),
        work,
        MeasurementSlots(bound.circuit.instructions),
        cirq_style=True,
    )
    assert_source(bound.circuit.instructions, expected)
    assert all(
        not gate.params and gate.matrix is None for gate in bound.circuit.instructions
    )
    actual = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(qrom, QECGatesCost())
    assert actual["t"] + actual["tdg"] == model.total_t_count()
    assert actual["measure"] == actual["reset"] == model.measurement
    assert original.t_complexity().t == 8 * sympy.Symbol("_c0") * target_width**2
    with pytest.raises(DecomposeNotImplementedError):
        original.decompose_bloq()


@pytest.mark.parametrize(
    "source_width,target_width,repeats", ((1, 8, 2), (2, 3, 2), (3, 3, 1))
)
def test_composed_lookup_full_complex_spectator_every_history_and_zero_work(
    source_width: int, target_width: int, repeats: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcTan

    cirq = pytest.importorskip("cirq")
    declaration = ArctangentDeclaration(source_width, target_width)
    builder = ComponentProgramBuilder(declaration.signature)
    values = builder.inputs
    for _ in range(repeats):
        values = builder.call_program(
            arctan_lookup_program(source_width, target_width), values
        )
    program = builder.finalize(values)
    logical = source_width + 1 + target_width
    ports = (
        tuple(range(source_width)),
        (source_width,),
        tuple(range(source_width + 1, logical)),
    )
    work = tuple(range(logical + 1, logical + source_width))
    bound = bind_register_program(program, ports, clean=work, borrowed=(logical,))
    assert bound.outputs == ports
    assert all(logical not in gate.wires for gate in bound.circuit.instructions)
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    stride = 1 << len(work)
    initial = torch.zeros(1 << (logical + 1 + len(work)), dtype=torch.complex128)
    initial[::stride] = amplitudes
    expected = torch.zeros_like(initial)
    matrix = torch.as_tensor(
        cirq.unitary(ArcTan(source_width, target_width)), dtype=torch.complex128
    )
    mapped = amplitudes.reshape(-1, 2)
    for _ in range(repeats):
        mapped = matrix @ mapped
    expected[::stride] = mapped.flatten()
    circuit = DynamicCircuit(
        bound.circuit.n_wires, inputs=initial, dtype=torch.complex128
    )
    circuit.append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=events):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (events / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, stride)[:, 1:]) == 0
    if repeats == 2:
        torch.testing.assert_close(expected, initial, rtol=0, atol=0)
