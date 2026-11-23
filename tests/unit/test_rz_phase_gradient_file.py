"""Locked source Rz factory, symbolic arithmetic model and complete tensor evidence."""

from collections import Counter
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sympy import Symbol

from examples.algorithms.rz_phase_gradient_catalog import GradientRzExample
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import rz_via_phase_gradient

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(rz_via_phase_gradient.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        ("rz_via_phase_gradient.ipynb", "75e37a757fb74d78b06704638dd34a78cb42c7c2"),
        ("rz_via_phase_gradient.py", "d3ec380d24af7dfb8a115b6fe88074ad53e22e93"),
        ("rz_via_phase_gradient_test.py", "4fb5e79871ad3943caad86206771575039146f7d"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return rz_via_phase_gradient


def assert_resource_model(op: GradientRzExample, source: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = op.resource_model()
    expected = get_cost_value(source, QECGatesCost())
    assert actual.logical_qubits == get_cost_value(source, QubitCount())
    for field in fields(GateCost):
        assert getattr(actual, field.name) == getattr(expected, field.name)


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_four_bit_factory(adjoint: bool, source_module: Any) -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.bookkeeping import Cast

    op = GradientRzExample(adjoint=adjoint)
    source = source_module._rz_via_phase_gradient.make()
    if adjoint:
        source = source.adjoint()
    assert [(r.name, r.dtype.num_qubits, r.side.name) for r in op.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    assert [dict(r.dtype.parameters) for r in list(op.signature)[1:]] == [
        {"fractional_width": r.dtype.num_frac, "signed": r.dtype.signed}
        for r in list(source.signature)[1:]
    ]
    ports = {"q": (0,), "angle": (1, 2, 3, 4), "phase_grad": (5, 6, 7, 8)}
    stages: list[str] = []
    work, bits = (9, 10, 11), (0, 1, 2)

    def leaf(
        bloq: Any, inputs: SourcePorts, _desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Cast):
            stages.append("cast")
            return None
        if isinstance(bloq, ControlledAddOrSubtract):
            stages.append("add_or_subtract")
            assert inputs == {
                "ctrl": ports["q"],
                "a": ports["angle"],
                "b": ports["phase_grad"],
            }
            assert bloq.add_when_ctrl_is_on is not adjoint
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    work,
                    dict(zip(work, bits, strict=True)),
                    preserve_components=True,
                ),
                inputs,
            )
        return None

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert stages == ["cast", "cast", "add_or_subtract", "cast", "cast"]
    if not adjoint:
        ((child, count),) = source.bloq_counts().items()
        label = (
            f"ControlledAddOrSubtract({child.a_dtype}, {child.b_dtype}, "
            f"add_on_one={child.add_when_ctrl_is_on})"
        )
        assert op.call_graph().leaf_counts == {label: count}
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    gates = circuit.to_ir().instructions
    assert_source_dependencies(gates, expected)
    counts = Counter(g.name for g in gates)
    assert op.qubit_count == 12
    assert counts["t"] + counts["tdg"] == 12
    assert counts["cx"] == 41
    assert counts["measure"] == counts["reset"] == 3
    assert_resource_model(op, source)


@pytest.mark.parametrize("widths", [(1, 1), (1, 4), (3, 4), (4, 4)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_cost_boundaries(
    widths: tuple[int, int],
    adjoint: bool,
    source_module: Any,
) -> None:
    from qualtran import QUInt

    left, right = widths
    op = GradientRzExample(left, right, adjoint)
    source = source_module.RzViaPhaseGradient(QUInt(left), QUInt(right))
    assert_resource_model(op, source.adjoint() if adjoint else source)


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_symbolic_cost(adjoint: bool, source_module: Any) -> None:
    from qualtran import QUInt

    n = Symbol("n")
    op = GradientRzExample(n, n, adjoint)
    source = source_module.RzViaPhaseGradient(QUInt(n), QUInt(n))
    assert_resource_model(op, source.adjoint() if adjoint else source)
    assert op.resource_model().clifford == 9 * n - (10 if adjoint else 8)
    with pytest.raises(ValueError, match="Bind the Rz"):
        DynamicCircuit(12).append(op)


def test_unordered_symbolic_widths_fail_source_admission(source_module: Any) -> None:
    from qualtran import QUInt
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n = Symbol("n")
    source = source_module.RzViaPhaseGradient(QUInt(1), QUInt(n))
    with pytest.raises(TypeError, match="truth value of Relational"):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(TypeError, match="truth value of Relational"):
        GradientRzExample(1, n).resource_model()


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("word", [1, 2])
def test_original_tensor_fixture_all_branches(
    width: int, word: int, source_module: Any
) -> None:
    from qualtran import BloqBuilder, QBit, QFxp, Signature
    from qualtran.bloqs.basic_gates import IntState, Rz
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

    op = GradientRzExample(width, width)
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    raw = circuit.to_ir().instructions
    angle_bits = tuple(
        Instruction("x", (wire,))
        for wire in range(1, 1 + width)
        if word >> (width - wire) & 1
    )
    gradient = tuple(range(1 + width, 1 + 2 * width))
    pre = phase_gradient_state_ir(gradient, exponent=-1)
    post = phase_gradient_state_ir(gradient, exponent=-1, inverse=True)
    theta = 4 * pi * word / 2**width
    dtype = QFxp(width, width)
    load_angle = IntState(
        dtype.to_fixed_width_int(theta / (4 * pi), require_exact=False), bitsize=width
    )
    load_gradient = PhaseGradientState(width)
    builder, ports = BloqBuilder.from_signature(Signature.build_from_dtypes(q=QBit()))
    q, a, g = builder.add(
        source_module.RzViaPhaseGradient(angle_dtype=dtype, phasegrad_dtype=dtype),
        q=ports["q"],
        angle=builder.add(load_angle),
        phase_grad=builder.add(load_gradient),
    )
    builder.add(load_angle.adjoint(), val=a)
    builder.add(load_gradient.adjoint(), phase_grad=g)
    matrix = torch.as_tensor(np.asarray(builder.finalize(q=q).tensor_contract()).copy())
    ideal = torch.as_tensor(np.asarray(Rz(theta).tensor_contract()).copy())
    torch.testing.assert_close(matrix, ideal, atol=2e-12, rtol=0)
    full = op.qubit_count
    indices = [0, 1 << (full - 1)]
    # Use two linearly independent complex inputs so the full tensor is checked.
    for vector in ((1, 1j), (1j, 1)):
        state = torch.zeros(1 << full, dtype=torch.complex128)
        expected = torch.zeros_like(state)
        local = torch.tensor(vector, dtype=torch.complex128) / sqrt(2)
        state[indices] = local
        expected[indices] = matrix @ local
        for history in product((0, 1), repeat=width - 1):
            actual = forced_branch(
                state, (*angle_bits, *pre, *raw, *post, *angle_bits), history
            )
            torch.testing.assert_close(
                actual, expected * 2 ** (-(width - 1) / 2), atol=2e-12, rtol=0
            )


@pytest.mark.parametrize("signed", (False, True))
def test_logical_root_retains_unbound_cast_calls(signed: bool) -> None:
    from flagquantum.compiler.registers.phase.rz_phase_gradient_component import (
        rz_via_phase_gradient_program,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import verify_program_record
    from flagquantum.core.ir.program.register_types import (
        fixed_point_register_type,
        unsigned_register_type,
    )

    dtype = (
        fixed_point_register_type(3, 2, signed=True)
        if signed
        else unsigned_register_type(Symbol("n", positive=True, integer=True))
    )
    record = rz_via_phase_gradient_program(dtype, dtype)
    assert verify_program_record(record) == ()
    assert len(record.functions[0].blocks[0].operations) == 5
    assert len([function for function in record.functions if not function.blocks]) == (
        4 if signed else 5
    )
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1, 2, 3), (4, 5, 6)), clean=(7, 8))
