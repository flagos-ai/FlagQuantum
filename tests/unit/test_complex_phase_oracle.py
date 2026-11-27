"""Locked nullable source decomposition and native coherent phase oracle."""

from __future__ import annotations

from cmath import exp
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, sqrt
from pathlib import Path
from typing import TYPE_CHECKING

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.complex_phase_oracle import ComplexPhaseOracle
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qrom import source_qrom_gates
from tests.unit.test_sparse_chemistry_prepare import _branch

if TYPE_CHECKING:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source(pytestconfig: pytest.Config) -> None:
    import kickmix
    from qualtran.bloqs.mean_estimation import complex_phase_oracle

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    root = Path(complex_phase_oracle.__file__).parent
    for name, digest in (
        ("complex_phase_oracle.py", "6abd4c2811b54184eb1ba33e06c1d11d919c852c"),
        ("complex_phase_oracle_test.py", "ab6dc2af6345641052518c37f2206b44dbd8efc3"),
    ):
        contents = (root / name).read_bytes()
        assert sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest() == digest
    pytestconfig.addinivalue_line("markers", "slow: original reference suite")


def _arctan_outputs(source: ArcTan, word: int) -> tuple[int, int, int]:
    """Validate the concrete tuple behind the source ArithmeticGate-wide return type."""
    result = source.apply(word, 0, 0)
    assert isinstance(result, tuple) and len(result) == 3
    assert all(isinstance(value, int) for value in result)
    return result


def fixture(n: int, precision: int, *, adjoint: bool = False) -> DynamicCircuit:
    word = opaque_register_type(n)
    oracle = ComplexPhaseOracle(
        (), (Register("selection", word),), (Register("target", word),), precision
    )
    work, slots = oracle.workspace_requirements()
    width = n + work
    encoder = FieldOperation(
        width, tuple(Instruction("cx", (i, n + i)) for i in range(n))
    )
    circuit = DynamicCircuit(width)
    circuit.append(
        oracle.make_on(
            encoder,
            through=tuple(range(n)),
            workspace=tuple(range(n, width)),
            classical_bits=tuple(range(slots)),
            adjoint=adjoint,
        )
    )
    return circuit


@pytest.mark.parametrize("n", (2, 3, 4, 5))
@pytest.mark.parametrize("precision", (5, 6, 7))
def test_original_phase_grid_and_all_work_zero(n: int, precision: int) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    source = ArcTan(n, precision)
    ir = fixture(n, precision).to_ir()
    work = ir.n_wires - n
    scale = sqrt(sum((x + 1) ** 2 + (x - 2) ** 2 for x in range(1 << n)))
    state = {x << work: complex(x + 1, x - 2) / scale for x in range(1 << n)}
    expected = {}
    for x in range(1 << n):
        returned, sign, magnitude = _arctan_outputs(source, x)
        assert returned == x
        expected[x << work] = state[x << work] * exp(
            (-1 if sign else 1) * 1j * pi * magnitude / (1 << precision)
        )
    if n == 2:
        count = sum(g.name == "measure" for g in ir.instructions)
        for outcomes in product((0, 1), repeat=count):
            actual = _branch(ir.instructions, ir.n_wires, outcomes, state)
            assert actual == pytest.approx(
                {i: a * 2 ** (-count / 2) for i, a in expected.items()}, abs=3e-12
            )
    else:
        for seed in (29, 42):
            assert execute_sparse_register(ir, state, seed=seed) == pytest.approx(
                expected, abs=3e-12
            )

    inverse = fixture(n, precision, adjoint=True).to_ir()
    assert execute_sparse_register(inverse, expected, seed=37) == pytest.approx(
        state, abs=3e-12
    )


@pytest.mark.parametrize("n,precision", ((2, 1), (2, 2), (2, 3), (3, 5), (5, 7)))
def test_source_models_and_direct_calls(n: int, precision: int) -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import (
        ComplexPhaseOracle as SourceOracle,
    )
    from qualtran.bloqs.mean_estimation.complex_phase_oracle_test import ExampleSelect
    from qualtran.cirq_interop.testing import GateHelper
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = SourceOracle(ExampleSelect(n), precision)
    native = ComplexPhaseOracle(
        (),
        (Register("selection", opaque_register_type(n)),),
        (Register("target", opaque_register_type(n)),),
        precision,
    )
    model = SymbolicResourceFormula(0, n, 0, 2 * n)
    actual = native.source_resource_model(model, model)
    expected = get_cost_value(source, QECGatesCost())
    assert (
        actual.t,
        actual.and_bloq,
        actual.clifford,
        actual.rotation,
        actual.measurement,
    ) == (
        expected.t,
        expected.and_bloq,
        expected.clifford,
        expected.rotation,
        expected.measurement,
    )
    assert actual.logical_qubits == get_cost_value(source, QubitCount())

    helper = GateHelper(source)
    original_calls = tuple(cirq.flatten_to_ops(cirq.decompose_once(helper.operation)))
    assert original_calls[0].gate == source.encoder
    assert original_calls[-1] == cirq.inverse(original_calls[0])
    assert original_calls[-2] == cirq.inverse(original_calls[1])
    sign, *fraction = original_calls[1].qubits[n:]
    assert len(original_calls) == 2 * precision + 4
    gates = fixture(n, precision).to_ir().instructions
    actual_ladder = tuple(g for g in gates if g.name == "cphase")
    assert len(actual_ladder) == 2 * precision
    for i, (pos, neg, digit) in enumerate(
        zip(original_calls[2:-2:2], original_calls[3:-2:2], fraction, strict=True)
    ):
        plus, minus = actual_ladder[2 * i : 2 * i + 2]
        assert pos.qubits == neg.qubits == (sign, digit)
        positive, negative = pos.gate, neg.gate
        assert isinstance(positive, cirq.ControlledGate)
        assert isinstance(positive.sub_gate, cirq.ZPowGate)
        assert isinstance(negative, cirq.CZPowGate)
        assert tuple(positive.control_values.expand()) == ((0,),)
        assert plus.wires == minus.wires == (2 * n, 2 * n + 1 + i)
        assert plus.params["theta"] == pi * positive.sub_gate.exponent
        assert minus.params["theta"] == pi * negative.exponent
    gate_counts = Counter(g.name for g in gates)
    assert gate_counts["measure"] == gate_counts["reset"]
    assert gate_counts["t"] + gate_counts["tdg"] == 4 * gate_counts["measure"]


@pytest.mark.parametrize("n,precision", ((2, 3), (3, 5), (5, 7)))
def test_lookup_stages_match_source_qrom_for_literal_arctan_words(
    n: int, precision: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan
    from qualtran.bloqs.data_loading.qrom import QROM

    ir = fixture(n, precision).to_ir()
    gates = ir.instructions
    phases = [i for i, g in enumerate(gates) if g.name == "cphase"]
    start, end = phases[0] - 1, phases[-1] + 1
    source = ArcTan(n, precision)
    data = tuple(_arctan_outputs(source, x) for x in range(1 << n))
    qrom = QROM(
        (np.asarray([r[1] for r in data]), np.asarray([r[2] for r in data])),
        (n,),
        (1, precision),
        ((), ()),
    )
    sign = 2 * n
    fraction = tuple(range(sign + 1, sign + 1 + precision))
    workspace = tuple(range(sign + 1 + precision, ir.n_wires))
    reference = source_qrom_gates(
        qrom,
        {
            "selection": tuple(range(n, 2 * n)),
            "target0_": (sign,),
            "target1_": fraction,
        },
        workspace,
        tuple(range(len(workspace))),
    )
    assert_source(gates[n:start], reference)
    assert_source(gates[end:-n], reference)
    assert gates[:n] == tuple(Instruction("cx", (i, n + i)) for i in range(n))
    assert gates[-n:] == tuple(reversed(gates[:n]))

    def bits(value: int, width: int) -> tuple[int, ...]:
        return tuple(value >> i & 1 for i in reversed(range(width)))

    clean = (0,) * len(workspace)
    initial, angle = [], []
    for word, direction, magnitude in data:
        copies = bits(word, n) * 2
        initial.append((*copies, 0, *(0,) * precision, *clean))
        angle.append((*copies, direction, *bits(magnitude, precision), *clean))
    events = sum(gate.name == "measure" for gate in reference)
    assert certified_and_channel_map(gates[n:start], tuple(initial)) == (
        tuple(angle),
        events,
    )
    assert certified_and_channel_map(gates[end:-n], tuple(angle)) == (
        tuple(initial),
        events,
    )


def test_controlled_protocol_and_non_self_adjoint_encoder() -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import (
        ComplexPhaseOracle as SourceOracle,
    )
    from qualtran.bloqs.mean_estimation.complex_phase_oracle_test import ExampleSelect

    oracle = ComplexPhaseOracle(
        (Register("control", quantum_bit_type()),),
        (Register("selection", opaque_register_type(2)),),
        (Register("high", quantum_bit_type()), Register("low", quantum_bit_type())),
        3,
    )
    source = SourceOracle(ExampleSelect(2, 1), 3)
    assert oracle.wire_symbols() == cirq.circuit_diagram_info(source).wire_symbols
    assert [(r.name, r.bitsize) for r in oracle.signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    work, slots = oracle.workspace_requirements()
    width = 3 + work
    encoder = FieldOperation(
        width,
        (
            Instruction("phase", (0,), {"theta": 0.3}),
            Instruction("ccx", (0, 1, 3)),
            Instruction("ccx", (0, 2, 4)),
        ),
    )
    body = oracle.make_on(
        encoder,
        through=(0, 1, 2),
        workspace=tuple(range(3, width)),
        classical_bits=tuple(range(slots)),
    )
    circuit = DynamicCircuit(width).append(body)
    assert circuit.to_ir().instructions[-1].params["theta"] == pytest.approx(-0.3)
    data = {i << work: complex(i + 1, i - 2) / sqrt(264) for i in range(8)}
    # The source test's controlled encoder is diagram-only. This executable
    # encoder instead has a supplied coherent controlled-copy definition.
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    expected = {
        index: amp
        * exp(-1j * pi * _arctan_outputs(ArcTan(2, 3), i & 3 if i >> 2 else 0)[2] / 8)
        for i, (index, amp) in enumerate(data.items())
    }
    count = sum(g.name == "measure" for g in circuit.to_ir().instructions)
    for history in product((0, 1), repeat=count):
        assert _branch(
            circuit.to_ir().instructions, width, history, data
        ) == pytest.approx(
            {i: a * 2 ** (-count / 2) for i, a in expected.items()},
            abs=3e-12,
        )
    with pytest.raises(ValueError, match="another child's quantum partition"):
        oracle.make_on(
            FieldOperation(width, (Instruction("x", (width - 1,)),)),
            through=(0, 1, 2),
            workspace=tuple(range(3, width)),
            classical_bits=tuple(range(slots)),
        )


def test_original_diagram_fixture_and_opaque_controlled_encoder() -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import (
        ComplexPhaseOracle as SourceOracle,
    )
    from qualtran.bloqs.mean_estimation.complex_phase_oracle_test import ExampleSelect
    from qualtran.cirq_interop.testing import GateHelper

    source_encoder = ExampleSelect(3, 1)
    source = SourceOracle(source_encoder, 5)
    word = opaque_register_type(3)
    native = ComplexPhaseOracle(
        (Register("control", quantum_bit_type()),),
        (Register("selection", word),),
        (Register("target", word),),
        5,
    )
    assert native.wire_symbols() == cirq.circuit_diagram_info(source).wire_symbols
    assert native.wire_symbols() == ("@", "ROTy", "ROTy", "ROTy")
    with pytest.raises(TypeError, match="unexpected keyword argument 'control'"):
        cirq.decompose_once(GateHelper(source_encoder).operation)
