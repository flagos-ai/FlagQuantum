"""Native mean estimation against the locked Bernoulli and Grover fixtures."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import acos, pi, sqrt
from pathlib import Path
from typing import Any

import cirq
import numpy as np
import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.mean_estimation_operator import (
    CodeForRandomVariable,
    MeanEstimationOperator,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_mean_files() -> None:
    import kickmix
    from qualtran.bloqs.mean_estimation import mean_estimation_operator

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    root = Path(mean_estimation_operator.__file__).parent
    for name, digest in (
        ("__init__.py", "ce76a8d10a5909daeb7b1113c8cc7f26cc31b410"),
        ("mean_estimation_operator.py", "1e5b5d7db7b2c397fad8107d3c0e330958c2bb7e"),
        (
            "mean_estimation_operator_test.py",
            "bbc6d9ff32f223237e9506892cb5cb9ed53176fe",
        ),
    ):
        data = (root / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    assert all(
        line.lstrip().startswith("#") or not line.strip()
        for line in (root / "__init__.py").read_text().splitlines()
    )


def _reference(synthesizer: Any, encoder: Any, precision: int = 5) -> Any:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator as SourceMean,
    )

    return SourceMean(SourceCode(synthesizer, encoder), precision)


def _operation(
    n: int,
    target_width: int,
    marked_item: int,
    marked_value: int,
    *,
    probability: float | None,
    precision: int = 5,
    control_value: int | None = None,
    power: int = 1,
    adjoint: bool = False,
) -> tuple[MeanEstimationOperator, CircuitIR]:
    shape = (
        opaque_register_type(n)
        if probability is None
        else bounded_unsigned_register_type(n, 2)
    )
    selection_reg = Register("selection" if probability is None else "q", shape)
    target_reg = Register(
        "target" if probability is None else "t", opaque_register_type(target_width)
    )
    code = CodeForRandomVariable((selection_reg,), (selection_reg,), (target_reg,))
    op = MeanEstimationOperator(code, precision) ** power
    control_count = int(control_value is not None)
    if control_value is not None:
        op = op.controlled(control_value)
    data_width = n + control_count
    data = tuple(range(control_count, data_width))
    work, slots = op.workspace_requirements()
    width = data_width + work
    if probability is None:
        synth = tuple(Instruction("h", (i,)) for i in data)
    else:
        synth = (
            Instruction("ry", (data[0],), {"theta": 2 * acos(sqrt(1 - probability))}),
            *(Instruction("cx", (data[0], t)) for t in data[1:]),
        )
    negated = tuple(
        Instruction("x", (wire,))
        for i, wire in enumerate(data)
        if not marked_item >> (n - 1 - i) & 1
    )
    local_cast = Circuit(1).x(0).control(n, tuple(range(1, n + 1)))
    encoder = Circuit(width).compose(CircuitIR(width, negated))
    for i in range(target_width):
        if not marked_value >> (target_width - i - 1) & 1:
            continue
        target = data_width + i
        if n <= 2:
            name = "cx" if n == 1 else "ccx"
            encoder.compose(CircuitIR(width, (Instruction(name, (*data, target)),)))
        else:
            encoder.compose(local_cast, qubits=(target, *data))
    encoder.compose(CircuitIR(width, negated))
    bound = op.make_on(
        FieldOperation(width, synth),
        FieldOperation(width, encoder.to_ir().instructions),
        selection=data,
        workspace=tuple(range(data_width, width)),
        classical_bits=tuple(range(slots)),
        control=0 if control_count else None,
        adjoint=adjoint,
    )
    return op, DynamicCircuit(width).append(bound).to_ir()


def _logical_matrix(ir: CircuitIR, n: int) -> np.ndarray:
    work = ir.n_wires - n
    dimension = 1 << n
    measurements = sum(gate.name == "measure" for gate in ir.instructions)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        state = _branch(
            ir.instructions,
            ir.n_wires,
            (0,) * measurements,
            {column << work: 1 + 0j},
            normalize_measurements=True,
        )
        for wire_state, amplitude in state.items():
            assert not wire_state & ((1 << work) - 1)
            matrix[wire_state >> work, column] = amplitude
    return matrix


def _theorem(
    matrix: np.ndarray, prepared: np.ndarray, mean: float, s: float, c: float
) -> None:
    values, states = cirq.linalg.unitary_eig(matrix)
    hav = np.abs(np.sin(np.angle(values) / 2))
    normalized = abs(mean) / sqrt(1 + s**2)
    mask = (normalized / (1 + c * s) <= hav) & (hav <= normalized / (1 - c * s))
    observed = np.sum(np.abs(states[:, mask].conj().T @ prepared) ** 2)
    assert observed >= 1 - 2 / c**2 > 0


@pytest.mark.parametrize("n", (1, 2))
@pytest.mark.parametrize(
    "probability,value,bits,c",
    (
        (1e-4, 3, 2, 100 / 7),
        (1 / 2500, 2, 2, 50 / 4),
        (1 / 2500, 1, 1, 5),
        (1 / 16, 1, 1, 1.5),
    ),
)
def test_original_bernoulli_full_matrix_and_theorem(
    n: int, probability: float, value: int, bits: int, c: float
) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )

    synth = BernoulliSynthesizer(probability, n)
    encoder = BernoulliEncoder(probability, (0, value), n, bits)
    source = _reference(synth, encoder)
    _, ir = _operation(n, bits, (1 << n) - 1, value, probability=probability)
    matrix = _logical_matrix(ir, n)
    np.testing.assert_allclose(matrix, cirq.unitary(source), atol=3e-12, rtol=0)
    _theorem(matrix, cirq.unitary(synth)[:, 0], encoder.mu, sqrt(encoder.s_square), c)


@pytest.mark.parametrize("n,c", ((5, 4), (4, 2), (2, sqrt(2))))
def test_original_grover_full_matrix_and_theorem(n: int, c: float) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        GroverEncoder,
        GroverSynthesizer,
    )

    synth = GroverSynthesizer(n)
    encoder = GroverEncoder(n, 1, 1)
    source = _reference(synth, encoder)
    _, ir = _operation(n, 1, 1, 1, probability=None)
    matrix = _logical_matrix(ir, n)
    np.testing.assert_allclose(matrix, cirq.unitary(source), atol=3e-12, rtol=0)
    _theorem(matrix, cirq.unitary(synth)[:, 0], encoder.mu, sqrt(encoder.s_square), c)


@pytest.mark.parametrize("control_value,power", ((1, 1), (0, 1), (0, 2), (1, -2)))
def test_control_and_power_protocols(control_value: int, power: int) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )

    source = _reference(
        BernoulliSynthesizer(0.1, 1), BernoulliEncoder(0.1, (0, 1), 1, 1), 2
    )
    origin = (source**power).controlled(control_values=(control_value,))
    native, ir = _operation(
        1,
        1,
        1,
        1,
        probability=0.1,
        precision=2,
        control_value=control_value,
        power=power,
    )
    assert native.wire_symbols() == cirq.circuit_diagram_info(origin).wire_symbols
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in native.signature] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in origin.signature
    ]
    matrix = np.linalg.matrix_power(cirq.unitary(source), power)
    expected = cirq.unitary(
        cirq.MatrixGate(matrix).controlled(control_values=(control_value,))
    )
    np.testing.assert_allclose(cirq.unitary(origin), expected, atol=3e-12, rtol=0)
    np.testing.assert_allclose(_logical_matrix(ir, 2), expected, atol=3e-12, rtol=0)
    graph = native.source_call_graph()
    _, before = origin.call_graph(max_depth=1)
    expected_weights = sorted(before.values())
    assert (
        sorted(
            weight
            for (edge, _), weight in graph.direct_calls.items()
            if edge == graph.root
        )
        == expected_weights
    )
    count = sum(g.name == "measure" for g in ir.instructions)
    work = ir.n_wires - 2
    vector = np.array([1 + 2j, 3 - 4j, 5 + 3j, 7 - 2j]) / sqrt(117)
    normalize = vector / np.linalg.norm(vector)
    state = {i << work: complex(a) for i, a in enumerate(normalize)}
    wanted = {i << work: complex(a) for i, a in enumerate(expected @ normalize)}
    for branch in product((0, 1), repeat=count):
        assert _branch(
            ir.instructions, ir.n_wires, branch, state, normalize_measurements=True
        ) == pytest.approx(wanted, abs=3e-12)


def test_source_order_cost_and_complete_adjoint_histories() -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )
    from qualtran.cirq_interop.testing import GateHelper
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    p = 1e-4
    source = _reference(BernoulliSynthesizer(p, 2), BernoulliEncoder(p, (0, 3), 2, 2))
    native, ir = _operation(2, 2, 3, 3, probability=p)
    assert native.wire_symbols() == cirq.circuit_diagram_info(source).wire_symbols
    original_calls = tuple(
        cirq.flatten_to_ops(cirq.decompose_once(GateHelper(source).operation))
    )
    assert tuple(op.gate for op in original_calls) == (source.select, source.reflect)
    assert native.source_call_graph().leaf_counts == {
        "ComplexPhaseOracle": 1,
        "ReflectionUsingPrepare": 1,
    }
    encoder = SymbolicResourceFormula(0, 0, 0, 4, toffoli=2)
    prep = ComponentCost("Prepare", 2, GateCost(clifford=1, rotation=1))
    modeled = native.source_resource_model(encoder, encoder, prep, prep)
    qec = get_cost_value(source, QECGatesCost())
    assert (
        modeled.t,
        modeled.toffoli,
        modeled.and_bloq,
        modeled.clifford,
        modeled.rotation,
        modeled.measurement,
    ) == (
        qec.t,
        qec.toffoli,
        qec.and_bloq,
        qec.clifford,
        qec.rotation,
        qec.measurement,
    )
    assert modeled.logical_qubits == get_cost_value(source, QubitCount()) == 22
    counts = Counter(g.name for g in ir.instructions)
    assert (counts["cphase"], counts["measure"], counts["reset"]) == (10, 5, 5)
    assert [g.name for g in ir.instructions[:2]] == ["ccx", "ccx"]
    assert [g.name for g in ir.instructions[-2:]] == ["ry", "cx"]

    # Compare the locked reflection's exact mark on the named temporary
    # target. Its global -1 is emitted as a π phase on that set work bit.
    source_reflection = tuple(
        cirq.flatten_to_ops(cirq.decompose_once(GateHelper(source.reflect).operation))
    )
    inverse_prep, opening, mark, scalar, closing, prepare = source_reflection
    assert inverse_prep == cirq.inverse(prepare)
    assert opening == closing and opening.gate == cirq.X
    assert opening.qubits == mark.qubits[-1:]
    assert isinstance(scalar.gate, cirq.GlobalPhaseGate)
    assert scalar.gate.coefficient == -1
    source_mark = reference_arithmetic_gates(
        mark.gate,
        {"controls": (0, 1), "target": (2,)},
        (3,),
        {3: 0},
        keep_pauli_leaves=True,
    )
    phase_end = len(ir.instructions) - 2
    flag = Instruction("x", (2,))
    expected_reflection = (
        flag,
        *source_mark,
        Instruction("phase", (2,), {"theta": pi}),
        flag,
    )
    assert_source(
        ir.instructions[phase_end - len(expected_reflection) : phase_end],
        expected_reflection,
    )

    _, inverse = _operation(2, 2, 3, 3, probability=p, adjoint=True)
    matrix = cirq.unitary(source)
    vector = np.array([1 + 2j, 3 - 4j, 5 + 3j, 7 - 2j])
    vector /= np.linalg.norm(vector)
    work = ir.n_wires - 2
    state = {i << work: complex(a) for i, a in enumerate(vector)}
    for direction, desired in ((ir, matrix), (inverse, matrix.conj().T)):
        expected = {i << work: complex(a) for i, a in enumerate(desired @ vector)}
        for branch in product((0, 1), repeat=counts["measure"]):
            assert _branch(
                direction.instructions,
                ir.n_wires,
                branch,
                state,
                normalize_measurements=True,
            ) == pytest.approx(expected, abs=3e-12)


def test_register_match_and_source_power_refusal() -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )

    left = Register("q", bounded_unsigned_register_type(1, 2))
    right = Register("q", bounded_unsigned_register_type(2, 2))
    target = Register("t", opaque_register_type(1))
    with pytest.raises(ValueError, match="same selection"):
        CodeForRandomVariable((left,), (right,), (target,))
    with pytest.raises(AssertionError):
        _reference(BernoulliSynthesizer(0.1, 1), BernoulliEncoder(0.1, (0, 1), 2, 1))
    native = MeanEstimationOperator(CodeForRandomVariable((left,), (left,), (target,)))
    source = _reference(
        BernoulliSynthesizer(0.1, 1), BernoulliEncoder(0.1, (0, 1), 1, 1)
    )
    with pytest.raises(ValueError, match="nonzero integer"):
        _ = native**0
    with pytest.raises(ValueError, match="positive integer"):
        _ = source**0
