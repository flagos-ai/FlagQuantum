"""Whole-file acceptance for the locked grouped conjunction and notebook."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sympy import Symbol

from examples.algorithms.control_predicate_catalog import (
    PredicateExample,
    predicate_examples,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.control_predicate_resource_model import (
    control_predicate_resource_formula,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXAMPLES = predicate_examples()


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.mcmt import ctrl_spec_and

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(ctrl_spec_and.__file__).parent
    for name, digest in (
        ("ctrl_spec_and.py", "3fa7edd03c6f488edc5b11358c5aa22819ccef5b"),
        ("ctrl_spec_and_test.py", "f0f8638db9882c8c75f598effba197f80b927e29"),
        ("ctrl_spec_and.ipynb", "e96d6fa66990fc52bb679a7892edef94d6fcd3c9"),
    ):
        data = (folder / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return ctrl_spec_and


@pytest.mark.parametrize("name", tuple(EXAMPLES))
def test_original_signatures_calls_and_cost(name: str, source_module: Any) -> None:
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    native = EXAMPLES[name]
    original = getattr(source_module, "_" + name).make()
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in native.signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in original.signature
    ]
    ((child, multiplicity),) = original.build_call_graph(None).items()
    call = native.source_child()
    assert multiplicity == 1
    if isinstance(child, And):
        assert call.controls == (child.cv1, child.cv2) == (Symbol("cv1"), Symbol("cv2"))
    else:
        assert isinstance(child, MultiAnd)
        assert call.controls == (
            tuple(child.cvs)
            if isinstance(native.activation, tuple)
            else original.n_ctrl_qubits
        )
    for direction in (False, True):
        qec = get_cost_value(
            original.adjoint() if direction else original, QECGatesCost()
        )
        model = control_predicate_resource_formula(
            native.activation, uncompute=direction
        )
        assert (model.and_bloq, model.clifford, model.measurement) == (
            qec.and_bloq,
            qec.clifford,
            qec.measurement,
        )
    width = get_cost_value(original, QubitCount())
    assert model.logical_qubits == width


def _physical_pair(
    native: PredicateExample, original: Any
) -> tuple[int, tuple[Instruction, ...], tuple[Instruction, ...]]:
    assert native.patterns is not None and isinstance(native.width, int)
    n = native.width
    cursor = 0
    inputs = {}
    for port, group in zip(native.controls, native.patterns, strict=True):
        end = cursor + len(group.bits())
        inputs[port.name] = tuple(range(cursor, end))
        cursor = end
    work = tuple(range(n, 2 * n - 1))
    outputs = {**inputs, "junk": work[:-1], "target": work[-1:]}
    slots = dict(zip(work, range(n - 1), strict=True))
    forward = DynamicCircuit(2 * n - 1).append(native).to_ir().instructions
    erase = DynamicCircuit(2 * n - 1).append(native.adjoint()).to_ir().instructions
    for component, gates, start, returned in (
        (original, forward, inputs, outputs),
        (original.adjoint(), erase, outputs, inputs),
    ):
        expected = reference_arithmetic_gates(
            component, start, work, slots, expected_outputs=returned
        )
        assert_source(gates, expected)
    return n, forward, erase


@pytest.mark.parametrize("name", tuple(EXAMPLES)[:4])
def test_original_append_gates_feedback_and_complex_state(
    name: str, source_module: Any
) -> None:
    native = EXAMPLES[name]
    original = getattr(source_module, "_" + name).make()
    n, forward, erase = _physical_pair(native, original)
    assert tuple(original._flat_cvs) == native.activation
    costs = Counter(gate.name for gate in forward)
    assert costs["t"] + costs["tdg"] == 4 * (n - 1)
    assert sum(gate.name == "measure" for gate in erase) == n - 1
    value = int("".join(str(v) for v in original._flat_cvs), 2)
    choices = range(1 << n) if n == 4 else (0, value, value ^ 1, value ^ (1 << (n - 1)))
    initial = {v << (n - 1): complex(i + 1, 1 - i) for i, v in enumerate(choices)}
    expected = {}
    for v in choices:
        prefix = 0
        matching = True
        for i, bit in enumerate(original._flat_cvs):
            matching &= ((v >> (n - i - 1)) & 1) == bit
            if i:
                prefix = (prefix << 1) | matching
        index = v << (n - 1)
        expected[index | prefix] = initial[index]
    actual = _branch(forward, 2 * n - 1, (), initial)
    assert actual == pytest.approx(expected, abs=2e-12)
    histories = (
        product((0, 1), repeat=n - 1)
        if n == 4
        else (
            (0,) * (n - 1),
            (1,) * (n - 1),
            tuple(i % 2 for i in range(n - 1)),
        )
    )
    for history in histories:
        restored = _branch(erase, 2 * n - 1, history, actual)
        assert restored == pytest.approx(
            {k: a * 2 ** (-(n - 1) / 2) for k, a in initial.items()}, abs=2e-12
        )


@pytest.mark.parametrize("values", [(4,), (3, 5)])
def test_original_classical_truth_tables(
    values: tuple[int, ...], source_module: Any
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.simulation.classical_sim import get_classical_truth_table

    original = source_module.CtrlSpecAnd(
        CtrlSpec(tuple(QUInt(4) for _ in values), tuple(np.array(x) for x in values))
    )
    native = PredicateExample(
        tuple(
            Register(f"ctrl_{i}", unsigned_register_type(4)) for i in range(len(values))
        ),
        tuple(ControlRegister(4, (x,)) for x in values),
    )
    n, forward, _ = _physical_pair(native, original)
    _, _, table = get_classical_truth_table(original)
    basis = {
        tuple(int(v) for v in inputs): tuple(int(v) for v in outputs)
        for inputs, outputs in table
    }
    actual = _branch(
        forward, 2 * n - 1, (), {i << (n - 1): 1 + 0j for i in range(1 << n)}
    )
    assert len(actual) == len(basis) == 1 << n
    for output, amp in actual.items():
        packed = output >> (n - 1)
        inputs = tuple((packed >> j) & 15 for j in range(n - 4, -1, -4))
        assert amp == pytest.approx(1)
        junk = (output & ((1 << (n - 1)) - 1)) >> 1
        assert (*inputs, junk, output & 1) == basis[inputs]


@pytest.mark.parametrize("form", [None, 0, [0]])
def test_source_single_bit_refusal(form: Any, source_module: Any) -> None:
    from qualtran import CtrlSpec

    with pytest.raises(ValueError, match="at least 2"):
        source_module.CtrlSpecAnd(CtrlSpec() if form is None else CtrlSpec(cvs=form))
    native = PredicateExample(
        (
            Register(
                "ctrl_0", quantum_bit_type(), (1,) if isinstance(form, list) else ()
            ),
        ),
        (ControlRegister(1, (1 if form is None else 0,)),),
    )
    with pytest.raises(ValueError, match="at least two"):
        _ = native.signature
    with pytest.raises(ValueError, match="at least two"):
        DynamicCircuit(1).append(native)
