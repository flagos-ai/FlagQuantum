"""Verify the locked fixture policies against native program and circuit checks."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.arithmetic.addition import add_ir
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, ValueRef, program_to_circuit
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.classical_addition import register_add_classical_values
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckResult,
    CheckStatus,
    RegisterExample,
    apply_pytest_result,
    assert_classical_action,
    check_example,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_program_lowering import packed_circuit

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def source_testing() -> tuple[Any, Any]:
    import kickmix
    from qualtran import conftest, testing

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    root = Path(testing.__file__).parent
    for name, digest in (
        ("conftest.py", "9699d1a2feb3497898606342ef711f94f2a8f596"),
        ("testing.py", "e06e8f1f785390053dfea68da8178132d38185a1"),
        ("testing_test.py", "5f34d93ee9c226abf03331b2c47dee6ab1221ad6"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return conftest, testing


def lower_fixture(record: ProgramRecord) -> CircuitIR:
    compiled = program_to_circuit(lower_register_program(record))
    assert compiled.state == "supported_exact", compiled.diagnostics
    assert compiled.canonical_payload is not None
    return CircuitIR.from_dict(compiled.canonical_payload)


def parallel_example() -> ProgramRecord:
    circuit = Circuit(3)
    circuit.append(FieldOperation(3, tuple(Instruction("x", (i,)) for i in range(3))))
    return packed_circuit(circuit.to_ir(), (3,))


@pytest.mark.parametrize("kind", CHECKS)
def test_five_owned_fixture_checks(kind: CheckKind) -> None:
    dtype = unsigned_register_type(3)
    example = RegisterExample(
        parallel_example, lower_fixture, lambda: {"x": 3}, ((dtype, dtype),)
    )
    apply_pytest_result(kind, check_example(example, kind))


@pytest.mark.ftqc_reference
def test_original_parallel_decomposition_state(source_testing: tuple[Any, Any]) -> None:
    from qualtran.bloqs.for_testing import TestParallelCombo

    circuit = Circuit.from_ir(lower_fixture(parallel_example()), dtype=torch.complex128)
    original = TestParallelCombo()
    source_testing[1].assert_valid_bloq_decomposition(original)
    matrix = torch.stack(
        [
            Circuit.from_ir(
                circuit.to_ir(),
                inputs=torch.eye(8, dtype=torch.complex128)[i],
                dtype=torch.complex128,
            )
            .state()
            .flatten()
            for i in range(8)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        matrix,
        torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mutation", ("port_width", "edge_missing", "producer_reused", "undeclared_result")
)
def test_owned_graph_checks_surface_invalid_fixtures(mutation: str) -> None:
    record = parallel_example()
    function = record.functions[0]
    block = function.blocks[0]
    (op,) = block.operations
    assert function.registers is not None
    if mutation == "port_width":
        (register,) = function.registers
        function = replace(
            function,
            registers=(
                Register(
                    register.name,
                    unsigned_register_type(4),
                    register.shape,
                    register.side,
                ),
            ),
        )
        # The width disagreement with the packed callee's signature must surface.
    else:
        if mutation == "edge_missing":
            op = replace(op, operands=(ValueRef("entry", 99, "register"),))
        elif mutation == "producer_reused":
            block = replace(block, operations=(op, op))
        else:
            block = replace(block, return_values=(ValueRef("entry", 99, "register"),))
        if mutation == "edge_missing":
            block = replace(block, operations=(op,))
        function = replace(function, blocks=(block,))
    bad = replace(record, functions=(function, *record.functions[1:]))
    for kind in ("decompose", "qtyping"):
        assert (
            check_example(RegisterExample(lambda: bad, lower_fixture), kind).status
            is CheckStatus.FAIL
        )


@pytest.mark.parametrize(
    "case,expected",
    (
        (0, CheckStatus.UNVERIFIED),
        (1, CheckStatus.FAIL),
        (2, CheckStatus.UNVERIFIED),
        (3, CheckStatus.FAIL),
    ),
)
def test_original_qtype_matrix_cases(case: int, expected: CheckStatus) -> None:
    types = (
        (signed_register_type(4), unsigned_register_type(4)),
        (signed_register_type(4), signed_register_type(5)),
        (unsigned_register_type(4), fixed_point_register_type(4, 4)),
        (unsigned_register_type(4), fixed_point_register_type(4, 2)),
    )
    example = RegisterExample(
        parallel_example, lower_fixture, type_edges=(types[case],)
    )
    result = check_example(example, "qtyping")
    assert result.status is expected
    exception = (
        pytest.fail.Exception if expected is CheckStatus.FAIL else pytest.skip.Exception
    )
    with pytest.raises(exception):
        apply_pytest_result("qtyping", result)


def test_factory_availability_and_independent_count_failures() -> None:
    assert (
        check_example(RegisterExample(lambda: "CNOT 0 1"), "make").status
        is CheckStatus.FAIL
    )
    assert (
        check_example(
            RegisterExample(parallel_example, atomic=True), "decompose"
        ).status
        is CheckStatus.NA
    )
    assert (
        check_example(RegisterExample(parallel_example), "decompose").status
        is CheckStatus.MISSING
    )
    assert (
        check_example(RegisterExample(parallel_example), "counts").status
        is CheckStatus.MISSING
    )
    assert (
        check_example(RegisterExample(parallel_example, lower_fixture), "counts").status
        is CheckStatus.UNVERIFIED
    )
    wrong = RegisterExample(parallel_example, lower_fixture, lambda: {"x": 2})
    result = check_example(wrong, "counts")
    assert result.status is CheckStatus.FAIL
    with pytest.raises(pytest.fail.Exception, match="Declared instruction counts"):
        apply_pytest_result("counts", result)


@pytest.mark.ftqc_reference
def test_source_pytest_dispatch_and_known_typing_defect(
    source_testing: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    conftest, testing = source_testing
    assert tuple(name for name, _ in conftest._TESTFUNCS) == CHECKS
    for name in ("gf16_inverse", "ec_add", "guided_hamiltonian_symb"):
        with pytest.raises(pytest.xfail.Exception):
            conftest.assert_bloq_example_serializes_for_pytest(
                SimpleNamespace(name=name)
            )

    def fail(example: Any) -> None:
        raise testing.BloqCheckException.fail("bad typed edge")

    monkeypatch.setattr(testing, "assert_bloq_example_qtyping", fail)
    assert conftest.assert_bloq_example_qtyping_for_pytest(object()) is None
    with pytest.raises(pytest.fail.Exception, match="bad typed edge"):
        apply_pytest_result("qtyping", CheckResult(CheckStatus.FAIL, "bad typed edge"))


@pytest.mark.ftqc_reference
def test_original_signed_classical_action_grid_and_zero_work(
    source_testing: tuple[Any, Any],
) -> None:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = Add(QInt(3))
    ir = CircuitIR(8, add_ir((0, 1, 2), (3, 4, 5), (6, 7), (0, 1)))

    def signed(word: int) -> int:
        return word if word < 4 else word - 8

    def declared(a: int, b: int) -> tuple[Any, int]:
        return register_add_classical_values(a, b, 3, signed=True)

    def decomposed(a: int, b: int) -> tuple[int, int]:
        basis = ((a % 8) << 3 | (b % 8)) << 2
        output = execute_sparse_register(ir, {basis: 1}, seed=27)
        result, amplitude = max(output.items(), key=lambda item: abs(item[1]))
        assert result & 3 == 0
        assert amplitude == pytest.approx(1, abs=2e-12)
        assert all(
            abs(value) < 2e-12 for word, value in output.items() if word != result
        )
        return signed(result >> 5), signed((result >> 2) & 7)

    values = range(-4, 4)
    assert_classical_action(declared, decomposed, a=values, b=values)
    assert_classical_action(source.call_classically, decomposed, a=values, b=values)
    with pytest.raises(AssertionError):
        assert_classical_action(lambda a, b: (a, b), decomposed, a=values, b=values)

    counts = Counter(g.name for g in ir.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement == 2
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq == 8
    state = torch.zeros(256, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for a, b in product(values, repeat=2):
        _, target = declared(a, b)
        phase = torch.tensor(complex(a + 5, b - 7), dtype=state.dtype)
        state[((a % 8) << 3 | (b % 8)) << 2] = phase
        expected[((a % 8) << 3 | (target % 8)) << 2] = phase
    norm = state.norm()
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(state / norm, ir.instructions, history),
            expected / (2 * norm),
            rtol=0,
            atol=2e-12,
        )
