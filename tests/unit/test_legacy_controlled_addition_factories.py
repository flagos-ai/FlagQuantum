"""Complete controlled-addition constructor closure uses the accepted typed owner."""

from collections import Counter
from itertools import product

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_arithmetic_factories import (
    arithmetic_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import ResolvedArgument
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _resolve
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_typed_addition_component import integer_types

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 5), (8, 8)))
@pytest.mark.parametrize("cv", (0, 1))
def test_named_and_direct_decomposition_match_source_gates_feedback_and_model(
    kind: str, left: int, right: int, cv: int
) -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    _, _, a, b = integer_types(kind, left, right)
    source = CAdd(a, b, cv)
    logical = 1 + left + right
    order = tuple(reversed(range(logical)))
    ports = {"ctrl": order[:1], "a": order[1 : left + 1], "b": order[left + 1 :]}
    work = tuple(range(logical, logical + left + right - 1))
    for original in (source, source.decompose_bloq()):
        bound = bind_register_program(
            _resolve(original), tuple(ports.values()), clean=work
        )
        slots = {
            g.wires[0]: int(g.metadata["classical_bit"])
            for g in bound.circuit.instructions
            if g.name == "measure"
        }
        expected = reference_arithmetic_gates(source, ports, work, slots)
        assert_source(bound.circuit.instructions, expected)
        assert bound.outputs == tuple(ports.values())
        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(source, QECGatesCost())
        assert (
            counts["t"] + counts["tdg"]
            == model.total_t_count()
            == 4 * (left + right - 1)
        )
        assert (
            counts["measure"]
            == counts["reset"]
            == model.measurement
            == left + right - 1
        )
        assert counts["x"] == 4 * left * (1 - cv)
        assert get_cost_value(source, QubitCount()) == logical + len(work)


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("cv", (0, 1))
def test_all_raw_branches_full_complex_spectator_clean_work_and_append(
    kind: str, left: int, right: int, cv: int
) -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    _, _, a, b = integer_types(kind, left, right)
    source = CAdd(a, b, cv)
    logical, scratch = 1 + left + right, left + right - 1
    ports = ((0,), tuple(range(1, left + 1)), tuple(range(left + 1, logical)))
    work = tuple(range(logical + 1, logical + 1 + scratch))
    bound = bind_register_program(_resolve(source), ports, clean=work)
    coeff = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coeff, 7 - 3 * coeff)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + scratch), dtype=torch.complex128)
    initial[:: 1 << scratch] = amplitudes
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << scratch] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for history in product((0, 1), repeat=scratch):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (scratch / 2), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << scratch)[:, 1:]) == 0
    for precision in (torch.complex64, torch.complex128):
        circuit = DynamicCircuit(
            logical + 1 + scratch, inputs=initial.to(precision), dtype=precision
        )
        circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=4, seed=37, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(4, -1),
            expected.to(precision).expand(4, -1),
            atol=2e-6 if precision == torch.complex64 else 3e-12,
            rtol=0,
        )


@pytest.mark.parametrize("cv", (False, True, 0.0, 1.0))
def test_source_numeric_control_transport_matches_integer_dependency(
    cv: bool | float,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    for original in (CAdd(QUInt(2), cv=cv), CAdd(QUInt(2), cv=cv).decompose_bloq()):
        bound = bind_register_program(
            _resolve(original), ((0,), (1, 2), (3, 4)), clean=(5, 6, 7)
        )
        reference = bind_register_program(
            _resolve(CAdd(QUInt(2), cv=int(cv))),
            ((0,), (1, 2), (3, 4)),
            clean=(5, 6, 7),
        )
        assert_source(bound.circuit.instructions, reference.circuit.instructions)


def test_defaults_symbolic_declaration_and_large_original_factory() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import (
        CAdd,
        _cadd_large,
        _cadd_small,
    )

    dtype = unsigned_register_type(2)
    factory = arithmetic_factories(encode_constant=lambda dtype, value: ())[
        "qualtran.bloqs.arithmetic.controlled_addition.CAdd"
    ]
    default = factory("custom", None, {"a_dtype": dtype})
    assert isinstance(default, ProgramRecord)
    assert default.functions[0].registers is not None
    assert (
        default.functions[0].registers[1].dtype
        is default.functions[0].registers[2].dtype
    )
    assert (
        factory("custom", default.functions[0].registers, {"a_dtype": dtype}) == default
    )
    n = sympy.Symbol("n", integer=True, positive=True)
    for source in (CAdd(QUInt(n)), CAdd(QUInt(1), QUInt(n))):
        for depth in (0, 1, 2):
            symbolic = _resolve(source, depth=depth)
            assert symbolic.functions[0].blocks == ()
            with pytest.raises(ValueError, match="component bodies"):
                bind_register_program(symbolic, ())
    for maker, width in ((_cadd_small, 3), (_cadd_large, 1000)):
        program = _resolve(maker.make(), depth=0)
        ports = (
            (0,),
            tuple(range(1, width + 1)),
            tuple(range(width + 1, 2 * width + 1)),
        )
        work = tuple(range(2 * width + 1, 4 * width))
        bound = bind_register_program(program, ports, clean=work)
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts["t"] + counts["tdg"] == 4 * (2 * width - 1)
        assert counts["measure"] == counts["reset"] == 2 * width - 1


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"a_dtype": unsigned_register_type(2), "extra": 1},
        {"a_dtype": 2},
        {"a_dtype": opaque_register_type(2)},
        {"a_dtype": unsigned_register_type(2), "b_dtype": "invalid"},
        {"a_dtype": unsigned_register_type(2), "b_dtype": unsigned_register_type(1)},
        *(
            {"a_dtype": unsigned_register_type(2), "cv": cv}
            for cv in (2, -1, 0.5, "1", float("nan"), float("inf"))
        ),
    ),
)
def test_malformed_constructor_metadata_refuses(
    arguments: dict[str, ResolvedArgument],
) -> None:
    factory = arithmetic_factories(encode_constant=lambda dtype, value: ())[
        "qualtran.bloqs.arithmetic.controlled_addition.CAdd"
    ]
    with pytest.raises(SerializationError):
        factory("bad", None, arguments)


def test_signature_mismatch_refuses() -> None:
    factory = arithmetic_factories(encode_constant=lambda dtype, value: ())[
        "qualtran.bloqs.arithmetic.controlled_addition.CAdd"
    ]
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"a_dtype": unsigned_register_type(2)})
