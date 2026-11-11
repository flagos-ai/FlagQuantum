"""Truth table enumeration and text retain the unchanged source call chain."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.bit_encoding import bit_value_domain
from flagquantum.simulation.bounded_integer_values import bounded_integer_domain
from flagquantum.simulation.classical_truth_table import (
    TruthTable,
    classical_truth_table,
    format_classical_truth_table,
)
from flagquantum.simulation.fixed_point_encoding import fixed_point_domain
from flagquantum.simulation.montgomery_values import montgomery_value_domain
from flagquantum.simulation.ones_complement_encoding import ones_complement_domain
from tests.unit.test_classical_action_tensor import signature_from_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_adapter(
    signature: Signature,
    domain: Callable[[Register], Iterable[object]],
    evaluate: Callable[[dict[str, object]], Sequence[object]],
) -> Any:
    """Convert only metadata and explicit callbacks at the optional test boundary."""
    registers = tuple(
        SimpleNamespace(
            name=register.name,
            shape=register.shape,
            dtype=SimpleNamespace(get_classical_domain=lambda r=register: domain(r)),
            side=register.side,
        )
        for register in signature
    )
    return SimpleNamespace(
        signature=SimpleNamespace(
            lefts=lambda: (r for r in registers if r.side & RegisterSide.LEFT),
            rights=lambda: (r for r in registers if r.side & RegisterSide.RIGHT),
        ),
        call_classically=lambda **values: evaluate(values),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "bit",
        "cbit",
        "uint",
        "signed",
        "bounded",
        "fixed",
        "signed_fixed",
        "ones",
        "montgomery",
    ],
)
def test_existing_value_domains_match_source_table_and_text(kind: str) -> None:
    from qualtran import (
        BQUInt,
        CBit,
        QBit,
        QFxp,
        QInt,
        QIntOnesComp,
        QMontgomeryUInt,
        QUInt,
    )
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table as source_format,
    )
    from qualtran.simulation.classical_sim import get_classical_truth_table

    types: dict[str, Any] = {
        "bit": QBit(),
        "cbit": CBit(),
        "uint": QUInt(3),
        "signed": QInt(3),
        "bounded": BQUInt(3, 5),
        "fixed": QFxp(3, 2),
        "signed_fixed": QFxp(3, 1, signed=True),
        "ones": QIntOnesComp(3),
        "montgomery": QMontgomeryUInt(3, 5),
    }
    dtype = types[kind]
    source_signature = SourceSignature([SourceRegister("word", dtype)])
    signature = signature_from_source(source_signature)
    domains: dict[str, Callable[[], Iterable[object]]] = {
        "bit": bit_value_domain,
        "cbit": bit_value_domain,
        "uint": lambda: range(8),
        "signed": lambda: range(-4, 4),
        "bounded": lambda: bounded_integer_domain(3, 5),
        "fixed": lambda: fixed_point_domain(3, 2),
        "signed_fixed": lambda: fixed_point_domain(3, 1, signed=True),
        "ones": lambda: ones_complement_domain(3),
        "montgomery": lambda: montgomery_value_domain(3, 5),
    }

    def evaluate(values: dict[str, object]) -> Sequence[object]:
        return (values["word"],)

    source = SimpleNamespace(
        signature=source_signature, call_classically=lambda **values: evaluate(values)
    )
    expected = get_classical_truth_table(source)
    actual = classical_truth_table(
        signature, lambda register: domains[kind](), evaluate
    )
    assert actual == expected
    assert format_classical_truth_table(*actual) == source_format(*expected)


@pytest.mark.parametrize(
    "domains",
    [((2, 0, 2), (1, 0)), ((), (1, 2)), ((0, 1), ()), ((np.int8(1),), (np.uint16(3),))],
)
def test_cartesian_order_duplicates_empty_domains_and_output_identity(
    domains: tuple[Sequence[object], Sequence[object]],
) -> None:
    from qualtran.simulation.classical_sim import get_classical_truth_table

    signature = Signature.build(a=2, b=2)
    outputs: list[Sequence[object]] = []
    trace: list[object] = []

    def domain(register: Register) -> Iterable[object]:
        trace.append(("domain", register.name))
        return domains[0 if register.name == "a" else 1]

    def evaluate(values: dict[str, object]) -> Sequence[object]:
        trace.append(tuple(values.items()))
        result = [values["b"], values["a"]]
        outputs.append(result)
        return result

    actual = classical_truth_table(signature, domain, evaluate)
    native_trace = list(trace)
    native_outputs = list(outputs)
    outputs.clear()
    trace.clear()
    expected = get_classical_truth_table(source_adapter(signature, domain, evaluate))
    assert actual == expected and trace == native_trace
    assert native_trace[:2] == [("domain", "a"), ("domain", "b")]
    assert len(actual[2]) == len(domains[0]) * len(domains[1])
    for row, output in zip(actual[2], native_outputs, strict=True):
        assert row[1] is output
    assert signature.get_left("a").shape == ()


@pytest.mark.parametrize("shape", [(1,), (0,), (1, 0), (2, 2)])
def test_all_input_shapes_are_refused_before_any_domain(shape: tuple[int, ...]) -> None:
    from qualtran.simulation.classical_sim import get_classical_truth_table

    bit = next(iter(Signature.build(x=1))).dtype
    signature = Signature((Register("scalar", bit), Register("array", bit, shape)))
    for action in (
        lambda: classical_truth_table(
            signature,
            lambda r: pytest.fail("domain"),
            lambda v: pytest.fail("evaluate"),
        ),
        lambda: get_classical_truth_table(
            source_adapter(
                signature,
                lambda r: pytest.fail("domain"),
                lambda v: pytest.fail("evaluate"),
            )
        ),
    ):
        with pytest.raises(NotImplementedError) as error:
            action()
        assert str(error.value) == ""


@pytest.mark.parametrize(
    "mode", ["empty", "prepare", "discard", "paired_names", "shaped_output"]
)
def test_directional_ports_and_empty_input_cartesian_identity(mode: str) -> None:
    from qualtran.simulation.classical_sim import get_classical_truth_table

    bit = next(iter(Signature.build(x=1))).dtype
    registers = {
        "empty": (),
        "prepare": (Register("x", bit, side=RegisterSide.RIGHT),),
        "discard": (Register("x", bit, side=RegisterSide.LEFT),),
        "paired_names": (
            Register("x", bit, side=RegisterSide.RIGHT),
            Register("x", bit, side=RegisterSide.LEFT),
        ),
        "shaped_output": (Register("out", bit, (2,), RegisterSide.RIGHT),),
    }
    signature = Signature(registers[mode])
    output: Sequence[object] = (
        [np.array([0, 1])] if mode == "shaped_output" else ["raw", None]
    )
    actual = classical_truth_table(
        signature, lambda r: bit_value_domain(), lambda v: output
    )
    expected = get_classical_truth_table(
        source_adapter(signature, lambda r: bit_value_domain(), lambda v: output)
    )
    assert actual[:2] == expected[:2]
    assert [row[0] for row in actual[2]] == [row[0] for row in expected[2]]
    assert all(row[1] is output for row in actual[2])
    assert len(actual[2]) == (2 if tuple(signature.lefts()) else 1)


@pytest.mark.parametrize("failure", ["domain", "iteration", "evaluate"])
def test_provider_errors_propagate_with_the_same_trace(failure: str) -> None:
    from qualtran.simulation.classical_sim import get_classical_truth_table

    signature = Signature.build(a=1, b=1)
    trace: list[object] = []
    error = RuntimeError("sentinel")

    def values(register: Register) -> Iterable[object]:
        trace.append(("iterate", register.name))
        yield 0
        if failure == "iteration":
            raise error
        yield 1

    def domain(register: Register) -> Iterable[object]:
        trace.append(("domain", register.name))
        if failure == "domain":
            raise error
        return values(register)

    def evaluate(inputs: dict[str, object]) -> Sequence[object]:
        trace.append(tuple(inputs.items()))
        raise error

    expected: list[object] | None = None
    for run in (
        lambda: classical_truth_table(signature, domain, evaluate),
        lambda: get_classical_truth_table(source_adapter(signature, domain, evaluate)),
    ):
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is error
        if expected is None:
            expected = list(trace)
        else:
            assert trace == expected
        trace.clear()


@pytest.mark.parametrize(
    "names",
    [
        ((), ()),
        (("a",), ("b",)),
        (("long input", "β"), ("same", "same")),
        (("a\nb",), ("",)),
    ],
)
@pytest.mark.parametrize(
    "rows", [[], [((), ())], [((0, -1, 1j), (None, np.uint8(3), np.array([1, 2])))]]
)
def test_formatting_exact_separator_newline_and_raw_objects(
    names: tuple[Sequence[str], Sequence[str]],
    rows: TruthTable,
) -> None:
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table as source_format,
    )

    actual = format_classical_truth_table(*names, rows)
    assert actual == source_format(*names, rows)
    assert actual.startswith("  ".join(names[0]) + "  |  " + "  ".join(names[1]) + "\n")


def test_table_evaluator_uses_native_graph_and_complete_measured_branches() -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import get_classical_truth_table

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.simulation.component_classical_state import ComponentClassicalState
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_classical_state import add_values, order
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add

    record, source = wrapped_add(1)
    signature = Signature(record.functions[0].registers or ())

    def evaluate(values: dict[str, object]) -> Sequence[object]:
        state = ComponentClassicalState(record, order(record), values, add_values)
        outputs = state.simulate()
        return tuple(outputs[r.name] for r in signature.rights())

    actual = classical_truth_table(
        signature, lambda r: range(2 ** int(r.bitsize)), evaluate
    )
    assert actual == get_classical_truth_table(source)

    def integer(value: object) -> int:
        assert isinstance(value, (int, np.integer))
        return int(value)

    matrix = torch.zeros((64, 64), dtype=torch.complex128)
    for inputs, outputs in actual[2]:
        assert all(isinstance(v, (int, np.integer)) for v in (*inputs, *outputs))
        a, b, spectator = map(integer, inputs)
        x, y, z = map(integer, outputs)
        matrix[(x << 4) | (y << 1) | z, (a << 4) | (b << 1) | spectator] = 1
    torch.testing.assert_close(
        matrix, torch.from_numpy(source.tensor_contract()), atol=2e-14, rtol=0
    )
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    initial = torch.zeros(256, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    initial[::4] = amplitudes / amplitudes.norm()
    expected = torch.zeros_like(initial)
    expected[::4] = matrix @ initial[::4]
    for history in product((0, 1), repeat=4):
        branch = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(branch, expected / 4, atol=1e-12, rtol=0)
        assert torch.count_nonzero(branch.reshape(-1, 4)[:, 1:]) == 0
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "h": 12,
        "t": 8,
        "tdg": 8,
        "cx": 40,
        "s": 4,
        "measure": 4,
        "cz": 4,
        "reset": 4,
    }
    model = get_cost_value(source, QECGatesCost())
    assert model.and_bloq == 4 and model.clifford == 22 and model.measurement == 4
