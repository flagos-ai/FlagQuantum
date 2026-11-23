"""Complete partition constructors retain source storage views and zero gates."""

from importlib import import_module
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_partition_factories import partition_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolve_partition(source: Any, depth: int = 1) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, depth), 0, partition_factories()
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("widths", ((0, 0), (0, 3), (3, 0), (1, 2), (2, 3), (8, 8)))
@pytest.mark.parametrize("reverse", (False, True))
def test_pair_constructors_source_order_no_primitives_and_complex_append(
    widths: tuple[int, int], reverse: bool
) -> None:
    from qualtran.bloqs.bookkeeping import Join2, Split2
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    np = pytest.importorskip("numpy")
    source = (Join2 if reverse else Split2)(*widths)
    wires = tuple(reversed(range(sum(widths))))
    inputs: tuple[tuple[int, ...], ...] = (
        (wires[: widths[0]], wires[widths[0] :]) if reverse else (wires,)
    )
    ports = (
        {"y1": np.asarray(inputs[0]), "y2": np.asarray(inputs[1])}
        if reverse
        else {"x": np.asarray(wires)}
    )
    _, views = source.as_cirq_op(None, **ports)
    expected = tuple(tuple(int(w) for w in word.flat) for word in views.values())
    for depth in (0, 1, 2):
        bound = bind_register_program(
            resolve_partition(source, depth), inputs, clean=(0,) if not wires else ()
        )
        assert bound.outputs == expected
        assert not bound.events and not bound.circuit.instructions
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == sum(widths)
        if sum(widths) > 6:
            continue
        for precision in (torch.complex64, torch.complex128):
            state = torch.randn(
                1 << (sum(widths) + 1),
                dtype=precision,
                generator=torch.Generator().manual_seed(52),
            )
            state /= state.norm()
            circuit = Circuit(sum(widths) + 1, inputs=state, dtype=precision)
            circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
            torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "encoding", ("QAny", "QUInt", "QInt", "QIntOnesComp", "BQUInt", "QMontgomeryUInt")
)
@pytest.mark.parametrize("reverse", (False, True))
def test_shaped_typed_partition_source_metadata_aliases_and_tensor(
    encoding: str, reverse: bool
) -> None:
    np = pytest.importorskip("numpy")
    source_module = import_module("qualtran")
    dtype = getattr(source_module, encoding)
    word = dtype(2, 3) if encoding in ("BQUInt", "QMontgomeryUInt") else dtype(2)
    parts = (
        source_module.Register("words", word, shape=(2,)),
        source_module.Register("tail", source_module.QAny(1)),
    )
    source = import_module("qualtran.bloqs.bookkeeping").Partition(
        5, parts, partition=not reverse
    )
    wires = (4, 3, 2, 1, 0)
    inputs = ((4, 3), (2, 1), (0,)) if reverse else (wires,)
    bound = bind_register_program(resolve_partition(source), inputs)
    assert bound.outputs == ((wires,) if reverse else ((4, 3), (2, 1), (0,)))
    assert not bound.events and not bound.circuit.instructions
    ports = (
        {"words": np.asarray(wires[:4]).reshape(2, 2), "tail": np.asarray(wires[4:])}
        if reverse
        else {"x": np.asarray(wires)}
    )
    _, views = source.as_cirq_op(None, **ports)
    assert bound.outputs == tuple(
        tuple(int(w) for w in row)
        for value in views.values()
        for row in value.reshape(-1, value.shape[-1])
    )
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        32, 32
    )
    torch.testing.assert_close(
        matrix, torch.eye(32, dtype=torch.complex128), rtol=0, atol=0
    )


@pytest.mark.parametrize("example", ("_partition", "_split2", "_join2"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_all_original_examples_logical_symbolic_disposition(
    example: str, depth: int
) -> None:
    source = getattr(
        import_module("qualtran.bloqs.bookkeeping.partition"), example
    ).make()
    program = resolve_partition(source, depth)
    assert (
        program.functions[0].registers
        == _library(source, depth).entries[0].declaration.registers
    )
    assert bool(program.functions[0].blocks) == (example == "_partition")


def test_partition_validation_defaults_truth_values_and_symbolic_shapes() -> None:
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping import Partition

    n = sympy.Symbol("n", integer=True, positive=True)
    source = Partition(n, (SourceRegister("part", QAny(1), shape=(n,)),))
    program = resolve_partition(source, 0)
    assert (
        program.functions[0].registers
        == _library(source, 0).entries[0].declaration.registers
    )
    assert not program.functions[0].blocks
    factories = partition_factories()
    part = Register("part", opaque_register_type(2))
    factory = factories["qualtran.bloqs.bookkeeping.partition.Partition"]
    result = factory("custom", None, {"n": 2, "regs": (part,)})
    assert isinstance(result, ProgramRecord)
    assert (
        factory("custom", result.functions[0].registers, {"n": 2, "regs": (part,)})
        == result
    )
    for truth in (0, 1, 2, 0.0, 1.5, "", "yes"):
        expected = Partition(2, (SourceRegister("part", QAny(2)),), partition=truth)
        assert (
            resolve_partition(expected, 0).functions[0].registers
            == _library(expected, 0).entries[0].declaration.registers
        )
    invalid: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"n": 2, "regs": "bad"},
        {"n": 2, "regs": ()},
        {"n": "bad", "regs": (part,)},
        {"n": 3, "regs": (part,)},
        {"n": 4, "regs": (part, part)},
    )
    for args in invalid:
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"n": 2, "regs": (part,)})
    for kind in ("Split2", "Join2"):
        factory = factories["qualtran.bloqs.bookkeeping.partition." + kind]
        with pytest.raises(SerializationError):
            factory("bad", None, {})
        symbolic = factory("custom", None, {"n1": n, "n2": 2})
        assert isinstance(symbolic, FunctionRecord) and not symbolic.blocks


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dimensions", [("n",), (2, "n"), ("n", 0, 3)])
def test_symbolic_partition_wire_roundtrip_and_binding_refusal(
    reverse: bool, dimensions: tuple[Any, ...]
) -> None:
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping import Partition

    n = sympy.Symbol("n", integer=True, positive=True)
    shape = tuple(n if value == "n" else value for value in dimensions)
    part = SourceRegister("part", QAny(1), shape=shape)
    source = Partition(part.total_bits(), (part,), partition=not reverse)
    for depth in (0, 1, 2):
        program = resolve_partition(source, depth)
        assert (
            program.functions[0].registers
            == _library(source, depth).entries[0].declaration.registers
        )
        assert not program.functions[0].blocks
        with pytest.raises(ValueError):
            bind_register_program(program, ())
