"""Literal nested constant controls preserve phase, feedback and private work."""

import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    import torch

    import_module("cirq")
    import_module("kickmix")
    from qualtran import (
        CtrlSpec,
        DecomposeNotImplementedError,
        DecomposeTypeError,
        QInt,
        QMontgomeryUInt,
        QUInt,
    )
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_constant_control_factories import (
    constant_control_factories,
)
from flagquantum.compiler.registers.arithmetic.constant_addition_component import (
    constant_addition_program,
)
from flagquantum.compiler.registers.arithmetic.nested_controlled_constant_component import (
    nested_controlled_constant_addition_program,
)
from flagquantum.compiler.registers.component_control import quantum_control_signature
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_modular_addition_component import bind_case
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources() -> None:
    for name, digest in (
        ("qualtran._infra.controlled", "6fabe760e793f1b39f825d76f80e92b6e5fa5c28"),
        ("qualtran._infra.adjoint", "0601a6248eec488301b31dd7c800a6ba14a7fb27"),
        (
            "qualtran.bloqs.arithmetic.addition",
            "33df3f7e021900acd65181baed2abf4d0264b248",
        ),
        (
            "qualtran.bloqs.arithmetic.bitwise",
            "24835990709339b96cde9b2302a2525c09345ab5",
        ),
    ):
        module = import_module(name)
        assert module.__file__ is not None
        raw = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    kickmix = import_module("kickmix")
    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")


@pytest.mark.parametrize(
    "kind,width,constant,control_count",
    (
        ("uint", 2, 1, 2),
        ("uint", 3, 3, 2),
        ("uint", 3, 0, 2),
        ("signed", 3, -3, 2),
        ("montgomery", 3, 3, 2),
        ("uint", 4, 5, 2),
        ("uint", 2, 1, 3),
        ("uint", 3, 0, 3),
        ("signed", 2, -2, 3),
        ("montgomery", 2, 1, 3),
        ("uint", 1, 1, 4),
        ("uint", 1, 1, 5),
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_source_calls_full_primitive_stream_all_complex_branches_and_resources(
    kind: str, width: int, constant: int, control_count: int, adjoint: bool
) -> None:
    dtype = RegisterType(kind, width)
    source_type: Any = {"uint": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[
        kind
    ]
    source = AddK(source_type(width), constant)
    source = source.adjoint() if adjoint else source
    for _ in range(control_count):
        source = source.controlled()
    pattern = constant % (1 << width)
    record = nested_controlled_constant_addition_program(
        dtype, pattern, adjoint=adjoint, control_count=control_count
    )
    assert len(record.functions[0].blocks[0].operations) == 5
    assert [r.name for r in record.functions[0].registers or ()] == (
        ["ctrl2", "ctrl", "x"]
        if control_count == 2
        else [*(f"ctrl{i}" for i in range(control_count, 1, -1)), "ctrl", "x"]
    )
    terminals = {
        f.name
        for f in record.functions
        if not f.blocks
        or any(o.opcode != "call" for b in f.blocks for o in b.operations)
        or any(r.side is not RegisterSide.THRU for r in f.registers or ())
    }
    # Reuse the complete Always(Add/Adjoint(Add)) owner. Its exact adjoint may
    # schedule independent Split(a)/Split(b) differently from source creation
    # order; the full primitive dependency stream is checked below.
    base = constant_addition_program(dtype, pattern, adjoint=adjoint, control_value=1)
    assert matches_component_definition(
        record, "constant_always", base_for_always(base)
    )
    terminals.add("constant_always")
    pairs = compare_layers(record, source, terminals)
    for name, child in pairs.items():
        if name == "constant_always":
            assert type(child).__name__ == "Always"
        elif type(child).__name__ == "CtrlSpecAnd" or (
            type(child).__name__ == "Adjoint"
            and type(child.subbloq).__name__ == "CtrlSpecAnd"
        ):
            child.decompose_bloq()
        elif name in terminals:
            with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
                child.decompose_bloq()
    logical = width + control_count
    inputs = (
        *((i,) for i in range(control_count)),
        tuple(range(control_count, logical)),
    )
    bound = bind_register_program(
        record,
        inputs,
        clean=tuple(
            range(logical, logical + width + max(width - 1, control_count - 1))
        ),
    )
    registers = record.functions[0].registers
    assert registers is not None
    ports = {r.name: wires for r, wires in zip(registers, inputs, strict=True)}
    allocations = tuple(
        w for event in bound.events if event.kind == "allocate" for w in event.wires
    )
    expected_gates = reference_arithmetic_gates(
        source,
        ports,
        allocations,
        MeasurementSlots(bound.circuit.instructions),
        reuse_allocations=False,
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected_gates)
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    if control_count == 2:
        assert counts["ccx"] == model.toffoli == 2 * pattern.bit_count()
    else:
        assert counts["ccx"] == model.toffoli == 0
    expected_measurements = (
        width
        - 1
        + (2 * (control_count - 1) * pattern.bit_count() if control_count >= 3 else 0)
    )
    assert (
        counts["measure"]
        == counts["reset"]
        == model.measurement
        == expected_measurements
    )
    assert counts["t"] + counts["tdg"] + 4 * counts["ccx"] == model.total_t_count()
    physical = max(w for g in bound.circuit.instructions for w in g.wires) + 1
    assert physical == get_cost_value(source, QubitCount())
    assert physical == logical + width + max(
        width - 1, control_count - 1 if control_count >= 3 and pattern else 0
    )
    work = physical - logical
    matrix = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    values = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, 3 - 2 * values)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << physical, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << work] = amplitudes
    expected[:: 1 << work] = matrix @ amplitudes
    measured = counts["measure"]
    for history in product((0, 1), repeat=measured):
        state = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            state, expected / 2 ** (measured / 2), rtol=0, atol=3e-12
        )
    circuit = DynamicCircuit(physical, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(physical, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=3e-12
    )


def base_for_always(record: ProgramRecord) -> ProgramRecord:
    """Select the complete existing Always owner as a definition certificate."""
    owner = next(f for f in record.functions if f.name == "constant_always")
    return replace(
        record, functions=(owner, *(f for f in record.functions if f is not owner))
    )


@pytest.mark.parametrize("controls", (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_constructor_metadata_roundtrip_and_reference_free_resolution(
    controls: int, adjoint: bool, depth: int
) -> None:
    from flagquantum.ecosystem._component_library import _constant_bits

    source = AddK(QUInt(3), -1)
    source = source.adjoint() if adjoint else source
    for _ in range(controls):
        source = source.controlled()
    encoded = serialize_component_library(_library(source, depth))
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded

    def resolve() -> ProgramRecord:
        base = _factories()
        mapping = {
            **base,
            **constant_control_factories(
                encode_constant=_constant_bits,
                controlled_factory=base.get("qualtran._infra.controlled.Controlled"),
                adjoint_factory=base.get("qualtran._infra.adjoint.Adjoint"),
            ),
        }
        return resolve_component_program(library, 0, mapping)

    actual = resolve()
    dtype = RegisterType("uint", 3)
    expected = (
        nested_controlled_constant_addition_program(
            dtype, 7, adjoint=adjoint, control_count=controls
        )
        if controls >= 2
        else constant_addition_program(
            dtype, 7, adjoint=adjoint, control_value=1 if controls else None
        )
    )
    assert bind_case(actual)[0] == bind_case(expected)[0]
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert resolve() == actual


@pytest.mark.parametrize("control_count", (2, 3))
def test_symbolic_constant_is_never_encoded_or_silently_executed(
    control_count: int,
) -> None:
    import sympy

    def forbidden(dtype: RegisterType, value: object) -> tuple[int, ...]:
        raise AssertionError("An unresolved source constant has no numeric encoding")

    source = AddK(QUInt(3), sympy.Symbol("k"))
    for _ in range(control_count):
        source = source.controlled()
    record = resolve_component_program(
        _library(source, 0), 0, constant_control_factories(encode_constant=forbidden)
    )
    assert not record.functions[0].blocks
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(
            record,
            (
                *((i,) for i in range(control_count)),
                tuple(range(control_count, control_count + 3)),
            ),
        )


def test_scoped_constructor_requires_provenance_and_valid_control_metadata() -> None:
    from flagquantum.ecosystem._component_library import _constant_bits

    mapping = constant_control_factories(encode_constant=_constant_bits)
    forward = mapping["qualtran.bloqs.arithmetic.addition.AddK"](
        "word", None, {"dtype": RegisterType("uint", 3), "k": 3}
    )
    assert isinstance(forward, ProgramRecord)
    child = forward.functions[0]
    control = mapping["qualtran._infra.controlled.Controlled"]
    spec = ControlSpecification(((quantum_bit_type(), np.asarray(1)),))
    with pytest.raises(SerializationError, match="context"):
        control(
            "main", None, {"subbloq": replace(child, name="unknown"), "ctrl_spec": spec}
        )
    with pytest.raises(SerializationError, match="signature"):
        control("main", (), {"subbloq": child, "ctrl_spec": spec})
    for values in (np.asarray(0), np.asarray([1]), np.asarray(1.5)):
        with pytest.raises(SerializationError, match="positive QBit"):
            control(
                "main",
                None,
                {
                    "subbloq": child,
                    "ctrl_spec": ControlSpecification(((quantum_bit_type(), values),)),
                },
            )
    alien = FunctionRecord("alien", (), registers=child.registers)
    marker = FunctionRecord(
        "owned_elsewhere",
        (),
        registers=tuple(
            quantum_control_signature(
                Signature(child.registers or ()), ((quantum_bit_type(), ()),)
            )
        ),
    )
    calls: list[FunctionRecord] = []

    def fallback(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        original = arguments["subbloq"]
        assert isinstance(original, FunctionRecord)
        calls.append(original)
        return marker

    other = constant_control_factories(
        encode_constant=_constant_bits, controlled_factory=fallback
    )
    assert (
        other["qualtran._infra.controlled.Controlled"](
            "main", None, {"subbloq": alien, "ctrl_spec": spec}
        )
        is marker
    )
    assert calls == [alien]


@pytest.mark.parametrize("mode", ("negative", "negative_nested", "grouped", "packed"))
def test_predicate_constructor_keeps_negative_and_grouped_source_recipes(
    mode: str,
) -> None:
    from flagquantum.ecosystem._component_library import _constant_bits
    from tests.unit.test_chebyshev_component import assert_all_small_raw_histories

    base = AddK(QUInt(3), 3)
    if mode == "negative_nested":
        base = base.controlled()
    spec = (
        CtrlSpec(cvs=np.asarray([1, 0]))
        if mode == "grouped"
        else CtrlSpec(qdtypes=QUInt(2), cvs=2) if mode == "packed" else CtrlSpec(cvs=0)
    )
    source = base.controlled(ctrl_spec=spec)
    library = _library(source, 0)
    record = resolve_component_program(
        library, 0, constant_control_factories(encode_constant=_constant_bits)
    )
    registers = record.functions[0].registers
    assert registers is not None
    inputs: list[tuple[int, ...]] = []
    ports: dict[str, tuple[int, ...]] = {}
    logical = 0
    for register in registers:
        wires: list[int] = []
        for _ in register.all_idxs():
            word = tuple(range(logical, logical + int(register.bitsize)))
            inputs.append(word)
            wires.extend(word)
            logical += len(word)
        ports[register.name] = tuple(wires)
    private = 6 if mode in {"grouped", "packed"} else 5
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(logical, logical + private))
    )
    assert len(record.functions[0].blocks[0].operations) == 3
    expected = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        ports,
        tuple(w for e in bound.events if e.kind == "allocate" for w in e.wires),
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
        reuse_allocations=False,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected)
    operator = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    assert_all_small_raw_histories(bound.circuit, logical, operator)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["measure"] == model.measurement
    assert counts["t"] + counts["tdg"] + 4 * counts["ccx"] == model.total_t_count()
    assert bound.circuit.n_wires == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("depth", (0, 1))
@pytest.mark.parametrize("control_count", (3, 4, 5))
def test_recursive_xor_constructor_roundtrip_uses_the_shared_word_owner(
    depth: int, control_count: int
) -> None:
    from qualtran.bloqs.arithmetic.bitwise import XorK

    from flagquantum.compiler.registers.bits.bitwise_component_control import (
        nested_controlled_xor_constant_program,
    )
    from flagquantum.ecosystem._component_library import _constant_bits

    source = XorK(QUInt(2), 1)
    for _ in range(control_count):
        source = source.controlled()
    encoded = serialize_component_library(_library(source, depth))
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        actual = resolve_component_program(
            library, 0, constant_control_factories(encode_constant=_constant_bits)
        )
    expected = nested_controlled_xor_constant_program(
        RegisterType("uint", 2), 1, control_count=control_count
    )
    assert bind_case(actual)[0] == bind_case(expected)[0]
