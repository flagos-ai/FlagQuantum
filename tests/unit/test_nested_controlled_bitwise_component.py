"""Nested XOR controls keep the literal Toffoli view and complete operators."""

from collections import Counter
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    import torch

    import_module("cirq")
    import_module("kickmix")
    from qualtran import DecomposeTypeError, QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.registers.bits.bitwise_component_control import (
    nested_controlled_xor_constant_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterSide, RegisterType
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_word_source() -> None:
    for name, digest in (
        ("qualtran._infra.controlled", "6fabe760e793f1b39f825d76f80e92b6e5fa5c28"),
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
    "kind,width,value",
    (
        ("uint", 1, 1),
        ("uint", 3, 0),
        ("uint", 3, 3),
        ("uint", 4, 5),
        ("signed", 3, -3),
        ("montgomery", 3, 3),
    ),
)
def test_complete_source_word_calls_primitives_and_full_complex_operator(
    kind: str, width: int, value: int
) -> None:
    source_type: Any = {"uint": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[
        kind
    ]
    source = XorK(source_type(width), value).controlled().controlled()
    pattern = value % (1 << width)
    record = nested_controlled_xor_constant_program(RegisterType(kind, width), pattern)
    terminals = {
        f.name
        for f in record.functions
        if any(o.opcode != "call" for b in f.blocks for o in b.operations)
        or any(r.side is not RegisterSide.THRU for r in f.registers or ())
    }
    pairs = compare_layers(record, source, terminals)
    for name, child in pairs.items():
        if name in terminals:
            with pytest.raises(DecomposeTypeError):
                child.decompose_bloq()
    inputs = ((0,), (1,), tuple(range(2, width + 2)))
    bound = bind_register_program(record, inputs)
    registers = record.functions[0].registers
    assert registers is not None
    ports = {r.name: wires for r, wires in zip(registers, inputs, strict=True)}
    logical = width + 2
    assert bound.circuit.n_wires == logical
    assert not bound.events
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        ccx=pattern.bit_count()
    )
    expected = reference_arithmetic_gates(
        source,
        ports,
        (),
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected)
    operator = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), operator, rtol=0, atol=2e-12
    )
    model = get_cost_value(source, QECGatesCost())
    assert (
        model.toffoli == pattern.bit_count()
        and model.measurement == model.and_bloq == 0
    )
    assert get_cost_value(source, QubitCount()) == width + 2
