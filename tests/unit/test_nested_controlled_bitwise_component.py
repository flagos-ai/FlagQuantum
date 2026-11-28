"""Nested XOR controls keep the literal Toffoli view and complete operators."""

from collections import Counter
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from itertools import product
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
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_word_source() -> None:
    for name, digest in (
        (
            "qualtran.bloqs.basic_gates.toffoli",
            "a007be2ef25aa7dc5fda4d793498041a8f6b5a48",
        ),
        ("qualtran._infra.controlled", "6fabe760e793f1b39f825d76f80e92b6e5fa5c28"),
        (
            "qualtran.bloqs.mcmt.controlled_via_and",
            "3947ddf86be37f4ff3ba58fcc316a8a64922f84f",
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


@pytest.mark.parametrize("control_count", (2, 3))
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
    kind: str, width: int, value: int, control_count: int
) -> None:
    source_type: Any = {"uint": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[
        kind
    ]
    source = XorK(source_type(width), value)
    for _ in range(control_count):
        source = source.controlled()
    pattern = value % (1 << width)
    record = nested_controlled_xor_constant_program(
        RegisterType(kind, width), pattern, control_count=control_count
    )
    terminals = {
        f.name
        for f in record.functions
        if any(o.opcode != "call" for b in f.blocks for o in b.operations)
        or any(r.side is not RegisterSide.THRU for r in f.registers or ())
    }
    pairs = compare_layers(record, source, terminals)
    for name, child in pairs.items():
        if name in terminals:
            if type(child).__name__ in {"And", "MultiAnd", "CtrlSpecAnd"} or (
                type(child).__name__ == "Adjoint"
                and type(child.subbloq).__name__ == "CtrlSpecAnd"
            ):
                # These shared measured predicates have separate complete
                # primitive/branch certificates; they are not source refusals.
                child.decompose_bloq()
            else:
                with pytest.raises(DecomposeTypeError):
                    child.decompose_bloq()
    inputs = (
        *((i,) for i in range(control_count)),
        tuple(range(control_count, width + control_count)),
    )
    logical = width + control_count
    clean = tuple(range(logical, logical + 2)) if control_count == 3 else ()
    bound = bind_register_program(record, inputs, clean=clean)
    registers = record.functions[0].registers
    assert registers is not None
    ports = {r.name: wires for r, wires in zip(registers, inputs, strict=True)}
    actual = Counter(g.name for g in bound.circuit.instructions)
    if control_count == 2:
        assert bound.circuit.n_wires == logical
        assert not bound.events
        assert actual == Counter(ccx=pattern.bit_count())
    expected = reference_arithmetic_gates(
        source,
        ports,
        clean,
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected)
    operator = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    if control_count == 2 or not pattern:
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical), operator, rtol=0, atol=2e-12
        )
    else:
        work = bound.circuit.n_wires - logical
        coefficients = torch.tensor(
            [complex(2 * i - 3, i + 1) for i in range(1 << logical)],
            dtype=torch.complex128,
        )
        coefficients /= coefficients.norm()
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        initial[:: 1 << work] = coefficients
        measured = actual["measure"]
        assert measured == 2 * pattern.bit_count()
        expected_state = torch.zeros_like(initial)
        expected_state[:: 1 << work] = operator @ coefficients / (2 ** (measured / 2))
        for history in product((0, 1), repeat=measured):
            torch.testing.assert_close(
                forced_branch(initial, bound.circuit.instructions, history),
                expected_state,
                rtol=0,
                atol=2e-12,
            )
    model = get_cost_value(source, QECGatesCost())
    if control_count == 2:
        assert model.toffoli == pattern.bit_count()
        assert model.measurement == model.and_bloq == 0
        assert get_cost_value(source, QubitCount()) == width + 2
    else:
        assert model.toffoli == 0
        assert model.and_bloq == model.measurement == actual["measure"]
        assert get_cost_value(source, QubitCount()) >= logical
