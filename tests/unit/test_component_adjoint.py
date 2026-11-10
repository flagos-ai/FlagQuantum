"""Pinned graph adjoints retain specialized bodies and exact state phases."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_adjoint import adjoint_component_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_add_calls import executable_add
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def compare(record: ProgramRecord, source: Any) -> None:
    assert verify_program_record(record) == ()
    observer = _source(Signature([]))
    assert _native_trace(record) == _source_trace(observer, source)
    assert tuple(
        observer.signature(Signature(record.functions[0].registers or ()))
    ) == tuple(source.signature)


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_split_join_reversal_and_double_adjoint(
    width: int, shape: tuple[int, ...]
) -> None:
    signature = Signature([Register("word", RegisterType("uint", width), shape)])
    native = ComponentProgramBuilder(signature)
    source = _source(signature)
    returned = []
    for builder in (native, source):
        outputs = np.empty(shape, dtype=object)
        for index in signature.get_left("word").all_idxs():
            word = np.asarray(builder.inputs["word"])[index]
            bits = builder.split(word)
            outputs[index] = builder.join(bits[::-1], signature.get_left("word").dtype)
        returned.append(outputs if shape else outputs[()])
    record = native.finalize({"word": returned[0]})
    original = source.builder.finalize(word=returned[1])
    definitions = record.functions[1:]
    mapping = {
        definitions[i].name: definitions[i ^ 1].name for i in range(len(definitions))
    }
    inverse = adjoint_component_program(record, mapping)
    if shape == (0, 2):
        # Pinned _adjoint_final_soqs omits the declared empty output key.
        with pytest.raises(ValueError, match="word"):
            original.adjoint()
        assert inverse.functions[0].registers == record.functions[0].registers
    else:
        compare(inverse, original.adjoint())
        compare(
            adjoint_component_program(inverse, mapping), original.adjoint().adjoint()
        )
    assert all(a is b for a, b in zip(definitions, inverse.functions[1:], strict=True))


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("width", [0, 1, 3])
def test_allocate_free_adjoint_uses_real_directional_definitions(
    width: int, dirty: bool
) -> None:
    from qualtran import BloqBuilder, QAny
    from qualtran.bloqs.bookkeeping import Allocate, Free

    dtype = RegisterType("opaque", width)
    definitions = (
        allocate_word_function("allocate", dtype, dirty),
        free_word_function("free", dtype, dirty),
    )
    native = ComponentProgramBuilder(Signature([]), definitions)
    word = native.call_dict("allocate", {})["reg"]
    native.call_dict("free", {"reg": word})
    record = native.finalize({})
    bb = BloqBuilder()
    value = bb.add(Allocate(QAny(width), dirty=dirty))
    bb.add(Free(QAny(width), dirty=dirty), reg=value)
    expected = bb.finalize().adjoint()
    inverse = adjoint_component_program(
        record, {"allocate": "free", "free": "allocate"}
    )
    compare(inverse, expected)
    assert [op.callee for op in inverse.functions[0].blocks[0].operations] == [
        "allocate",
        "free",
    ]


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_serial_h_graph_complex_amplitudes_and_source_order(repeats: int) -> None:
    record, source = _hierarchy(1, repeats)
    inverse = adjoint_component_program(record, {"h": "h"})
    compare(inverse, source.adjoint())
    bound = bind_register_program(inverse, ((0,),))
    initial = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128) / 30**0.5
    expected = torch.as_tensor(
        np.einsum(
            "ij,j->i",
            source.adjoint().tensor_contract(),
            initial.numpy(),
            optimize=False,
        )
    )
    torch.testing.assert_close(
        forced_branch(initial, bound.circuit.instructions, ()), expected
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": repeats}


@pytest.mark.parametrize("repeats", [1, 2])
def test_specialized_measured_add_adjoint_exact_gates_all_histories_and_work_zero(
    repeats: int,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    signature = Signature(
        [
            Register("a", RegisterType("uint", 2)),
            Register("b", RegisterType("uint", 3)),
            Register("spectator", RegisterType("uint", 1)),
        ]
    )
    definitions = (*executable_add(), *executable_add(adjoint=True))
    native = ComponentProgramBuilder(signature, definitions)
    source = _source(signature)
    add = Add(QUInt(2), QUInt(3))
    outputs = {name: native.inputs[name] for name in ("a", "b")}
    expected_outputs = {name: source.inputs[name] for name in ("a", "b")}
    for _ in range(repeats):
        outputs = native.call_dict("add", outputs)
        expected_outputs = source.builder.add_d(add, **expected_outputs)
    record = native.finalize({**outputs, "spectator": native.inputs["spectator"]})
    original = source.builder.finalize(
        **expected_outputs, spectator=source.inputs["spectator"]
    )
    inverse = adjoint_component_program(record, {"add": "add_adjoint"})
    compare(inverse, original.adjoint())
    assert all(
        op.callee == "add_adjoint" for op in inverse.functions[0].blocks[0].operations
    )
    bound = bind_register_program(inverse, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    measurements = tuple(g for g in bound.circuit.instructions if g.name == "measure")
    expected_gates = tuple(
        g
        for start in range(0, 2 * repeats, 2)
        for g in reference_arithmetic_gates(
            add.adjoint(),
            {"a": (0, 1), "b": (2, 3, 4)},
            (6, 7),
            {
                g.wires[0]: g.metadata["classical_bit"]
                for g in measurements[start : start + 2]
            },
        )
    )
    assert_source(bound.circuit.instructions, expected_gates)
    model = get_cost_value(original.adjoint(), QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == {
        name: count * repeats
        for name, count in {
            "cx": 20,
            "h": 6,
            "t": 4,
            "tdg": 4,
            "s": 2,
            "measure": 2,
            "cz": 2,
            "reset": 2,
        }.items()
    }
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 8 * repeats
    assert counts["measure"] == model.measurement == 2 * repeats
    assert get_cost_value(original.adjoint(), QubitCount()) == 8
    initial = torch.zeros(256, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial[::4] = amplitudes
    expected = torch.zeros_like(initial)
    for prefix in range(64):
        a, b, spectator = prefix >> 4, (prefix >> 1) & 7, prefix & 1
        target = (a << 4) | (((b - repeats * a) & 7) << 1) | spectator
        expected[target << 2] = initial[prefix << 2]
    for history in product((0, 1), repeat=2 * repeats):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected / 2**repeats, rtol=0, atol=1e-12)
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0
    restored = adjoint_component_program(inverse, {"add_adjoint": "add"})
    compare(restored, original.adjoint().adjoint())
    forward = bind_register_program(restored, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    for history in product((0, 1), repeat=2 * repeats):
        torch.testing.assert_close(
            forced_branch(expected, forward.circuit.instructions, history),
            initial / 2**repeats,
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("mapping", [{}, {"h": "missing"}, {"h": "main"}])
def test_missing_or_entry_adjoint_target_rejected(mapping: dict[str, str]) -> None:
    record, _ = _hierarchy(1)
    with pytest.raises(ValueError, match="missing adjoint"):
        adjoint_component_program(record, mapping)


def test_wrong_direction_signature_and_invalid_unused_body_rejected() -> None:
    definitions = executable_add()
    record, _ = _hierarchy(1)
    record = replace(record, functions=(*record.functions, *definitions))
    with pytest.raises(ValueError, match="incompatible adjoint signature"):
        adjoint_component_program(record, {"h": "add"})
    bad = replace(definitions[0], name="invalid", blocks=())
    record = replace(record, functions=(*record.functions, bad))
    with pytest.raises(ValueError, match="invalid component graph"):
        adjoint_component_program(record, {"h": "h"})


def test_custom_entry_block_preserved_and_forward_payload_cleared() -> None:
    record, _ = _hierarchy(1)
    function = record.functions[0]
    custom = replace(
        function,
        name="custom",
        entry="body",
        blocks=(replace(function.blocks[0], name="body"),),
    )
    record = replace(
        record,
        functions=(custom, *record.functions[1:]),
        canonical_payload={"witness": 36},
    )
    inverse = adjoint_component_program(record, {"h": "h"}, "custom")
    assert inverse.canonical_payload is None
    assert record.canonical_payload == {"witness": 36}
    assert inverse.functions[0].entry == "body"
    assert inverse.functions[0].blocks[0].name == "body"
    assert verify_program_record(inverse) == ()


@pytest.mark.parametrize("side", ["allocate", "free"])
@pytest.mark.parametrize("dirty", [False, True])
def test_entry_direction_flips_real_owned_lifetime(side: str, dirty: bool) -> None:
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Allocate, Free

    from flagquantum.core.ir.program.registers import RegisterSide

    dtype = RegisterType("opaque", 2)
    register = Register(
        "reg",
        dtype,
        side=RegisterSide.RIGHT if side == "allocate" else RegisterSide.LEFT,
    )
    native = ComponentProgramBuilder(
        Signature([register]),
        (
            allocate_word_function("allocate", dtype, dirty),
            free_word_function("free", dtype, dirty),
        ),
    )
    source = _source(Signature([register]))
    if side == "allocate":
        outputs = native.call_dict("allocate", {})
        expected_outputs = source.builder.add_d(Allocate(QAny(2), dirty=dirty))
    else:
        outputs = native.call_dict("free", native.inputs)
        expected_outputs = source.builder.add_d(
            Free(QAny(2), dirty=dirty), **source.inputs
        )
    record = native.finalize(outputs)
    expected = source.builder.finalize(**expected_outputs)
    inverse = adjoint_component_program(
        record, {"allocate": "free", "free": "allocate"}
    )
    compare(inverse, expected.adjoint())
    assert inverse.functions[0].registers == (register.adjoint(),)


def test_distinct_unitary_bodies_reverse_order_and_conjugate_complex_phase() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, TGate

    record, _ = _hierarchy(1)
    h = record.functions[1]
    t = replace(
        h,
        name="t",
        blocks=(
            replace(
                h.blocks[0],
                operations=tuple(
                    replace(op, opcode="t") if op.opcode == "h" else op
                    for op in h.blocks[0].operations
                ),
            ),
        ),
    )
    tdg = replace(
        t,
        name="tdg",
        blocks=(
            replace(
                t.blocks[0],
                operations=tuple(
                    replace(op, opcode="tdg") if op.opcode == "t" else op
                    for op in t.blocks[0].operations
                ),
            ),
        ),
    )
    signature = Signature(
        [*(h.registers or ()), Register("spectator", RegisterType("bit", 1))]
    )
    native = ComponentProgramBuilder(signature, (h, t, tdg))
    source = _source(signature)
    outputs = {"q": native.inputs["q"]}
    expected_outputs = {"q": source.inputs["q"]}
    for name, gate in (("h", Hadamard()), ("t", TGate()), ("t", TGate())):
        outputs = native.call_dict(name, outputs)
        expected_outputs = source.builder.add_d(gate, **expected_outputs)
    record = native.finalize({**outputs, "spectator": native.inputs["spectator"]})
    expected = source.builder.finalize(
        **expected_outputs, spectator=source.inputs["spectator"]
    ).adjoint()
    inverse = adjoint_component_program(record, {"h": "h", "t": "tdg"})
    compare(inverse, expected)
    assert [op.callee for op in inverse.functions[0].blocks[0].operations] == [
        "tdg",
        "tdg",
        "h",
    ]
    bound = bind_register_program(inverse, ((0,), (1,)))
    initial = torch.tensor([1 + 2j, -3 + 1j, 4 - 2j, -1 - 5j], dtype=torch.complex128)
    initial /= initial.norm()
    result = np.einsum(
        "ij,j->i", expected.tensor_contract(), initial.numpy(), optimize=False
    )
    torch.testing.assert_close(
        forced_branch(initial, bound.circuit.instructions, ()), torch.as_tensor(result)
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"tdg": 2, "h": 1}


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_identity_return_permutation_retains_row_major_elements(
    classical: bool, shape: tuple[int, ...]
) -> None:
    from flagquantum.core.ir.program.register_types import classical_bit_type

    dtype = classical_bit_type() if classical else RegisterType("uint", 3)
    signature = Signature([Register("word", dtype, shape)])
    native = ComponentProgramBuilder(signature)
    source = _source(signature)
    outputs = []
    for builder in (native, source):
        values = np.asarray(builder.inputs["word"])
        result = values.reshape(-1)[::-1].reshape(shape)
        outputs.append(result if shape else result[()])
    record = native.finalize({"word": outputs[0]})
    expected = source.builder.finalize(word=outputs[1])
    inverse = adjoint_component_program(record, {})
    if shape == (0, 2):
        with pytest.raises(ValueError, match="word"):
            expected.adjoint()
    else:
        compare(inverse, expected.adjoint())
    assert len(inverse.functions[0].blocks[0].operations) == 0


def test_untyped_target_cannot_stand_in_for_register_adjoint() -> None:
    record, _ = _hierarchy(1)
    record = replace(record, functions=(*record.functions, *executable_add()))
    with pytest.raises(ValueError, match="incompatible adjoint signature"):
        adjoint_component_program(record, {"h": "gates"})


def test_direction_flip_rejects_invalid_retained_caller_without_mutating_input() -> (
    None
):
    from flagquantum.core.ir.program.registers import RegisterSide

    dtype = RegisterType("opaque", 2)
    signature = Signature([Register("reg", dtype, side=RegisterSide.RIGHT)])
    builder = ComponentProgramBuilder(
        signature,
        (
            allocate_word_function("allocate", dtype, False),
            free_word_function("free", dtype, False),
        ),
    )
    record = builder.finalize(builder.call_dict("allocate", {}))
    caller = ComponentProgramBuilder(signature, record.functions, name="caller")
    caller_record = caller.finalize(caller.call_dict("main", {}))
    record = replace(record, functions=(*record.functions, caller_record.functions[0]))
    assert verify_program_record(record) == ()
    with pytest.raises(ValueError, match="invalid constructed register Program"):
        adjoint_component_program(record, {"allocate": "free"})
    assert verify_program_record(record) == ()
