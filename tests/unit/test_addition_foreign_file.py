# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Callback fixtures adapted from frozen Qualtran v0.7.0 addition_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition_test.py

"""Complete addition factories and consumers retain existing native owners."""

from collections import Counter
from collections.abc import Mapping
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_registers import deserialize_registers
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition import (
    out_of_place_add_ir,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
    out_of_place_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from flagquantum.simulation.classical_addition import (
    add_constant_classical_value,
    add_integer_values,
)
from flagquantum.simulation.classical_controlled_values import (
    controlled_classical_values,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
FACTORIES = (
    "_add_symb",
    "_add_small",
    "_add_large",
    "_add_diff_size_regs",
    "_add_oop_symb",
    "_add_oop_small",
    "_add_oop_large",
    "_add_k",
    "_add_k_small",
    "_add_k_large",
)


def _signature(source: Any) -> Signature:
    from qualtran.serialization.registers import registers_to_proto

    return Signature(
        deserialize_registers(registers_to_proto(source.signature).SerializeToString())
    )


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_all_original_objects_roundtrip_full_library_metadata(
    factory_name: str,
) -> None:
    from qualtran.bloqs.arithmetic import addition
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
    from qualtran.testing import assert_bloq_example_serializes

    factory = getattr(addition, factory_name)
    source = factory.make()
    assert_bloq_example_serializes(factory)
    for depth in (0, 1):
        original = bloqs_to_proto(source, max_depth=depth)
        wire = original.SerializeToString(deterministic=True)
        native = deserialize_component_library(wire)
        assert serialize_component_library(native) == wire
        assert (
            deserialize_component_library(serialize_component_library(native)) == native
        )
        assert source in bloqs_from_proto(original)
        declaration = native.entries[0].declaration
        assert declaration.registers == tuple(_signature(source))
        assert declaration.name.endswith("." + type(source).__name__)
        if depth == 0:

            def declared(
                name: str,
                registers: tuple[Register, ...] | None,
                arguments: Mapping[str, ResolvedArgument],
            ) -> FunctionRecord:
                assert registers is not None and arguments
                return FunctionRecord(name, (), registers=registers)

            resolved = resolve_component_program(
                native, 0, {declaration.name: declared}
            )
            assert resolved.functions[0].registers == declaration.registers


@pytest.mark.parametrize(
    "factory_name", ("_add_small", "_add_large", "_add_diff_size_regs")
)
@pytest.mark.parametrize("kind", CHECKS)
def test_all_concrete_add_factories_use_existing_five_checks(
    factory_name: str,
    kind: CheckKind,
) -> None:
    from qualtran.bloqs.arithmetic import addition

    source = getattr(addition, factory_name).make()
    lhs, rhs = tuple(_signature(source))
    left, right = int(lhs.bitsize), int(rhs.bitsize)
    words = (tuple(range(left)), tuple(range(left, left + right)))
    work = tuple(range(left + right, left + 2 * right - 1))
    original = reference_arithmetic_gates(
        source,
        dict(zip(("a", "b"), words, strict=True)),
        work,
        dict(zip(work, range(right - 1), strict=True)),
    )

    def make() -> ProgramRecord:
        return addition_component_program(
            left, right, left_dtype=lhs.dtype, right_dtype=rhs.dtype
        )

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(record, words, clean=work).circuit

    result = check_example(
        RegisterExample(
            make,
            lower,
            lambda: Counter(gate.name for gate in original),
            ((lhs.dtype, lhs.dtype), (rhs.dtype, rhs.dtype)),
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


def test_original_out_of_place_routes_have_distinct_linear_order() -> None:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder

    source = OutOfPlaceAdder(2, is_adjoint=True, include_most_significant_bit=False)
    qubits = cirq.LineQubit.range(6)
    literal = tuple(
        cirq.flatten_op_tree(
            source.decompose_from_registers(
                context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                a=np.array(qubits[:2]),
                b=np.array(qubits[2:4]),
                c=np.array(qubits[4:]),
            )
        )
    )
    scheduled = tuple(cirq.Circuit(literal).all_operations())
    # Both sequences come directly from the locked SDK, without native code.
    assert literal[1] == cirq.CNOT(qubits[0], qubits[4])
    assert scheduled[1] == cirq.CNOT(qubits[3], qubits[5])
    assert Counter(literal) == Counter(scheduled)


@pytest.mark.parametrize("width", (1, 2, 3, 5))
@pytest.mark.parametrize("include_msb", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_sdk_sided_decomposition_matches_native_stream(
    width: int,
    include_msb: bool,
    adjoint: bool,
) -> None:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = OutOfPlaceAdder(
        width, is_adjoint=adjoint, include_most_significant_bit=include_msb
    )
    signature = Signature(_signature(source))
    qubits = tuple(reversed(cirq.LineQubit.range(3 * width + int(include_msb))))
    imported = import_cirq_register_gate(source.on(*qubits), signature=signature)
    assert imported is not None
    words = (tuple(range(width)), tuple(range(width, 2 * width)))
    c = tuple(range(2 * width, 3 * width + int(include_msb)))
    operands = (*words, c) if adjoint else words
    kwargs: dict[str, Any] = dict(
        clean=() if adjoint else c[::-1],
        allow_workspace_outputs=not adjoint,
        input_ownership=tuple((wire, False) for wire in c) if adjoint else (),
    )
    actual = bind_register_program(imported, operands, **kwargs)
    expected = bind_register_program(
        out_of_place_addition_program(width, include_msb=include_msb, adjoint=adjoint),
        operands,
        **kwargs,
    )
    ports = {"a": words[0], "b": words[1], "c": c}
    outputs = {"a": words[0], "b": words[1]} if adjoint else ports
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in actual.circuit.instructions
        if gate.name == "measure"
    }
    graph_slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in expected.circuit.instructions
        if gate.name == "measure"
    }
    assert slots == graph_slots
    literal_source = reference_arithmetic_gates(
        source, ports, (), slots, cirq_style=True
    )
    graph_source = reference_arithmetic_gates(
        source,
        ports if adjoint else {"a": words[0], "b": words[1]},
        () if adjoint else c[::-1],
        graph_slots,
        expected_outputs=outputs,
    )
    raw_native = out_of_place_add_ir(
        *words,
        c,
        include_msb=include_msb,
        adjoint=adjoint,
        classical_bits=tuple(slots[wire] for wire in c[::-1][1:]) if adjoint else (),
    )
    assert raw_native == literal_source
    assert expected.circuit.instructions == graph_source
    model = get_cost_value(source, QECGatesCost())
    # The importer keeps measured bodies atomic; pure bodies use typed graph order.
    # The locked SDK generator and graph exporter have different linear schedules.
    sdk_source = literal_source if model.measurement else graph_source
    assert actual.circuit.instructions == sdk_source
    actual_counts = Counter(gate.name for gate in actual.circuit.instructions)
    assert actual_counts == Counter(gate.name for gate in graph_source)
    assert actual_counts["t"] + actual_counts["tdg"] == model.total_t_count()
    assert actual_counts["measure"] == actual_counts["reset"] == model.measurement
    assert actual.outputs == expected.outputs
    assert actual.circuit.n_wires == expected.circuit.n_wires
    assert CircuitIR.from_json(actual.circuit.to_json()) == actual.circuit
    assert not any(gate.name == "any" for gate in actual.circuit.instructions)
    if width <= 3:
        count = len(c)
        coefficients = torch.arange(1, (1 << (2 * width + 1)) + 1, dtype=torch.float64)
        amplitudes = torch.complex(coefficients, 3 - 2 * coefficients)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(
            1 << (3 * width + int(include_msb) + 1), dtype=torch.complex128
        )
        correct = torch.zeros_like(initial)
        for a, b, spectator in product(range(1 << width), range(1 << width), (0, 1)):
            index = ((a << width) | b) * 2 + spectator
            empty = (((a << width) | b) << (count + 1)) | spectator
            occupied = empty | (((a + b) % (1 << count)) << 1)
            initial[empty] = amplitudes[index]
            correct[occupied] = amplitudes[index]
        histories = tuple(product((0, 1), repeat=count - 1)) if adjoint else ((),)

        def compare_branch(
            empty_state: torch.Tensor, sum_state: torch.Tensor, history: tuple[int, ...]
        ) -> None:
            incoming = sum_state if adjoint else empty_state
            result = forced_branch(incoming, actual.circuit.instructions, history)
            graph_result = forced_branch(
                incoming, expected.circuit.instructions, history
            )
            target = empty_state if adjoint else sum_state
            torch.testing.assert_close(
                result, target / 2 ** (len(history) / 2), rtol=0, atol=3e-12
            )
            torch.testing.assert_close(graph_result, result, rtol=0, atol=3e-12)
            if adjoint:
                assert (
                    torch.count_nonzero(result.reshape(-1, 1 << (count + 1))[:, 2:])
                    == 0
                )

        for history in histories:
            compare_branch(initial, correct, history)
        if width <= 2:
            # Check the complete small isometry, not only one coherent input.
            for a, b, spectator in product(
                range(1 << width), range(1 << width), (0, 1)
            ):
                empty = (((a << width) | b) << (count + 1)) | spectator
                occupied = empty | (((a + b) % (1 << count)) << 1)
                basis_empty, basis_sum = torch.zeros_like(initial), torch.zeros_like(
                    correct
                )
                basis_empty[empty] = basis_sum[occupied] = 1
                for history in histories:
                    compare_branch(basis_empty, basis_sum, history)


@pytest.mark.parametrize(
    "width,k,x,cvs,controls,signed,result",
    [
        (5, 1, 2, (), (), False, 3),
        (5, 3, 2, (1,), (1,), False, 5),
        (5, 2, 0, (1, 0), (1, 0), False, 2),
        (5, 1, 2, (1, 0, 1), (0, 0, 0), False, 2),
        (5, 2, 0, (1, 0), (1, 0), True, 2),
        (6, -3, 2, (), (), True, -1),
    ],
)
def test_complete_original_controlled_constant_callbacks(
    width: int,
    k: int,
    x: int,
    cvs: tuple[int, ...],
    controls: tuple[int, ...],
    signed: bool,
    result: int,
) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QInt, QUInt
    from qualtran.bloqs.arithmetic.addition import AddK

    source: Any = AddK((QInt if signed else QUInt)(width), k)
    values: dict[str, object] = {"x": x}

    def child(**kwargs: Any) -> dict[str, object]:
        return {"x": add_constant_classical_value(kwargs["x"], k, width, signed=signed)}

    if cvs:
        source = source.controlled(CtrlSpec(cvs=cvs))
        values["ctrl"] = np.asarray(controls)
        actual = controlled_classical_values(
            values, ("ctrl",), ("x",), (np.asarray(cvs),), child
        )
    else:
        actual = child(**values)
    assert isinstance(actual, dict) and actual["x"] == result
    original = source.on_classical_vals(**values)
    assert set(actual) == set(original)
    for name in original:
        np.testing.assert_array_equal(actual[name], original[name])
    decomposed = source.decompose_bloq().call_classically(**values)
    assert decomposed[-1] == result


def test_original_arithmetic_sdk_protocol_is_distinct_from_sided_erasure() -> None:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from sympy import Symbol

    for adjoint, full in product((False, True), repeat=2):
        source = OutOfPlaceAdder(
            2, is_adjoint=adjoint, include_most_significant_bit=full
        )
        assert source.registers() == ([2, 2], [2, 2], [2] * (2 + int(full)))
        with pytest.raises(NotImplementedError, match="no need"):
            source.with_registers(*source.registers())
        for a, b, c in product(range(4), repeat=3):
            assert source.apply(a, b, c) == (a, b, add_integer_values(c, a + b))
        assert source.adjoint().include_most_significant_bit == full
        assert (source**-1).include_most_significant_bit is True
    with pytest.raises(ValueError, match="Symbolic bitsize"):
        OutOfPlaceAdder(Symbol("n")).registers()
