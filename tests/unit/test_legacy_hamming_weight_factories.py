"""The source Hamming constructor reuses exact owned compression and erasure."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hamming_weight_factories import (
    hamming_weight_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.bits.hamming_weight import (
    hamming_weight_requirements,
)
from flagquantum.compiler.registers.bits.hamming_weight_component import (
    hamming_weight_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5, 7, 8, 9, 16))
def test_original_constructor_matches_certified_program_bound_stream_and_resources(
    width: int,
) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = HammingWeightCompute(width)
    program = resolve_component_program(
        _library(source, 0), 0, hamming_weight_factories()
    )
    expected = hamming_weight_program(width)
    assert verify_program_record(program) == ()
    assert program == _rename_program(expected, program.functions[0].name)
    junk, output = hamming_weight_requirements(width)
    inputs = (tuple(range(width)),)
    clean = tuple(range(width, width + junk + output))
    actual = bind_register_program(
        program, inputs, clean=clean, allow_workspace_outputs=True
    )
    direct = bind_register_program(
        expected, inputs, clean=clean, allow_workspace_outputs=True
    )
    assert actual == direct
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in actual.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement == 0


@pytest.mark.parametrize(
    "width",
    (
        0,
        sympy.Symbol("n"),
        sympy.Symbol("n", integer=True, positive=True),
    ),
)
def test_symbolic_and_empty_source_widths_keep_encoded_declarations(width: Any) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    library = _library(HammingWeightCompute(width), 0)
    program = resolve_component_program(library, 0, hamming_weight_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize(
    "arguments", ({}, {"bitsize": True}, {"bitsize": 2.0}, {"bitsize": 3, "extra": 0})
)
def test_invalid_metadata_refuses(arguments: dict[str, Any]) -> None:
    with pytest.raises(SerializationError):
        next(iter(hamming_weight_factories().values()))("main", None, arguments)


def test_integer_symbol_validation_preserves_actual_output_width() -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    library = _library(HammingWeightCompute(sympy.Symbol("n", integer=True)), 0)
    entry = library.entries[0]
    ports = entry.declaration.registers
    assert ports is not None
    output = ports[-1]
    wrong = Register(
        output.name,
        replace(output.dtype, quantum_bits=output.bitsize + 1),
        side=output.side,
    )
    declaration = replace(entry.declaration, registers=(*ports[:-1], wrong))
    with pytest.raises(SerializationError, match="signature"):
        resolve_component_program(
            replace(library, entries=(replace(entry, declaration=declaration),)),
            0,
            hamming_weight_factories(),
        )


def test_original_integer_expression_wire_loses_population_count_semantics() -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    width = sympy.Integer(5)
    source = HammingWeightCompute(width)
    restored = bloqs_from_proto(bloqs_to_proto(source, max_depth=0))[0]
    assert isinstance(restored, HammingWeightCompute)
    assert type(restored.bitsize) is int
    assert source.junk_bitsize == 4 and restored.junk_bitsize == 3
    with pytest.raises(SerializationError, match="signature"):
        resolve_component_program(_library(source, 0), 0, hamming_weight_factories())
    direct = next(iter(hamming_weight_factories().values()))(
        "main", None, {"bitsize": width}
    )
    assert isinstance(direct, ProgramRecord)
    root = direct.functions[0]
    assert root.blocks
    assert root.registers is not None and root.registers[1].bitsize == 4


# The original width-one generic decomposition fails at depth one; existing
# source-negative evidence is reused instead of counting an unexecuted case.
@pytest.mark.parametrize(
    "width,depth", ((1, 0), *((w, d) for w in (2, 3, 5, 8) for d in (0, 1)))
)
def test_actual_inverse_constructor_uses_exact_owner_and_preserves_resources(
    width: int, depth: int
) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    original = HammingWeightCompute(width).adjoint()
    library = _library(original, depth)
    program = resolve_component_program(library, 0, hamming_weight_factories())
    assert verify_program_record(program) == ()
    junk, out = hamming_weight_requirements(width)
    total = width + junk + out
    inputs = (
        tuple(reversed(range(width))),
        tuple(range(width, width + junk)),
        tuple(range(width + junk, total)),
    )
    ownership = tuple((wire, False) for word in inputs[1:] for wire in word)
    actual = bind_register_program(program, inputs, input_ownership=ownership)
    expected = bind_register_program(
        hamming_weight_program(width, adjoint=True), inputs, input_ownership=ownership
    )
    assert actual == expected
    assert actual.outputs == (inputs[0],)
    counts = Counter(g.name for g in actual.circuit.instructions)
    assert counts["measure"] == counts["reset"] == junk
    assert get_cost_value(original, QECGatesCost()).measurement == junk
    assert "any" not in counts
    if width == 1:
        assert inputs[1] == () and counts == {"cx": 1}


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_resolved_forward_inverse_append_all_complex_branches_and_zero_outputs(
    width: int,
) -> None:
    from itertools import product
    from math import sqrt

    import torch
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.core.ir import CircuitIR
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch

    source = HammingWeightCompute(width)
    forward = resolve_component_program(
        _library(source, 0), 0, hamming_weight_factories()
    )
    inverse = resolve_component_program(
        _library(source.adjoint(), 0), 0, hamming_weight_factories()
    )
    junk, out = hamming_weight_requirements(width)
    total = width + junk + out + 1
    clean = tuple(range(width, width + junk + out))
    first = bind_register_program(
        forward, (tuple(range(width)),), clean=clean, allow_workspace_outputs=True
    )
    second = bind_register_program(
        inverse, first.outputs, input_ownership=tuple((w, False) for w in clean)
    )
    gates = (*first.circuit.instructions, *second.circuit.instructions)
    circuit = DynamicCircuit(total).append(FieldOperation(total, gates))
    assert circuit.to_ir().instructions == gates
    assert CircuitIR.from_json(circuit.to_ir().to_json()) == circuit.to_ir()
    logical = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64).reshape(
        1 << width, 2
    )
    amplitudes = torch.complex(logical, logical.square().remainder(13) - 4)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    addresses = (torch.arange(1 << width)[:, None] << (junk + out + 1)) | torch.arange(
        2
    )
    initial[addresses] = amplitudes
    for history in product((0, 1), repeat=junk):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            initial / sqrt(2**junk),
            atol=4e-12,
            rtol=0,
        )
    assert second.outputs == first.outputs[:1]
    if width == 1:
        assert first.outputs[1] == ()


@pytest.mark.parametrize("width", (0, sympy.Symbol("n", integer=True, positive=True)))
def test_inverse_symbolic_and_zero_widths_preserve_declarations(width: Any) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    library = _library(HammingWeightCompute(width).adjoint(), 0)
    program = resolve_component_program(library, 0, hamming_weight_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


def test_double_inverse_and_compilation_scoped_unknown_context() -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    source = Adjoint(Adjoint(HammingWeightCompute(1)))
    program = resolve_component_program(
        _library(source, 0), 0, hamming_weight_factories()
    )
    expected = hamming_weight_program(1)
    assert bind_register_program(
        program, ((0,),), clean=(1,), allow_workspace_outputs=True
    ) == bind_register_program(
        expected, ((0,),), clean=(1,), allow_workspace_outputs=True
    )
    key = "qualtran._infra.adjoint.Adjoint"
    factories = hamming_weight_factories()
    base = factories["qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute"](
        "child", None, {"bitsize": 1}
    )
    assert not isinstance(base, FunctionRecord)
    root = base.functions[0]
    unknown = replace(root, blocks=())
    for child in (unknown, FunctionRecord("different", (), registers=root.registers)):
        with pytest.raises(SerializationError, match="context"):
            factories[key]("inverse", None, {"subbloq": child})
    with pytest.raises(SerializationError, match="context"):
        hamming_weight_factories()[key]("inverse", None, {"subbloq": root})
    with pytest.raises(SerializationError, match="signature"):
        factories[key]("inverse", root.registers, {"subbloq": root})


def test_unknown_inverse_delegates_only_to_the_explicit_owner() -> None:
    from collections.abc import Mapping

    from flagquantum.compiler.formats.legacy_component_resolution import (
        ResolvedArgument,
    )

    seen: list[Mapping[str, ResolvedArgument]] = []

    def fallback(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        seen.append(arguments)
        return FunctionRecord(name, (), registers=registers)

    unknown = FunctionRecord("foreign", (), registers=())
    arguments = {"subbloq": unknown}
    factories = hamming_weight_factories(adjoint_factory=fallback)
    actual = factories["qualtran._infra.adjoint.Adjoint"]("inverse", (), arguments)
    assert actual == FunctionRecord("inverse", (), registers=()) and seen == [arguments]


def test_actual_inverse_metadata_compiles_and_appends_without_reference_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    library = _library(HammingWeightCompute(3).adjoint(), 0)
    original_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError(
                "Reference SDK import during native inverse compilation"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    program = resolve_component_program(library, 0, hamming_weight_factories())
    bound = bind_register_program(
        program,
        ((0, 1, 2), (3,), (4, 5)),
        input_ownership=((3, False), (4, False), (5, False)),
    )
    circuit = DynamicCircuit(6).append(FieldOperation(6, bound.circuit.instructions))
    assert circuit.to_ir().instructions == bound.circuit.instructions
