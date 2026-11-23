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
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5, 7, 8, 9, 16))
def test_original_constructor_matches_certified_program_bound_stream_and_resources(
    width: int,
) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting import QECGatesCost, get_cost_value

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
    assert type(restored.bitsize) is int
    assert source.junk_bitsize == 4 and restored.junk_bitsize == 3
    with pytest.raises(SerializationError, match="signature"):
        resolve_component_program(_library(source, 0), 0, hamming_weight_factories())
    direct = next(iter(hamming_weight_factories().values()))(
        "main", None, {"bitsize": width}
    )
    assert isinstance(direct, FunctionRecord)
    assert not direct.blocks
    assert direct.registers is not None and direct.registers[1].bitsize == 4
