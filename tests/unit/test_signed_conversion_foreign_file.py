"""Retain complete signed-conversion sided ports through existing SDK adapters."""

from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.bits.signed_encoding_component import (
    signed_encoding_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.ecosystem.cirq._register_program_export import (
    export_bound_register_cirq,
)
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")


def _source_named_program(width: int, adjoint: bool) -> tuple[ProgramRecord, Signature]:
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        SignedIntegerToTwosComplement,
    )

    source: Any = SignedIntegerToTwosComplement(width)
    if adjoint:
        source = source.adjoint()
    signature = Signature(
        tuple(
            Register(
                register.name,
                RegisterType(
                    (
                        "ones_complement"
                        if type(register.dtype).__name__ == "QIntOnesComp"
                        else "signed"
                    ),
                    width,
                ),
                side=RegisterSide(register.side.value),
            )
            for register in source.signature
        )
    )
    graph = signed_encoding_program(width, adjoint=adjoint)
    builder = ComponentProgramBuilder(signature, graph.functions, name="sdk")
    input_name = next(signature.lefts()).name
    output_name = next(signature.rights()).name
    result = builder.call_dict("main", {"x": builder.inputs[input_name]})
    return builder.finalize({output_name: result["y"]}), signature


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("adjoint", [False, True])
def test_sided_source_names_aliases_feedback_and_all_complex_branches(
    width: int, adjoint: bool
) -> None:
    record, signature = _source_named_program(width, adjoint)
    word = tuple(range(width))
    work = tuple(range(width, 2 * width - 2))
    bound = bind_register_program(record, (word,), clean=work, entry="sdk")
    incoming = np.asarray(tuple(reversed(cirq.LineQubit.range(width))), dtype=object)
    input_name = next(signature.lefts()).name
    output_name = next(signature.rights()).name
    exported, outputs = export_bound_register_cirq(
        bound,
        signature,
        input_quregs={input_name: incoming},
        feedback_key_prefix="signed",
    )
    np.testing.assert_array_equal(outputs[output_name], incoming)
    imported = import_cirq_register_optree(
        exported.circuit,
        signature=signature,
        in_quregs={input_name: incoming},
        out_quregs=outputs,
    )
    restored = bind_register_program(imported, (word,), clean=work)
    assert restored.outputs == bound.outputs == (word,)
    assert restored.circuit.n_wires == bound.circuit.n_wires
    assert CircuitIR.from_json(restored.circuit.to_json()) == restored.circuit
    assert restored.circuit.instructions == bound.circuit.instructions
    total = bound.circuit.n_wires + 1
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    coefficients = torch.arange(1, (1 << (width + 1)) + 1).double()
    values = torch.complex(coefficients, 2 - 3 * coefficients)
    values /= values.norm()
    mask = (1 << (width - 1)) - 1
    for raw in range(1 << width):
        sign = raw >> (width - 1)
        converted = (raw & ~mask) | (
            ((raw & mask) + (-sign if adjoint else sign)) & mask
        )
        for spectator in (0, 1):
            initial[(raw << (len(work) + 1)) | spectator] = values[2 * raw + spectator]
            expected[(converted << (len(work) + 1)) | spectator] = values[
                2 * raw + spectator
            ]
    for history in product((0, 1), repeat=len(work)):
        actual = forced_branch(initial, restored.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected * 2 ** (-len(work) / 2), rtol=0, atol=4e-14
        )


def test_actual_source_factory_serialization_retains_sided_signature() -> None:
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        _signed_to_twos,
    )
    from qualtran.testing import assert_bloq_example_serializes

    assert_bloq_example_serializes(_signed_to_twos)
