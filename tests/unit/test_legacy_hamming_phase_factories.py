"""Full stored Hamming phase sources reuse count, phase and erasure evidence."""

from itertools import product

import pytest
import sympy
import torch

from examples.algorithms.hamming_phasing_catalog import HammingPhasingExample
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hamming_phase_factories import (
    hamming_phase_factories,
)
from flagquantum.compiler.registers.phase.hamming_phasing_component import (
    hamming_phasing_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "gradient,width,exponent",
    tuple(product((False, True), (2, 3, 4), (0.375, 1.0, -1.1934341))),
)
def test_stored_complete_graph_and_accepted_primitive_stream(
    gradient: bool, width: int, exponent: float
) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasing,
        HammingWeightPhasingViaPhaseGradient,
    )

    source = (
        HammingWeightPhasingViaPhaseGradient if gradient else HammingWeightPhasing
    )(width, exponent, 0.2)
    record = resolve_component_program(
        _library(source, 0), 0, hamming_phase_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, hamming_phase_factories()
            )
            == record
        )
    example = HammingPhasingExample(width, exponent, 0.2, via_gradient=gradient)
    logical = width + example.gradient_width
    wires = tuple(reversed(range(logical)))
    ports = (wires[:width], wires[width:]) if gradient else (wires,)
    clean = tuple(range(logical, example.qubit_count))
    if gradient and exponent == 0.375:
        with pytest.raises(ValueError):
            bind_register_program(record, ports, clean=clean)
        from qualtran import BloqError

        with pytest.raises(BloqError, match="register dtypes are not consistent"):
            next(
                iter(source.phase_oracle.decompose_bloq().bloq_instances)
            ).bloq.decompose_bloq()
        return
    bound = bind_register_program(record, ports, clean=clean)
    accepted = bind_register_program(
        hamming_phasing_program(width, exponent, 0.2, via_gradient=gradient),
        ports,
        clean=clean,
    )
    assert bound.circuit == accepted.circuit
    assert bound.events == accepted.events
    assert bound.outputs == accepted.outputs == ports
    if not gradient and width <= 3:
        contiguous = bind_register_program(record, (tuple(range(width)),), clean=clean)
        torch.testing.assert_close(
            logical_operator(contiguous.circuit, width),
            torch.tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("gradient", (False, True))
def test_original_examples_keep_signature_and_source_root(gradient: bool) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        _hamming_weight_phasing,
        _hamming_weight_phasing_via_phase_gradient,
    )

    source = (
        _hamming_weight_phasing_via_phase_gradient
        if gradient
        else _hamming_weight_phasing
    ).make()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, hamming_phase_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())


@pytest.mark.parametrize("gradient", (False, True))
def test_symbolic_exponent_keeps_source_success_root(gradient: bool) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasing,
        HammingWeightPhasingViaPhaseGradient,
    )

    from examples.algorithms.binary_field_arithmetic import FieldOperation

    theta = sympy.Symbol("theta")
    source = (
        HammingWeightPhasingViaPhaseGradient if gradient else HammingWeightPhasing
    )(3, theta, 0.2)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, hamming_phase_factories())
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    if gradient:
        assert len(record.functions[0].blocks[0].operations) == 3
        with pytest.raises(ValueError):
            bind_register_program(
                record, ((0, 1, 2), tuple(range(3, 10))), clean=tuple(range(10, 20))
            )
    else:
        bound = bind_register_program(record, ((0, 1, 2),), clean=(3, 4, 5))
        numeric = (
            Circuit(6)
            .append(FieldOperation(6, bound.circuit.instructions))
            .bind_parameters({"theta": 0.375})
            .to_ir()
        )
        expected = bind_register_program(
            hamming_phasing_program(3, 0.375, 0.2), ((0, 1, 2),), clean=(3, 4, 5)
        )
        assert numeric.instructions == expected.circuit.instructions


def test_symbolic_precision_direct_and_metadata_validation() -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing

    source = HammingWeightPhasing(3, 0.375, sympy.Symbol("eps"))
    record = resolve_component_program(
        _library(source, 0), 0, hamming_phase_factories()
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    direct, gradient = tuple(hamming_phase_factories().values())
    assert direct("default", None, {"bitsize": 3}) == direct(
        "default", None, {"bitsize": 3, "exponent": 1.0, "eps": 1e-10}
    )
    for arguments in ({}, {"bitsize": True}, {"bitsize": 3, "unknown": 0}):
        with pytest.raises(SerializationError):
            direct("bad", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        direct("bad", (), {"bitsize": 3})
    with pytest.raises(SerializationError, match="precision"):
        gradient("bad", None, {"bitsize": 3, "eps": sympy.Symbol("eps")})


@pytest.mark.parametrize("gradient", (False, True))
def test_one_bit_original_root_keeps_empty_junk_logical_calls(gradient: bool) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasing,
        HammingWeightPhasingViaPhaseGradient,
    )

    source = (
        HammingWeightPhasingViaPhaseGradient if gradient else HammingWeightPhasing
    )(1, 0.375, 0.2)
    record = resolve_component_program(
        _library(source, 0), 0, hamming_phase_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())


def test_empty_gradient_declaration_retains_source_signature() -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasingViaPhaseGradient,
    )

    source = HammingWeightPhasingViaPhaseGradient(0, 0.375, 0.2)
    assert source.signature.get_left("phase_grad").bitsize == 1
    with pytest.raises(TypeError, match="x"):
        source.decompose_bloq()
    gradient = tuple(hamming_phase_factories().values())[1]
    record = gradient("empty", None, {"bitsize": 0, "exponent": 0.375, "eps": 0.2})
    assert isinstance(record, FunctionRecord) and record.registers is not None
    assert not record.blocks
    assert record.registers[0].name == "phase_grad"
    assert record.registers[0].dtype.quantum_bits == 1
