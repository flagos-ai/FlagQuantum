"""Sparse alias selection follows the source QROM word compatibility rules."""

from collections import Counter
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    import sympy
    import torch
    from qualtran._infra.data_types import (
        QGF,
        BQUInt,
        QFxp,
        QInt,
        QMontgomeryUInt,
    )
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_alias_preparation_factories import (
    alias_preparation_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    montgomery_register_type,
    ones_complement_register_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_SPARSE = "qualtran.bloqs.state_preparation.state_preparation_alias_sampling.SparseStatePreparationAliasSampling"


def storage(
    kind: str,
) -> tuple[QInt | QMontgomeryUInt | QFxp | QGF, RegisterType]:
    if kind == "signed":
        return QInt(2), signed_register_type(2)
    if kind == "montgomery":
        modulus = sympy.Symbol("modulus", integer=True)
        return QMontgomeryUInt(2, modulus), montgomery_register_type(2, modulus)
    if kind in ("fraction", "integer"):
        fraction = 2 if kind == "fraction" else 0
        return QFxp(2, fraction), fixed_point_register_type(2, fraction)
    if kind == "bit_field":
        return QGF(2, 1), FiniteFieldEncoding(2, 1).register_type()
    assert kind == "bit_signed"
    return QInt(1), signed_register_type(1)


@pytest.mark.parametrize(
    "kind", ("signed", "montgomery", "fraction", "integer", "bit_field", "bit_signed")
)
def test_nominal_selection_source_calls_full_complex_channel_and_inverse(
    kind: str,
) -> None:
    foreign, dtype = storage(kind)
    width = int(foreign.num_qubits)
    maximum = (1 << width) - 1
    indices, alternates, keep = (
        np.array([0, maximum]),
        np.array([maximum, 0]),
        np.array([1, 0]),
    )
    source = SparseStatePreparationAliasSampling(
        SourceRegister("selection", foreign), indices, alternates, keep, 1, 2
    )
    arguments: dict[str, ResolvedArgument] = {
        "selection_registers": (Register("selection", dtype),),
        "index": indices,
        "alt": alternates,
        "keep": keep,
        "mu": 1,
        "sum_of_unnormalized_probabilities": 2,
    }
    factories = alias_preparation_factories()
    program = factories[_SPARSE]("main", None, arguments)
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    assert _native_trace(program) == _source_trace(SourceCalls, source.decompose_bloq())
    signature = program.functions[0].registers
    assert signature is not None and signature[0].dtype == dtype
    positions = []
    offset = 0
    for register in signature:
        length = int(register.total_bits())
        positions.append(tuple(range(offset, offset + length)))
        offset += length
    ports = tuple(positions)
    logical = offset
    # Existing two-support-row QROM/comparison owners need two clean bits.
    clean = (logical + 1, logical + 2)
    bound = bind_register_program(program, ports, clean=clean)
    raw = factories[_SPARSE](
        "main",
        None,
        {
            **arguments,
            "selection_registers": (
                Register("selection", unsigned_register_type(width)),
            ),
        },
    )
    assert isinstance(raw, ProgramRecord)
    accepted = bind_register_program(raw, ports, clean=clean)
    assert bound.circuit == accepted.circuit and bound.events == accepted.events
    assert bound.outputs == ports
    if kind != "montgomery" and not kind.startswith("bit_"):
        # Field and Montgomery codecs have separately retained source limits.
        for depth in (0, 1):
            restored = resolve_component_program(
                _library(source, depth), 0, alias_preparation_factories()
            )
            restored_bound = bind_register_program(restored, ports, clean=clean)
            assert (
                restored_bound.circuit == bound.circuit
                and restored_bound.outputs == ports
            )
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    contiguous = bind_register_program(program, ports, clean=(logical, logical + 1))
    torch.testing.assert_close(
        logical_operator(contiguous.circuit, logical), matrix, atol=1e-11, rtol=0
    )
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "inverse", None, {"subbloq": program.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord) and verify_program_record(inverse) == ()
    backward = bind_register_program(inverse, ports, entry="inverse", clean=clean)
    assert _native_trace(inverse, "inverse") == _source_trace(
        SourceCalls, source.adjoint().decompose_bloq()
    )
    raw_inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "raw_inverse", None, {"subbloq": raw.functions[0]}
    )
    assert isinstance(raw_inverse, ProgramRecord)
    raw_backward = bind_register_program(
        raw_inverse, ports, entry="raw_inverse", clean=clean
    )
    assert (
        backward.circuit == raw_backward.circuit
        and backward.events == raw_backward.events
    )
    assert backward.outputs == raw_backward.outputs == ports
    values = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, 1 - 2 * values)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 3), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[::4] = amplitudes
    expected[::4] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for circuit, start, target in (
        (bound.circuit, initial, expected),
        (backward.circuit, expected, initial),
    ):
        appended = (
            DynamicCircuit(logical + 3)
            .append(FieldOperation(logical + 3, circuit.instructions))
            .to_ir()
        )
        assert appended.instructions == circuit.instructions
        assert CircuitIR.from_json(appended.to_json()) == appended
        count = sum(gate.name == "measure" for gate in circuit.instructions)
        for history in product((0, 1), repeat=count):
            actual = forced_branch(start, circuit.instructions, history)
            torch.testing.assert_close(
                actual, target / 2 ** (count / 2), atol=1e-11, rtol=0
            )
            assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0
    raw_source = SparseStatePreparationAliasSampling(
        SourceRegister(
            "selection", source.qrom_bloq.signature.get_left("target0_").dtype
        ),
        indices,
        alternates,
        keep,
        1,
        2,
    )
    assert get_cost_value(source, QECGatesCost()) == get_cost_value(
        raw_source, QECGatesCost()
    )
    assert get_cost_value(source, QubitCount()) == get_cost_value(
        raw_source, QubitCount()
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["measure"] == counts["reset"] == 1


def test_noncompatible_word_and_hardcoded_selection_name_still_refuse() -> None:
    factory = alias_preparation_factories()[_SPARSE]
    for dtype, name in (
        (fixed_point_register_type(2, 1), "selection"),
        (fixed_point_register_type(2, 0, signed=True), "selection"),
        (ones_complement_register_type(2), "selection"),
        (FiniteFieldEncoding(3, 1).register_type(), "selection"),
        (signed_register_type(2), "renamed"),
    ):
        with pytest.raises(SerializationError, match="source selection name"):
            factory(
                "main",
                None,
                {
                    "selection_registers": (Register(name, dtype),),
                    "index": (0, 3),
                    "alt": (3, 0),
                    "keep": (1, 0),
                    "mu": 1,
                    "sum_of_unnormalized_probabilities": 2,
                },
            )


def test_nominal_constructor_append_is_source_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError("Source SDK imported during native alias execution")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    program = alias_preparation_factories()[_SPARSE](
        "main",
        None,
        {
            "selection_registers": (Register("selection", signed_register_type(2)),),
            "index": (0, 3),
            "alt": (3, 0),
            "keep": (1, 0),
            "mu": 1,
            "sum_of_unnormalized_probabilities": 2,
        },
    )
    assert isinstance(program, ProgramRecord)
    bound = bind_register_program(
        program, ((0, 1), (2,), (3,), (4, 5), (6,), (7,)), clean=(8, 9)
    )
    circuit = DynamicCircuit(10).append(FieldOperation(10, bound.circuit.instructions))
    assert circuit.to_ir().instructions == bound.circuit.instructions


@pytest.mark.parametrize("sparse", (False, True))
@pytest.mark.parametrize(
    "mu", (sympy.Integer(1), sympy.Symbol("mu", positive=True, integer=True))
)
def test_known_data_symbolic_word_width_retains_source_root_and_refusal(
    sparse: bool, mu: sympy.Expr
) -> None:
    alternate = np.array([3, 0] if sparse else [1, 0, 3, 2])
    keep = np.array([1, 0] if sparse else [1, 0, 1, 0])
    foreign = SourceRegister("selection", BQUInt(2, 4))
    arguments: dict[str, ResolvedArgument] = {
        "selection_registers": (
            Register("selection", bounded_unsigned_register_type(2, 4)),
        ),
        "alt": alternate,
        "keep": keep,
        "mu": mu,
        "sum_of_unnormalized_probabilities": 2,
    }
    source: SparseStatePreparationAliasSampling | StatePreparationAliasSampling
    if sparse:
        indices = np.array([0, 3])
        source = SparseStatePreparationAliasSampling(
            foreign, indices, alternate, keep, mu, 2
        )
        arguments["index"] = indices
    else:
        source = StatePreparationAliasSampling(foreign, alternate, keep, mu, 2)
    factory_key = _SPARSE if sparse else _SPARSE.replace("SparseState", "State")
    factories = alias_preparation_factories()
    record = factories[factory_key]("main", None, arguments)
    assert isinstance(record, ProgramRecord) and verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    restored = resolve_component_program(
        _library(source, 0), 0, alias_preparation_factories()
    )
    assert verify_program_record(restored) == ()
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "inverse", None, {"subbloq": record.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord) and verify_program_record(inverse) == ()
    assert _native_trace(inverse, "inverse") == _source_trace(
        SourceCalls, source.adjoint().decompose_bloq()
    )
    with pytest.raises(NotImplementedError, match="does not support tensor simulation"):
        source.tensor_contract()
    ports = (
        ((0, 1), (2,), (3,), (4, 5), (6,), (7,))
        if sparse
        else ((0, 1), (2,), (3, 4), (5,), (6,))
    )
    for program, entry in ((record, "main"), (inverse, "inverse")):
        with pytest.raises(ValueError):
            bind_register_program(program, ports, entry=entry, clean=(8, 9, 10))
