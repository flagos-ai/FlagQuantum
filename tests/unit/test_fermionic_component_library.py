"""Stored fermionic Fourier roots retain literal gates and absolute phase."""

from collections import Counter
from itertools import product

import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.fermionic_fourier import (
    two_bit_fermionic_fourier_program,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.ecosystem._fermionic_component_library import (
    fermionic_library_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "k,n,adjoint", tuple(product((-2, 0, 1, 2), (-3, 1, 4), (False, True)))
)
def test_fermionic_constructor_exact_decomposition(
    k: int, n: int, adjoint: bool
) -> None:
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

    source = TwoBitFFFT(k, n, is_adjoint=adjoint)
    record = resolve_component_program(
        _library(source, 0), 0, fermionic_library_factories()
    )
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, fermionic_library_factories()
            )
            == record
        )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(record, ((1,), (0,)))
    accepted = bind_register_program(
        two_bit_fermionic_fourier_program(k, n), ((1,), (0,))
    )
    assert bound.circuit == accepted.circuit
    assert bound.events == ()
    assert bound.outputs == ((1,), (0,))
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "rz": 1,
        "s": 3,
        "h": 6,
        "cx": 3,
        "tdg": 1,
        "t": 1,
    }
    # The source graph ignores its direction flag and retains the scalar phase.
    contiguous = bind_register_program(record, ((0,), (1,)))
    torch.testing.assert_close(
        logical_operator(contiguous.circuit, 2),
        torch.tensor(source.decompose_bloq().tensor_contract(), dtype=torch.complex128),
        atol=3e-12,
        rtol=0,
    )


def test_fermionic_defaults_example_symbolic_precision_and_validation() -> None:
    from qualtran.bloqs.qft.two_bit_ffft import _two_bit_ffft

    source = _two_bit_ffft.make()
    record = resolve_component_program(
        _library(source, 0), 0, fermionic_library_factories()
    )
    factory = next(iter(fermionic_library_factories().values()))
    assert factory("main", None, {"k": 2, "n": 3}) == record
    assert factory("main", None, {"k": 2, "n": 3, "eps": sympy.Symbol("eps")}) == record
    invalid: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"k": 2, "n": 0},
        {"k": True, "n": 3},
        {"k": 2, "n": 3, "eps": True},
        {"k": 2, "n": 3, "is_adjoint": 2},
        {"k": 2, "n": 3, "extra": 1},
    )
    for arguments in invalid:
        with pytest.raises(SerializationError):
            factory("bad", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"k": 2, "n": 3})


@pytest.mark.parametrize("symbolic_denominator", (False, True))
def test_fermionic_symbolic_rotation_reuses_existing_parameter_binding(
    symbolic_denominator: bool,
) -> None:
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.simulation.unitary import get_unitary

    theta = sympy.Symbol("theta")
    k, n = (1, theta) if symbolic_denominator else (theta, 3)
    source = TwoBitFFFT(k, n, eps=sympy.Symbol("eps"))
    record = resolve_component_program(
        _library(source, 0), 0, fermionic_library_factories()
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(record, ((0,), (1,)))
    circuit = Circuit(2, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    numeric = 2.0
    expected_source = (
        TwoBitFFFT(1, numeric) if symbolic_denominator else TwoBitFFFT(numeric, 3)
    )
    torch.testing.assert_close(
        get_unitary(circuit.bind_parameters({"theta": numeric})),
        torch.tensor(
            expected_source.decompose_bloq().tensor_contract(), dtype=torch.complex128
        ),
        atol=3e-12,
        rtol=0,
    )
