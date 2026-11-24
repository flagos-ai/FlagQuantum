"""Resolve original Phase constructors, scalars and explicit positive child pairs."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_phased_block_factories import (
    phased_block_factories,
)
from flagquantum.ecosystem._default_linear_combination_oracles import _positive_pair
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_phase_source() -> None:
    import kickmix
    from qualtran.bloqs.block_encoding import phase

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("phase.py", "d7e6b03237f63238e511a4e7aa5056808a646593"),
        ("phase_test.py", "c41738844b4eb08517b4c3245d7cc47f1d6bdf1b"),
    ):
        raw = (Path(phase.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def scalar_terminals(record: Any) -> set[str]:
    return {function.name for function in record.functions if function.registers == ()}


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_factory_recursive_calls_full_phase_and_work(
    depth: int, inverse: bool
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.phase import _phase_block_encoding

    source = _phase_block_encoding.make()
    if inverse:
        source = Adjoint(source)
    if depth == 2:
        with pytest.raises(ValueError, match="inhomogeneous shape"):
            _library(source, depth)
        return
    record = resolve_component_program(
        _library(source, depth), 0, phased_block_factories()
    )
    compare_layers(record, source, atomic_leaves(record) | scalar_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


def test_nested_and_auxiliary_phase_exact_inverse() -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary

    forward = Phase(
        Phase(
            Unitary(TGate(), ancilla_bitsize=1, resource_bitsize=1), phi=0.125, eps=1e-8
        ),
        phi=-0.375,
        eps=2e-8,
    )
    for source in (forward, Adjoint(forward)):
        record = resolve_component_program(
            _library(source, 0), 0, phased_block_factories()
        )
        compare_layers(record, source, atomic_leaves(record) | scalar_terminals(record))
        bound, logical = bind(record)
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical),
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=1e-10,
            rtol=0,
        )


def test_symbolic_nonlinear_phase_and_sdk_free_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import sympy
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    theta = sympy.Symbol("theta", real=True)
    source = Phase(Unitary(TGate()), phi=sympy.sin(theta), eps=0)
    library = _library(source, 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(library, 0, phased_block_factories())
        bound, logical = bind(record)
        circuit = Circuit(bound.circuit.n_wires, dtype=torch.complex128)
        circuit.h(0)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        actual = circuit.bind_parameters({"theta": 0.37}).state().flatten()
    expected = torch.zeros_like(actual)
    stride = 1 << (bound.circuit.n_wires - logical)
    from cmath import exp
    from math import pi, sin, sqrt

    expected[0] = torch.tensor(exp(1j * pi * sin(0.37)), dtype=torch.complex128) / sqrt(
        2
    )
    expected[stride] = expected[0] * exp(1j * pi / 4)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def test_explicit_measured_child_replaces_constructor_dependency() -> None:
    from qualtran.bloqs.block_encoding.phase import Phase

    from flagquantum.compiler.formats.legacy_linear_combination_factories import (
        linear_combination_factories,
    )
    from flagquantum.ecosystem._default_linear_combination_oracles import (
        default_linear_combination_prepare_pair,
        default_linear_combination_select_pair,
    )
    from flagquantum.ecosystem._linear_combination_component_library import (
        capture_linear_combination_context,
    )
    from tests.unit.test_default_linear_combination_oracles import default_source

    child = default_source()
    context = capture_linear_combination_context(
        child,
        prepare_owner=default_linear_combination_prepare_pair,
        select_owner=default_linear_combination_select_pair,
    )
    source = Phase(child, phi=-0.125, eps=0)
    library = _library(source, 0)
    entry = next(
        e for e in library.entries if e.declaration.name.endswith(".LinearCombination")
    )
    phases = phased_block_factories(
        child_owner=lambda _: (context.program(), context.program(adjoint=True))
    )
    children = linear_combination_factories(
        {f"legacy_component_{entry.identifier}": context},
        adjoint_factory=phases["qualtran._infra.adjoint.Adjoint"],
    )
    record = resolve_component_program(library, 0, {**phases, **children})
    compare_layers(record, source, atomic_leaves(record) | scalar_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


def test_phase_constructor_rejects_missing_native_owner_and_invalid_payload() -> None:
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.errors import SerializationError

    child = FunctionRecord("unknown", (), registers=tuple(Signature.build(system=1)))
    factory = phased_block_factories()["qualtran.bloqs.block_encoding.phase.Phase"]
    with pytest.raises(SerializationError, match="child, phi and eps"):
        factory("main", None, {"block_encoding": child, "phi": 0.25})
    with pytest.raises(SerializationError, match="Missing exact"):
        factory("main", None, {"block_encoding": child, "phi": 0.25, "eps": 0})


@pytest.mark.parametrize("pattern", ((1,), (0,), (1, 0), (1, 0, 1)))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_controlled_phase_and_predicate_inverse(
    pattern: tuple[int, ...], inverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.phase import _phase_block_encoding

    phase = _phase_block_encoding.make()
    positive = _positive_pair(phase.block_encoding)
    source = phase.controlled(
        CtrlSpec(cvs=pattern[0] if len(pattern) == 1 else pattern)
    )
    if inverse:
        source = Adjoint(source)
    record = resolve_component_program(
        _library(source, 0),
        0,
        phased_block_factories(positive_child_owner=lambda _: positive),
    )
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )
