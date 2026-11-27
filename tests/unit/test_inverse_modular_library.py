"""Kaliski source graphs retain symbols, source adjoints and every endpoint."""

import sys
from itertools import accumulate

import pytest
import sympy

from flagquantum.compiler.formats.inverse_modular_library import (
    InverseGraphKind,
    inverse_modular_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_inverse_factories import (
    modular_inverse_factories,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from tests.unit.register_library_checks import bind_quantum_library
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_modular_inverse_factories import _direct
from tests.unit.test_symbolic_modular_library import _canonical

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_N, _P = sympy.symbols("n p", positive=True, integer=True)
_CASES: tuple[tuple[InverseGraphKind, SymbolicInt, SymbolicInt, bool], ...] = (
    *(
        (kind, n, 5, False)
        for kind in (
            "_KaliskiIterationStep3",
            "_KaliskiIterationStep4",
            "_KaliskiIterationStep5",
        )
        for n in (_N, 3, 4)
    ),
    *(
        ("_KaliskiIteration", n, p, False)
        for n, p in ((_N, 5), (_N, _P), (3, _P), (4, _P), (3, 5))
    ),
    *(("_KaliskiModInverseImpl", n, p, False) for n, p in ((3, _P), (4, _P), (3, 5))),
    *(
        ("KaliskiModInverse", n, p, inverse)
        for n, p in ((3, _P), (4, _P), (3, 5))
        for inverse in (False, True)
    ),
)


@pytest.mark.parametrize("kind,width,modulus,inverse", _CASES)
def test_actual_source_libraries_match_all_calls_arguments_and_edges(
    kind: InverseGraphKind,
    width: SymbolicInt,
    modulus: SymbolicInt,
    inverse: bool,
) -> None:
    from qualtran.bloqs.mod_arithmetic import mod_division

    arguments = {"bitsize": width}
    if not kind.startswith("_KaliskiIterationStep"):
        arguments["mod"] = modulus
    source = getattr(mod_division, kind)(
        **arguments, **({"uncompute": inverse} if kind == "KaliskiModInverse" else {})
    )
    library = inverse_modular_component_library(kind, width, modulus, uncompute=inverse)
    assert _canonical(library) == _canonical(_library(source, 1))
    assert not library.entries[0].counts


@pytest.mark.parametrize(
    "kind,inverse",
    (
        ("_KaliskiIterationStep3", False),
        ("_KaliskiIterationStep4", False),
        ("_KaliskiIterationStep5", False),
        ("_KaliskiIteration", False),
        ("_KaliskiModInverseImpl", False),
        ("KaliskiModInverse", False),
        ("KaliskiModInverse", True),
    ),
)
def test_native_graphs_feed_existing_exact_owners_with_sdk_imports_blocked(
    kind: InverseGraphKind,
    inverse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = _direct(kind, inverse)
    registers = tuple(Signature(direct.functions[0].registers or ()).lefts())
    widths = [int(r.bitsize) for r in registers]
    ends = tuple(accumulate(widths, initial=0))
    inputs = tuple(tuple(range(a, b)) for a, b in zip(ends[:-1], ends[1:], strict=True))
    n = ends[-1]
    expected = bind_quantum_library(direct, inputs, n)
    for name in ("qualtran", "cirq", "kickmix"):
        monkeypatch.setitem(sys.modules, name, None)
    library = inverse_modular_component_library(kind, 3, 5, uncompute=inverse)
    record = resolve_component_program(library, 0, modular_inverse_factories())
    actual = bind_quantum_library(record, inputs, n)
    assert actual.circuit == expected.circuit and actual.outputs == expected.outputs
    assert tuple(
        (e.kind, e.wires, e.instruction_offset) for e in actual.events
    ) == tuple((e.kind, e.wires, e.instruction_offset) for e in expected.events)
    symbolic = inverse_modular_component_library(
        kind,
        _N if kind.startswith("_KaliskiIterationStep") else 3,
        _P,
        uncompute=inverse,
    )
    declaration = resolve_component_program(symbolic, 0, modular_inverse_factories())
    with pytest.raises(ValueError, match="component bodies"):
        bind_quantum_library(declaration, inputs, n)
