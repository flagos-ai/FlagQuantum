"""Preserve complete logical top-level source paths and their execution limits."""

from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import sympy

from flagquantum.algorithms.first_quantization_preparation_formula import (
    source_first_quantization_prepare_formula,
    source_first_quantization_prepare_gate_cost,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._first_quantization_component_library import (
    first_quantization_component_factories,
)
from flagquantum.ecosystem._projectile_component_library import (
    projectile_component_factories,
)
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_momentum_source_declarations import frozen_source as frozen_source
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."
N = sympy.Symbol("n", positive=True, integer=True)


@pytest.fixture(scope="module", autouse=True)
def root_sources(frozen_source: None) -> None:
    ordinary = import_module(BASE + "select_and_prepare")
    assert ordinary.__file__ is not None
    root = Path(ordinary.__file__).parent
    for path, blob in (
        ("select_and_prepare.py", "5f5001c60acedf729fc4c4122431494047fb2f2e"),
        (
            "projectile/select_and_prepare.py",
            "dde8e7eb6c1802cf71d67a3338d5cc4214e6e963",
        ),
    ):
        data = (root / path).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob


def source(projectile: bool, select: bool, width: Any) -> Any:
    module = import_module(
        BASE + ("projectile." if projectile else "") + "select_and_prepare"
    )
    name = (
        ("Select" if select else "Prepare")
        + "FirstQuantization"
        + ("WithProj" if projectile else "")
    )
    fields = dict(num_bits_p=width, eta=2, num_atoms=3, lambda_zeta=3, m_param=4)
    if projectile:
        fields["num_bits_n"] = width + 1
    return getattr(module, name)(**fields)


@pytest.mark.parametrize("projectile", (False, True))
@pytest.mark.parametrize("select", (False, True))
@pytest.mark.parametrize("width", (0, 1, N))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_top_connections_and_original_recursive_boundaries(
    projectile: bool,
    select: bool,
    width: Any,
    adjoint: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran import BloqError

    original = source(projectile, select, width)
    if adjoint:
        original = original.adjoint()
    owners = (
        projectile_component_factories
        if projectile
        else first_quantization_component_factories
    )
    library = _library(original, 0)
    if not projectile and not select and width == 0:
        with pytest.raises(BloqError, match="does not accept Soquets"):
            original.decompose_bloq()
        result = resolve_component_program(library, 0, owners())
        assert not result.functions[0].blocks
        return
    expected = _named_inputs(_source_trace(SourceCalls, original.decompose_bloq()))
    import sys

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        result = resolve_component_program(library, 0, owners())
        assert verify_program_record(result) == ()
        assert _named_inputs(_native_trace(result)) == expected
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(result, (), clean=tuple(range(128)))
    if select and width in (0, N):
        with pytest.raises(
            (ValueError, BloqError), match="symbolic data type|not an available Soquet"
        ):
            _library(original, 1)
    else:
        deeper = resolve_component_program(_library(original, 1), 0, owners())
        assert _named_inputs(_native_trace(deeper)) == expected


@pytest.mark.parametrize("projectile", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("width", (1, 2, N, N + 1))
def test_complete_prepare_symbolic_models_keep_source_failures_separate(
    projectile: bool, adjoint: bool, width: Any
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = source(projectile, False, width)
    if adjoint:
        original = original.adjoint()
    options = {"num_bits_n": width + 1} if projectile else {}
    gates = source_first_quantization_prepare_gate_cost(
        width, 2, 3, 3, m_param=4, adjoint=adjoint, **options
    )
    expected = get_cost_value(original, QECGatesCost()).asdict()
    for key, value in gates.asdict().items():
        assert sympy.simplify(value - expected[key]) == 0
    if projectile and adjoint:
        with pytest.raises(RuntimeError):
            get_cost_value(original, QubitCount())
        with pytest.raises(ValueError, match="no valid peak width"):
            source_first_quantization_prepare_formula(
                width, 2, 3, 3, m_param=4, adjoint=adjoint, **options
            )
    else:
        native = source_first_quantization_prepare_formula(
            width, 2, 3, 3, m_param=4, adjoint=adjoint, **options
        )
        difference = sympy.simplify(
            native.logical_qubits - get_cost_value(original, QubitCount())
        )
        assert difference == 0


@pytest.mark.parametrize("projectile", (False, True))
def test_symbolic_precision_metadata_threads_the_actual_prepare_chain(
    projectile: bool,
) -> None:
    from attrs import evolve
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = evolve(source(projectile, False, 2), num_bits_rot_aa=N, num_bits_t=N + 1)
    owners = (
        projectile_component_factories
        if projectile
        else first_quantization_component_factories
    )
    result = resolve_component_program(_library(original, 0), 0, owners())
    assert _named_inputs(_native_trace(result)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq())
    )
    native = source_first_quantization_prepare_gate_cost(
        2,
        2,
        3,
        3,
        m_param=4,
        num_bits_rot_aa=N,
        num_bits_t=N + 1,
        num_bits_n=3 if projectile else None,
    )
    expected = get_cost_value(original, QECGatesCost()).asdict()
    for key, value in native.asdict().items():
        assert sympy.simplify(value - expected[key]) == 0
